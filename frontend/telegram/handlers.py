"""
Telegram update handlers.

Each chat_id is a user identity; a user may have several persisted
conversations ("sessions") under backend/sessions/<user_id>/, but only one
is "active" at a time (see backend/session_store.py — /new starts another,
/sessions switches between them). Because `respond()` pauses the graph with
`interrupt()` between turns, every message after the first *within a
session* is normally a *resume*, not a fresh invocation — told apart by
checking `aget_state(config).interrupts`: non-empty means the graph is
genuinely parked at respond()'s interrupt(), waiting for this session's
next input.

A turn can also crash mid-execution (an unhandled exception, or the process
being killed) *before* reaching that interrupt(). LangGraph only checkpoints
completed supersteps, so the next `aget_state` in that case comes back with
pending `.tasks` but no `.interrupts` — the graph is not paused on purpose,
it just never finished. Blindly resuming there would silently continue a
half-finished internal step, so that case is instead handled by rolling back
to the last checkpoint that *does* have a real interrupt (see
`_find_last_interrupted_state`) and asking the user whether to retry the
request that crashed, rather than resuming automatically.
"""

import logging
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes
from langchain_core.messages import HumanMessage
from langgraph.types import Command, StateSnapshot

from backend import session_store
from backend.graph.session_graph import get_session_graph, is_cached
from backend.memory.updater import update_user_memory
from backend.models import UserMemory

from frontend.telegram.formatter import format_for_telegram
from frontend.telegram.stream_runner import run_turn, send_long_message

logger = logging.getLogger(__name__)

_TITLE_MAX_LEN = 40

_AFFIRMATIVE_REPLIES = {
    "yes", "y", "yeah", "yep", "yup", "sure",
    "ok", "okay", "continue", "go ahead", "please continue", "confirm",
}


def _make_title(user_text: str) -> str:
    text = user_text.strip().replace("\n", " ")
    return text if len(text) <= _TITLE_MAX_LEN else text[: _TITLE_MAX_LEN - 1] + "…"


def _fresh_state(user_text: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_text)],
        "user_query": user_text,
        "weather_info": [],
        "attractions": [],
        "hotels": [],
        "errors": [],
        "execution_plan": [],
        "current_intent": None,
        "current_itinerary": None,
        "refinement_request": [],
        "subgraphs_to_run": [],
    }


def _past_requests(values: dict) -> list[str]:
    """Every user ask in this conversation, oldest first: the original
    user_query plus every refinement_request accumulated since (both fields
    are additive across turns — see OrchestratorState in backend/models.py)."""
    requests = [values["user_query"]] if values.get("user_query") else []
    requests.extend(values.get("refinement_request") or [])
    return requests


def _format_history_reminder(values: dict) -> str:
    requests = _past_requests(values)
    if len(requests) <= 1:
        return ""
    bullets = "\n".join(f"{i}. {r}" for i, r in enumerate(requests, start=1))
    return f"📝 *Everything you've asked for so far:*\n{bullets}"


def _format_recap(itinerary, history_reminder: str, closing: str) -> str:
    parts = [format_for_telegram(itinerary) if itinerary is not None
             else "_No itinerary has been generated yet._"]
    if history_reminder:
        parts.append(history_reminder)
    parts.append(closing)
    return "\n\n".join(parts)


async def _find_last_interrupted_state(graph, config: dict) -> Optional[StateSnapshot]:
    """Walk this thread's checkpoint history backward for the most recent
    state genuinely parked at an interrupt() call. Used to recover when the
    latest checkpoint belongs to a turn that crashed mid-execution instead of
    completing — resuming there directly would just retry a broken internal
    step, so we roll back to the last real pause point instead."""
    async for snapshot in graph.aget_state_history(config):
        if snapshot.interrupts:
            return snapshot
    return None


async def _maybe_show_recall_recap(bot, chat_id: int, state_snapshot: StateSnapshot) -> None:
    """Shows the current itinerary + full request history when genuinely
    parked at an interrupt — used both when a session is explicitly switched
    to (handle_session_callback) and when a text message is the first one
    handled after a fresh process-level load (handle_message)."""
    if not state_snapshot.interrupts:
        return
    recap = _format_recap(
        state_snapshot.values.get("current_itinerary"),
        _format_history_reminder(state_snapshot.values),
        "_Here's where you left off — anything you'd like to change?_",
    )
    await send_long_message(bot, chat_id, recap)


async def _build_retry_input(
    graph, config: dict, user_id: str, retry_text: str
) -> tuple[Command | dict, dict]:
    """Build the (graph_input, resume_config) pair to retry a crashed turn:
    resume from the last checkpoint that has a real interrupt() if one
    exists (the node that crashed re-runs from scratch — LangGraph only
    checkpoints completed supersteps, not partial node execution), otherwise
    there was no prior interrupt at all and the turn starts fresh."""
    last_good = await _find_last_interrupted_state(graph, config)
    if last_good is not None:
        resume_config = {"configurable": {**last_good.config["configurable"], "user_id": user_id}}
        return Command(resume=retry_text), resume_config
    return _fresh_state(retry_text), config


async def _maybe_ask_about_crash(
    bot, chat_id: int, user_id: str, session_id: str,
    state_snapshot: StateSnapshot, pending_confirmation: bool,
) -> bool:
    """If the last turn crashed mid-execution and we haven't already asked,
    surfaces the stale itinerary + what was being worked on and asks whether
    to retry. Returns True if it just asked — the caller should stop and wait
    for the next message to answer, rather than touching the graph now."""
    if not state_snapshot.tasks or state_snapshot.interrupts or pending_confirmation:
        return False
    past = _past_requests(state_snapshot.values)
    crashed_request = past[-1] if past else "your last request"
    recap = _format_recap(
        state_snapshot.values.get("current_itinerary"),
        _format_history_reminder(state_snapshot.values),
        f'⚠️ _I was interrupted while working on:_ "{crashed_request}"\n\n'
        "_Want me to continue with that request? (yes/no)_",
    )
    await send_long_message(bot, chat_id, recap)
    session_store.set_pending_confirmation(user_id, session_id, True)
    return True


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "✈️ Hi! Tell me about the trip you're planning — destination, dates, "
        "and anything you care about — and I'll put together an itinerary. "
        "You can keep refining it after I reply.\n\n"
        "/new starts a fresh conversation, /sessions lists your saved ones, "
        "/resume retries if I got interrupted mid-turn, /update_memory refreshes "
        "what I remember about your preferences."
    )


async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_chat.id)
    meta = session_store.create_session(user_id)
    session_store.set_active_session_id(user_id, meta.session_id)
    await update.message.reply_text("Started a new conversation. Tell me about the trip!")


async def handle_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_chat.id)
    sessions = session_store.list_sessions(user_id)
    if not sessions:
        await update.message.reply_text("You don't have any saved conversations yet.")
        return

    active_id = session_store.get_active_session_id(user_id)
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅ ' if meta.session_id == active_id else ''}{meta.title}",
            callback_data=f"switch_session:{meta.session_id}",
        )]
        for meta in sessions
    ]
    await update.message.reply_text(
        "Your saved conversations — tap one to switch to it:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_session_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = str(update.effective_chat.id)
    session_id = query.data.split(":", 1)[1]
    chat_id = update.effective_chat.id

    meta = session_store.get_session(user_id, session_id)
    if meta is None:
        await query.edit_message_text("That conversation no longer exists.")
        return

    session_store.set_active_session_id(user_id, session_id)
    await query.edit_message_text(f'Switched to "{meta.title}".')

    # Explicitly switching to a session is always a "recall" — surface
    # whatever state it's in, same as handle_message does on a fresh load.
    graph = await get_session_graph(user_id, session_id)
    config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
    state_snapshot = await graph.aget_state(config)

    await _maybe_show_recall_recap(context.bot, chat_id, state_snapshot)
    await _maybe_ask_about_crash(
        context.bot, chat_id, user_id, session_id, state_snapshot, meta.pending_confirmation
    )

    if not state_snapshot.interrupts and not state_snapshot.tasks:
        await context.bot.send_message(chat_id=chat_id, text="Send a message to continue.")


async def handle_update_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_chat.id)
    await update.message.reply_text("Updating your travel preferences from recent conversations...")
    memory = await update_user_memory(user_id)
    await update.message.reply_text(_format_memory_summary(memory))


def _format_memory_summary(memory: UserMemory) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"• {item}" for item in items) if items else "  (none yet)"

    return (
        "Here's what I've learned about your travel preferences:\n\n"
        f"🗺 Attractions:\n{bullets(memory.attraction_preferences)}\n\n"
        f"🏨 Hotels:\n{bullets(memory.hotel_preferences)}\n\n"
        f"📋 Itinerary style:\n{bullets(memory.itinerary_preferences)}"
    )


async def handle_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last-resort catch-all: anything raised by any handler (including an
    unhandled failure inside run_turn's graph execution) lands here via
    python-telegram-bot's error-handler dispatch, rather than failing silently.
    This is a different layer from _maybe_ask_about_crash above — that
    recovers from a *known* mid-execution crash on the *next* turn by reading
    checkpoint state; this reports an exception happening right now that
    nothing downstream caught."""
    logger.exception("Unhandled error while processing update", exc_info=context.error)

    if isinstance(update, Update) and update.effective_chat:
        error_detail = f"{type(context.error).__name__}: {context.error}"
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"⚠️ Something went wrong while processing that:\n`{error_detail}`",
            parse_mode=ParseMode.MARKDOWN,
        )


async def handle_resume(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Explicit alternative to the automatic crash-recovery prompt in
    handle_message: lets the user retry a crashed turn on demand instead of
    waiting to be asked on their next message. Invoking the command is itself
    the confirmation, so — unlike _maybe_ask_about_crash — this skips the
    yes/no round trip and retries immediately."""
    chat_id = update.effective_chat.id
    user_id = str(chat_id)

    session_meta = session_store.get_or_create_active_session(user_id)
    session_id = session_meta.session_id

    graph = await get_session_graph(user_id, session_id)
    config = {"configurable": {"thread_id": session_id, "user_id": user_id}}
    state_snapshot = await graph.aget_state(config)

    if state_snapshot.interrupts or not state_snapshot.tasks:
        await update.message.reply_text(
            "Nothing to resume — the conversation isn't stuck mid-turn."
        )
        return

    past = _past_requests(state_snapshot.values)
    retry_text = past[-1] if past else "continue with my last request"
    graph_input, resume_config = await _build_retry_input(graph, config, user_id, retry_text)

    session_store.set_pending_confirmation(user_id, session_id, False)
    logger.info("chat %s session %s: /resume retrying crashed turn", chat_id, session_id)

    await run_turn(
        graph=graph,
        graph_input=graph_input,
        config=resume_config,
        bot=context.bot,
        chat_id=chat_id,
        log_dir=session_store.session_dir(user_id, session_id),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = str(chat_id)
    user_text = update.message.text

    session_meta = session_store.get_or_create_active_session(user_id)
    session_id = session_meta.session_id

    # Must be checked before get_session_graph, which populates the cache —
    # tells a genuine session recall (process restart, or first message back
    # after eviction) apart from continuing an already-active conversation.
    is_recall = not is_cached(user_id, session_id)
    graph = await get_session_graph(user_id, session_id)

    # thread_id ties this conversation to its persisted checkpoint state;
    # user_id lets graph nodes log memory evidence (see backend/memory/capture.py)
    config = {"configurable": {"thread_id": session_id, "user_id": user_id}}

    state_snapshot = await graph.aget_state(config)

    if state_snapshot.interrupts:
        # Genuinely parked at respond()'s interrupt() — the normal resume point.
        if is_recall:
            await _maybe_show_recall_recap(context.bot, chat_id, state_snapshot)
        graph_input = Command(resume=user_text)
        resume_config = config
        logger.info("chat %s session %s: resuming interrupted graph", chat_id, session_id)

    elif state_snapshot.tasks:
        # Last turn crashed mid-execution instead of reaching interrupt() —
        # never resume that half-finished step directly; confirm first.
        just_asked = await _maybe_ask_about_crash(
            context.bot, chat_id, user_id, session_id,
            state_snapshot, session_meta.pending_confirmation,
        )
        if just_asked:
            logger.info(
                "chat %s session %s: detected an aborted turn, asking to retry",
                chat_id, session_id,
            )
            return  # hold this message; the user's next one is the answer

        # This message is the answer to that question.
        session_store.set_pending_confirmation(user_id, session_id, False)
        past = _past_requests(state_snapshot.values)
        confirmed = user_text.strip().lower() in _AFFIRMATIVE_REPLIES
        retry_text = (past[-1] if past else user_text) if confirmed else user_text

        graph_input, resume_config = await _build_retry_input(graph, config, user_id, retry_text)
        logger.info(
            "chat %s session %s: recovering from an aborted turn (confirmed=%s)",
            chat_id, session_id, confirmed,
        )

    else:
        graph_input = _fresh_state(user_text)
        resume_config = config
        logger.info("chat %s session %s: starting fresh turn", chat_id, session_id)

    # A brand-new session still has its default "New trip" title — replace
    # it with something derived from what the user actually said.
    title = _make_title(user_text) if session_meta.title == "New trip" else None
    session_store.touch_session(user_id, session_id, title=title)

    await run_turn(
        graph=graph,
        graph_input=graph_input,
        config=resume_config,
        bot=context.bot,
        chat_id=chat_id,
        log_dir=session_store.session_dir(user_id, session_id),
    )

