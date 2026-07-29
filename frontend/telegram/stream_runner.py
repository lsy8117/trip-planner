"""
Core bridge between LangGraph's astream_events() and Telegram.

Design note — this differs from a "normal" astream_events consumer because
the graph never reaches a natural END on a turn: `respond()` calls
`interrupt(reply)`, which pauses execution. `astream_events` simply stops
yielding events when that happens (it does not raise). So "the turn is
over" is detected by draining the stream to completion and then reading
`aget_state(config)` — the reliable pattern already validated in this
project's test suite — rather than looking for an on_chain_end/LangGraph
event.

Flow per turn:
  1. Send one "status" message.
  2. As node/tool events stream in, edit that status message in place
     (throttled + "message is not modified" errors swallowed).
  3. Once the stream is exhausted, read graph state:
       - if state.interrupts exists → graph paused at respond();
         format that interrupt value and send it as the final reply.
       - otherwise → graph ran to true END with no interrupt (shouldn't
         normally happen given the current respond.py, but handled so a
         bug there doesn't hang the bot silently).
  4. Delete the status message.
"""

import asyncio
import logging
import time
from pathlib import Path

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError

from frontend.telegram.progress import ProgressTracker
from frontend.telegram.formatter import format_for_telegram
from frontend.telegram.logging import log_graph_event, log_final_state

logger = logging.getLogger(__name__)

TELEGRAM_MAX_LEN = 4096
MIN_EDIT_INTERVAL_S = 1.0  # avoid Telegram edit-rate limits on fast graphs


async def run_turn(graph, graph_input, config: dict, bot: Bot, chat_id: int, log_dir: Path) -> None:
    tracker = ProgressTracker()

    status_message = await bot.send_message(chat_id=chat_id, text=tracker.render())
    last_edit = time.monotonic()

    thread_id = config["configurable"]["thread_id"]

    async for event in graph.astream_events(graph_input, config=config, version="v2"):
        # Full trace (node I/O, tool calls, tool results) → log file only.
        # Independent of what the tracker below shows the user.
        log_graph_event(log_dir, thread_id, event)

        kind = event.get("event")
        name = event.get("name", "")

        if kind == "on_chain_start" and name in _known_node_names(tracker):
            tracker.on_node_start(name)
        elif kind == "on_chain_end" and name in _known_node_names(tracker):
            tracker.on_node_end(name)
        elif kind == "on_tool_start":
            tracker.on_tool_start(name, event.get("data", {}).get("input"))
        else:
            continue

        now = time.monotonic()
        if now - last_edit >= MIN_EDIT_INTERVAL_S:
            await _safe_edit(bot, chat_id, status_message.message_id, tracker.render())
            last_edit = now

    # Stream drained — the graph is either paused at interrupt() or truly done.
    state_snapshot = await graph.aget_state(config)

    # respond()'s interrupt() is an internal exception, not a normal return,
    # so its on_chain_end never arrived above — finalize() checks off
    # whatever was left "in progress" so the permanent message ends clean.
    tracker.finalize()
    await _safe_edit(bot, chat_id, status_message.message_id, tracker.render())

    # Full final state (messages, weather_info, attractions, hotels,
    # current_itinerary, execution_plan) → log file only.
    log_final_state(log_dir, thread_id, state_snapshot.values)

    final_text: str
    if state_snapshot.interrupts:
        interrupt_value = state_snapshot.interrupts[0].value
        final_text = format_for_telegram(interrupt_value)
    elif state_snapshot.values.get("current_itinerary") is not None:
        # Fallback if the graph ended without interrupting
        final_text = format_for_telegram(state_snapshot.values["current_itinerary"])
    else:
        final_text = "Something went wrong and I don't have a result to show — please try again."

    # Progress message is left in place (never deleted) — the itinerary is
    # sent as a new message below it, so the user keeps the full trail.
    await send_long_message(bot, chat_id, final_text)


def _known_node_names(tracker: ProgressTracker) -> set:
    from frontend.telegram.progress import NODE_CONFIG
    return set(NODE_CONFIG.keys())


async def _safe_edit(bot: Bot, chat_id: int, message_id: int, text: str) -> None:
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
        )
    except TelegramError as e:
        if "message is not modified" not in str(e).lower():
            logger.warning("edit_message_text failed: %s", e)


async def send_long_message(bot: Bot, chat_id: int, text: str) -> None:
    """Split and send text exceeding Telegram's 4096-char message limit."""
    if len(text) <= TELEGRAM_MAX_LEN:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
        return

    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        if len(current) + len(line) > TELEGRAM_MAX_LEN:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)

    for chunk in chunks:
        await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
        await asyncio.sleep(0.05)  # gentle pacing between chunks
