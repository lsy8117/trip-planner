# backend/memory/updater.py
"""
Manually-triggered pipeline: reads new evidence (generated itineraries +
the user's own messages) across a user's sessions and refreshes their
long-term UserMemory. See backend/memory/capture.py for what writes that
evidence, and backend/memory/store.py for where everything lives.
"""
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend import session_store
from backend.memory import store
from backend.memory.condense import condense_itinerary
from backend.memory.prompts import MEMORY_UPDATE_SYSTEM_PROMPT
from backend.models import MemoryEvent, MemoryEventKind, MemoryPreferences, SessionMemoryState, UserMemory
from backend.utilities import get_llm
from backend.config import settings


def _format_session_transcript(session_id: str, events: list[MemoryEvent]) -> str:
    lines = [f"### Session {session_id}"]
    for event in events:
        if event.kind == MemoryEventKind.USER_MESSAGE:
            lines.append(f"User: {event.text}")
        elif event.kind == MemoryEventKind.ITINERARY_GENERATED and event.itinerary is not None:
            lines.append("Itinerary generated:\n" + condense_itinerary(event.itinerary))
    return "\n".join(lines)


async def update_user_memory(user_id: str, session_ids: Optional[list[str]] = None) -> UserMemory:
    """Re-analyze new conversation evidence and refresh the user's long-term
    memory.

    Defaults to scanning every session this user has, each read from its own
    watermark (so a session created after the last memory update is still
    picked up in full, with no special-casing needed). Pass `session_ids` to
    restrict the run to specific sessions instead — e.g. to manually pick
    which conversation(s) to analyze, or to re-run a single one in
    isolation. Sessions actually read (whether auto-selected or explicitly
    passed) have their watermark advanced; sessions you don't include here
    are left untouched and will still be picked up by a future run.
    """
    target_session_ids = session_ids or [meta.session_id for meta in session_store.list_sessions(user_id)]

    existing_memory = store.read_user_memory(user_id)

    session_transcripts: list[str] = []
    new_watermarks: dict[str, float] = {}

    for session_id in target_session_ids:
        state = store.read_session_memory_state(user_id, session_id)
        events = store.read_memory_events(user_id, session_id, since_ts=state.last_memory_sync_at)
        if not events:
            continue
        session_transcripts.append(_format_session_transcript(session_id, events))
        new_watermarks[session_id] = max(event.ts for event in events)

    if not session_transcripts:
        return existing_memory  # nothing new anywhere — leave memory as-is

    llm = get_llm(provider=settings.memory_update_model_provider, model=settings.memory_update_model) \
        .with_structured_output(MemoryPreferences, method="json_schema") \
        .with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 1.0},
        )

    messages = [
        SystemMessage(content=MEMORY_UPDATE_SYSTEM_PROMPT),
        SystemMessage(content="Existing preference memory:\n" + existing_memory.model_dump_json(
            indent=2, include={"attraction_preferences", "hotel_preferences", "itinerary_preferences"},
        )),
        HumanMessage(content="\n\n".join(session_transcripts)),
    ]

    result: MemoryPreferences = await llm.ainvoke(messages)

    updated_memory = UserMemory(
        user_id=user_id,
        attraction_preferences=result.attraction_preferences,
        hotel_preferences=result.hotel_preferences,
        itinerary_preferences=result.itinerary_preferences,
        last_updated=time.time(),
    )
    store.write_user_memory(user_id, updated_memory)

    for session_id, watermark in new_watermarks.items():
        store.write_session_memory_state(user_id, session_id, SessionMemoryState(last_memory_sync_at=watermark))

    return updated_memory
