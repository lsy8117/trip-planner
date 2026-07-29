"""
Called from graph nodes to append the two kinds of evidence the memory
updater analyzes — user messages and generated itineraries — to a session's
memory_events.jsonl (see backend/memory/store.py).

Identity (user_id, session_id) comes from the LangGraph RunnableConfig
passed into each node, not from OrchestratorState — it's run-time identity,
not conversation content. Both lookups no-op silently if either id is
missing, which happens for any invocation that doesn't set
config["configurable"]["user_id"] (e.g. the test suite, or ad-hoc scripts
like test_multiround.py) — event capture is best-effort instrumentation and
must never break a graph turn.
"""
import time
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from backend.memory import store
from backend.models import MemoryEvent, MemoryEventKind, OrchestratorState


def _ids_from_config(config: Optional[RunnableConfig]) -> tuple[Optional[str], Optional[str]]:
    configurable = (config or {}).get("configurable", {})
    return configurable.get("user_id"), configurable.get("thread_id")


def log_user_message(state: OrchestratorState, config: Optional[RunnableConfig]) -> None:
    """Call at the top of classify_request — by the time it runs, on both a
    fresh turn and a resume-from-interrupt, the latest HumanMessage is
    already the last item in state['messages']."""
    user_id, session_id = _ids_from_config(config)
    if not user_id or not session_id:
        return

    messages = state.get("messages") or []
    if not messages or not isinstance(messages[-1], HumanMessage):
        return

    store.append_memory_event(
        user_id, session_id,
        MemoryEvent(kind=MemoryEventKind.USER_MESSAGE, ts=time.time(), text=messages[-1].content),
    )


def log_itinerary_generated(state: OrchestratorState, config: Optional[RunnableConfig]) -> None:
    """Call in respond(), right before interrupt() — state['current_itinerary']
    is the itinerary about to be shown to the user."""
    user_id, session_id = _ids_from_config(config)
    itinerary = state.get("current_itinerary")
    if not user_id or not session_id or itinerary is None:
        return

    store.append_memory_event(
        user_id, session_id,
        MemoryEvent(kind=MemoryEventKind.ITINERARY_GENERATED, ts=time.time(), itinerary=itinerary),
    )
