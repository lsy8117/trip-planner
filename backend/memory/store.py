"""
Filesystem access for the long-term memory system.

Two kinds of state, deliberately kept apart:

  backend/sessions/<user_id>/<session_id>/
      memory_events.jsonl   # append-only raw evidence for this session —
                             # user messages + generated itineraries (see
                             # backend/memory/capture.py for what writes it)
      memory_state.json     # SessionMemoryState — a watermark: how far the
                             # memory updater has already read into this
                             # session's event log

  backend/memory/data/<user_id>.json
      UserMemory — the distilled long-term preferences for this user,
      rebuilt by backend/memory/updater.py from events across all of their
      sessions.

memory_events.jsonl/memory_state.json live under the session folder (not
here) because they're per-conversation; UserMemory lives here because it's
per-user and spans every session. session_dir() is reused from
session_store.py rather than duplicated.
"""
from pathlib import Path

from backend.config import settings
from backend.session_store import session_dir
from backend.models import MemoryEvent, SessionMemoryState, UserMemory


def memory_events_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "memory_events.jsonl"


def session_memory_state_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "memory_state.json"


def user_memory_path(user_id: str) -> Path:
    return settings.memory_dir / f"{user_id}.json"


# ── session event log ──────────────────────────────────────────────────

def append_memory_event(user_id: str, session_id: str, event: MemoryEvent) -> None:
    path = memory_events_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")


def read_memory_events(user_id: str, session_id: str, since_ts: float = 0.0) -> list[MemoryEvent]:
    """Events strictly after `since_ts`. Pass the default (0.0) to read a
    session's whole event log, e.g. when it has no watermark yet."""
    path = memory_events_path(user_id, session_id)
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = MemoryEvent.model_validate_json(line)
        if event.ts > since_ts:
            events.append(event)
    return events


# ── per-session memory watermark ───────────────────────────────────────

def read_session_memory_state(user_id: str, session_id: str) -> SessionMemoryState:
    path = session_memory_state_path(user_id, session_id)
    if not path.exists():
        return SessionMemoryState()  # last_memory_sync_at=0.0 — never synced
    return SessionMemoryState.model_validate_json(path.read_text())


def write_session_memory_state(user_id: str, session_id: str, state: SessionMemoryState) -> None:
    path = session_memory_state_path(user_id, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json())


# ── per-user long-term memory ──────────────────────────────────────────

def read_user_memory(user_id: str) -> UserMemory:
    path = user_memory_path(user_id)
    if not path.exists():
        return UserMemory(user_id=user_id)
    return UserMemory.model_validate_json(path.read_text())


def write_user_memory(user_id: str, memory: UserMemory) -> None:
    path = user_memory_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory.model_dump_json(indent=2))
