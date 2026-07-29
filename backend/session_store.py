"""
Filesystem registry for persisted conversations ("sessions").

Layout, one folder per conversation:

    backend/sessions/<user_id>/<session_id>/
        meta.json          # SessionMeta — title, timestamps
        checkpoints.db      # LangGraph AsyncSqliteSaver db (see graph/session_graph.py)
        agent_runs.jsonl    # per-turn debug log (see frontend/telegram/logging.py)

    backend/sessions/<user_id>/active.json   # {"session_id": "..."} — which
                                              # session this user is currently
                                              # talking to

`user_id` is the frontend-level identity (e.g. Telegram chat_id, as a str).
Nothing here is Telegram-specific — any frontend can reuse this registry.
"""
import json
import time
import uuid
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from backend.config import settings


class SessionMeta(BaseModel):
    session_id: str
    title: str
    created_at: float
    updated_at: float
    # True once we've asked the user whether to retry a turn that crashed
    # mid-execution, until their next message answers it (see handle_message).
    pending_confirmation: bool = False


def _user_dir(user_id: str) -> Path:
    return settings.sessions_dir / user_id


def session_dir(user_id: str, session_id: str) -> Path:
    return _user_dir(user_id) / session_id


def session_db_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "checkpoints.db"


def _meta_path(user_id: str, session_id: str) -> Path:
    return session_dir(user_id, session_id) / "meta.json"


def _active_path(user_id: str) -> Path:
    return _user_dir(user_id) / "active.json"


def _read_meta(user_id: str, session_id: str) -> Optional[SessionMeta]:
    path = _meta_path(user_id, session_id)
    if not path.exists():
        return None
    return SessionMeta.model_validate_json(path.read_text())


def _write_meta(user_id: str, meta: SessionMeta) -> None:
    _meta_path(user_id, meta.session_id).write_text(meta.model_dump_json())


def create_session(user_id: str, title: str = "New trip") -> SessionMeta:
    """
    - create a new random session_id
    - use user_id and session_id to create a new session folder
    - create a new SessionMeta with the current timestamp, and write it to meta.json
    - return the new SessionMeta
    """
    session_id = uuid.uuid4().hex[:12]
    session_dir(user_id, session_id).mkdir(parents=True, exist_ok=True)
    now = time.time()
    meta = SessionMeta(session_id=session_id, title=title, created_at=now, updated_at=now)
    _write_meta(user_id, meta)
    return meta


def list_sessions(user_id: str) -> list[SessionMeta]:
    user_dir = _user_dir(user_id)
    if not user_dir.exists():
        return []
    sessions = []
    for child in user_dir.iterdir():
        if not child.is_dir():
            continue
        meta = _read_meta(user_id, child.name)
        if meta is not None:
            sessions.append(meta)
    return sorted(sessions, key=lambda m: m.updated_at, reverse=True)


def get_session(user_id: str, session_id: str) -> Optional[SessionMeta]:
    return _read_meta(user_id, session_id)


def touch_session(user_id: str, session_id: str, title: Optional[str] = None) -> None:
    """Bump updated_at, and set the title if one is provided (e.g. once the
    first user message or trip destination is known)."""
    meta = _read_meta(user_id, session_id)
    if meta is None:
        return
    meta.updated_at = time.time()
    if title:
        meta.title = title
    _write_meta(user_id, meta)


def set_pending_confirmation(user_id: str, session_id: str, pending: bool) -> None:
    """Marks whether we're waiting on the user's yes/no answer about retrying
    a turn that crashed mid-execution instead of reaching a real interrupt."""
    meta = _read_meta(user_id, session_id)
    if meta is None:
        return
    meta.pending_confirmation = pending
    _write_meta(user_id, meta)


def get_active_session_id(user_id: str) -> Optional[str]:
    path = _active_path(user_id)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("session_id")


def set_active_session_id(user_id: str, session_id: str) -> None:
    """
    - Get the path to the user's active.json file, which store the active session_id per user_id
    - Create the parent directory if it doesn't exist
    - Write the current session_id to the active.json file in JSON format, update user active session information
    """
    path = _active_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": session_id}))


def get_or_create_active_session(user_id: str) -> SessionMeta:
    """
    - Get the user's active session_id from active.json that belongs to the user_id
    - If the active session_id exists, retrieve the corresponding SessionMeta from meta.json
    - If the active session_id does not exist, create a new session with a new session_id, write the new session_id to active.json, and return the new SessionMeta
    """
    session_id = get_active_session_id(user_id)
    meta = get_session(user_id, session_id) if session_id else None
    if meta is None:
        meta = create_session(user_id)
        set_active_session_id(user_id, meta.session_id)
    return meta
