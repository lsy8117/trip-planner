# backend/graph/session_graph.py
"""
Binds a compiled orchestrator graph to one session's own SQLite checkpoint
db (see backend/session_store.py for the folder layout), and caches it for
the life of the process so repeated turns in the same conversation reuse
the same connection instead of reopening it every message.
"""
import asyncio

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.graph.orchestrator import build_graph
from backend.session_store import session_db_path

_cache: dict[str, tuple] = {}  # key -> (graph, saver, conn)
_lock = asyncio.Lock() # Lock on SqliteSaver database


def is_cached(user_id: str, session_id: str) -> bool:
    """True if this session's graph is already loaded in this process. Used to
    tell a genuine session "recall" (first message since a process restart, or
    since this session was last evicted) apart from continuing an
    already-active in-memory conversation — see frontend/telegram/handlers.py."""
    return f"{user_id}:{session_id}" in _cache


async def get_session_graph(user_id: str, session_id: str):
    """Return the compiled graph for this (user_id, session_id), building and
    caching a fresh AsyncSqliteSaver-backed one on first use."""
    key = f"{user_id}:{session_id}"
    if key in _cache: # always be None if the coroutine does not have the Lock
        return _cache[key][0]

    async with _lock:
        if key in _cache:  # re-check after acquiring the lock
            return _cache[key][0]

        db_path = session_db_path(user_id, session_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(db_path)
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        graph = build_graph(checkpointer=saver)
        _cache[key] = (graph, saver, conn)
        return graph
