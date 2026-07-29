"""
Writes structured JSONL records of every graph run to a log file:
node transitions (with their input/output state), tool calls, and tool
results, plus the complete final state after each turn.

None of this is ever sent to the user — it's purely for offline
debugging/observability. Each session gets its own logger + file, living
alongside that session's checkpoint db (see backend/session_store.py), so a
conversation's log travels with the rest of its folder. Kept as separate
loggers (propagate=False) so they don't get mixed into the console INFO
logging set up in bot.py.
"""

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Event kinds worth persisting in full. Deliberately excludes
# on_chat_model_stream (raw token chunks) — pure noise for this purpose,
# not state. Widen this set if you want token-level traces too.
_LOGGED_EVENT_KINDS = {"on_chain_start", "on_chain_end", "on_tool_start", "on_tool_end"}

_loggers: dict[str, logging.Logger] = {}  # keyed by resolved log_dir


def _get_logger(log_dir: Path) -> logging.Logger:
    key = str(log_dir)
    if key in _loggers:
        return _loggers[key]

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"agent_runs.{key}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(
        log_dir / "agent_runs.jsonl", maxBytes=20_000_000, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))  # each line is already a JSON blob
    logger.addHandler(handler)
    _loggers[key] = logger
    return logger


def _json_default(obj: Any):
    """Best-effort serialization for LangChain messages / Pydantic models,
    which json.dumps doesn't know how to handle natively."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "content") and hasattr(obj, "type"):  # BaseMessage-like
        return {"type": getattr(obj, "type", obj.__class__.__name__), "content": obj.content}
    return str(obj)


def _write(log_dir: Path, record: dict) -> None:
    record["ts"] = time.time()
    logger = _get_logger(log_dir)
    try:
        logger.info(json.dumps(record, default=_json_default))
    except Exception:
        logger.info(json.dumps({"ts": record["ts"], "error": "failed to serialize record"}))


def log_graph_event(log_dir: Path, thread_id: str, event: dict) -> None:
    """Call this for every event yielded by astream_events — it filters
    internally to the kinds worth keeping."""
    kind = event.get("event")
    if kind not in _LOGGED_EVENT_KINDS:
        return
    _write(log_dir, {
        "thread_id": thread_id,
        "kind": kind,
        "name": event.get("name"),
        "data": event.get("data"),  # includes input/output — messages, tool args, tool results
    })


def log_final_state(log_dir: Path, thread_id: str, state_values: dict) -> None:
    """Call once per turn after the stream drains — captures the complete
    orchestrator state (messages, weather_info, attractions, hotels,
    current_itinerary, execution_plan, etc.) at that point."""
    _write(log_dir, {
        "thread_id": thread_id,
        "kind": "final_state",
        "state": state_values,
    })
