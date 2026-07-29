import functools

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from traitlets import Any, Callable
from backend.config import settings

import asyncio
import random
import httpx
from enum import Enum
from langchain_core.tools import ToolException
from pydantic import ValidationError


def get_llm(provider=None, model=None, **kwargs) -> BaseChatModel:
    """
    Factory function that returns a configured LangChain BaseChatModel
    based on the LLM_PROVIDER env var. This is the ONLY place where
    LLM clients should be instantiated.

    Frontier providers: gemini, groq
    Local providers: ollama, llamacpp

    Extra kwargs (e.g. temperature, max_tokens) are passed through to
    the underlying chat model constructor.
    """
    if not provider:
        provider = settings.llm_provider.lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        
        if not model:
            model="gemma-4-26b-a4b-it"

        return ChatGoogleGenerativeAI(
            model=model,    # gemma-4-31b-it Swith if needed
            api_key=settings.gemini_api_key,
            **kwargs,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        
        if not model:
            model="openai/gpt-oss-20b"

        return ChatGroq(
            model=model,
            api_key=settings.groq_api_key,
            **kwargs,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            **kwargs,
        )

    elif provider == "llamacpp":
        from langchain_community.chat_models import ChatLlamaCpp

        if not settings.llamacpp_model_path:
            raise ValueError("LLAMACPP_MODEL_PATH is not set in .env")

        return ChatLlamaCpp(
            model_path=settings.llamacpp_model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Must be one of: gemini, groq, ollama, llamacpp"
        )
    

class FailureCategory(Enum):
    TRANSIENT = "transient"
    ARGUMENT = "argument"
    FATAL = "fatal"


def classify_tool_error(exc: Exception) -> FailureCategory:
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, asyncio.TimeoutError)):
        return FailureCategory.TRANSIENT
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 500, 502, 503, 504):
            return FailureCategory.TRANSIENT
        if status in (401, 403):
            return FailureCategory.FATAL
        return FailureCategory.ARGUMENT
    if isinstance(exc, (ValueError, ValidationError, KeyError)):
        return FailureCategory.ARGUMENT
    return FailureCategory.FATAL


def resilient(max_transient_retries: int = 3, base_delay: float = 1.0):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    category = classify_tool_error(exc)

                    if category is FailureCategory.TRANSIENT and attempt < max_transient_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue

                    tag = {
                        FailureCategory.TRANSIENT: "TRANSIENT_EXHAUSTED",
                        FailureCategory.ARGUMENT: "ARGUMENT_ERROR",
                        FailureCategory.FATAL: "FATAL_ERROR",
                    }[category]
                    raise ToolException(f"{tag}: {exc}") from exc
                else:
                    if isinstance(result, dict) and result.get("error"):
                        raise ToolException(f"ARGUMENT_ERROR: {result['error']}")
                    return result
        return wrapped
    return decorator

def tool_error_handler(exc: Exception) -> str:
    """Passed as ToolNode(handle_tool_errors=...). Catches anything that never went through
    `resilient` at all — e.g. a schema-validation ValidationError raised by ToolNode's own
    argument parsing, before the tool body ever runs."""
    if isinstance(exc, ToolException):
        return str(exc)  # already tagged by resilient()
    category = classify_tool_error(exc)
    tag = {
        FailureCategory.TRANSIENT: "TRANSIENT_EXHAUSTED",
        FailureCategory.ARGUMENT: "ARGUMENT_ERROR",
        FailureCategory.FATAL: "FATAL_ERROR",
    }[category]
    return f"{tag}: {exc}"

# ── loop-guard nodes (parameterized per subgraph) ─────────────────────────

ARGUMENT_ERROR_LIMIT = 3


def make_check_tool_results() -> Callable:
    async def check_tool_results(state: dict) -> dict:
        counts = dict(state.get("tool_failure_counts") or {})
        immediate_exit_message = None

        tail = []
        for m in reversed(state.get("messages") or []):
            if isinstance(m, ToolMessage):
                tail.append(m)
            else:
                break

        for msg in tail:
            name = msg.name or "unknown_tool"
            if getattr(msg, "status", None) != "error":
                counts.pop(name, None)
                continue

            content = msg.content if isinstance(msg.content, str) else ""
            if content.startswith("ARGUMENT_ERROR:"):
                counts[name] = counts.get(name, 0) + 1
            else:  # TRANSIENT_EXHAUSTED or FATAL_ERROR
                immediate_exit_message = f"{name}: {content}"

        return {"tool_failure_counts": counts, "tool_failure_message": immediate_exit_message}
    return check_tool_results


def make_route_after_tools(
    agent_node: str, success_node: str | None = None, give_up_node: str = "give_up"
) -> Callable:
    def route_after_tools(state: dict) -> str:
        if state.get("tool_failure_message"):
            return give_up_node
        counts = state.get("tool_failure_counts") or {}
        if any(c >= ARGUMENT_ERROR_LIMIT for c in counts.values()):
            return give_up_node

        # Tail = the consecutive ToolMessages from the most recent tool-call round.
        tail = []
        for m in reversed(state.get("messages") or []):
            if isinstance(m, ToolMessage):
                tail.append(m)
            else:
                break

        round_succeeded = bool(tail) and all(getattr(m, "status", None) != "error" for m in tail)
        if round_succeeded and success_node:
            return success_node

        return agent_node
    return route_after_tools


def make_give_up_node(output_defaults: dict[str, Any]) -> Callable:
    async def give_up(state: dict) -> dict:
        immediate = state.get("tool_failure_message")
        if immediate:
            error_msg = immediate
        else:
            counts = state.get("tool_failure_counts") or {}
            failing = [n for n, c in counts.items() if c >= ARGUMENT_ERROR_LIMIT]
            error_msg = f"Gave up after {ARGUMENT_ERROR_LIMIT} attempts to correct arguments for: {', '.join(failing)}"
        return {**output_defaults, "error": error_msg}
    return give_up