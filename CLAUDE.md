# Travel Planning AI Agent

## Overview
An agentic travel assistant handling end-to-end trip planning: weather lookups,
tourist attractions, route planning, itinerary generation, and hotel recommendations.

**Current phase:** Prototyping with Telegram bot as the primary chat interface.
**Frontend roadmap:** Telegram bot (now) → Streamlit (internal demo) → Next.js or Telegram Web App (production).
**LLM strategy:** Provider-agnostic — must be easy to swap between cloud APIs and local models.

## Stack
- **Orchestration**: LangGraph (agent state machine and graph)
- **LLM layer**: Abstracted via LangChain's `BaseChatModel` — see LLM Configuration below
- **Telegram**: `python-telegram-bot` (polling mode, no public URL needed)
- **Web search**: Tavily
- **Weather**: OpenWeatherMap API
- **Backend**: FastAPI (future REST layer for Streamlit/Next.js; not needed for Telegram bot)
- **Frontend roadmap**: Telegram bot → Streamlit → Next.js or Telegram Web App

## Project Structure
\```
backend/                      # all agent, tool, and API logic
├── agents/
│   └── travel_graph.py       # main LangGraph graph definition
├── tools/                    # tool functions (weather, search, places, routing)
├── models/
│   └── state.py              # LangGraph TypedDict state (includes session_id)
├── llm/                      # LLM provider abstraction layer
│   ├── base.py               # get_llm() factory — only place LLM is instantiated
│   ├── google.py
│   ├── groq.py
│   └── local.py              # Ollama / llama.cpp backends
├── api/                      # FastAPI routes (used when Next.js frontend is added)
│   └── main.py
└── config.py                 # pydantic-settings, all env var access goes here
frontend/                     # all user-facing interface code
└── telegram/
    └── bot.py                # python-telegram-bot handlers, current entry point
.env
pyproject.toml
CLAUDE.md
\```

## Setup
```bash
uv sync
cp .env.example .env          # fill in keys for your chosen LLM provider + Telegram token
uv run python -m frontend.telegram.bot    # start the Telegram bot (polling, no ngrok needed)
```

To test: search your bot's username on Telegram and start chatting.

## Dev Commands
```bash
uv run python -m frontend.telegram.bot        # run Telegram bot
uv run fastapi dev backend/api/main.py        # run API server (future use)
uv run pytest                                 # run all tests
uv run pytest tests/backend/                  # backend tests only
uv run pytest tests/frontend/                 # frontend tests only
uv run ruff check . --fix                     # lint
uv run mypy backend/ frontend/                # type check
```

## LLM Configuration
The LLM is never instantiated directly in agent or tool code.
Always call `backend/llm/base.py::get_llm()` which reads `LLM_PROVIDER` from env and
returns the appropriate `BaseChatModel`. This is the single place to change when
switching providers.

Supported providers:

| LLM_PROVIDER value | Backend              | Required env vars                        |
|--------------------|----------------------|------------------------------------------|
| `google`           | gemma-4-26b-a4b-it   | `OPENAI_API_KEY`                         |
| `groq`             | Llama 3.3 70B        | `GROQ_API_KEY`                           |
| `ollama`           | Local via Ollama     | `OLLAMA_BASE_URL`, `OLLAMA_MODEL`        |
| `llamacpp`         | Local via llama.cpp  | `LLAMACPP_MODEL_PATH`                    |

Switching providers requires only changing `LLM_PROVIDER` in `.env` — no code changes.

## Environment Variables
See `.env.example`. Always update it when adding new keys.

## LLM — set ONE provider
LLM_PROVIDER=google        # google | groq | ollama | llamacpp
GEMINI_API_KEY=
GROQ_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
LLAMACPP_MODEL_PATH=

## Environment Variables
See `.env.example`. Required:
- GEMINI_API_KEY
- TAVILY_API_KEY
- OPENWEATHERMAP_API_KEY
- TELEGRAM_BOT_TOKEN           # from @BotFather on Telegram

## Agent Architecture
The main graph (src/agents/travel_graph.py) uses LangGraph with this flow:
  parse_intent → [weather | attractions | hotels | routing] → synthesize → respond

- State is a `TypedDict` in `backend/models/state.py`; includes `session_id` mapped to Telegram `chat_id` for per-user memory
- Each node is an async function; tools are standalone async functions in `backend/tools/`
- `frontend/telegram/bot.py` is the sole entry point for now; future frontends call the same graph

## Telegram Bot Specifics
- Use **polling mode** during development (`Application.run_polling()`) — no public URL or ngrok needed
- Switch to **webhook mode** only when deploying to a server
- Map Telegram `chat_id` → `session_id` for per-user conversation state
- Use `InlineKeyboardMarkup` for quick-reply options (e.g. "🌤 Weather", "🏨 Hotels", "🗺 Itinerary")
- Long responses: split at 4096 chars (Telegram message limit) or use `parse_mode=ParseMode.MARKDOWN`
- For streaming LLM responses: send an initial message, then use `bot.edit_message_text()` to update it as tokens arrive

## Frontend / Backend Boundary
- `frontend/` handles all user interaction: message parsing, formatting, UI-specific logic
- `backend/` handles all agent logic: LangGraph graph, tools, LLM calls, data models
- **`frontend/` must never be imported by `backend/`** — the boundary is strictly one-way
- `frontend/` calls into `backend/agents/travel_graph.py` only via its public `invoke()` / `astream()` interface
- When Next.js is added, `frontend/nextjs/` talks to `backend/api/` via HTTP — the graph itself is unchanged

## Future Frontend Migration
- **Streamlit**: add `frontend/streamlit/app.py`; imports and calls the graph directly (same Python process)
- **Next.js**: add routes to `backend/api/main.py`; `frontend/nextjs/` calls those endpoints over HTTP
- **Telegram Web App**: React mini-app under `frontend/nextjs/`; served via the same FastAPI backend

## Conventions
- Async everywhere — use `httpx.AsyncClient`, never `requests`; `python-telegram-bot` v20+ is fully async
- All data models use Pydantic v2
- Tool functions return structured dicts, not raw strings
- Type hints required on all function signatures
- Use `pydantic-settings` via `backend/config.py` for all env var access — no bare `os.getenv()`
- Do not call `get_llm()` inside tool functions; only graph nodes hold the LLM instance

## Do Not
- Instantiate any LLM client directly outside of `backend/llm/`
- Import anything from `frontend/` inside `backend/`
- Put Telegram handler logic inside agent or tool code
- Use synchronous HTTP clients
- Hardcode provider-specific model names outside of `backend/llm/`
- Use webhook mode locally — polling is simpler and requires no tunnel
- Modify `uv.lock` manually