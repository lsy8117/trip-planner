# Trip Planner

An agentic travel-planning assistant, currently exposed through a Telegram bot. Tell it
where and when you want to go, and it looks up weather, searches attractions and hotels,
and builds a day-by-day itinerary — then keeps refining that itinerary as you send
follow-up requests in the same conversation.

## How it works

The bot is a thin frontend over a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine (`backend/graph/orchestrator.py`). Each user message is a turn through the
graph:

1. **classify_request** — an LLM call figures out whether this is a new trip or a
   refinement of the current one, and which of the three research subgraphs need to run.
2. **weather / attraction / hotel** — independent subgraphs (each its own small
   tool-calling loop) that run sequentially or in parallel depending on what changed.
   Weather comes from OpenWeatherMap, attractions from a Tavily web search, hotels from a
   dedicated search tool.
3. **itinerary / itinerary_patch** — assembles a full day-by-day plan, or patches just
   the part the user asked to change.
4. **respond** — sends the itinerary back and pauses the graph (via LangGraph's
   `interrupt()`) waiting for the next message, so a conversation can go on indefinitely
   without losing state.

Each conversation ("session") is checkpointed to its own SQLite file under
`backend/sessions/<user_id>/<session_id>/`, so it survives a bot restart. A separate,
manually-triggered memory pipeline (`backend/memory/`, run via the `/update_memory`
command) distills a user's message history and generated itineraries into longer-term
preferences.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- API keys for Groq (LLM), Tavily (web search), and OpenWeatherMap (weather)

## Setup

```bash
uv sync
```

Create a `.env` file in the project root (there's no `.env.example` checked in yet — see
below for the full list of keys `backend/config.py` reads):

```bash
LLM_PROVIDER=groq
GEMINI_API_KEY=
GROQ_API_KEY=
TAVILY_API_KEY=
OPENWEATHERMAP_API_KEY=
RAPIDAPI_KEY=
GOOGLE_MAPS_API_KEY=
TELEGRAM_BOT_TOKEN=
```

Only `GROQ_API_KEY`, `TAVILY_API_KEY`, `OPENWEATHERMAP_API_KEY`, and
`TELEGRAM_BOT_TOKEN` are required to run the bot as currently configured (Groq is the
default LLM provider for every node).

Run the bot:

```bash
uv run python -m frontend.telegram.bot
```

It runs in polling mode — no public URL or tunnel needed. Search for your bot's username
on Telegram and start chatting.

## Bot commands

- Just send a message describing a trip to start planning
- `/new` — start a fresh conversation without losing the previous one
- `/sessions` — switch between your past conversations
- `/update_memory` — manually re-run the long-term preference-memory update

## Development

```bash
uv run pytest                                  # run all tests
uv run pytest backend/test/test_graph.py -v    # graph tests only
uv run python backend/test/test_multiround.py  # interactive CLI, streams node-by-node progress
uv run ruff check . --fix                      # lint
uv run mypy backend/ frontend/                 # type check
```

See [CLAUDE.md](CLAUDE.md) for the full architecture reference, directory layout, and
conventions used across the codebase.

## Roadmap

- **Now:** Telegram bot (polling)
- **Next:** Streamlit internal demo (`frontend/streamlit/`, scaffolded but not yet built)
- **Later:** Next.js or Telegram Web App frontend, talking to a FastAPI layer under
  `backend/api/`
