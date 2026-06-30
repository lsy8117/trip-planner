import json
from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.config import settings
from backend.utilities import get_llm
from backend.models import AttractionInfo, AttractionsList, DayWeatherInfo

import httpx

# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

tavily_client = TavilyClient(api_key=settings.tavily_api_key)


@tool
async def search_attraction(search_query: str):
    """
    Search with tavily client with the custom search query, and return maximum 5 results.

    Args:
        search_query: terms to search with tavily client.

    Returns:
        dict with search query, answer and a list of search results.
    """
    try:
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5, include_answer=True
        )
        return response
    except Exception as e:
        return {"error": str(e), "results": []}


attraction_tools = [search_attraction]

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

def merge_attractions(
    existing: list[AttractionInfo],
    incoming: list[AttractionInfo],
) -> list[AttractionInfo]:
    """Dedup by name across tool loop iterations."""
    by_name = {a.name: a for a in existing}
    for a in incoming:
        by_name[a.name] = a
    return list(by_name.values())


class AttractionState(TypedDict):
    # Input fields (written by orchestrator)
    user_query: str
    weather_info: list[DayWeatherInfo]          # optional context; may be empty list

    # Internal working memory
    messages: Annotated[list[BaseMessage], add_messages]

    # Output field (read by orchestrator)
    attractions: Annotated[list[AttractionInfo], merge_attractions]


class AttractionInput(TypedDict):
    user_query: str
    weather_info: list[DayWeatherInfo]


class AttractionOutput(TypedDict):
    attractions: list[AttractionInfo]


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

ATTRACTION_AGENT_SYSTEM_PROMPT = """
** Role **
You are a helpful assistant specialized at recommending tourist attractions
based on user requirement or past preferences.

** Task & Responsibility **
Your responsibility is to look through the message list, identify where the
user is traveling to and the respective dates, then generate customized search
terms to search for tourist attractions that best match user requirements.

** Important **
Do not fabricate any tourist attractions. Always use the search_attraction tool
to search for attraction information.

** Output **
If attraction search is successful, simply pass down the information to the next
step.
"""

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def _build_attraction_agent():
    llm_w_tavily = get_llm().bind_tools(attraction_tools, parallel_tool_calls=False)

    async def attraction_agent(state: AttractionState) -> dict:
        # Inject weather context into the user query if available
        context = state["user_query"]
        if state.get("weather_info"):
            weather_lines = "\n".join(
                f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"
                for w in state["weather_info"]
            )
            context += f"\n\nWeather forecast:\n{weather_lines}"

        response = await llm_w_tavily.ainvoke([
            SystemMessage(content=ATTRACTION_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=context),
            *state["messages"],
        ])
        return {"messages": [response]}

    return attraction_agent


def attraction_router(state: AttractionState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "extract_attractions"   # always route to extraction, never jump to END from agent


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


def _build_extract_attractions():
    extractor = get_llm(provider='groq', model='qwen/qwen3-32b').with_structured_output(AttractionsList)
    # extractor = get_llm(provider='groq', model='openai/gpt-oss-120b').with_structured_output(AttractionsList, method="json_schema")

    async def extract_attractions(state: AttractionState) -> dict:
        """
        Parse Tavily search results into structured AttractionInfo objects.
        Uses a separate LLM extractor (with_structured_output) over chunked text,
        then deduplicates by name.
        """
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not tool_messages:
            return {"attractions": []}

        try:
            tool_result = json.loads(tool_messages[-1].content)
        except (json.JSONDecodeError, TypeError):
            return {"attractions": []}

        all_attractions: list[AttractionInfo] = []
        seen: set[str] = set()

        for result in tool_result.get("results", []):
            content = f"Title: {result['title']}\n\nContent: {result['content']}"

            for chunk in _chunk_text(content, chunk_size=800, overlap=100):
                extracted = await extractor.ainvoke([
                    SystemMessage(
                        "Extract tourist attractions from the search result below. "
                        "Only include attractions explicitly mentioned. "
                        "Set fields to null if not mentioned. Be concise."
                    ),
                    HumanMessage(content=chunk),
                ])
                for attraction in extracted.attractions:
                    key = attraction.name.lower()
                    if key not in seen:
                        seen.add(key)
                        all_attractions.append(attraction)

        return {"attractions": all_attractions}

    return extract_attractions


async def geocode_attractions(state: AttractionState) -> dict:
    """
    Enrich attractions that are missing coordinates via Google Geocoding API.
    Attractions that already have lat/lon are passed through unchanged.
    """
    geocoded = []
    async with httpx.AsyncClient() as client:
        for attraction in state["attractions"]:
            # Skip geocoding if coordinates already present
            if attraction.latitude is not None and attraction.longitude is not None:
                geocoded.append(attraction)
                continue

            try:
                geo = await client.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={
                        "address": attraction.name,
                        "key": settings.google_maps_api_key,
                    },
                )
                geo_data = geo.json()
                if geo_data.get("results"):
                    location = geo_data["results"][0]["geometry"]["location"]
                    geocoded.append(attraction.model_copy(update={
                        "latitude": location["lat"],
                        "longitude": location["lng"],
                    }))
                else:
                    geocoded.append(attraction)
            except Exception:
                geocoded.append(attraction)

    return {"attractions": geocoded}


# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_attraction_subgraph() -> StateGraph:
    attraction_agent = _build_attraction_agent()
    extract_attractions = _build_extract_attractions()

    builder = StateGraph(AttractionState, input=AttractionInput, output=AttractionOutput)

    builder.add_node("attraction_agent", attraction_agent)
    builder.add_node("tools", ToolNode(attraction_tools))
    builder.add_node("extract_attractions", extract_attractions)
    builder.add_node("geocode_attractions", geocode_attractions)

    builder.set_entry_point("attraction_agent")

    builder.add_conditional_edges(
        "attraction_agent",
        attraction_router,
        {
            "tools": "tools",
            "extract_attractions": "extract_attractions",
        },
    )
    builder.add_edge("tools", "attraction_agent")          # loop: results go back to agent for next search term
    builder.add_edge("extract_attractions", "geocode_attractions")
    builder.add_edge("geocode_attractions", END)           # structured + geocoded output ready

    return builder.compile()


attraction_subgraph = build_attraction_subgraph()