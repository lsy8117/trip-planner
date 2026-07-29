import json
from typing import Annotated, Callable, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import ToolException, tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from tavily import TavilyClient

from backend.config import settings
from backend.utilities import ARGUMENT_ERROR_LIMIT, get_llm, resilient, tool_error_handler, make_check_tool_results, make_route_after_tools, make_give_up_node
from backend.models import AttractionInfo, AttractionsList, DayWeatherInfo, merge_attractions

import httpx

# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

tavily_client = TavilyClient(api_key=settings.tavily_api_key)


@tool
@resilient(max_transient_retries=3, base_delay=1.0)
async def search_attraction(search_query: str):
    """
    Search with tavily client with the custom search query, and return maximum 5 results.

    Args:
        search_query: terms to search with tavily client.

    Returns:
        dict with search query, answer, and a list of search results — each result
        includes the full scraped source article (raw_content) in addition to
        Tavily's short snippet (content).
    """
    try:
        response = tavily_client.search(
            query=search_query, search_depth="basic", max_results=5,
            include_answer=True, include_raw_content=True,
        )
        return response
    except Exception as e:
        return {"error": str(e), "results": []}


attraction_tools = [search_attraction]

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class AttractionState(TypedDict):
    # Input fields (written by orchestrator)
    user_query: str
    weather_info: list[DayWeatherInfo]          # optional context; may be empty list
    attraction_requirements: Optional[list[str]]
    attraction_preferences: Optional[list[str]]
    past_search_queries: list[str]              # queries already run in earlier turns; may be empty list

    # Internal working memory
    messages: Annotated[list[BaseMessage], add_messages]

    # Output fields (read by orchestrator)
    attractions: Annotated[list[AttractionInfo], merge_attractions]
    search_queries_used: list[str]


class AttractionInput(TypedDict):
    user_query: str
    weather_info: list[DayWeatherInfo]
    attraction_requirements: Optional[list[str]]
    attraction_preferences: Optional[list[str]]
    edit_instruction: Optional[str]
    past_search_queries: list[str]


class AttractionOutput(TypedDict):
    attractions: list[AttractionInfo]
    search_queries_used: list[str]
    error: Optional[str]


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

If an EDIT INSTRUCTION is provided below, this is a targeted follow-up request —
search ONLY for what the edit instruction asks for. Do not re-search broadly for
general attractions in the destination; the existing attraction list is already
finalized except for this specific change.

If NO edit instruction is provided, this is a first-time or full-replan search —
identify the destination and dates from the conversation and search broadly.

If a list of PAST SEARCHES is provided below, those queries have already been run
earlier in this conversation and their results are already captured — do not repeat
them. Use them as reference to figure out what's already been covered and generate
a new, different query for whatever information is still missing.

** Important **
- Do not fabricate any tourist attractions. Always use the search_attraction tool
to search for attraction information.
- Generate a search query that is as SPECIFIC as possible to what's being asked
for — not a generic "top attractions in <city>" query if a targeted edit is given.
- If tool search failed, you may analyze the reason and try to generate a new search
query to re-search.
- The generated search query should consider the weather forecast if available, unless
user past preferences or requirements indicate otherwise.
- When generating search queries, consider user's requirements, preferences, ALL past 
requests (if any), and past search queries already run in this conversation. Do not 
repeat past searches.
- If user requirements or preferences contradict each other, prioritize requirements 
over preferences.
- You may also look through user requests and identify any specific requirements or 
preferences the user has stated, and use them to generate more targeted search queries.


** Output **
If attraction search is successful, simply pass down the information to the next
step.
"""

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def _build_attraction_agent_v1():
    # llm_w_tavily = get_llm(provider='groq',model='openai/gpt-oss-20b').bind_tools(attraction_tools, parallel_tool_calls=False)
    llm_w_tavily = get_llm(provider=settings.attraction_model_provider, model=settings.attraction_model).bind_tools(attraction_tools)

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

def _build_attraction_agent():
    llm_w_tavily = get_llm(provider=settings.attraction_model_provider, model=settings.attraction_model).bind_tools(attraction_tools)

    async def attraction_agent(state: AttractionState) -> dict:
        if state.get("edit_instruction"):
            context = f"EDIT INSTRUCTION: {state['edit_instruction']}"
            if state.get("existing_attractions"):
                existing_names = ", ".join(a.name for a in state["existing_attractions"])
                context += f"\n\nAttractions already in the plan (do not re-search these): {existing_names}"
        else:
            context = state["user_query"]

        if state.get("weather_info"):
            weather_lines = "\n".join(
                f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"
                for w in state["weather_info"]
            )
            context += f"\n\nWeather forecast:\n{weather_lines}"

        if state.get("attraction_requirements"):
            requirements = ", ".join(state["attraction_requirements"])
            context += f"\n\nUser requirements: {requirements}"
        
        if state.get("attraction_preferences"):
            preferences = ", ".join(state["attraction_preferences"])
            context += f"\n\nUser preferences: {preferences}"

        if state.get("past_search_queries"):
            past = "\n".join(f"- {q}" for q in state["past_search_queries"])
            context += f"\n\nPAST SEARCHES already run in this conversation (do not repeat):\n{past}"

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


def _collect_search_queries(messages: list[BaseMessage]) -> list[str]:
    """Pull every search_query arg the agent actually issued to search_attraction this run."""
    queries = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc.get("name") == "search_attraction":
                q = tc.get("args", {}).get("search_query")
                if q:
                    queries.append(q)
    return queries


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start: start + chunk_size])
        start += chunk_size - overlap
    return chunks


# Cap how much of a raw scraped article we run through the extractor — full pages
# can run tens of thousands of characters; attraction mentions are almost always
# in the first few thousand, and this bounds the extractor LLM calls per result.
MAX_SOURCE_CHARS = 6000


def _build_extract_attractions():
    # extractor = get_llm(provider='groq', model='qwen/qwen3.6-27b').with_structured_output(AttractionsList)
    extractor = get_llm(provider=settings.extraction_model_provider, model=settings.extraction_model) \
        .with_structured_output(AttractionsList, method="json_schema") \
        .with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 1.0},
        )

    async def extract_attractions(state: AttractionState) -> dict:
        queries_used = _collect_search_queries(state["messages"])
        tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if not tool_messages:
            return {"attractions": [], "search_queries_used": queries_used, "error": state.get("tool_failure_message")}

        all_attractions: list[AttractionInfo] = []
        seen: set[str] = set()
        any_success = False

        async def extract_from_text(text: str) -> None:
            for chunk in _chunk_text(text[:MAX_SOURCE_CHARS], chunk_size=800, overlap=100):
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

        for tm in tool_messages:
            if getattr(tm, "status", None) == "error":
                continue  # skip failed searches — don't try to json.loads an error tag

            try:
                tool_result = json.loads(tm.content)
            except (json.JSONDecodeError, TypeError):
                continue

            any_success = True

            # Tavily's synthesized answer is itself a useful, pre-summarized source
            if tool_result.get("answer"):
                await extract_from_text(f"Search summary: {tool_result['answer']}")

            for result in tool_result.get("results", []):
                # Prefer the full scraped source article over Tavily's short snippet;
                # fall back to the snippet if the page couldn't be fetched/cleaned.
                body = result.get("raw_content") or result.get("content") or ""
                content = f"Title: {result['title']}\n\nContent: {body}"
                await extract_from_text(content)

        error = None if any_success else (state.get("tool_failure_message") or "All attraction searches failed.")
        return {"attractions": all_attractions, "search_queries_used": queries_used, "error": error}

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

def route_after_tools(state: AttractionState) -> str:
    counts = state.get("tool_failure_counts") or {}
    if state.get("tool_failure_message") or any(c >= ARGUMENT_ERROR_LIMIT for c in counts.values()):
        return "extract_attractions"   # give up on more searching, extract whatever succeeded
    return "attraction_agent"

# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_attraction_subgraph() -> StateGraph:
    attraction_agent = _build_attraction_agent()
    extract_attractions = _build_extract_attractions()

    builder = StateGraph(AttractionState, input=AttractionInput, output=AttractionOutput)

    check_tool_results = make_check_tool_results()
    route_after_tools = make_route_after_tools(agent_node="attraction_agent", success_node="extract_attractions")
    give_up = make_give_up_node({"attractions": []})

    builder.add_node("attraction_agent", attraction_agent)
    builder.add_node("tools", ToolNode(attraction_tools, handle_tool_errors=tool_error_handler))
    builder.add_node("extract_attractions", extract_attractions)
    builder.add_node("geocode_attractions", geocode_attractions)

    builder.add_node("check_tool_results", check_tool_results)
    builder.add_node("give_up", give_up)

    builder.set_entry_point("attraction_agent")

    builder.add_conditional_edges(
        "attraction_agent",
        attraction_router,
        {
            "tools": "tools",
            "extract_attractions": "extract_attractions",
        },
    )
    builder.add_edge("tools", "check_tool_results")
    builder.add_conditional_edges("check_tool_results", route_after_tools, {
        "attraction_agent": "attraction_agent",
        "extract_attractions": "extract_attractions",
    })
    builder.add_edge("extract_attractions", "geocode_attractions")
    builder.add_edge("geocode_attractions", END)           # structured + geocoded output ready

    return builder.compile()


attraction_subgraph = build_attraction_subgraph()