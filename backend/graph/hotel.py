import json
from typing import Annotated, Optional
from typing_extensions import TypedDict

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.config import settings
from backend.utilities import get_llm
from backend.models import AttractionInfo, DayWeatherInfo, HotelInfo, HotelOffer

# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

@tool
async def search_hotels(
    city: str,
    chk_in: str,
    chk_out: str,
    adults: int = 1,
    limit: int = 5,
    sort: str = "best_value",
) -> dict:
    """
    Search for available hotels in a city for given dates using the Xotelo API.

    Args:
        city: City name, e.g. "Tokyo"
        chk_in: Check-in date "YYYY-MM-DD"
        chk_out: Check-out date "YYYY-MM-DD"
        adults: Number of adults (default 1)
        limit: Number of hotels to return (default 5)
        sort: Sort order for /api/list — "best_value", "popularity", or "distance" (default "best_value")

    Returns:
        dict with city, dates, and a list of hotels with rates from multiple OTAs.
    """
    BASE = "https://xotelo-hotel-prices.p.rapidapi.com"
    headers = {
        "X-RapidAPI-Key": settings.rapidapi_key,
        "X-RapidAPI-Host": "xotelo-hotel-prices.p.rapidapi.com",
    }

    async with httpx.AsyncClient(timeout=15, headers=headers) as client:
        # Step 1: search hotels by city name directly (accommodation search returns hotel_key per item)
        r = await client.get(f"{BASE}/api/search", params={"query": city, "location_type": "accommodation"})
        r.raise_for_status()
        data = r.json()
        search_list = (data.get("result") or {}).get("list", [])

        if not search_list:
            # Fallback: try /api/list with a geo search for location_key
            r2 = await client.get(f"{BASE}/api/search", params={"query": city, "location_type": "geo"})
            r2.raise_for_status()
            geo_list = (r2.json().get("result") or {}).get("list", [])
            if not geo_list:
                return {"error": f"Location not found: {city}", "hotels": []}
            location_key = geo_list[0]["location_key"]
            r3 = await client.get(f"{BASE}/api/list", params={"location_key": location_key, "limit": limit, "sort": sort})
            r3.raise_for_status()
            search_list = (r3.json().get("result") or {}).get("list", [])

        # Step 2: fetch rates for each hotel
        hotels = []
        for h in search_list[:limit]:
            hotel_key = h.get("hotel_key") or h.get("key")
            if not hotel_key:
                continue

            r = await client.get(f"{BASE}/api/rates", params={
                "hotel_key": hotel_key,
                "chk_in": chk_in,
                "chk_out": chk_out,
                "adults": adults,
            })
            r.raise_for_status()
            rates = (r.json().get("result") or {}).get("rates", [])

            hotels.append({
                "name": h.get("name"),
                "hotel_key": hotel_key,
                "rating": h.get("rating"),
                "address": h.get("address"),
                "offers": [{"ota": rate.get("name"), "price_usd": rate.get("rate")} for rate in rates],
            })

    return {"city": city, "chk_in": chk_in, "chk_out": chk_out, "hotels": hotels}


hotel_tools = [search_hotels]

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class HotelState(TypedDict):
    # Input fields (written by orchestrator)
    user_query: str
    weather_info: list[DayWeatherInfo]          # optional context; may be empty list
    attractions: list[AttractionInfo]           # optional context; may be empty list

    # Internal working memory
    messages: Annotated[list[BaseMessage], add_messages]

    # Output field (read by orchestrator)
    hotels: list[HotelInfo]


class HotelInput(TypedDict):
    user_query: str
    weather_info: list[DayWeatherInfo]
    attractions: list[AttractionInfo]


class HotelOutput(TypedDict):
    hotels: list[HotelInfo]


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

HOTEL_AGENT_SYSTEM_PROMPT = """
** Role **
You are a helpful assistant specialized in finding hotel options for travelers.

** Task & Responsibility **
Look through the conversation history to identify the destination city and travel dates (check-in and check-out),
then use the search_hotels tool to retrieve available options with prices.

You may choose the sort order based on user preference:
- "best_value" — default, balance of price and rating
- "popularity" — most booked hotels first
- "distance" — closest to city center first

You are not asked to generate any travel plan or suggestions. 

** Important **
- Always use the search_hotels tool. Do not fabricate hotel information.
- If hotels have already been found in a previous tool call, do not search again — proceed to finish.

** Output **
If attraction search is successful, simply pass down the information to the next
step.
"""

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

def _build_hotel_agent():
    llm_w_hotel_tools = get_llm().bind_tools(hotel_tools)

    async def hotel_agent(state: HotelState) -> dict:
        # Inject available context into the user query
        context = state["user_query"]
        if state.get("weather_info"):
            weather_lines = "\n".join(
                f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"
                for w in state["weather_info"]
            )
            context += f"\n\nWeather forecast:\n{weather_lines}"
        if state.get("attractions"):
            attraction_names = ", ".join(a.name for a in state["attractions"][:5])
            context += f"\n\nNearby attractions the user wants to visit: {attraction_names}"

        response = await llm_w_hotel_tools.ainvoke([
            SystemMessage(content=HOTEL_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=context),
            *state["messages"],
        ])
        return {"messages": [response]}

    return hotel_agent


def hotel_router(state: HotelState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "parse_hotel_results"   # always parse explicitly, never jump to END from agent


async def parse_hotel_results(state: HotelState) -> dict:
    """
    Parse the last ToolMessage into structured HotelInfo objects.
    This is the only node that writes to `hotels` — guarantees output key is always set.
    """
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    if not tool_messages:
        return {"hotels": []}

    try:
        tool_result = json.loads(tool_messages[-1].content)
    except (json.JSONDecodeError, TypeError):
        return {"hotels": []}

    if tool_result.get("error"):
        return {"hotels": []}

    hotels = [
        HotelInfo(
            name=h["name"],
            hotel_key=h["hotel_key"],
            rating=h.get("rating"),
            address=h.get("address"),
            offers=[HotelOffer(**o) for o in h.get("offers", [])],
        )
        for h in tool_result.get("hotels", [])
        if h.get("name") and h.get("hotel_key")
    ]
    return {"hotels": hotels}


# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_hotel_subgraph() -> StateGraph:
    hotel_agent = _build_hotel_agent()

    builder = StateGraph(HotelState, input=HotelInput, output=HotelOutput)

    builder.add_node("hotel_agent", hotel_agent)
    builder.add_node("tools", ToolNode(hotel_tools))
    builder.add_node("parse_hotel_results", parse_hotel_results)

    builder.set_entry_point("hotel_agent")

    builder.add_conditional_edges(
        "hotel_agent",
        hotel_router,
        {
            "tools": "tools",
            "parse_hotel_results": "parse_hotel_results",
        },
    )
    builder.add_edge("tools", "hotel_agent")           # loop: agent reviews results, decides if done
    builder.add_edge("parse_hotel_results", END)       # structured output ready, exit

    return builder.compile()


hotel_subgraph = build_hotel_subgraph()