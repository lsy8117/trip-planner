from langchain_core.messages import HumanMessage, SystemMessage

from backend.utilities import get_llm
from backend.models import AttractionInfo, DayWeatherInfo, HotelInfo, PlannedItinerary

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

ITINERARY_AGENT_SYSTEM_PROMPT = """
You are an expert specialized at trip planning. Your role is to help the user
plan a trip based on gathered information.
You are provided with user query, searched results on destination weather,
tourist attractions, and hotel options. Your role is to look through the
weather_info, attractions and hotels provided, and then plan a trip
itinerary based on these gathered information.

** Important **
- Do not come up with any other new information. Strictly use information
  provided in the context.
- Use the default planning preferences unless the user stated otherwise.

## Default Planning Preferences
- Consider weather conditions when selecting attractions. E.g. Prefer indoor
  activities on a rainy day.
- Attractions closer together should be planned on the same day to save travel time.
- Choose more popular and iconic attractions unless the user stated otherwise.
- Choose hotels that are more convenient to stay and travel.
"""


# ─────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────
# Unlike the other agents, itinerary does not need a subgraph — it has no tools
# and no internal loop. It is a single LLM call that synthesizes all prior results.
# It lives directly as a node in the orchestrator graph.

def build_itinerary_node_v1():
    """
    Returns an async node function that synthesizes weather, attractions, and
    hotels from OrchestratorState into a final itinerary.

    Import this and add it directly to the orchestrator builder:
        builder.add_node("itinerary", build_itinerary_node())
    """
    itinerary_llm = get_llm()

    async def itinerary_agent(state: dict) -> dict:
        """
        Reads weather_info, attractions, and hotels from orchestrator state.
        All three may be empty if upstream agents failed — handled gracefully.
        """
        context_parts = []

        weather_info: list[DayWeatherInfo] = state.get("weather_info") or []
        if weather_info:
            weather_lines = "\n".join(
                f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"
                for w in weather_info
            )
            context_parts.append(f"Weather forecast:\n{weather_lines}")

        attractions: list[AttractionInfo] = state.get("attractions") or []
        if attractions:
            attraction_lines = "\n".join(
                f"- {a.name} ({a.attraction_type})"
                + (f" at ({a.latitude:.4f}, {a.longitude:.4f})" if a.latitude and a.longitude else "")
                + (f": {a.description[:100]}" if a.description and a.description != "null" else "")
                for a in attractions
            )
            context_parts.append(f"Available attractions:\n{attraction_lines}")

        hotels: list[HotelInfo] = state.get("hotels") or []
        if hotels:
            hotel_lines = "\n".join(
                f"- {h.name}"
                + (f" (rating: {h.rating})" if h.rating else "")
                + (f", from ${min(o.price_usd for o in h.offers):.0f}/night" if h.offers else "")
                for h in hotels
            )
            context_parts.append(f"Hotel options:\n{hotel_lines}")

        if not context_parts:
            # Graceful fallback: no upstream data available
            context_parts.append(
                "No structured data was retrieved. Please create a general itinerary "
                "based on the user's query alone."
            )

        structured_context = "\n\n".join(context_parts)

        # Orchestrator-level messages hold the original HumanMessage(user_query).
        # Pass them through so the itinerary agent has full conversation context.
        response = await itinerary_llm.ainvoke([
            SystemMessage(content=ITINERARY_AGENT_SYSTEM_PROMPT),
            *state.get("messages", []),
            HumanMessage(
                content=f"Here is the gathered information:\n\n{structured_context}\n\n"
                        f"Please create a detailed day-by-day itinerary."
            ),
        ])

        # Write back to orchestrator messages so the response is visible to the caller
        return {"messages": [response]}

    return itinerary_agent

def build_itinerary_node():
    itinerary_llm = get_llm(provider='groq',model='openai/gpt-oss-120b').with_structured_output(PlannedItinerary, method="json_schema")

    async def itinerary_agent(state: dict) -> dict:
        context_parts = []

        weather_info: list[DayWeatherInfo] = state.get("weather_info") or []
        if weather_info:
            weather_lines = "\n".join(
                f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"
                for w in weather_info
            )
            context_parts.append(f"Weather forecast:\n{weather_lines}")

        attractions: list[AttractionInfo] = state.get("attractions") or []
        if attractions:
            attraction_lines = "\n".join(
                f"- {a.name} ({a.attraction_type})"
                + (f" at ({a.latitude:.4f}, {a.longitude:.4f})" if a.latitude and a.longitude else "")
                + (f": {a.description[:100]}" if a.description and a.description != "null" else "")
                for a in attractions
            )
            context_parts.append(f"Available attractions:\n{attraction_lines}")

        hotels: list[HotelInfo] = state.get("hotels") or []
        if hotels:
            hotel_lines = "\n".join(
                f"- {h.name}"
                + (f" (rating: {h.rating})" if h.rating else "")
                + (f", from ${min(o.price_usd for o in h.offers):.0f}/night" if h.offers else "")
                for h in hotels
            )
            context_parts.append(f"Hotel options:\n{hotel_lines}")

        if not context_parts:
            context_parts.append(
                "No structured data was retrieved. Create a general itinerary "
                "based on the user's query alone."
            )

        structured_context = "\n\n".join(context_parts)

        # with_structured_output doesn't stream — it returns the full Pydantic object
        itinerary: PlannedItinerary = await itinerary_llm.ainvoke([
            SystemMessage(content=ITINERARY_AGENT_SYSTEM_PROMPT),
            *state.get("messages", []),
            HumanMessage(
                content=f"Here is the gathered information:\n\n{structured_context}\n\n"
                        f"Please create a detailed day-by-day itinerary."
            ),
        ])

        return {"itinerary": itinerary}

    return itinerary_agent