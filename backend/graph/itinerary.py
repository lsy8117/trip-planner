from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from backend.utilities import get_llm
from backend.models import (
    AttractionInfo,
    DayWeatherInfo,
    HotelInfo,
    PlannedActivity,
    PlannedDay,
    PlannedItinerary,
    PlannedItineraryDraft,
)
from backend.config import settings

def _format_weather_line(w: DayWeatherInfo) -> str:
    if w.error:
        return f"- {w.date}: unavailable ({w.error})"
    return f"- {w.date}: {w.condition}, {w.temp_min_c}–{w.temp_max_c}°C"


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
- The information that could be used for planning includes: weather_info, attractions, hotels.
- You should not use any other information, e.g. attractions or hotels that are not
  provided in the context. If the information is not provided, you should not make up any information.
- The itinerary should be planned based on the user's query and preferences.
- Use the default planning preferences unless the user stated otherwise.
- You should consider ALL the user's requirements in the current conversations, and if they
  contradict with the default planning preferences or user preferences, follow the user's requirements.
- You should also consider the availability of the attractions and hotels when planning the itinerary. 
  If an attraction or hotel is not available, you should not include it in the itinerary.

## Default Planning Preferences
- Consider weather conditions when selecting attractions. E.g. Prefer indoor
  activities on a rainy day.
- Attractions closer together should be planned on the same day to save travel time.
- Choose more popular and iconic attractions unless the user stated otherwise.
- Choose hotels that are more convenient to stay and travel.
- Always include lunch and dinner time in the itinerary
- Every day of the trip must have a hotel assigned — pick one hotel from the provided
  options and keep the traveler at that same hotel across consecutive nights unless
  they ask to switch. Only skip this if the hotel options list is empty.

** Selecting hotels and attractions **
- Do not transcribe hotel or attraction details yourself. Instead, reference them by key:
  - For an activity drawn from the provided attractions list, set `attraction_key` to that
    attraction's exact name and leave `attraction_type`/`description` blank — they will be
    filled in automatically from the source data.
  - `attraction_key` is null ONLY for logistics activities: meals, travel/transit, hotel
    check-in/out, rest, or free time. For these, `attraction_type` must be one of the fixed
    logistics categories provided by the schema.
  - Any real point of interest — a landmark, museum, tour, viewpoint, or anything else someone
    would sightsee — MUST be selected via `attraction_key` from the provided attractions list.
    If it is not in that list, do NOT include it, under any name or category. Never invent a
    place to visit; only plan with attractions that were actually provided.
  - For a day's hotel, set `hotel_key` to the exact hotel_key of your chosen hotel from the
    provided hotel options. EVERY day must have a hotel_key set — reuse the same hotel_key on
    consecutive nights unless the user wants to change hotels partway through. Leave it null
    only if the hotel options list is empty.
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
            weather_lines = "\n".join(_format_weather_line(w) for w in weather_info)
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

def _format_context(
    weather_info: list[DayWeatherInfo],
    attractions: list[AttractionInfo],
    hotels: list[HotelInfo],
) -> str:
    """Render available weather/attractions/hotels as text, including the keys
    (attraction name, hotel_key) the LLM should reference rather than transcribe."""
    context_parts = []

    if weather_info:
        weather_lines = "\n".join(_format_weather_line(w) for w in weather_info)
        context_parts.append(f"Weather forecast:\n{weather_lines}")

    if attractions:
        attraction_lines = "\n".join(
            f"- {a.name} ({a.attraction_type})"
            + (f" at ({a.latitude:.4f}, {a.longitude:.4f})" if a.latitude and a.longitude else "")
            + (f": {a.description[:100]}" if a.description and a.description != "null" else "")
            for a in attractions
        )
        context_parts.append(f"Available attractions (reference by exact name as attraction_key):\n{attraction_lines}")

    if hotels:
        hotel_lines = "\n".join(
            f"- {h.name} (hotel_key: {h.hotel_key})"
            + (f" (rating: {h.rating})" if h.rating else "")
            + (f", from ${min(o.price_usd for o in h.offers):.0f}/night" if h.offers else "")
            for h in hotels
        )
        context_parts.append(f"Hotel options (reference by hotel_key):\n{hotel_lines}")

    if not context_parts:
        context_parts.append(
            "No structured data was retrieved. Create a general itinerary "
            "based on the user's query alone."
        )

    return "\n\n".join(context_parts)


def _resolve_itinerary(
    draft: PlannedItineraryDraft,
    weather_info: list[DayWeatherInfo],
    attractions: list[AttractionInfo],
    hotels: list[HotelInfo],
) -> PlannedItinerary:
    """Reattach the real weather/attraction/hotel objects the LLM selected by key,
    instead of trusting anything it transcribed itself."""
    weather_map = {w.date: w for w in weather_info}
    attraction_map = {a.name: a for a in attractions}
    hotel_map = {h.hotel_key: h for h in hotels}

    resolved_days = []
    last_hotel: Optional[HotelInfo] = None
    for day in draft.days:
        resolved_activities = []
        for act in day.activities:
            attraction = attraction_map.get(act.attraction_key) if act.attraction_key else None
            if attraction:
                resolved_activities.append(PlannedActivity(
                    time=act.time,
                    name=attraction.name,
                    attraction_type=attraction.attraction_type,
                    description=attraction.description,
                    estimated_duration_hours=act.estimated_duration_hours,
                    latitude=attraction.latitude,
                    longitude=attraction.longitude,
                    notes=act.notes,
                    attraction_key=attraction.name,
                    booking_required=attraction.booking_required,
                    estimated_fee_usd=attraction.estimated_fee_usd,
                    fee_notes=attraction.fee_notes,
                ))
            else:
                # Custom (logistics-only) activity with no matching attraction — attraction_type
                # is a free-form str set by the LLM (see PlannedActivityDraft.attraction_type),
                # not the CustomActivityType enum.
                resolved_activities.append(PlannedActivity(
                    time=act.time,
                    name=act.name,
                    attraction_type=act.attraction_type or "other_logistics",
                    description=act.description or act.name,
                    estimated_duration_hours=act.estimated_duration_hours,
                    notes=act.notes,
                ))

        # Every day should have a hotel. The LLM is instructed to always set hotel_key,
        # but if it doesn't (or the key doesn't match), carry the previous day's hotel
        # forward — travelers typically keep the same hotel across consecutive nights —
        # falling back to the first available hotel on day one. Only stays None if no
        # hotel data exists at all (hotel search found nothing).
        hotel = hotel_map.get(day.hotel_key) if day.hotel_key else None
        if hotel is None:
            hotel = last_hotel or (hotels[0] if hotels else None)
        last_hotel = hotel

        resolved_days.append(PlannedDay(
            date=day.date,
            weather_summary=day.weather_summary,
            weather=weather_map.get(day.date),
            activities=resolved_activities,
            hotel=hotel,
        ))

    return PlannedItinerary(
        destination=draft.destination,
        trip_summary=draft.trip_summary,
        days=resolved_days,
        total_estimated_budget_usd=draft.total_estimated_budget_usd,
        travel_tips=draft.travel_tips,
    )


def build_itinerary_node():
    itinerary_llm = get_llm(provider=settings.itinerary_model_provider, model=settings.itinerary_model).with_structured_output(
        PlannedItineraryDraft#, method="json_schema"
    ) \
        .with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 1.0},
        )

    async def itinerary_agent(state: dict) -> dict:
        weather_info: list[DayWeatherInfo] = state.get("weather_info") or []
        attractions: list[AttractionInfo] = state.get("attractions") or []
        hotels: list[HotelInfo] = state.get("hotels") or []
        instruction = state.get("itinerary_edit_instruction", "")

        structured_context = _format_context(weather_info, attractions, hotels)

        messages = [SystemMessage(content=ITINERARY_AGENT_SYSTEM_PROMPT)]

        if state.get("current_itinerary"):
            messages.append(HumanMessage(content=f"Current itinerary:\n{state['current_itinerary'].model_dump_json()}"))
        
        messages.append(HumanMessage(content=f"Here is the gathered information:\n\n{structured_context}\n\n"
                            f"Please create a detailed day-by-day itinerary."))
        
        if state.get("itinerary_edit_instruction"):
            messages.append(HumanMessage(content=f"Edit instruction: {instruction}"))
        
        if state.get("refinement_request"):
            messages.append(HumanMessage(content=f"User's past edit requests: {state.get('refinement_request', [])}"))
        


        # with_structured_output doesn't stream — it returns the full Pydantic object
        try:
            draft: PlannedItineraryDraft = await itinerary_llm.ainvoke(messages)
        except Exception as e:
            return {"errors": [f"Failed to generate itinerary: {e}"]}

        itinerary = _resolve_itinerary(draft, weather_info, attractions, hotels)
        return {"current_itinerary": itinerary, "errors": []}

    return itinerary_agent

PATCH_SYSTEM_PROMPT = """
You are editing an EXISTING trip itinerary. You will be given the current
itinerary in full, an edit instruction and user's past edit requests.

Rules:
- Return the complete itinerary structure, but change ONLY what the edit
  instruction asks for.
- Do not alter days, activities, hotels, or timing that are not directly
  affected by the instruction.
- If the instruction references new attraction/hotel data, use only the
  data provided — do not fabricate.
- Make sure the final itinerary meets the user's requirements and preferences, 
  including any past edit requests.
- You should consider the availability of the attractions and hotels when editing the itinerary.

** Selecting hotels and attractions **
- Do not transcribe hotel or attraction details yourself. Reference them by key:
  - For an activity drawn from the attractions list, set `attraction_key` to that attraction's
    exact name and leave `attraction_type`/`description` blank.
  - `attraction_key` is null ONLY for logistics activities (meals, travel/transit, check-in/out,
    rest, free time), using one of the fixed logistics categories for `attraction_type`. Any real
    point of interest not in the provided attractions list must be left out — never invented.
  - For a day's hotel, set `hotel_key` to the exact hotel_key of your chosen hotel. EVERY day
    must keep a hotel_key set — do not drop it unless the edit instruction specifically removes
    the hotel for that night.
  - For unaffected days/activities carried over unchanged from the current itinerary, still set
    `attraction_key`/`hotel_key` to the same keys they already had, if known.
"""

def build_itinerary_patch_node():
    llm = get_llm(provider=settings.itinerary_patch_model_provider, model=settings.itinerary_patch_model).with_structured_output(
        PlannedItineraryDraft#, method="json_schema"
    ) \
        .with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 1.0},
        )

    async def itinerary_patch(state: dict) -> dict:
        current = state["current_itinerary"]
        instruction = state.get("itinerary_edit_instruction", "")

        weather_info: list[DayWeatherInfo] = state.get("weather_info") or []
        attractions: list[AttractionInfo] = state.get("attractions") or []
        hotels: list[HotelInfo] = state.get("hotels") or []

        structured_context = _format_context(weather_info, attractions, hotels)

        try:
            patched: PlannedItineraryDraft = await llm.ainvoke([
                SystemMessage(content=PATCH_SYSTEM_PROMPT),
                HumanMessage(content=f"Current itinerary:\n{current.model_dump_json()}"),
                HumanMessage(content=f"Available information:\n\n{structured_context}"),
                HumanMessage(content=f"Edit instruction: {instruction}"),
                HumanMessage(content=f"User's past edit requests: {state.get('refinement_request', [])}"),
            ])
        except Exception as e:
            return {"errors": [f"Failed to patch itinerary: {e}"]}

        itinerary = _resolve_itinerary(patched, weather_info, attractions, hotels)
        return {"current_itinerary": itinerary, "errors": []}

    return itinerary_patch