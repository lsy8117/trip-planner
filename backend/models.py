import difflib
import json
import math
import re
from typing import Annotated, List, Optional, TypedDict
from annotated_types import Annotated
from operator import add
from pydantic import BaseModel, Field, field_validator
from datetime import date
from enum import Enum

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class DayWeatherInfo(BaseModel):
    location: str = Field(..., examples=["Chicago"])
    date: str = Field(..., examples=["2026-06-20"])
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    condition: Optional[str] = None
    error: Optional[str] = None

class AttractionInfo(BaseModel):
    name: str = Field(..., description='Name of the attraction or event.')
    attraction_type: str = Field(..., description='Tourist attraction type.', examples=['Museum','Historical Site','Shopping','Cruise Tour'])
    description: str = Field(..., description='Brief description of the attraction extracted from the search results.')
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    booking_required: Optional[bool] = None
    estimated_fee_usd: Optional[float] = None
    fee_notes: Optional[str] = None
    availability: Optional[str] = Field(None, description='Availability information for the attraction, e.g. "Open daily 9am-5pm", "Closed on Mondays".')

class AttractionsList(BaseModel):
    attractions: list[AttractionInfo]

class HotelOffer(BaseModel):
    ota: str
    price_usd: float

class HotelInfo(BaseModel):
    name: str
    hotel_key: str
    rating: Optional[float] = None
    address: Optional[str] = None
    offers: list[HotelOffer] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Reducer for attractions (dedup by fuzzy name + geolocation)
# ─────────────────────────────────────────────

_NAME_SIMILARITY_THRESHOLD = 0.85   # difflib ratio on normalized names, 0-1
_GEO_DISTANCE_THRESHOLD_KM = 0.3    # ~300m — same landmark despite geocoding jitter


def _normalize_attraction_name(name: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", name.lower()).strip()


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _normalize_attraction_name(a), _normalize_attraction_name(b)).ratio()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _is_same_attraction(a: AttractionInfo, b: AttractionInfo) -> bool:
    """Two attractions are duplicates if their names are near-identical AND, when both
    have coordinates, those coordinates sit within a small radius of each other. If
    either is missing coordinates (e.g. before geocoding has run), name similarity
    alone decides."""
    if _name_similarity(a.name, b.name) < _NAME_SIMILARITY_THRESHOLD:
        return False

    has_coords = (
        a.latitude is not None and a.longitude is not None
        and b.latitude is not None and b.longitude is not None
    )
    if not has_coords:
        return True

    return _haversine_km(a.latitude, a.longitude, b.latitude, b.longitude) <= _GEO_DISTANCE_THRESHOLD_KM


def merge_attractions(
        existing: list[AttractionInfo],
        incoming: list[AttractionInfo],
) -> list[AttractionInfo]:
    merged = list(existing)
    for candidate in incoming:
        match_idx = next(
            (i for i, current in enumerate(merged) if _is_same_attraction(current, candidate)),
            None,
        )
        if match_idx is None:
            merged.append(candidate)
        else:
            merged[match_idx] = candidate  # newer record wins (e.g. more complete fields)
    return merged




class PlannedActivity(BaseModel):
    time: str = Field(description="Suggested time, e.g. '9:00 AM'")
    name: str = Field(description="Name of the attraction or activity")
    attraction_type: str = Field(description="e.g. 'museum', 'landmark', 'restaurant'")
    description: str = Field(description="Brief description of what to do/see here")
    estimated_duration_hours: float = Field(description="How long to spend here")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = Field(None, description="Practical tips, booking info, weather considerations")
    attraction_key: Optional[str] = Field(None, description="Name of the source AttractionInfo this activity was resolved from, if any")
    booking_required: Optional[bool] = None
    estimated_fee_usd: Optional[float] = None
    fee_notes: Optional[str] = None

class PlannedDay(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    weather_summary: Optional[str] = Field(None, description="Brief weather note for the day, e.g. 'Light drizzle, 17–26°C'")
    weather: Optional[DayWeatherInfo] = Field(None, description="Exact forecast data for this date, resolved from weather_info")
    activities: list[PlannedActivity]
    hotel: Optional[HotelInfo] = Field(None, description="Recommended hotel for this night")

class PlannedItinerary(BaseModel):
    destination: str
    trip_summary: str = Field(description="2-3 sentence overview of the trip plan")
    days: list[PlannedDay]
    total_estimated_budget_usd: Optional[float] = Field(None, description="Rough total budget estimate if inferable from hotel/attraction prices")
    travel_tips: list[str] = Field(default_factory=list, description="General tips for the trip")


# ─────────────────────────────────────────────
# Draft schemas — what the LLM actually fills in.
#
# The LLM selects existing hotels/attractions/weather by key instead of
# transcribing their full structured data (which is unreliable, especially
# for nested numeric fields like hotel offers). The real objects are
# reattached from graph state after generation — see itinerary.py's
# `_resolve_itinerary`.
# ─────────────────────────────────────────────

class CustomActivityType(str, Enum):
    """Non-touristy logistics activities the itinerary agent may invent without a
    matching attraction_key. Any actual point of interest — sightseeing, landmark,
    museum, tour, etc. — MUST be resolved via attraction_key from the provided
    attractions list, or left out entirely if it isn't in that list."""
    MEAL = "meal"
    TRAVEL = "travel"
    CHECK_IN = "check_in"
    CHECK_OUT = "check_out"
    REST = "rest"
    FREE_TIME = "free_time"
    OTHER_LOGISTICS = "other_logistics"


class PlannedActivityDraft(BaseModel):
    time: str = Field(description="Suggested time, e.g. '9:00 AM'")
    name: str = Field(
        description="Name of the activity. For custom (non-attraction_key) activities, use a "
                    "generic label like 'Lunch' or 'Travel to hotel' — never a specific place "
                    "name that isn't in the provided attractions list."
    )
    attraction_key: Optional[str] = Field(
        None,
        description="Exact name of the matching attraction from the provided attractions list. "
                    "Null ONLY for logistics activities (meal, travel, check-   in/out, rest, free "
                    "time) that are not a specific point of interest."
    )
    attraction_type: Optional[str] = Field(
        None,
        description="Either the type of the matching attraction (if attraction_key is set), or simply 'Logistics' for a custom activity. "
                    "Logistics activities includes meal, travel, check-in/out, rest."
    )
    # attraction_type: Optional[CustomActivityType] = Field(
    #     None,
    #     description="Only set when attraction_key is null, and only to one of these fixed "
    #                 "logistics categories. Never used to label a real point of interest — any "
    #                 "actual attraction must go through attraction_key or be omitted."
    # )
    description: Optional[str] = Field(None, description="Brief description of what to do/see here. Only needed if attraction_key is null.")
    estimated_duration_hours: float = Field(description="How long to spend here")
    notes: Optional[str] = Field(None, description="Practical tips, booking info, weather considerations")

class PlannedDayDraft(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    weather_summary: Optional[str] = Field(None, description="Brief weather note or planning consideration for the day")
    activities: list[PlannedActivityDraft]
    hotel_key: Optional[str] = Field(
        None,
        description="hotel_key of the selected hotel from the provided hotel options for this night. "
                    "Every day MUST have one set — if staying at the same hotel across multiple nights, "
                    "repeat the same hotel_key on each of those days. Only null if the hotel options "
                    "list is empty."
    )

class PlannedItineraryDraft(BaseModel):
    destination: str
    trip_summary: str = Field(description="2-3 sentence overview of the trip plan")
    days: list[PlannedDayDraft]
    total_estimated_budget_usd: Optional[float] = Field(None, description="Rough total budget estimate if inferable from hotel/attraction prices")
    travel_tips: list[str] = Field(default_factory=list, description="General tips for the trip")

class TripIntent(BaseModel):
    """Parsed user intent for a trip planning request."""
    
    # Core trip parameters
    destination: str = Field(
        description="City or region the user wants to visit, e.g. 'Tokyo' or 'Kyoto, Japan'"
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Trip start date. Null if user only specified a season or month."
    )
    end_date: Optional[date] = Field(
        default=None,
        description="Trip end date. Null if user gave a duration instead."
    )
    duration_days: Optional[int] = Field(
        default=None,
        description="Number of days for the trip. Derived from dates if both given."
    )
    
    # Preference signals — used by subgraphs
    budget_level: Optional[str] = Field(
        default=None,
        description="One of: budget, mid-range, luxury. Null if unspecified."
    )
    traveler_count: Optional[int] = Field(
        default=1,
        description="Number of travelers."
    )
    traveler_type: Optional[str] = Field(
        default=None,
        description="E.g. solo, couple, family, group."
    )
    
    # Hotel-specific signals
    # hotel_location_preference: Optional[str] = Field(
    #     default=None,
    #     description="Preferred area for hotels, e.g. 'near Shinjuku', 'city centre'."
    # )
    # hotel_amenities: Optional[list[str]] = Field(
    #     default_factory=list,
    #     description="Requested hotel features, e.g. ['pool', 'breakfast included']."
    # )
    # hotel_preferences: Optional[list[str]] = Field(
    #     default_factory=list,
    #     description="Requested hotel preferences indicated by past user conversation, e.g. ['pool', 'breakfast included']."
    # )

    hotel_requirements: Optional[list[str]] = Field(
        default_factory=list,
        description="Explicit hotel requirements indicated by in the current user conversation, e.g. ['must have pool']."
    )
    
    # Attraction-specific signals
    # avoid: Optional[list[str]] = Field(
    #     default_factory=list,
    #     description="Things the user explicitly wants to avoid, e.g. ['crowded tourist spots']."
    # )
    # pace: Optional[str] = Field(
    #     default=None,
    #     description="One of: relaxed, moderate, packed. How busy the day schedule should be."
    # )
    # attraction_preferences: Optional[list[str]] = Field(
    #     default_factory=list,
    #     description="Attraction preferences indicated by past user conversation, e.g. ['prefers historical sites', 'always wants to go iconic landmarks']"
    # )

    attraction_requirements: Optional[list[str]] = Field(
        default_factory=list,
        description="Requirements about WHAT kind of places/activities to include or exclude — "
                    "any category, type, or characteristic of an attraction, even if phrased as "
                    "'a day for X'. If the requirement names or implies a type of place or activity "
                    "(museums, historical sites, shopping, hiking, live shows, nightlife, natural "
                    "scenery, accessibility, etc.), it belongs here, NOT in itinerary_requirements — "
                    "the attraction search agent can only act on requirements listed here. "
                    "E.g. ['must be wheelchair accessible', 'want to see natural scenery / hiking trails', "
                    "'want to watch an iconic live show', 'no shopping', 'must include a day for hiking']."
    )

    # itinerary_preferences: Optional[list[str]] = Field(
    #     default_factory=list,
    #     description="Itinerary preferences indicated by past user conversation, e.g. ['enjoys relaxed pacing', 'activities start no earlier than 10am']"
    # )

    itinerary_requirements: Optional[list[str]] = Field(
        default_factory=list,
        description="Requirements about SCHEDULE/STRUCTURE only — pacing, timing, or day allocation "
                    "that does NOT name or imply any type of place or activity. If it references what "
                    "kind of attraction to visit, it belongs in attraction_requirements instead, even "
                    "if phrased as 'a day for X'. E.g. ['keep the pace relaxed, max 2 activities per day', "
                    "'no early mornings', 'leave one day completely free/unplanned', "
                    "'be back at the hotel by 6pm each day']."
    )
    
    # Raw user input — preserved for context
    raw_request: str = Field(
        description="The original user message, verbatim."
    )

class ChangeScope(str, Enum):
    MINOR_EDIT = "minor_edit"     # e.g. swap one attraction, change one hotel
    FULL_PLAN = "full_plan"   # destination/dates/major structural change

class RefinementDelta(BaseModel):
    updated_intent: TripIntent
    subgraphs_to_run: list[str]  # subset of ["weather", "attractions", "hotels"]
    is_first_turn: bool = False
    change_scope: ChangeScope
    itinerary_edit_instruction: str


# ─────────────────────────────────────────────
# Orchestrator state
# ─────────────────────────────────────────────

class OrchestratorState(TypedDict):
    # Conversation messages (user query lives here as HumanMessage)
    messages: Annotated[list[BaseMessage], add_messages]
    current_intent: Optional[TripIntent]  # the active parsed intent
    current_itinerary: Optional[PlannedItinerary]  # the itinerary being refined
    refinement_request: Annotated[list[str], add]
    itinerary_edit_instruction: Optional[str]  # concise instruction for the itinerary agent
    subgraphs_to_run: list[str]

    # Raw user query string passed into each subgraph
    user_query: str

    # Results written by each subgraph
    weather_info: Annotated[list[DayWeatherInfo], add]
    attractions: Annotated[list[AttractionInfo], merge_attractions]
    hotels: Annotated[list[HotelInfo], add]

    # Search queries the attraction agent has already run, across all turns —
    # passed back into the attraction subgraph so it doesn't repeat a search.
    attraction_search_history: Annotated[list[str], add]

    # Error(s) from the current turn's itinerary/itinerary_patch run only — no reducer,
    # so each turn's node overwrites it outright (set on failure, cleared on success)
    # instead of accumulating stale errors from earlier turns. Safe because exactly one
    # of itinerary/itinerary_patch runs per turn (see classify_request), so there's never
    # a concurrent write to merge.
    errors: list[str]

    # Execution plan: list of agent names (or lists of names for parallel groups)
    # e.g. ["weather", "attractions", "hotels"]  — fully sequential
    # e.g. [["weather", "attractions"], "hotels"] — parallel first group, then hotels
    execution_plan: list


# ─────────────────────────────────────────────
# Memory Management
# ─────────────────────────────────────────────

class UserMemory(BaseModel):
    user_id: str
    attraction_preferences: list[str] = Field(default_factory=list)
    hotel_preferences: list[str] = Field(default_factory=list)
    itinerary_preferences: list[str] = Field(default_factory=list)
    # 0 == never updated. Defaulting to "now" here would be misleading — a
    # freshly-constructed blank memory hasn't actually been analyzed yet.
    last_updated: float = 0.0

class MemoryPreferences(BaseModel):
    """Structured output schema for the memory-update LLM call — just the
    three revised lists. user_id/last_updated are set by code, not the LLM."""
    attraction_preferences: list[str] = Field(default_factory=list)
    hotel_preferences: list[str] = Field(default_factory=list)
    itinerary_preferences: list[str] = Field(default_factory=list)

class SessionMemoryState(BaseModel):
    # 0 == this session has never been read by the memory updater, so a
    # fresh run should read its event log from the very start.
    last_memory_sync_at: float = 0.0

class MemoryEventKind(str, Enum):
    USER_MESSAGE = "user_message"
    ITINERARY_GENERATED = "itinerary_generated"

class MemoryEvent(BaseModel):
    """One line in a session's memory_events.jsonl — the raw evidence the
    memory updater analyzes. Only `text` (for USER_MESSAGE) or `itinerary`
    (for ITINERARY_GENERATED) is set, depending on `kind`."""
    kind: MemoryEventKind
    ts: float
    text: Optional[str] = None
    itinerary: Optional[PlannedItinerary] = None