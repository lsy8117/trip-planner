"""
Orchestrator graph for the trip planning agent system.
 
Execution plan controls agent order dynamically per-request:
    Sequential:  execution_plan = ["weather", "attractions", "hotels"]
    Parallel:    execution_plan = [["weather", "attractions"], "hotels"]
 
The itinerary node always runs last (after the plan is exhausted) and is not
part of the execution_plan — it is a fixed final step.
"""
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from operator import add
 
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from langgraph.types import Send
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from backend.models import AttractionInfo, OrchestratorState, DayWeatherInfo, HotelInfo, PlannedItinerary, TripIntent
from backend.graph.weather import weather_subgraph
from backend.graph.attraction import attraction_subgraph
from backend.graph.hotel import hotel_subgraph
from backend.graph.itinerary import build_itinerary_node, build_itinerary_patch_node
from backend.graph.respond import respond
from backend.graph.classify_request import classify_request

# ─────────────────────────────────────────────
# Plan node
# ─────────────────────────────────────────────
 
def plan_execution(state: OrchestratorState) -> dict:
    """
    Decide execution order. Called once at the start of every user request.
    Modify this logic to encode your routing strategy.
    """
    # Default: sequential weather → attractions → hotels
    # Swap to [["weather", "attractions"], "hotels"] for parallel first pass
    return {
        "execution_plan": ["weather", "attractions", "hotels"]
    }


# ─────────────────────────────────────────────
# Separate init from routing
# ─────────────────────────────────────────────

def initialize_plan(state: OrchestratorState) -> dict:
    """Runs ONCE at START. Sets the execution plan."""
    return {
        "execution_plan": ["weather", "attractions", "hotels"]
    }

# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

def route_next(state: OrchestratorState) -> str | list:
    """
    Runs after each subgraph completes.
    Peeks at the current plan head — does NOT reset the plan.
    """
    plan = state.get("execution_plan") or []

    # if not plan:
    #     if state.get("change_scope") == "minor_edit":
    #         return "itinerary_patch"
    #     return "itinerary"
    next_step = plan[0]

    if isinstance(next_step, list):
        return [Send(agent, {**state, "execution_plan": plan[1:]}) for agent in next_step]

    return next_step
 
 
 
# ─────────────────────────────────────────────
# Subgraph wrapper nodes
# Each wrapper:
#   1. Maps orchestrator state → subgraph input
#   2. Invokes the subgraph
#   3. Maps subgraph output → orchestrator state
#   4. Pops itself from the execution plan
# ─────────────────────────────────────────────
 
def _pop_plan(state: OrchestratorState) -> list:
    """Remove the first item from the execution plan (handles both str and list heads)."""
    plan = state.get("execution_plan") or []
    return plan[1:]
 
 
def make_weather_node(subgraph):
    async def weather_node(state: OrchestratorState) -> dict:
        result = await subgraph.ainvoke({
            "user_query": state["user_query"],
            # messages intentionally omitted — subgraph starts fresh
        })
        return {
            "weather_info": result.get("weather_info") or [],
            "execution_plan": _pop_plan(state),
        }
    return weather_node
 
 
def make_attraction_node(subgraph):
    async def attraction_node(state: OrchestratorState) -> dict:
        past_queries = state.get("attraction_search_history") or []
        trip_intent = state.get("current_intent")
        # getattr with a default handles both trip_intent being None (first turn before
        # classify_request runs) and attraction_preferences not yet being a TripIntent field.
        attraction_requirements = getattr(trip_intent, "attraction_requirements", None) or []
        attraction_preferences = getattr(trip_intent, "attraction_preferences", None) or []
        result = await subgraph.ainvoke({
            "user_query": state["user_query"],
            # Pass weather_info if available so the attraction agent can consider it
            "weather_info": state.get("weather_info") or [],
            "past_search_queries": past_queries,
            "attraction_requirements": attraction_requirements,
            "attraction_preferences": attraction_preferences,
        })
        new_queries = result.get("search_queries_used") or []
        return {
            "attractions": result.get("attractions") or [],
            # Only append queries not already recorded — add-reducer would otherwise
            # accumulate duplicates if the agent re-runs an old query anyway.
            "attraction_search_history": [q for q in new_queries if q not in past_queries],
            "execution_plan": _pop_plan(state),
        }
    return attraction_node
 
 
def make_hotel_node(subgraph):
    async def hotel_node(state: OrchestratorState) -> dict:
        result = await subgraph.ainvoke({
            "user_query": state["user_query"],
            "weather_info": state.get("weather_info") or [],
            "attractions": state.get("attractions") or [],
        })
        return {
            "hotels": result.get("hotels") or [],
            "execution_plan": _pop_plan(state),
        }
    return hotel_node
 
 
# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """checkpointer defaults to an in-memory MemorySaver (lost on restart).
    Pass a persistent one (e.g. AsyncSqliteSaver — see graph/session_graph.py)
    for conversations that must survive a process restart."""
    builder = StateGraph(OrchestratorState)

    # Wrap subgraphs so they map state and pop the plan
    weather_node = make_weather_node(weather_subgraph)
    attraction_node = make_attraction_node(attraction_subgraph)
    hotel_node = make_hotel_node(hotel_subgraph)
    itinerary_node = build_itinerary_node()
    

    builder.add_node("classify_request", classify_request)
    builder.add_node("weather", weather_node)
    builder.add_node("attraction", attraction_node)
    builder.add_node("hotel", hotel_node)
    builder.add_node("itinerary", itinerary_node)
    builder.add_node("itinerary_patch", build_itinerary_patch_node())
    builder.add_node("respond", respond)

    targets = {"weather": "weather", "attraction": "attraction",
               "hotel": "hotel", "itinerary": "itinerary", "itinerary_patch": "itinerary_patch"}

    builder.add_edge(START, "classify_request")

    builder.add_conditional_edges("classify_request", route_next, targets)
    builder.add_conditional_edges("weather", route_next, targets)
    builder.add_conditional_edges("attraction", route_next, targets)
    builder.add_conditional_edges("hotel", route_next, targets)

    builder.add_edge("itinerary", "respond")
    builder.add_edge("itinerary_patch", "respond")
    builder.add_edge("respond", "classify_request")

    return builder.compile(checkpointer=checkpointer or MemorySaver())


orchestrator = build_graph()
 
# ─────────────────────────────────────────────
# Entry point helper
# ─────────────────────────────────────────────
 
async def plan_trip(user_query: str) -> OrchestratorState:
    """Convenience wrapper for invoking the orchestrator."""
    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=user_query)],
        "current_intent": None,
        "current_itinerary": None,
        "refinement_request": [],
        "itinerary_edit_instruction": None,
        "subgraphs_to_run": [],
        "user_query": user_query,
        "weather_info": [],
        "attractions": [],
        "hotels": [],
        "attraction_search_history": [],
        "errors": [],
        "execution_plan": [],      # will be populated by plan_execution node
    }
    return await orchestrator.ainvoke(initial_state)


