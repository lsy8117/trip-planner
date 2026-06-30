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
from langgraph.graph.message import add_messages
from langgraph.types import Send
 
from backend.models import AttractionInfo, DayWeatherInfo, HotelInfo, PlannedItinerary
from backend.graph.weather import weather_subgraph
from backend.graph.attraction import attraction_subgraph
from backend.graph.hotel import hotel_subgraph
from backend.graph.itinerary import build_itinerary_node
 
 
# ─────────────────────────────────────────────
# Reducer for attractions (dedup by name)
# ─────────────────────────────────────────────
 
def merge_attractions(
    existing: list[AttractionInfo],
    incoming: list[AttractionInfo],
) -> list[AttractionInfo]:
    by_name = {a.name: a for a in existing}
    for a in incoming:
        by_name[a.name] = a
    return list(by_name.values())
 
 
# ─────────────────────────────────────────────
# Orchestrator state
# ─────────────────────────────────────────────
 
class OrchestratorState(TypedDict):
    # Conversation messages (user query lives here as HumanMessage)
    messages: Annotated[list[BaseMessage], add_messages]
    current_intent: Optional[TripIntent]      # the active parsed intent
    current_itinerary: Optional[PlannedItinerary]  # the itinerary being refined
    refinement_request: Optional[str] 
 
    # Raw user query string passed into each subgraph
    user_query: str

    subgraphs_to_run: list[str]
 
    # Results written by each subgraph
    weather_info: Annotated[list[DayWeatherInfo], add]
    attractions: Annotated[list[AttractionInfo], merge_attractions]
    hotels: Annotated[list[HotelInfo], add]
 
    # Errors collected across subgraphs (concurrent-safe)
    errors: Annotated[list[str], add]
 
    # Execution plan: list of agent names (or lists of names for parallel groups)
    # e.g. ["weather", "attractions", "hotels"]  — fully sequential
    # e.g. [["weather", "attractions"], "hotels"] — parallel first group, then hotels
    execution_plan: list
    itinerary: PlannedItinerary | None 
 
 
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

    if not plan:
        return "itinerary"

    next_step = plan[0]

    if isinstance(next_step, list):
        return [Send(agent, state) for agent in next_step]

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
        result = await subgraph.ainvoke({
            "user_query": state["user_query"],
            # Pass weather_info if available so the attraction agent can consider it
            "weather_info": state.get("weather_info") or [],
        })
        return {
            "attractions": result.get("attractions") or [],
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

def build_orchestrator() -> StateGraph:
    builder = StateGraph(OrchestratorState)

    builder.add_node("initialize_plan", initialize_plan)
    builder.add_node("itinerary", build_itinerary_node())
    builder.add_node("weather", make_weather_node(weather_subgraph))
    builder.add_node("attractions", make_attraction_node(attraction_subgraph))
    builder.add_node("hotels", make_hotel_node(hotel_subgraph))

    # initialize_plan runs exactly once
    builder.add_edge(START, "initialize_plan")

    # After init, route to first agent
    builder.add_conditional_edges(
        "initialize_plan",
        route_next,
        {"weather": "weather", "attractions": "attractions",
         "hotels": "hotels", "itinerary": "itinerary"},
    )

    # After each subgraph, route directly — no more plan_execution in the loop
    builder.add_conditional_edges(
        "weather", route_next,
        {"weather": "weather", "attractions": "attractions",
         "hotels": "hotels", "itinerary": "itinerary"},
    )
    builder.add_conditional_edges(
        "attractions", route_next,
        {"weather": "weather", "attractions": "attractions",
         "hotels": "hotels", "itinerary": "itinerary"},
    )
    builder.add_conditional_edges(
        "hotels", route_next,
        {"weather": "weather", "attractions": "attractions",
         "hotels": "hotels", "itinerary": "itinerary"},
    )

    builder.add_edge("itinerary", END)

    return builder.compile()
 
 
orchestrator = build_orchestrator()
 
 
# ─────────────────────────────────────────────
# Entry point helper
# ─────────────────────────────────────────────
 
async def plan_trip(user_query: str) -> OrchestratorState:
    """Convenience wrapper for invoking the orchestrator."""
    initial_state: OrchestratorState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "weather_info": [],
        "attractions": [],
        "hotels": [],
        "errors": [],
        "execution_plan": [],      # will be populated by plan_execution node
        "itinerary": None, 
    }
    return await orchestrator.ainvoke(initial_state)
 