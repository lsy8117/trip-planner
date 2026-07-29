# backend/nodes/classify_request.py

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from backend.utilities import get_llm
from backend.memory.capture import log_user_message
from backend.models import TripIntent, RefinementDelta, OrchestratorState
from backend.config import settings

SYSTEM_PROMPT = """
## Role
You are a travel planning assistant. You are specialized in understanding user requests
and overlooking the entire planning process. 

## Task & Responsibilities
Given the conversation history and current itinerary (if any), determine:
1. Is this a first-time request or a refinement request?
    - If this is a first-time request, extract the user's travel intent and preferences 
into a TripIntent object.
    - If this is a refinement request, extract what the user wants to change and update 
the existing TripIntent accordingly.
2. Determine which subgraphs need to be run. 
3. If this is a refinement request, determine how the user wants to change the 
itinerary plan and whether the itinerary needs a complete replan or just a minor 
adjustment.

Additionally determine change_scope:
- "minor_edit": user wants to swap/add/remove some attraction, hotel, or activity,
  or make a small tweak, while keeping the rest of the plan intact.
- "full_plan": need to plan an entirely new itinerary, either because this is a
first-time request or the user wants a substantially different plan.

## Disambiguating attraction_requirements vs itinerary_requirements
These are easy to mix up — use this rule: if the requirement names or implies ANY
type/category of place or activity (museums, historical sites, shopping, hiking, live
shows, nightlife, natural scenery, accessibility, etc.), it goes in
attraction_requirements — even if it's phrased as "a day for X" or "no day of Y".
Only put something in itinerary_requirements if it is PURELY about schedule/pacing/
structure and does not reference any kind of place or activity (e.g. "no early
mornings", "max 2 activities per day", "leave a day free", "back at the hotel by 6pm").
This distinction matters: only attraction_requirements reaches the attraction search
agent, so a requirement misfiled under itinerary_requirements will never be searched for.

Return a JSON object with:
- updated_intent: the full updated TripIntent (merge changes into the existing one)
- subgraphs_to_run: list of ["weather", "attraction", "hotel"] that need to be run, for example:
  - If this is a first-time request → ["weather", "attraction", "hotel"]
  - If only hotel preferences changed → ["hotel"]
  - If user preferences for attractions changed → ["attraction"]
  - If dates changed → ["weather", "hotel"] (attractions rarely change by date)
  - If destination changed → ["weather", "attraction", "hotel"]  
  - If only the itinerary structure/narrative needs adjustment → []
- is_first_turn: true if there is no existing itinerary
- change_scope: "minor_edit" or "full_plan" as described above
- itinerary_edit_instruction: understand the user's request and provide a concise 
instruction for the itinerary agent to follow when editing/planning the itinerary.
You should consider all user's requirements and requests in the current conversation.
This should be a short, clear instruction that captures the user's intent and desired changes.
"""

async def classify_request(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    log_user_message(state, config)

    # Explicit non-reasoning model: gpt-oss (Groq's default here) burns its token
    # budget on hidden reasoning before emitting JSON, and fails validation with an
    # empty failed_generation on a schema this large (RefinementDelta + nested TripIntent).
    llm = get_llm(provider=settings.classify_request_model_provider, model=settings.classify_request_model) \
        .with_structured_output(RefinementDelta) \
        .with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 1.0},
        )
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    # Give it the existing context if this is a follow-up
    if state.get("current_intent"):
        messages.append(SystemMessage(content=f"""
            Current intent: {state['current_intent'].model_dump_json()}
            Current itinerary summary: {_summarize_itinerary(state.get('current_itinerary'))}
            """))
    

    past_requests = [state["user_query"]] if state.get("user_query") else []
    past_requests.extend(state.get("refinement_request") or [])
    messages.append(HumanMessage(content="\n".join(past_requests)))
    
    delta: RefinementDelta = await llm.ainvoke(messages)
    
    # Build the execution plan from what needs re-running
    plan = list(delta.subgraphs_to_run)
    if plan or delta.is_first_turn or delta.change_scope == "full_replan":
        plan.append("itinerary")
    else:
        plan.append("itinerary_patch")
    
    return {
        "current_intent": delta.updated_intent,
        "subgraphs_to_run": delta.subgraphs_to_run,
        "execution_plan": plan,
        "itinerary_edit_instruction": delta.itinerary_edit_instruction,
    }

def _summarize_itinerary(itinerary) -> str:
    if not itinerary:
        return "None yet."
    return f"{itinerary.destination}, {len(itinerary.days)} days planned"