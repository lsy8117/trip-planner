from langgraph.types import interrupt
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from backend.graph.orchestrator import OrchestratorState
from backend.memory.capture import log_itinerary_generated

async def respond(state: OrchestratorState, config: RunnableConfig = None) -> dict:
    reply = format_itinerary(state["current_itinerary"])

    errors = state.get("errors") or []
    if errors:
        error_banner = "\n".join(f"⚠️ {e}" for e in errors)
        reply_body = reply if isinstance(reply, str) else reply.model_dump_json(indent=2)
        reply = f"{error_banner}\n\n{reply_body}"

    log_itinerary_generated(state, config)

    # Pause here, surface the itinerary to the user, wait for follow-up
    user_input = interrupt(reply)

    reply_text = reply if isinstance(reply, str) else reply.model_dump_json(indent=2)
    return {
        "messages": [
            AIMessage(content=reply_text),
            HumanMessage(content=user_input),
        ],
        "refinement_request": [user_input],

    }

def format_itinerary(itinerary):
    """
    Pass the PlannedItinerary through unchanged; frontend/telegram/formatter.py
    is responsible for rendering it (see format_for_telegram).
    """
    if itinerary is None:
        return "No itinerary available yet."
    return itinerary