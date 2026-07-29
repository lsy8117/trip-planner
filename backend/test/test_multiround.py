"""
interactive_planner.py — Real-time streaming multi-turn travel planner

Run from the project root:
    uv run python interactive_planner.py

Type your trip request, watch progress stream in real time.
After each itinerary, type a follow-up to refine. Type 'quit' to exit.
"""

import sys
from pathlib import Path

# Walk up from backend/test/ to the project root
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

import asyncio
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from backend.graph.orchestrator import build_graph

NODE_LABELS = {
    "classify_request": "🔍  Understanding your request",
    "weather":          "🌤️  Fetching weather forecast",
    "attraction":       "🗺️  Searching attractions",
    "hotel":            "🏨  Searching hotels",
    "itinerary":        "📋  Building your itinerary",
    "respond":          "✅  Preparing response",
}

async def stream_and_display(graph, input_data, config):
    current_node = None

    async for event in graph.astream_events(input_data, config=config, version="v2"):
        event_type = event["event"]
        name = event.get("name", "")

        # Node start — print progress label
        if event_type == "on_chain_start" and name in NODE_LABELS:
            if name != current_node:
                current_node = name
                print(f"\n{NODE_LABELS[name]}...", flush=True)

        # Tool calls — show what the agent is querying
        elif event_type == "on_tool_start":
            tool_input = event.get("data", {}).get("input", {})
            detail = ""
            if isinstance(tool_input, dict):
                detail = (
                    tool_input.get("city")
                    or tool_input.get("search_query")
                    or tool_input.get("query", "")
                )
            print(f"   ↳ {name}({detail!r})", flush=True)

        # Stream LLM tokens only during itinerary generation
        elif event_type == "on_chat_model_stream" and current_node == "itinerary":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

        # Node end — show result counts for data-fetching nodes
        elif event_type == "on_chain_end" and name in NODE_LABELS:
            output = event.get("data", {}).get("output", {})
            if name == "weather":
                count = len(output.get("weather_info") or [])
                print(f"   ✓ {count} day(s) of forecast", flush=True)
            elif name == "attraction":
                count = len(output.get("attractions") or [])
                print(f"   ✓ {count} attraction(s) found", flush=True)
            elif name == "hotel":
                count = len(output.get("hotels") or [])
                print(f"   ✓ {count} hotel(s) found", flush=True)


async def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "interactive-session-001"}}

    print("=" * 60)
    print("  🌏 AI Travel Planner — Multi-turn")
    print("  Describe your trip to get started. Type 'quit' to exit.")
    print("=" * 60)

    # Turn 1 — initial request
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ("quit", "exit"):
        return

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user_query": user_input,
        "weather_info": [],
        "attractions": [],
        "hotels": [],
        "errors": [],
        "execution_plan": [],
        "current_intent": None,
        "current_itinerary": None,
        "refinement_request": [],
        "subgraphs_to_run": [],
    }

    print("\n" + "─" * 60)
    await stream_and_display(graph, initial_state, config)

    # Subsequent turns — refinement loop
    while True:
        state_snap = await graph.aget_state(config)

        if not state_snap.tasks:
            print("\n[Session ended]")
            break

        # Show the itinerary surfaced at interrupt()
        itinerary_text = state_snap.tasks[0].interrupts[0].value
        state = state_snap.values

        print("\n\n" + "=" * 60)
        print("📄 YOUR ITINERARY")
        print("=" * 60)
        print(itinerary_text)
        print("=" * 60)
        print(f"\n[{len(state.get('weather_info') or [])} weather days | "
              f"{len(state.get('attractions') or [])} attractions | "
              f"{len(state.get('hotels') or [])} hotels]")

        user_followup = input("\nYou (refine or 'quit'): ").strip()
        if user_followup.lower() in ("quit", "exit"):
            print("\nGoodbye! Safe travels 🌏")
            break

        print("\n" + "─" * 60)
        await stream_and_display(graph, Command(resume=user_followup), config)


if __name__ == "__main__":
    asyncio.run(main())