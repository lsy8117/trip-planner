"""
test_graph.py — Multi-turn travel planner graph tests

Run from the project root with:
    uv run pytest tests/test_graph.py -v
or for a quick smoke test without pytest:
    uv run python tests/test_graph.py

Tests are layered from cheapest to most expensive:
  Layer 1 — Unit tests: pure logic, no LLM calls, no API calls
  Layer 2 — Node tests: single node in isolation with mocked LLM
  Layer 3 — Integration smoke test: full graph with real LLMs (costs money/quota)
"""

import sys
from pathlib import Path

# Walk up from backend/test/ to the project root
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command

# ---------------------------------------------------------------------------
# Adjust this import path to match your actual project layout
# ---------------------------------------------------------------------------
from backend.graph.orchestrator import build_graph, OrchestratorState, route_next, _pop_plan
from backend.graph.classify_request import classify_request, _summarize_itinerary
from backend.graph.respond import respond, format_itinerary
from backend.models import TripIntent, RefinementDelta, PlannedItinerary, PlannedDay


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — Pure unit tests (no LLM, no network)
# ═══════════════════════════════════════════════════════════════════════════

class TestRouteNext:
    """
    route_next should return the correct next node name based on execution_plan.
    No LLM or network involved.
    """

    def _make_state(self, plan):
        return {
            "execution_plan": plan,
            "messages": [],
            "weather_info": [],
            "attractions": [],
            "hotels": [],
            "errors": [],
            "current_intent": None,
            "current_itinerary": None,
            "refinement_request": [],
            "subgraphs_to_run": [],
            "user_query": "test",
        }

    def test_empty_plan_routes_to_itinerary(self):
        from backend.graph.orchestrator import route_next
        result = route_next(self._make_state([]))
        assert result == "itinerary", f"Expected 'itinerary', got {result!r}"

    def test_weather_first(self):
        from backend.graph.orchestrator import route_next
        result = route_next(self._make_state(["weather", "attractions", "hotels"]))
        assert result == "weather"

    def test_attractions_first(self):
        from backend.graph.orchestrator import route_next
        result = route_next(self._make_state(["attractions", "hotels"]))
        assert result == "attractions"

    def test_hotels_only(self):
        from backend.graph.orchestrator import route_next
        result = route_next(self._make_state(["hotels"]))
        assert result == "hotels"

    def test_itinerary_in_plan_routes_to_itinerary_node(self):
        from backend.graph.orchestrator import route_next
        result = route_next(self._make_state(["itinerary"]))
        assert result == "itinerary"

    def test_parallel_group_returns_send_list(self):
        from backend.graph.orchestrator import route_next
        from langgraph.types import Send
        result = route_next(self._make_state([["weather", "attractions"], "hotels"]))
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        destinations = {s.node for s in result}
        assert destinations == {"weather", "attractions"}


class TestPopPlan:
    """_pop_plan should always remove the first item."""

    def test_pop_string_head(self):
        from backend.graph.orchestrator import _pop_plan
        state = {"execution_plan": ["weather", "attractions", "hotels"]}
        assert _pop_plan(state) == ["attractions", "hotels"]

    def test_pop_list_head(self):
        from backend.graph.orchestrator import _pop_plan
        state = {"execution_plan": [["weather", "attractions"], "hotels"]}
        assert _pop_plan(state) == ["hotels"]

    def test_pop_single_item(self):
        from backend.graph.orchestrator import _pop_plan
        state = {"execution_plan": ["hotels"]}
        assert _pop_plan(state) == []

    def test_pop_empty_is_safe(self):
        from backend.graph.orchestrator import _pop_plan
        state = {"execution_plan": []}
        assert _pop_plan(state) == []


class TestSummarizeItinerary:
    """_summarize_itinerary handles None and populated itineraries."""

    def test_none_returns_placeholder(self):
        from backend.graph.classify_request import _summarize_itinerary
        assert _summarize_itinerary(None) == "None yet."

    def test_populated_itinerary(self):
        from backend.graph.classify_request import _summarize_itinerary
        from backend.models import PlannedItinerary, PlannedDay, PlannedActivity

        mock = MagicMock()
        mock.destination = "Tokyo"
        mock.duration_days = 5
        mock.days = [MagicMock()] * 5

        result = _summarize_itinerary(mock)
        assert "Tokyo" in result
        assert "5" in result


class TestFormatItinerary:
    """format_itinerary must always return a string, never a raw Pydantic object."""

    def test_returns_string(self):
        from backend.graph.respond import format_itinerary
        from backend.models import PlannedItinerary

        mock_itinerary = MagicMock()
        mock_itinerary.destination = "Paris"
        mock_itinerary.duration_days = 3
        mock_itinerary.days = []
        mock_itinerary.model_dump_json.return_value = '{"destination": "Paris"}'

        result = format_itinerary(mock_itinerary)
        assert isinstance(result, str), (
            "format_itinerary must return a str — interrupt() requires a string. "
            f"Got {type(result)} instead."
        )

    def test_none_is_handled(self):
        from backend.graph.respond import format_itinerary
        # Should not raise even if itinerary is None
        result = format_itinerary(None)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — Node-level tests with mocked LLM
# These test node logic without real API calls.
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyRequestNode:
    """
    classify_request should correctly set execution_plan and current_intent
    from the LLM's RefinementDelta output.
    """

    def _base_state(self, has_existing_itinerary=False):
        return {
            "messages": [HumanMessage(content="Plan a 5-day trip to Tokyo in July")],
            "current_intent": None,
            "current_itinerary": MagicMock() if has_existing_itinerary else None,
            "refinement_request": [],
            "subgraphs_to_run": [],
            "execution_plan": [],
            "user_query": "",
            "weather_info": [],
            "attractions": [],
            "hotels": [],
            "errors": [],
        }

    @pytest.mark.asyncio
    async def test_first_turn_sets_full_plan(self):
        """On first turn, all three subgraphs + itinerary should be in the plan."""
        from backend.graph.classify_request import classify_request
        from backend.models import RefinementDelta, TripIntent

        mock_delta = RefinementDelta(
            updated_intent=TripIntent(
                destination="Tokyo",
                duration_days=5,
                raw_request="Plan a 5-day trip to Tokyo in July",
            ),
            subgraphs_to_run=["weather", "attractions", "hotels"],
            is_first_turn=True,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_delta)
        mock_llm_with_structured = MagicMock()
        mock_llm_with_structured.ainvoke = AsyncMock(return_value=mock_delta)

        with patch("backend.graph.classify_request.get_llm") as mock_get_llm:
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm_with_structured
            result = await classify_request(self._base_state())

        assert "weather" in result["execution_plan"]
        assert "attractions" in result["execution_plan"]
        assert "hotels" in result["execution_plan"]
        assert "itinerary" in result["execution_plan"]
        assert result["execution_plan"][-1] == "itinerary", \
            "itinerary must always be last in the plan"

    @pytest.mark.asyncio
    async def test_hotel_only_refinement(self):
        """When only hotels change, plan should be ['hotels', 'itinerary']."""
        from backend.graph.classify_request import classify_request
        from backend.models import RefinementDelta, TripIntent

        mock_delta = RefinementDelta(
            updated_intent=TripIntent(
                destination="Tokyo",
                duration_days=5,
                hotel_location_preference="near Shinjuku",
                raw_request="Actually, I want hotels near Shinjuku",
            ),
            subgraphs_to_run=["hotels"],
            is_first_turn=False,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_delta)

        with patch("backend.graph.classify_request.get_llm") as mock_get_llm:
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm
            result = await classify_request(self._base_state(has_existing_itinerary=True))

        assert result["execution_plan"] == ["hotels", "itinerary"]
        assert result["subgraphs_to_run"] == ["hotels"]

    @pytest.mark.asyncio
    async def test_narrative_only_refinement_gives_empty_subgraphs(self):
        """When only narrative changes (e.g. 'make it more relaxed'), no subgraphs run."""
        from backend.graph.classify_request import classify_request
        from backend.models import RefinementDelta, TripIntent

        mock_delta = RefinementDelta(
            updated_intent=TripIntent(
                destination="Tokyo",
                duration_days=5,
                pace="relaxed",
                raw_request="Make the itinerary more relaxed",
            ),
            subgraphs_to_run=[],
            is_first_turn=False,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_delta)

        with patch("backend.graph.classify_request.get_llm") as mock_get_llm:
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm
            result = await classify_request(self._base_state(has_existing_itinerary=True))

        # narrative-only: no subgraphs, only itinerary rebuild
        assert result["execution_plan"] == ['itinerary']
        assert result["subgraphs_to_run"] == []


class TestRespondNode:
    """respond node should call interrupt() and return correct state updates."""

    @pytest.mark.asyncio
    async def test_interrupt_is_called_with_string(self):
        from backend.graph.respond import respond

        mock_itinerary = MagicMock()
        mock_itinerary.model_dump_json.return_value = '{"destination": "Paris"}'
        state = {
            "current_itinerary": mock_itinerary,
            "messages": [],
        }

        captured_interrupt_value = {}

        def mock_interrupt(value):
            captured_interrupt_value["value"] = value
            return "user follow-up message"  # simulates what comes back after resume

        with patch("backend.graph.respond.interrupt", side_effect=mock_interrupt):
            result = await respond(state)

        # interrupt must receive a string
        assert isinstance(captured_interrupt_value["value"], str), \
            "interrupt() must receive a string, not a Pydantic object"

        # result should have the AI reply and user follow-up in messages
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], AIMessage)
        assert isinstance(result["messages"][1], HumanMessage)
        assert result["refinement_request"] == ["user follow-up message"]


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — Graph structure tests (no LLM, just graph compilation)
# ═══════════════════════════════════════════════════════════════════════════

class TestGraphCompiles:
    """
    The graph should compile without errors. This catches broken edge
    references, missing nodes, and invalid conditional edge maps.
    """

    def test_build_graph_compiles(self):
        """build_graph() must not raise."""
        from backend.graph.orchestrator import build_graph
        try:
            graph = build_graph()
            assert graph is not None
        except Exception as e:
            pytest.fail(f"build_graph() raised an exception: {e}")

    def test_graph_has_checkpointer(self):
        """Graph must have a checkpointer for multi-turn to work."""
        from backend.graph.orchestrator import build_graph
        graph = build_graph()
        assert graph.checkpointer is not None, \
            "Graph must have a checkpointer (MemorySaver or AsyncSqliteSaver) for multi-turn conversation"

    def test_graph_nodes_exist(self):
        """All expected nodes should be present in the compiled graph."""
        from backend.graph.orchestrator import build_graph
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        required = {"classify_request", "weather", "attraction", "hotel", "itinerary", "respond"}
        missing = required - node_names
        assert not missing, f"Missing nodes in compiled graph: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 4 — Full multi-turn integration smoke test (REAL LLM + REAL APIs)
# Run manually, not in CI. Comment out @pytest.mark.skip to enable.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skip(reason="Integration test — requires real API keys and makes live calls")
class TestMultiTurnIntegration:
    """
    End-to-end smoke test for the multi-turn conversation loop.

    Turn 1: Initial trip request → full pipeline → interrupt at respond
    Turn 2: Refinement → selective subgraph re-run → interrupt at respond
    """

    @pytest.mark.asyncio
    async def test_two_turn_conversation(self):
        from backend.graph.orchestrator import build_graph
        from langgraph.types import Command

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-smoke-001"}}

        # ── Turn 1: Initial request ───────────────────────────────────────
        print("\n[Turn 1] Sending initial trip request...")
        initial_input = {
            "messages": [HumanMessage(content="Plan a 3-day trip to Tokyo in August 2026")],
            "user_query": "Plan a 3-day trip to Tokyo in August 2026",
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

        interrupted_value = None
        async for _ in graph.astream_events(initial_input, config=config, version="v2"):
            pass  # drain the stream

        # The graph is now paused at interrupt() — check state
        state_snap = await graph.aget_state(config)

        # Pending tasks means the graph is interrupted and waiting
        assert state_snap.tasks, "Graph should be interrupted at respond node"

        # The interrupt value is in the task's interrupts list
        interrupted_value = state_snap.tasks[0].interrupts[0].value
        assert interrupted_value is not None, "Interrupt should have a value"
        print(f"[Turn 1] Graph paused. Itinerary preview:\n{str(interrupted_value)[:300]}...")
        state = state_snap.values
        assert state.get("current_itinerary") is not None, "current_itinerary should be set after turn 1"
        assert state.get("current_intent") is not None, "current_intent should be set after turn 1"
        assert len(state.get("weather_info") or []) > 0, "weather_info should be populated"
        assert len(state.get("attractions") or []) > 0, "attractions should be populated"
        print(f"[Turn 1 ✓] Itinerary generated. Weather days: {len(state['weather_info'])}, "
              f"Attractions: {len(state['attractions'])}, Hotels: {len(state['hotels'])}")

        # ── Turn 2: Refinement (hotels only) ─────────────────────────────
        print("\n[Turn 2] Sending hotel refinement...")
        attractions_before = len(state.get("attractions") or [])
        weather_before = len(state.get("weather_info") or [])

        # Drain the stream — same pattern as Turn 1
        async for _ in graph.astream_events(
                Command(resume="I'd prefer hotels near Shinjuku station"),
                config=config,
                version="v2"
        ):
            pass

        # Check state after Turn 2
        state_snap2 = await graph.aget_state(config)
        assert state_snap2.tasks, "Graph should be interrupted again after turn 2"

        interrupted_value_2 = state_snap2.tasks[0].interrupts[0].value
        assert interrupted_value_2 is not None, "Interrupt should have a value after turn 2"

        state2 = state_snap2.values

        # Hotels should have been refreshed
        # Weather + attractions should be unchanged
        assert len(state2.get("weather_info") or []) == weather_before, \
            "Weather should NOT be re-fetched for a hotel-only refinement"
        assert len(state2.get("attractions") or []) >= attractions_before, \
            "Attractions should NOT be cleared by a hotel-only refinement"

        msgs = state2.get("messages") or []
        human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]

        # After turn 2: respond from turn 1 has now returned, adding 1 AIMessage
        # The turn 2 AIMessage won't appear until turn 3 resumes
        assert len(human_msgs) >= 2, f"Expected ≥2 HumanMessages, got {len(human_msgs)}"
        assert len(ai_msgs) >= 1, f"Expected ≥1 AIMessage, got {len(ai_msgs)}"
        # Note: the AIMessage for turn N is written to state when turn N+1 resumes respond.
        # So after turn 2, only turn 1's AIMessage is in state — turn 2's appears on turn 3.

        print(f"[Turn 2 ✓] Refinement applied. Messages in history: {len(msgs)}")
        print(f"           Hotels: {len(state2.get('hotels') or [])}")


# ═══════════════════════════════════════════════════════════════════════════
# Quick runner (no pytest needed)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running unit tests directly (no pytest)...\n")

    async def run_unit_checks():
        # route_next checks
        t = TestRouteNext()
        # t.test_empty_plan_routes_to_itinerary()
        t.test_weather_first()
        t.test_hotels_only()
        t.test_parallel_group_returns_send_list()
        print("✓ route_next tests passed")

        # _pop_plan checks
        p = TestPopPlan()
        p.test_pop_string_head()
        p.test_pop_list_head()
        p.test_pop_empty_is_safe()
        print("✓ _pop_plan tests passed")

        # format_itinerary check
        f = TestFormatItinerary()
        f.test_returns_string()
        f.test_none_is_handled()
        print("✓ format_itinerary tests passed")

        # Graph compilation
        g = TestGraphCompiles()
        g.test_build_graph_compiles()
        g.test_graph_has_checkpointer()
        g.test_graph_nodes_exist()
        print("✓ Graph compilation tests passed")

        print("\nAll unit tests passed. Run with pytest -v for full output.")
        print("To run the integration smoke test, remove @pytest.mark.skip from TestMultiTurnIntegration.")

    asyncio.run(run_unit_checks())
