"""
Tracks graph execution progress and renders it as a single cumulative
status string that stream_runner edits into one Telegram message in place.

── Integration point #2 ──────────────────────────────────────────────
NODE_CONFIG keys must match the exact strings passed to `builder.add_node(...)`
in orchestrator.py. As of the current graph that's:
    classify_request, weather, attraction, hotel, itinerary, respond
TOOL_CONFIG keys must match your @tool function names.
If you rename a node or add a new subgraph, add an entry here or it will
silently fall back to a generic "Working on it..." line.
───────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass, field


NODE_CONFIG = {
    "classify_request": ("🧠", "Understanding your request..."),
    "weather": ("🌤️", "Checking the weather..."),
    "attraction": ("🗺️", "Finding attractions..."),
    "hotel": ("🏨", "Looking up hotels..."),
    "itinerary": ("📋", "Putting your itinerary together..."),
    "respond": ("✍️", "Wrapping up..."),
}

TOOL_CONFIG = {
    "get_weather_forecast": "Fetching forecast",
    "search_attraction": "Searching for attractions",
    "search_hotels": "Searching hotel rates",
}

DEFAULT_NODE_ICON = "🤖"
DEFAULT_NODE_LABEL = "Working on it..."


@dataclass
class ProgressTracker:
    """One instance per turn. Call on_node_start/on_node_end/on_tool_start
    as events arrive, then render() to get the current status text."""

    completed: list[str] = field(default_factory=list)
    current: str | None = None
    current_detail: str | None = None

    def on_node_start(self, node_name: str) -> None:
        icon, label = NODE_CONFIG.get(node_name, (DEFAULT_NODE_ICON, DEFAULT_NODE_LABEL))
        self.current = f"{icon} {label}"
        self.current_detail = None

    def on_node_end(self, node_name: str) -> None:
        if self.current:
            # Strip the in-progress line into a checked-off completed line
            done_label = NODE_CONFIG.get(node_name, (None, "Done"))[1].rstrip(".")
            self.completed.append(f"✅ {done_label}")
        self.current = None
        self.current_detail = None

    def on_tool_start(self, tool_name: str, tool_input: dict | None = None) -> None:
        label = TOOL_CONFIG.get(tool_name, f"Running {tool_name}")
        query_hint = ""
        if tool_input:
            # Best-effort: surface the first short string arg as a hint
            for v in tool_input.values():
                if isinstance(v, str) and len(v) < 60:
                    query_hint = f" ({v})"
                    break
        self.current_detail = f"   ↳ {label}{query_hint}"

    def finalize(self) -> None:
        """Call once the stream has drained. respond() pausing via
        interrupt() means its on_chain_end never fires (the pause is an
        internal exception, not a normal return), so whatever node was
        still 'in progress' needs to be manually checked off here —
        otherwise the permanent progress message would be left showing
        a spinner-style line forever."""
        if self.current:
            label = self.current.split(" ", 1)[-1].rstrip(".")
            self.completed.append(f"✅ {label}")
        self.current = None
        self.current_detail = None

    def render(self) -> str:
        lines = ["🤖 Planning your trip...", ""]
        lines.extend(self.completed)
        if self.current:
            lines.append(self.current)
        if self.current_detail:
            lines.append(self.current_detail)
        return "\n".join(lines)
