"""
Converts the value your graph pauses with (`interrupt(reply)` in respond.py)
into Telegram Markdown.

── Integration point #3 ──────────────────────────────────────────────
`respond.py` currently does:

    reply = format_itinerary(state["current_itinerary"])   # returns the
                                                             # PlannedItinerary
                                                             # object itself
    user_input = interrupt(reply)

`interrupt()` needs a value the frontend can render, and right now that
value is the raw Pydantic object, not a string. This module handles BOTH
cases so it works whichever way you leave `respond.format_itinerary`:
  - if `interrupt_value` is already a string  → lightly escape and send it
  - if it's a PlannedItinerary (or dict)      → render it fully here

Field names below (`destination`, `duration_days`, `days`, `hotels`, ...)
match what itinerary.py's prompt/context-building code assumes. If your
actual `PlannedItinerary` model in models.py uses different field names,
update the `getattr(...)` calls below to match.
───────────────────────────────────────────────────────────────────────
"""

from typing import Any


def format_for_telegram(interrupt_value: Any) -> str:
    """Entry point called by stream_runner once the graph pauses at interrupt()."""
    if isinstance(interrupt_value, str):
        return _escape_markdown(interrupt_value)

    # Pydantic model → dict, or already a dict
    data = interrupt_value.model_dump() if hasattr(interrupt_value, "model_dump") else interrupt_value
    if isinstance(data, dict):
        return _format_itinerary_dict(data)

    # Fallback: don't lose the content even if the shape is unexpected
    print(f"Warning: unexpected interrupt value type {type(interrupt_value)}; falling back to str()")
    return _escape_markdown(str(interrupt_value))


def _format_itinerary_dict(data: dict) -> str:
    destination = data.get("destination", "your trip")
    # duration_days = data.get("duration_days")
    days = data.get("days") or []
    # hotels = data.get("hotels") or []

    lines = [f"✈️ *Your Travel Plan — {_escape_markdown(str(destination))}*"]
    # if duration_days:
    #     lines.append(f"_{duration_days} day(s)_")
    lines.append("")

    for i, day in enumerate(days, start=1):
        day_label = day.get("date") or f"Day {i}"
        lines.append(f"*📅 {_escape_markdown(str(day_label))}*")

        summary = day.get("weather_summary") or day.get("narrative")
        if summary:
            lines.append(_escape_markdown(str(summary)))

        activities = day.get("activities") or day.get("items") or []
        for act in activities:
            if isinstance(act, dict):
                time = act.get("time")
                name = act.get("name")
                note = act.get("notes")
                line = f"  • {_escape_markdown(str(time))} {_escape_markdown(str(name))}"
                if note:
                    line += f" — {_escape_markdown(str(note))}"
                lines.append(line)
            else:
                lines.append(f"  • {_escape_markdown(str(act))}")
        day_hotel = day.get("hotel")
        if day_hotel:
            hotel_name = day_hotel.get("name") if isinstance(day_hotel, dict) else str(day_hotel)
            lines.append(f"  🏨 {_escape_markdown(str(hotel_name))}")
            hotel_offers = day_hotel.get("offers") if isinstance(day_hotel, dict) else None
            if hotel_offers:
                for offer in hotel_offers:
                    ota = offer.get("ota") if isinstance(offer, dict) else None
                    price = offer.get("price_usd") if isinstance(offer, dict) else None
                    if ota and price is not None:
                        lines.append(f"    - {_escape_markdown(str(ota))}: ${price:.2f}")
        lines.append("")


    lines.append("")
    lines.append("_Want changes? Just tell me what to adjust._")
    return "\n".join(lines)


_MD_SPECIAL_CHARS = "_*[]()~`>#+-=|{}.!"


def _escape_markdown(text: str) -> str:
    """Minimal escaping for Telegram's legacy Markdown parse mode.
    (Switch to MarkdownV2 escaping if you change ParseMode in stream_runner.)"""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, f"\\{ch}")
    return text
