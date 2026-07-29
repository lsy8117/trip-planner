"""
Renders a PlannedItinerary into compact text for the memory-update prompt.

Kept isolated from updater.py on purpose: what's worth keeping (does pacing
feedback need fees? notes? weather?) will need retuning as real usage shows
what the extraction LLM actually needs — that tuning should happen here,
without touching the pipeline that calls it.
"""
from backend.models import PlannedItinerary

# Toggle which optional per-activity fields get included below.
_INCLUDE_FEES = True
_INCLUDE_NOTES = True


def condense_itinerary(itinerary: PlannedItinerary) -> str:
    lines = [f"Destination: {itinerary.destination}", f"Summary: {itinerary.trip_summary}"]

    for day in itinerary.days:
        lines.append(f"\n{day.date} ({len(day.activities)} activities):")
        if day.hotel:
            lines.append(f"  Hotel: {day.hotel.name}")
        for activity in day.activities:
            entry = f"  - {activity.time} {activity.name} [{activity.attraction_type}, {activity.estimated_duration_hours}h]"
            if _INCLUDE_FEES and activity.estimated_fee_usd:
                entry += f" (${activity.estimated_fee_usd})"
            if _INCLUDE_NOTES and activity.notes:
                entry += f" — {activity.notes}"
            lines.append(entry)

    return "\n".join(lines)
