# backend/graph/subgraphs/weather.py

import json
from typing import Annotated, List, Optional, TypedDict
from annotated_types import Annotated
from operator import add
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool

import requests
import httpx
from datetime import date, datetime, timedelta

from backend.models import DayWeatherInfo
# from backend.tools.weather import tools  # your existing weather tools
from backend.utilities import get_llm
from backend.config import settings

# ─────────────────────────────────────────────
# Sub-graph state (private to this subgraph)
# ─────────────────────────────────────────────

class WeatherState(TypedDict):
    # Input fields (written by orchestrator before invoking)
    user_query: str

    # Internal working memory (messages accumulate during tool loops)
    messages: Annotated[list[BaseMessage], add]

    # Output field (read by orchestrator after subgraph returns)
    weather_info: list[DayWeatherInfo] | None


class WeatherInput(TypedDict):
    """Only these fields flow IN from the orchestrator."""
    user_query: str


class WeatherOutput(TypedDict):
    """Only this field flows OUT to the orchestrator."""
    weather_info: list[DayWeatherInfo] | None

# ─────────────────────────────────────────────
# Weather tool 
# ─────────────────────────────────────────────

# Open-Meteo's free forecast endpoint only covers today .. today+15 days; requesting
# beyond that returns a 400. Requests further out get clipped to this window and the
# clipped-off days are reported back per-day as unavailable instead of failing outright.
MAX_FORECAST_DAYS = 15


@tool
def get_weather_forecast(city: str, start_date: str, end_date: str) -> dict:
    """
    Get a daily weather forecast for a city between two dates.

    Args:
        city: City name, e.g. "Tokyo" or "New York"
        start_date: "YYYY-MM-DD"
        end_date: "YYYY-MM-DD"

    Returns:
        dict with city info and a list of daily forecasts. Dates beyond the forecast
        provider's supported range (more than 15 days from today) come back with an
        "error" field instead of weather data, rather than failing the whole request.
    """
    # validate dates
    try:
        d1 = datetime.strptime(start_date, "%Y-%m-%d").date()
        d2 = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("Dates must be in YYYY-MM-DD format")
    if d2 < d1:
        raise ValueError("end_date must not be before start_date")

    # step 1: geocode city name -> lat/lon
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        raise ValueError(f"Could not find location: {city}")

    place = geo_data["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    resolved_name = place.get("name", city)

    # step 2: fetch forecast for date range, falling back to a clipped range if the
    # provider rejects it as too far out
    forecast_days, unavailable_reasons = _fetch_forecast_days(lat, lon, d1, d2)

    forecast = []
    for day in _date_range(d1, d2):
        day_str = day.isoformat()
        if day_str in forecast_days:
            forecast.append({"date": day_str, **forecast_days[day_str]})
        else:
            forecast.append({
                "date": day_str,
                "error": unavailable_reasons.get(day_str, "Forecast unavailable for this date"),
            })

    return {"city": resolved_name, "latitude": lat, "longitude": lon, "forecast": forecast}


def _date_range(start: date, end: date):
    for i in range((end - start).days + 1):
        yield start + timedelta(days=i)


def _request_forecast_daily(lat: float, lon: float, start: date, end: date) -> dict:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
            "timezone": "auto",
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("daily", {})


def _fetch_forecast_days(
    lat: float, lon: float, start: date, end: date
) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Request the forecast for [start, end], retrying with a range clipped to
    today+MAX_FORECAST_DAYS if the provider 400s the original range.

    Returns (date_str -> parsed day dict, date_str -> reason unavailable).
    """
    max_available_date = date.today() + timedelta(days=MAX_FORECAST_DAYS)

    try:
        daily = _request_forecast_daily(lat, lon, start, end)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status != 400:
            raise

        unavailable_reasons = {
            day.isoformat(): (
                f"Forecast unavailable: more than {MAX_FORECAST_DAYS} days from today, "
                "beyond the forecast provider's supported range."
            )
            for day in _date_range(max(start, max_available_date + timedelta(days=1)), end)
            if day > max_available_date
        }

        if start > max_available_date:
            # Nothing in the requested range is fetchable at all.
            return {}, unavailable_reasons

        daily = _request_forecast_daily(lat, lon, start, max_available_date)
        return _parse_daily(daily), unavailable_reasons

    return _parse_daily(daily), {}


def _parse_daily(daily: dict) -> dict[str, dict]:
    parsed = {}
    for i, day_str in enumerate(daily.get("time", [])):
        parsed[day_str] = {
            "temp_max_c": daily["temperature_2m_max"][i],
            "temp_min_c": daily["temperature_2m_min"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "condition": _weathercode_to_text(daily["weathercode"][i]),
        }
    return parsed


def _weathercode_to_text(code: int) -> str:
    mapping = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, f"Unknown ({code})")

tools = [get_weather_forecast]

# ─────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────

WEATHER_AGENT_SYSTEM_PROMPT = """
** Role **
You are a helpful assistant for providing weather forecasts in
a larger trip planning application.

** Task & Responsibility **
The user would describe a travel plan, and your task is to look through the conversation history 
and identify the destination(s) and the respective date(s) to query the weather forecast for 
those locations.
If the weather information is unavailable, stop any tool calling and inform the user as it is.
Do not generate any travel plans, since your task is restricted to weather query.

** Important **
You must use the get_weather tool to search for weather forecast information. 
Do not come out with weather information on your own.

** Output **
Present any weather information found with short, succinct descriptive text. 
Do not return anything else.
If weather information is unavailable, return a message indicating that the weather information is unavailable.
"""

def _build_weather_agent():
    llm = get_llm(provider=settings.weather_model_provider, model=settings.weather_model)  # called inside node factory, not at module level
    llm_w_tools = llm.bind_tools(tools)

    async def weather_agent(state: WeatherState):
        response = await llm_w_tools.ainvoke([
            SystemMessage(content=WEATHER_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=state["user_query"]),
            *state["messages"],
        ])
        return {"messages": [response]}

    return weather_agent


def weather_router(state: WeatherState) -> str:
    last = state["messages"][-1] if state["messages"] else None
    if last and getattr(last, "tool_calls", None):
        return "tools"
    return "parse_weather_results"  # ← always parse explicitly, never jump to END from agent


async def parse_weather_results(state: WeatherState) -> dict:
    """
    Extract structured WeatherResult from the last ToolMessage.
    This is the only node that writes to weather_info.
    """
    # Find the last ToolMessage (most recent tool response)
    tool_messages = [m for m in state["messages"] if m.type == "tool"]
    if not tool_messages:
        return {
            "weather_info": [DayWeatherInfo(
                location="Unknown", date="Unknown",
                error="No tool response found in messages.",
            )]
        }

    last_tool_msg = tool_messages[-1]

    try:
        tool_result = json.loads(last_tool_msg.content)
    except (json.JSONDecodeError, TypeError):
        return {
            "weather_info": [DayWeatherInfo(
                location="Unknown", date="Unknown",
                error=str(last_tool_msg.content),
            )]
        }

    location = tool_result["city"]
    day_weather_list = [
        DayWeatherInfo(
            location=location,
            date=entry["date"],
            temp_min_c=entry.get("temp_min_c"),
            temp_max_c=entry.get("temp_max_c"),
            precipitation_mm=entry.get("precipitation_mm"),
            condition=entry.get("condition"),
            error=entry.get("error"),
        )
        for entry in tool_result["forecast"]
    ]
    return {"weather_info": day_weather_list}


# ─────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────

def build_weather_subgraph() -> StateGraph:
    weather_agent = _build_weather_agent()

    builder = StateGraph(WeatherState, input=WeatherInput, output=WeatherOutput)

    builder.add_node("weather_agent", weather_agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("parse_weather_results", parse_weather_results)

    builder.set_entry_point("weather_agent")

    builder.add_conditional_edges(
        "weather_agent",
        weather_router,
        {
            "tools": "tools",
            "parse_weather_results": "parse_weather_results",  # done with tools
        },
    )
    builder.add_edge("tools", "weather_agent")          # loop back after each tool call
    builder.add_edge("parse_weather_results", END)      # structured output ready, exit

    return builder.compile()


weather_subgraph = build_weather_subgraph()