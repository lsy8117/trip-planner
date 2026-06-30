import httpx
from langchain_core.tools import tool

from backend.config import settings


##############################################
## OpenWeather API (Forecast not Available) ## 
##############################################

_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

@tool
async def get_weather(location: str, dates: list[str]) -> dict:
    """Get weather forecast for a location on specific dates.

    Args:
        location: City name, e.g. "London", "New York", "Tokyo".
        dates: List of dates in YYYY-MM-DD format, e.g. ["2024-06-20", "2024-06-21"].
               Forecast is available for up to 5 days from today.

    Returns:
        Dict with location info and per-date weather forecasts including
        min/max temperature, humidity, wind speed, conditions, and descriptions.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            _FORECAST_URL,
            params={
                "q": location,
                "appid": settings.openweathermap_api_key,
                "units": "metric",
            },
        )
        response.raise_for_status()
        data = response.json()

    # Group 3-hour forecast entries by date
    forecasts_by_date: dict[str, list] = {}
    for entry in data["list"]:
        entry_date = entry["dt_txt"].split(" ")[0]  # "2024-06-20 12:00:00" -> "2024-06-20"
        forecasts_by_date.setdefault(entry_date, []).append(entry)

    # Build per-date summary for each requested date
    daily_forecasts: dict[str, dict] = {}
    for target_date in dates:
        entries = forecasts_by_date.get(target_date)
        if not entries:
            daily_forecasts[target_date] = {
                "error": "No forecast available for this date (forecast covers up to 5 days ahead)"
            }
            continue

        temps = [e["main"]["temp"] for e in entries]
        feels = [e["main"]["feels_like"] for e in entries]
        humidities = [e["main"]["humidity"] for e in entries]
        wind_speeds = [e["wind"]["speed"] for e in entries]
        # "main" is the broad category: "Rain", "Snow", "Clear", "Clouds", "Thunderstorm", etc.
        conditions = list(dict.fromkeys(e["weather"][0]["main"] for e in entries))
        # "description" is the fine-grained detail: "light rain", "clear sky", etc.
        descriptions = list(dict.fromkeys(e["weather"][0]["description"] for e in entries))

        daily_forecasts[target_date] = {
            "temp_min_c": round(min(temps), 1),
            "temp_max_c": round(max(temps), 1),
            "feels_like_avg_c": round(sum(feels) / len(feels), 1),
            "humidity_percent": round(sum(humidities) / len(humidities)),
            "wind_speed_avg_mps": round(sum(wind_speeds) / len(wind_speeds), 1),
            "conditions": conditions,
            "descriptions": descriptions,
        }

    return {
        "location": data["city"]["name"],
        "country": data["city"]["country"],
        "coordinates": {
            "lat": data["city"]["coord"]["lat"],
            "lon": data["city"]["coord"]["lon"],
        },
        "forecasts": daily_forecasts,
    }