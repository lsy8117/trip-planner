import json
from typing import Annotated, List, Optional, TypedDict
from annotated_types import Annotated
from operator import add
from pydantic import BaseModel, Field, field_validator


class DayWeatherInfo(BaseModel):
    location: str = Field(..., examples=["Chicago"])
    date: str = Field(..., examples=["2026-06-20"])
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None
    precipitation_mm: Optional[float] = None
    condition: Optional[str] = None
    error: Optional[str] = None

class AttractionInfo(BaseModel):
    name: str = Field(..., description='Name of the attraction or event.')
    attraction_type: str = Field(..., description='Tourist attraction type.', examples=['Museum','Historical Site','Shopping','Cruise Tour'])
    description: str = Field(..., description='Brief description of the attraction extracted from the search results.')
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    booking_required: Optional[bool] = None
    estimated_fee_usd: Optional[float] = None
    fee_notes: Optional[str] = None

class AttractionsList(BaseModel):
    attractions: list[AttractionInfo]

class HotelOffer(BaseModel):
    ota: str
    price_usd: float

class HotelInfo(BaseModel):
    name: str
    hotel_key: str
    rating: Optional[float] = None
    address: Optional[str] = None
    offers: list[HotelOffer] = Field(default_factory=list)


class PlannedActivity(BaseModel):
    time: str = Field(description="Suggested time, e.g. '9:00 AM'")
    name: str = Field(description="Name of the attraction or activity")
    # attraction: AttractionInfo = Field(description="Attraction Information")
    attraction_type: str = Field(description="e.g. 'museum', 'landmark', 'restaurant'")
    description: str = Field(description="Brief description of what to do/see here")
    estimated_duration_hours: float = Field(description="How long to spend here")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    notes: Optional[str] = Field(None, description="Practical tips, booking info, weather considerations")

class PlannedDay(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    weather_summary: Optional[str] = Field(None, description="Brief weather note for the day, e.g. 'Light drizzle, 17–26°C'")
    activities: list[PlannedActivity]
    hotel: Optional[str] = Field(None, description="Recommended hotel name for this night")

class PlannedItinerary(BaseModel):
    destination: str
    trip_summary: str = Field(description="2-3 sentence overview of the trip plan")
    days: list[PlannedDay]
    total_estimated_budget_usd: Optional[float] = Field(None, description="Rough total budget estimate if inferable from hotel/attraction prices")
    travel_tips: list[str] = Field(default_factory=list, description="General tips for the trip")