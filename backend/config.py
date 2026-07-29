# backend/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root regardless of current working directory
ROOT_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: str = "groq"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    # ollama_base_url: str = "http://localhost:11434"
    # ollama_model: str = "llama3.2"
    # llamacpp_model_path: str = ""

    ## classify_request LLM
    classify_request_model: str = "qwen/qwen3.6-27b"
    classify_request_model_provider: str = "groq"

    ## weather agent LLM
    weather_model: str = "openai/gpt-oss-20b"
    weather_model_provider: str = "groq"

    ## attraction agent LLM 
    attraction_model: str = "openai/gpt-oss-20b"
    attraction_model_provider: str = "groq"
    extraction_model: str = "openai/gpt-oss-120b"
    extraction_model_provider: str = "groq"

    ## hotel agent LLM
    hotel_model: str = "openai/gpt-oss-20b"
    hotel_model_provider: str = "groq"

    ## itinerary agent LLM
    itinerary_model: str = "llama-3.3-70b-versatile"
    itinerary_model_provider: str = "groq"

    ## itinerary patch LLM
    itinerary_patch_model: str = "llama-3.3-70b-versatile"
    itinerary_patch_model_provider: str = "groq"

    ## memory update LLM
    memory_update_model: str = "openai/gpt-oss-20b"
    memory_update_model_provider: str = "groq"

    # Tools
    tavily_api_key: str = ""
    openweathermap_api_key: str = ""
    rapidapi_key: str = ""
    google_maps_api_key: str = ""

    # Frontend
    telegram_bot_token: str = ""

    sessions_dir: Path = ROOT_DIR / "backend" / "sessions"

    memory_dir: Path = ROOT_DIR / "backend" / "memory" / "data"


# Singleton instance — import this everywhere instead of re-instantiating
settings = Settings()