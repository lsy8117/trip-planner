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

    # # Telegram
    # telegram_bot_token: str

    # Tools
    tavily_api_key: str = ""
    openweathermap_api_key: str = ""
    rapidapi_key: str = ""
    # amadeus_api_key: str = ""
    # amadeus_api_secret: str = ""
    # google_places_api_key: str = ""
    google_maps_api_key: str = ""


# Singleton instance — import this everywhere instead of re-instantiating
settings = Settings()