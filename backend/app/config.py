"""Application configuration — loads environment variables via .env file.

Centralizes all config so no API key or setting is ever hardcoded.
Uses Pydantic Settings for type-safe, validated configuration.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: backend/app/config.py → backend/app → backend → project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars not defined here
    )

    # --- Google Gemini ---
    google_api_key: str = ""
    gemini_flash_model: str = "gemini-3.5-flash-lite"
    gemini_pro_model: str = "gemini-3.1-pro-preview"
    gemini_temperature: float = 0.0  # Deterministic for data analysis

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- LangSmith ---
    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "insightforge"

    # --- PostgreSQL (Week 8+) ---
    postgres_url: str = ""

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"

    # --- AWS (Week 12+) ---
    aws_region: str = ""

    # --- App ---
    data_dir: str = str(PROJECT_ROOT / "data")
    upload_dir: str = str(PROJECT_ROOT / "uploads")
    max_sandbox_timeout: int = 30  # seconds
    max_retries: int = 2  # Critic → Coder retry cap


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings (loaded once, reused everywhere)."""
    return Settings()


def get_llm(model_type: str = "flash"):
    """Create a Gemini LLM instance.

    Args:
        model_type: 'flash' for cheap/fast routing, 'pro' for complex reasoning.

    Returns:
        A ChatGoogleGenerativeAI instance ready to use.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()

    model_name = (
        settings.gemini_flash_model
        if model_type == "flash"
        else settings.gemini_pro_model
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=settings.gemini_temperature,
        max_retries=3,
    )
