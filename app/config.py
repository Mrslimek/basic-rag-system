from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration managed by Pydantic Settings.
    Reads from environment variables and falls back to clean local defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database Configuration (PostgreSQL with pgvector)
    # Using asyncpg driver for asynchronous database I/O
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_docs_db"
    
    # Embedding Model Configuration
    # intfloat/multilingual-e5-small produces 384-dimensional dense vectors
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIMENSION: int = 384
    
    # Local Docs Directory scanned by Taskiq worker
    DOCS_DIR: Path = Path(__file__).resolve().parent.parent / "docs_mock"

    # LLM Service Configuration (Ollama or OpenAI-compatible endpoint)
    # Uses installed local Ollama models (e.g. qwen2.5-coder:7b or qwen2.5-coder:3b)
    LLM_API_BASE: str = "http://127.0.0.1:8080/v1"
    LLM_API_KEY: str = "ollama"
    LLM_MODEL: str = "qwen2.5-coder-7b"

    # Backend API URL for Streamlit frontend
    BACKEND_API_URL: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()
