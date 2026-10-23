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

    # Database Configuration (PostgreSQL: source of truth for documents + graph extraction JSONB)
    # Using asyncpg driver for asynchronous database I/O
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_docs_db"

    # Qdrant vector store (single collection used by the LlamaIndex pipeline)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "doc_chunks"

    # Embedding Model Configuration
    # intfloat/multilingual-e5-small produces 384-dimensional dense vectors
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIMENSION: int = 384
    HF_CACHE_DIR: Path | None = Path.home() / "Projects" / "AI" / "models" / "hub"
    
    # Local Docs Directory scanned by Taskiq worker
    DOCS_DIR: Path = Path(__file__).resolve().parent.parent / "docs_mock"

    # LLM Service Configuration (Ollama or OpenAI-compatible endpoint)
    # Uses installed local Ollama models (e.g. qwen2.5-coder:7b or qwen2.5-coder:3b)
    LLM_API_BASE: str = "http://127.0.0.1:8080/v1"
    LLM_API_KEY: str = "ollama"
    LLM_MODEL: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"

    # Backend API URL for Streamlit frontend
    BACKEND_API_URL: str = "http://localhost:8000"

    # Neo4j graph store (bolt protocol)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "graphrag123"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()
