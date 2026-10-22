from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.config import get_settings

settings = get_settings()

# Engine creation: Postgres owns documents/chunks/graph tables only.
# Vectors live in Qdrant (app/vectorstore.py), so no pgvector codecs are registered here.
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an async database session per request.
    Handles rollback on exceptions and automatic cleanup on exit.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for standalone worker tasks / scripts outside FastAPI request scope.
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """
    Bootstrap database:
    1. Creates all tables defined in models.py if they do not exist.
    Vectors are NOT stored here anymore (Qdrant owns them, see app/vectorstore.py),
    so the pgvector extension is no longer required.
    """
    from app.models import Base

    active_engine = target_engine or engine

    async with active_engine.begin() as conn:
        # Create all tables (services, chunks, graph layer; HNSW index is gone)
        await conn.run_sync(Base.metadata.create_all)
