from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from app.config import get_settings

settings = get_settings()

# Engine creation:
# pgvector.sqlalchemy.Vector automatically handles string/list serialization for queries and distance operators.
# We do NOT register asyncpg binary codecs because SQLAlchemy's Vector type serializer already prepares values as strings,
# which would conflict with asyncpg's binary codec expecting raw list/ndarray.
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
    1. Ensures pgvector extension is installed (`CREATE EXTENSION IF NOT EXISTS vector`).
    2. Registers the pgvector asyncpg codec for direct vector type handling.
    3. Creates all tables defined in models.py if they do not exist.
    """
    from app.models import Base

    active_engine = target_engine or engine

    async with active_engine.begin() as conn:
        # Enable pgvector extension before creating tables with Vector columns
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create all tables (services, service_documentation_chunks, and HNSW indexes)
        await conn.run_sync(Base.metadata.create_all)
