import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


class Service(Base):
    """
    Represents a registered microservice in the architecture.
    Acts as the parent entity for all documentation chunks, tracking documentation source & metadata.
    """
    __tablename__ = "services"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # 1-to-many relationship with documentation chunks
    chunks: Mapped[list["ServiceDocumentationChunk"]] = relationship(
        "ServiceDocumentationChunk",
        back_populates="service",
        cascade="all, delete-orphan",
        order_by="ServiceDocumentationChunk.chunk_index",
    )


class ServiceDocumentationChunk(Base):
    """
    Represents a discrete semantic chunk of a microservice's documentation.
    Embeddings are 384-dimensional unit vectors (via intfloat/multilingual-e5-small).
    Indexed using HNSW with vector_ip_ops (inner product) for ultra-fast normalized vector search.
    """
    __tablename__ = "service_documentation_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    service_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("services.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # 384 dimensions for multilingual-e5-small
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship back to service parent entity for provenance tracking
    service: Mapped["Service"] = relationship("Service", back_populates="chunks")

    __table_args__ = (
        # HNSW Index for vector search:
        # We use inner product ('vector_ip_ops' / max_inner_product) because multilingual-e5-small
        # embeddings are normalized to unit length (||v|| = 1). For unit vectors, inner product
        # is mathematically identical to cosine similarity but runs significantly faster.
        Index(
            "ix_doc_chunks_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_ip_ops"},
        ),
    )
