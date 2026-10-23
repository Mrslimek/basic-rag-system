import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
    This row is the source of truth for chunk TEXT only; the embedding lives
    in Qdrant as a point with the SAME uuid id, indexed by the LlamaIndex
    pipeline (see app/llama_rag.py), and the graph lives in Neo4j via
    app/graph_rag.py.
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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship back to service parent entity for provenance tracking
    service: Mapped["Service"] = relationship("Service", back_populates="chunks")
