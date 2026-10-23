import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text, UniqueConstraint, CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
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
    This row is the source of truth for chunk TEXT and graph extraction results;
    the embedding lives in Qdrant as a point with the SAME uuid id, indexed by
    the LlamaIndex pipeline (see app/llama_rag.py).
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

    entities: Mapped[dict] = mapped_column(JSONB, nullable=True)
    relations: Mapped[dict] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationship back to service parent entity for provenance tracking
    service: Mapped["Service"] = relationship("Service", back_populates="chunks")


# --- Graph layer: canonical nodes, deduplicated edges, per-chunk evidence ---


class GraphEntity(Base):
    """
    Canonical graph node produced by entity resolution across chunks.

    `canonical_name` stores the normalized surface form (lowercased, trimmed);
    raw variants live in EntityAlias, so a wrong merge is corrected by editing
    one alias row instead of rewriting edges.
    """
    __tablename__ = "graph_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    canonical_name: Mapped[str] = mapped_column(String(300), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    aliases: Mapped[list["EntityAlias"]] = relationship(
        "EntityAlias", back_populates="entity", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Identity is (name, type): the same string may be both a service and a
        # database, and those must stay distinct nodes (type guard, lesson 3).
        UniqueConstraint("canonical_name", "entity_type", name="uq_graph_entities_name_type"),
    )


class EntityAlias(Base):
    """
    A surface form ("AuthService", "payments-svc") pointing at its canonical node.

    Lookup by `normalized` is the hot path for indexing (merge decisions) and
    retrieval (query entity -> start node).
    """
    __tablename__ = "entity_aliases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    alias: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("service_documentation_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

    entity: Mapped["GraphEntity"] = relationship("GraphEntity", back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("entity_id", "normalized", name="uq_entity_alias_per_node"),
    )


class GraphEdge(Base):
    """
    Unique typed relation between two canonical entities.

    Identity is the triple (source, target, relation): every chunk repeating
    the same statement produces an EdgeMention row, not a new edge. Raw
    predicate wording ("to validate tokens") goes to `description` and does
    not participate in deduplication.
    """
    __tablename__ = "graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    mentions: Mapped[list["EdgeMention"]] = relationship(
        "EdgeMention", back_populates="edge", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relation", name="uq_graph_edge_triple"),
        CheckConstraint("source_id <> target_id", name="ck_graph_edge_no_self_loop"),
    )


class EdgeMention(Base):
    """
    Proof that one chunk asserts one edge. The composite PK (edge_id, chunk_id)
    physically forbids the same chunk claiming the same fact twice — the
    deduplication is the constraint, not application code.
    """
    __tablename__ = "edge_mentions"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("graph_edges.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("service_documentation_chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    edge: Mapped["GraphEdge"] = relationship("GraphEdge", back_populates="mentions")
    chunk: Mapped["ServiceDocumentationChunk"] = relationship("ServiceDocumentationChunk")
