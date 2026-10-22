"""
Qdrant vector store — replaces the pgvector column for ANN similarity search.

Division of storage after this migration:
  Postgres  = source of truth: documents, chunk text, graph-extraction JSONB.
  Qdrant    = vector index only. One point == one chunk:
      id      -> the chunk's UUID (same key links the two stores, like a FK)
      vector  -> 384-dim unit-normalized e5-small embedding
      payload -> provenance metadata (service_id, service_name, doc_file_path, chunk_index)

  Retrieval: Qdrant returns point ids + scores -> we join back to Postgres for text.

Qdrant concepts mapped to SQL vocabulary:
  collection          ~ table (fixed vector size + distance metric schema)
  point               ~ row
  payload             ~ JSONB column; filterable after a payload index
  upsert(point)       ~ INSERT ... ON CONFLICT (id) DO UPDATE  (idempotent by id)
  delete by filter    ~ DELETE ... WHERE payload_field = value

Why COSINE here: e5-small embeddings are L2-normalized to unit length, so
cosine == dot product == what pgvector computed with vector_ip_ops. Same scores,
different engine.
"""
import asyncio
import logging
import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings

logger = logging.getLogger("vectorstore")
settings = get_settings()


@dataclass(frozen=True)
class ChunkVector:
    """One chunk's vector + payload, ready to be upserted into Qdrant."""
    chunk_id: uuid.UUID
    service_id: str
    service_name: str
    doc_file_path: str
    chunk_index: int
    embedding: list[float]


@dataclass(frozen=True)
class VectorHit:
    """A search hit: chunk id, similarity score in [-1..1], and stored payload."""
    chunk_id: uuid.UUID
    score: float
    payload: dict


@dataclass
class QdrantStore:
    """Async thin wrapper over the Qdrant REST client for our chunk collection."""
    client: AsyncQdrantClient
    collection: str

    async def ensure_collection(self) -> None:
        """Idempotent DDL: create the collection + payload indexes if absent."""
        cols = await self.client.get_collections()
        names = [c.name for c in cols.collections]
        if self.collection not in names:
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )
            # Payload index = the "CREATE INDEX" equivalent for filters:
            # without it, service_name filtering degrades to a full scan.
            for field in ("service_name", "service_id"):
                await self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created Qdrant collection '{self.collection}' ({settings.EMBEDDING_DIMENSION}d, cosine)")

    async def upsert_chunks(self, items: list[ChunkVector]) -> None:
        if not items:
            return
        points = [
            models.PointStruct(
                id=str(item.chunk_id),
                vector=item.embedding,
                payload={
                    "service_id": item.service_id,
                    "service_name": item.service_name,
                    "doc_file_path": item.doc_file_path,
                    "chunk_index": item.chunk_index,
                },
            )
            for item in items
        ]
        await self.client.upsert(collection_name=self.collection, points=points)

    async def delete_by_service(self, service_id: str) -> None:
        """Drop all vectors of one service (used before re-syncing its chunks).

        Without this, a re-sync inserts chunks with NEW uuids and the old
        points would linger as orphans, polluting search results with dead ids.
        """
        await self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="service_id",
                            match=models.MatchValue(value=service_id),
                        )
                    ]
                )
            ),
        )

    async def search(
        self,
        vector: list[float],
        top_k: int = 4,
        service_filter: str | None = None,
    ) -> list[VectorHit]:
        query_filter = None
        if service_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="service_name",
                        match=models.MatchValue(value=service_filter),
                    )
                ]
            )
        res = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            VectorHit(chunk_id=uuid.UUID(str(p.id)), score=float(p.score), payload=p.payload or {})
            for p in res.points
        ]


_client: AsyncQdrantClient | None = None
_store: QdrantStore | None = None


def get_vectorstore() -> QdrantStore:
    """Lazily-built shared store (client keeps a connection pool)."""
    global _client, _store
    if _store is None:
        _client = AsyncQdrantClient(url=settings.QDRANT_URL, timeout=30.0)
        _store = QdrantStore(client=_client, collection=settings.QDRANT_COLLECTION)
    return _store


async def ping() -> bool:
    """Best-effort reachability check for startup logs."""
    try:
        await asyncio.wait_for(get_vectorstore().client.get_collections(), timeout=5.0)
        return True
    except Exception as exc:
        logger.warning(f"Qdrant not reachable at {settings.QDRANT_URL}: {exc}")
        return False
