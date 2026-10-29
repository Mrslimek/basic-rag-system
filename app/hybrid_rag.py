"""
THE HYBRID branch of the Haystack pipeline — vector-first, graph-enriched.

Stage 1 reuses the Qdrant vector stage from app/vector_rag.py exactly like
the vector branch (same top_k, same service_filter semantics, same memoized
store). Stage 2 runs at retrieve time inside FactsExpander: for every
retrieved chunk the v2 graph schema is consulted — one hop backwards over
MENTIONS ((e:Entity)-[:MENTIONS]->(c:Chunk), the v2 direction — OPPOSITE to
v1) resolving the chunk's entities, then one semantic step outward from
those entities for the typed triplets. The facts land as a "Graph facts:"
text block appended to the chunk content; any Neo4j error degrades the
hybrid to the plain vector documents (one warning per call).

Neo4j is reached ONLY through graph_rag.query_cypher, and graph_rag is
imported lazily inside FactsExpander.run — importing app.hybrid_rag must
not pull the neo4j driver into sys.modules (the vector-only startup gate).
"""
import logging
from typing import Any

from haystack import Document, Pipeline, component

from app.vector_rag import build_retrieval_pipeline, build_vector_store

logger = logging.getLogger("hybrid_rag")


# Hop 1 of the hybrid expansion: chunk <- MENTIONS - entity (v2 direction),
# resolving each entity's display name + its type label (the writer's
# whitelisted SERVICE/QUEUE/... labels). Only $ids is injectable.
_MENTIONS_HOP_QUERY = (
    "MATCH (c:Chunk)<-[:MENTIONS]-(e:Entity) "
    "WHERE c.id IN $ids "
    "RETURN c.id AS chunk_id, e.name AS name, "
    "[l IN labels(e) WHERE l <> 'Entity'][0] AS type"
)

# Hop 2: one semantic step outward from the mentioned entities (v1 get_rel_map
# depth=1). MENTIONS is excluded so the walk never returns to chunk text.
_TRIPLETS_HOP_QUERY = (
    "MATCH (e:Entity)-[r]->(t:Entity) "
    "WHERE e.name IN $names AND type(r) <> 'MENTIONS' "
    "RETURN e.name AS source, type(r) AS relation, t.name AS target"
)


def build_hybrid_pipeline(
    top_k: int = 4,
    service_filter: str | None = None,
    collection_name: str | None = None,
) -> Pipeline:
    """
    Hybrid branch: text_embedder -> QdrantEmbeddingRetriever -> FactsExpander.

    The first two stages are the vector branch itself — app.vector_rag.
    build_retrieval_pipeline wires exactly make_text_embedder() +
    QdrantEmbeddingRetriever with the legacy meta.service_name filter dict
    over build_vector_store(collection_name) (None = default collection).
    FactsExpander then appends the Neo4j facts block without touching
    meta/id/score.

    Run contract:
        result = pipeline.run({"text_embedder": {"text": q}})
        result["facts_expander"]["documents"]  # list[Document], enriched content
    """
    pipeline = build_retrieval_pipeline(
        top_k=top_k,
        service_filter=service_filter,
        store=build_vector_store(collection_name),
    )
    pipeline.add_component("facts_expander", FactsExpander())
    pipeline.connect("retriever.documents", "facts_expander.documents")
    return pipeline


@component
class FactsExpander:
    """
    Appends each retrieved chunk's graph facts as a text block.

    Two batched queries per call (mentions hop + triplets hop) against the v2
    schema; chunks without mentions pass through untouched and an exotic
    empty result does the same without a warning. One Neo4j error => one
    logger.warning => the input documents are returned UNCHANGED — hybrid
    degrades to plain vector, the v1 contract.
    Inputs: "documents" (list[Document]) — the Qdrant retriever output.
    Outputs: {"documents": list[Document]} — same ids/meta/scores, content +
    the facts block for chunks that have graph facts.
    """

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        # Lazy import: graph_rag pulls the neo4j driver into sys.modules, and
        # app.hybrid_rag must stay neo4j-free at import time.
        from app import graph_rag

        try:
            facts_by_chunk = _expand_facts(graph_rag, documents)
        except Exception as exc:  # noqa: BLE001 - any graph failure must degrade to plain vector documents
            logger.warning(
                "Neo4j unavailable during hybrid expansion (%s); "
                "returning plain vector results.",
                exc,
            )
            return {"documents": documents}
        if not facts_by_chunk:
            return {"documents": documents}
        return {
            "documents": [
                _attach_facts(document, facts_by_chunk)
                for document in documents
            ]
        }


def _expand_facts(graph_rag: Any, documents: list[Document]) -> dict[str, str]:
    """Stages 2-3 per retrieved chunk: mentions hop, one triplets hop over the
    union of mentioned names, packaged into a facts block per chunk id."""
    chunk_ids = [document.id for document in documents if document.id]
    if not chunk_ids:
        return {}
    mentions_by_chunk = _collect_mentions(graph_rag, chunk_ids)
    if not mentions_by_chunk:
        return {}
    names = sorted(
        {name for mentions in mentions_by_chunk.values() for name, _ in mentions}
    )
    triplets = _semantic_triplets(graph_rag, names)
    return {
        chunk_id: _format_graph_facts(mentions, triplets)
        for chunk_id, mentions in mentions_by_chunk.items()
    }


def _collect_mentions(
    graph_rag: Any, chunk_ids: list[str]
) -> dict[str, list[tuple[str, str]]]:
    rows = graph_rag.query_cypher(_MENTIONS_HOP_QUERY, ids=chunk_ids)
    mentions_by_chunk: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        name = row.get("name")
        if not name:
            continue
        mention = (name, row.get("type") or "ENTITY")
        bucket = mentions_by_chunk.setdefault(str(row["chunk_id"]), [])
        if mention not in bucket:
            bucket.append(mention)
    return mentions_by_chunk


def _semantic_triplets(
    graph_rag: Any, names: list[str]
) -> list[tuple[str, str, str]]:
    rows = graph_rag.query_cypher(_TRIPLETS_HOP_QUERY, names=names)
    triplets: list[tuple[str, str, str]] = []
    for row in rows:
        source, relation, target = (
            row.get("source"),
            row.get("relation"),
            row.get("target"),
        )
        if source and relation and target:
            triplets.append((source, relation, target))
    return triplets


def _format_graph_facts(
    mentions: list[tuple[str, str]], triplets: list[tuple[str, str, str]]
) -> str:
    """The chunk's facts block in the v1 _format_graph_facts layout: mentioned
    entities with their type labels, then deduplicated "X -> REL -> Y" lines
    for the triplets whose source is one of this chunk's entities."""
    entity_line = "- Entities: " + ", ".join(
        f"{name} ({label})" for name, label in mentions
    )
    mentioned_names = {name for name, _ in mentions}
    lines = [entity_line]
    seen: set[tuple[str, str, str]] = set()
    for source, relation, target in triplets:
        if source not in mentioned_names:
            continue
        key = (source, relation, target)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {source} -> {relation} -> {target}")
    return "Graph facts:\n" + "\n".join(lines)


def _attach_facts(document: Document, facts_by_chunk: dict[str, str]) -> Document:
    """Stage 3 packaging: same id/meta/score as the vector document, content =
    original chunk + appended facts block; chunks without facts pass through
    untouched."""
    facts = facts_by_chunk.get(document.id)
    if not facts:
        return document
    return Document(
        id=document.id,
        content=f"{document.content}\n\n{facts}",
        meta=dict(document.meta),
        score=document.score,
    )