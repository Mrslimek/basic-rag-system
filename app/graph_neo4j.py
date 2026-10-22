"""
Neo4j-backed knowledge graph over the extracted chunk entities/relations.

The Postgres graph tables (models.py) hold the same semantics with relational
keys; this module is the native-graph twin of them: entity resolution happens
by MERGE on (canonical_name, entity_type), and multi-hop questions become a
single Cypher pattern match instead of a recursive CTE.

The resolver here is intentionally "safe but blind": normalize + exact match
only. No fuzzy merge, no LLM judge — payments-svc will NOT find Payment
Gateway Service yet. That gap is a known, deliberate next lesson.

Usage:
    python -m app.graph_neo4j ingest
    python -m app.graph_neo4j stats
    python -m app.graph_neo4j expand "billing" [max_hops]
    python -m app.graph_neo4j path "billing" "ha pg eu 1"
"""

import asyncio
import logging
import re
import sys
from functools import lru_cache

from neo4j import AsyncGraphDatabase
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db_context
from app.models import Service, ServiceDocumentationChunk

logger = logging.getLogger("graph_neo4j")


def normalize(text: str) -> str:
    """Surface form -> match key: lowercase, spaces/underscores/dashes unified."""
    return re.sub(r"[\s_\-]+", " ", text.strip().lower())


MAX_NAME_CHARS = 80  # sentence-shaped extractions like "Billing calls Auth via gRPC"
_JUNK_PREFIX = re.compile(r"^\d{3}\b")  # "200 OK", "404 Not Found"


def _is_junk(name: str) -> bool:
    """Cheap heuristic filter for extraction noise. Real fix = better extraction."""
    if len(name) > MAX_NAME_CHARS or _JUNK_PREFIX.match(name):
        return True
    if len(name.split()) > 6:
        return True
    return name.strip(" .:,;'\"`()") == ""


@lru_cache
def get_driver():
    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
    )


# Constraints make re-ingestion idempotent: MERGE finds what INSERT would duplicate.
_INIT_STATEMENTS = (
    "CREATE CONSTRAINT entity_name_type IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.canonical_name, e.entity_type) IS UNIQUE",
    "CREATE INDEX alias_normalized IF NOT EXISTS FOR (a:Alias) ON (a.normalized)",
)


async def init_graph() -> None:
    async with get_driver().session() as session:
        for stmt in _INIT_STATEMENTS:
            await session.run(stmt)


async def _upsert_entity(session, raw_name: str, raw_type: str) -> str:
    """
    Resolve-or-create a canonical node. Identity is (normalized name, type) —
    the same string as two types stays two nodes (type guard lesson).
    """
    norm = normalize(raw_name)
    etype = normalize(raw_type) or "concept"
    result = await session.run(
        """
        MERGE (e:Entity {canonical_name: $n, entity_type: $t})
        ON CREATE SET e.id = randomUUID(), e.display_name = $raw
        MERGE (a:Alias {normalized: $n})
        ON CREATE SET a.raw_first_seen = $raw
        MERGE (a)-[:REFERS_TO]->(e)
        RETURN e.id AS id
        """,
        n=norm, t=etype, raw=raw_name.strip(),
    )
    return (await result.single())["id"]


async def _lookup_entity(session, norm: str) -> str | None:
    """
    Find an existing node by normalized name regardless of type, preferring
    'service' — relation endpoints often reference entities this chunk's
    entity list forgot to mention, and we must not mint a duplicate for them.
    """
    result = await session.run(
        """
        MATCH (e:Entity {canonical_name: $n})
        RETURN e.id AS id, e.entity_type AS t
        ORDER BY CASE e.entity_type WHEN 'service' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        n=norm,
    )
    record = await result.single()
    return record["id"] if record else None


async def _upsert_edge(
    session, source_id: str, target_id: str, relation: str, chunk_id: str, doc: str
) -> bool:
    """
    One REL relationship per (source, target, relation) triple, mirroring
    UNIQUE(source_id, target_id, relation) of the Postgres twin. Evidence of
    which chunk asserted the fact accumulates in list properties instead of a
    mentions table. The self-loop guard replaces CHECK (source <> target).
    """
    if source_id == target_id:
        return False
    result = await session.run(
        """
        MATCH (s:Entity {id: $sid}), (t:Entity {id: $tid})
        MERGE (s)-[r:REL {name: $rel}]->(t)
        ON CREATE SET r.chunks = [$cid], r.docs = [$doc]
        ON MATCH  SET r.chunks = CASE WHEN $cid IN r.chunks THEN r.chunks ELSE r.chunks + $cid END,
                    r.docs   = CASE WHEN $doc IN r.docs   THEN r.docs   ELSE r.docs + $doc   END
        RETURN r
        """,
        sid=source_id, tid=target_id, rel=normalize(relation), cid=chunk_id, doc=doc,
    )
    await result.consume()
    return True


async def ingest() -> dict:
    """
    Push every chunk's JSONB extraction into Neo4j. Reads Postgres once,
    writes per-chunk entity/edge statements; returns counters.
    """
    await init_graph()
    stats = {"chunks": 0, "entities": 0, "relations": 0, "skipped_relations": 0, "junk": 0}

    async with get_db_context() as db:
        rows = (
            await db.execute(
                select(
                    Service.name,
                    ServiceDocumentationChunk.id,
                    ServiceDocumentationChunk.entities,
                    ServiceDocumentationChunk.relations,
                ).join(Service, Service.id == ServiceDocumentationChunk.service_id)
            )
        ).all()

    async with get_driver().session() as session:
        for doc, chunk_id, entities, relations in rows:
            stats["chunks"] += 1
            chunk_ids: dict[str, str] = {}  # normalized name -> entity id, this chunk only

            for ent in entities or []:
                name = (ent.get("name") or "").strip()
                if not name or _is_junk(name):
                    stats["junk"] += 1
                    continue
                chunk_ids[normalize(name)] = await _upsert_entity(
                    session, name, ent.get("type") or "concept"
                )
                stats["entities"] += 1

            for rel in relations or []:
                src_name = normalize(rel.get("source", ""))
                tgt_name = normalize(rel.get("target", ""))
                if not src_name or not tgt_name:
                    stats["skipped_relations"] += 1
                    continue
                # Endpoints: this chunk's entities first, then the global
                # catalog; last resort create under 'concept' (we only know
                # the name the relation cited).
                resolved = {}
                for side, norm in (("source", src_name), ("target", tgt_name)):
                    entity_id = chunk_ids.get(norm) or await _lookup_entity(session, norm)
                    if entity_id is None:
                        entity_id = await _upsert_entity(session, norm, "concept")
                    resolved[side] = entity_id
                if await _upsert_edge(
                    session,
                    resolved["source"],
                    resolved["target"],
                    rel.get("relation", ""),
                    str(chunk_id),
                    doc,
                ):
                    stats["relations"] += 1
                else:
                    stats["skipped_relations"] += 1

    logger.info("Neo4j ingest done: %s", stats)
    return stats


async def expand(entity_name: str, max_hops: int = 3, limit: int = 50) -> list[dict]:
    """All directed paths of 1..max_hops steps from the named entity."""
    hops = max(1, min(4, int(max_hops)))  # interpolated, must be a bounded int
    async with get_driver().session() as session:
        result = await session.run(
            f"""
            MATCH path = (start:Entity {{canonical_name: $n}})-[:REL*1..{hops}]->(other:Entity)
            RETURN [x IN nodes(path) | x.canonical_name] AS entities,
                   [r IN relationships(path) | r.name] AS relations,
                   [r IN relationships(path) | r.chunks] AS evidence,
                   length(path) AS hops
            ORDER BY hops
            LIMIT $limit
            """,
            n=normalize(entity_name), limit=limit,
        )
        return await result.data()


async def shortest_path(from_name: str, to_name: str) -> dict | None:
    """Undirected shortest chain — the multi-hop eval questions become one query."""
    async with get_driver().session() as session:
        # Named "start"/"target", not "from": `from` is a reserved word in
        # Python, and kwargs must match the Cypher parameter names exactly.
        result = await session.run(
            """
            MATCH (a:Entity {canonical_name: $start}), (b:Entity {canonical_name: $target})
            MATCH path = shortestPath((a)-[:REL*..6]-(b))
            RETURN [x IN nodes(path) | x.canonical_name] AS entities,
                   [r IN relationships(path) | r.name] AS relations,
                   length(path) AS hops
            """,
            start=normalize(from_name), target=normalize(to_name),
        )
        records = await result.data()
        return records[0] if records else None


async def graph_stats() -> dict:
    async with get_driver().session() as session:
        result = await session.run(
            """
            // count-subqueries, not MATCH...RETURN count(): on an empty graph a
            // plain MATCH produces zero ROWS, so the aggregation never runs and
            // single() returns None instead of zeros.
            RETURN count { MATCH (e:Entity) } AS entities,
                   count { MATCH ()-[r:REL]->() } AS relations,
                   count { MATCH (a:Alias) } AS aliases
            """
        )
        return dict((await result.single()))


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    # An empty/young graph legitimately references not-yet-existing labels;
    # the driver's schema warnings would bury the actual output.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    command = sys.argv[1] if len(sys.argv) > 1 else "ingest"

    if command == "ingest":
        print(await ingest())
    elif command == "stats":
        print(await graph_stats())
    elif command == "expand":
        rows = await expand(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
        for row in rows:
            line = row["entities"][0]
            for i, step in enumerate(row["relations"]):
                line += f" -[{step}]-> {row['entities'][i + 1]}"
            print(f"[{row['hops']} hop] {line}")
    elif command == "path":
        found = await shortest_path(sys.argv[2], sys.argv[3])
        print(found or "no chain found")
    else:
        raise SystemExit(f"unknown command: {command}")

    await get_driver().close()


if __name__ == "__main__":
    asyncio.run(_main())
