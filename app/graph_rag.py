"""
THE GRAPH branch of the Haystack pipeline — the Neo4j rebuild-and-retrieve
half, parallel to the vector branch in app/vector_rag.py.

  Postgres   = source of truth for chunk text (unchanged, worker-owned).
  Neo4j      = the graph store OWNED by this module, written only by
               Neo4jGraphWriter: Chunk nodes, Entity nodes carrying a
               per-type label plus a vector embedding, MENTIONS edges and
               LLM-extracted typed relations. The writer hand-rolls Cypher
               over the raw neo4j driver (no neo4j-haystack integration), so
               the schema is exactly the queries below — all Community 5.26
               native calls, no APOC.
  Extraction = a small Haystack pipeline: ChatPromptBuilder (extraction
               rules) -> OpenAIChatGenerator pinned to a json_schema
               response_format -> TripletParser that normalizes entity names
               and validates every triplet against the schema Literals.
  Retrieval  = EntityGraphRetriever over Neo4j's native 384-dim
               "entity-embeddings" vector index plus keyword tokens over
               entity names; a chunk's score is its best entity's score.

Uppercase ENTITIES/RELATIONS Literals are an extractor constraint, not
style: v1's SchemaLLMPathExtractor uppercased every type before Literal
validation, so lowercase Literals silently extracted zero triplets — the
uppercase enums in the JSON schema keep the 1.5B model on-spec.

Run the graph rebuild after corpus changes (mass LLM calls):
    uv run python -m app.graph_rag sync
"""
import json
import logging
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, LiteralString, cast, get_args

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from app.config import get_settings
from app.embeddings import make_text_embedder
from app.llm import get_chat_model
from app.vector_rag import _load_chunk_rows

# Reused, not duplicated: _load_chunk_rows is the cross-store chunk contract —
# one Postgres join yields the identical UUID + provenance in Qdrant and Neo4j.
logger = logging.getLogger("graph_rag")
settings = get_settings()

# Uppercase literals — see module docstring, this is a hard extractor constraint.
ENTITIES = Literal["SERVICE", "ENDPOINT", "DATABASE", "QUEUE", "PROTOCOL", "CONCEPT"]
RELATIONS = Literal[
    "USES",
    "PROVIDES",
    "PUBLISHES_TO",
    "SUBSCRIBES_TO",
    "CALLS",
    "STORES_IN",
    "AUTHENTICATES_VIA",
]
VALIDATION = [
    ("SERVICE", "PROVIDES", "ENDPOINT"),
    ("SERVICE", "CALLS", "SERVICE"),
    ("SERVICE", "STORES_IN", "DATABASE"),
    ("SERVICE", "PUBLISHES_TO", "QUEUE"),
    ("SERVICE", "SUBSCRIBES_TO", "QUEUE"),
    ("SERVICE", "USES", "PROTOCOL"),
    ("SERVICE", "USES", "SERVICE"),
    ("SERVICE", "USES", "ENDPOINT"),
    ("SERVICE", "AUTHENTICATES_VIA", "SERVICE"),
    ("SERVICE", "AUTHENTICATES_VIA", "PROTOCOL"),
    ("ENDPOINT", "USES", "PROTOCOL"),
]

ENTITY_TYPES = get_args(ENTITIES)
RELATION_TYPES = get_args(RELATIONS)

MAX_TRIPLETS_PER_CHUNK = 8  # v1 extractor budget, kept identical

# Graph schema: labels/relationship/index are the single source below, quoted
# into Cypher exclusively from these constants or the validated type enums.
CHUNK_LABEL = "Chunk"
ENTITY_LABEL = "Entity"
MENTIONS_REL = "MENTIONS"
ENTITY_EMBEDDING_INDEX = "entity-embeddings"

# Extraction rules carried over from v1's EXTRACT_PROMPT_TMPL (all meaningful
# rules, verbatim semantics). The chunk text itself is the prompt_builder's
# "text" input, and the triplet shape is enforced by the json_schema below.
EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction assistant. "
    "Kafka topics are QUEUE entities: the producer service PUBLISHES_TO a "
    "topic and the consuming service SUBSCRIBES_TO it. Data stores (PostgreSQL "
    "clusters, Redis instances) are DATABASE entities; services connect to "
    "them with STORES_IN. JWT scopes: a service requiring a scope from "
    "AuthService links to AuthService with AUTHENTICATES_VIA; never model "
    "scopes as QUEUE entities. Create QUEUE entities only for topic names "
    "that contain dots (e.g. service.event.action). USES covers general "
    "dependencies or reading another service's data. CALLS is a synchronous "
    "call from one service to another. PROVIDES means a service exposes an "
    "API specification (ENDPOINT entity). Do not invent facts that are not in "
    f"the text. Limit the output to at most {MAX_TRIPLETS_PER_CHUNK} paths."
)
EXTRACTION_USER_TEMPLATE = (
    "Extract the knowledge graph from the text below.\n"
    "-------\n"
    "{{text}}\n"
    "-------"
)


def normalize_entity_name(name: str) -> str:
    # Dotted names are Kafka topic identifiers: keep their internal punctuation
    # verbatim, only strip and lowercase.
    if "." in name:
        return name.strip().lower()
    split_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return re.sub(r"[-_\s]+", " ", split_camel).strip().lower()


@dataclass(frozen=True, slots=True)
class KGEntity:
    """A normalized entity mention: name + schema-valid type."""

    name: str
    type: str


@dataclass(frozen=True, slots=True)
class KGRelation:
    """A schema-validated typed relation between two KGEntity endpoints."""

    head: KGEntity
    relation: str
    tail: KGEntity


@dataclass(frozen=True, slots=True)
class KGRecord:
    """One LLM-extracted triplet, normalized and validated by TripletParser."""

    head: str
    head_type: str
    relation: str
    tail: str
    tail_type: str


def _triplet_json_schema() -> dict[str, Any]:
    """
    OpenAI json_schema response_format pinning the extraction payload.

    Mirror LlamaIndex's SchemaLLMPathExtractor wire shape:
    {"triplets": [{"head", "head_type", "relation", "tail", "tail_type"}]} —
    the local model was calibrated on exactly this envelope, and llama.cpp
    compiles it into a constrained grammar (strict mode). maxItems turns
    MAX_TRIPLETS_PER_CHUNK into a hard grammar bound: under greedy decoding
    an unbounded array lets the 1.5B continue emitting triplets forever
    (observed: finish_reason=length at the full 2048-token budget), while
    the bounded grammar forces the reply to close after N items.
    """
    triplet = {
        "type": "object",
        "properties": {
            "head": {"type": "string"},
            "head_type": {"type": "string", "enum": list(ENTITY_TYPES)},
            "relation": {"type": "string", "enum": list(RELATION_TYPES)},
            "tail": {"type": "string"},
            "tail_type": {"type": "string", "enum": list(ENTITY_TYPES)},
        },
        "required": ["head", "head_type", "relation", "tail", "tail_type"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "kg_triplets",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "triplets": {
                        "type": "array",
                        "items": triplet,
                        "maxItems": MAX_TRIPLETS_PER_CHUNK,
                    }
                },
                "required": ["triplets"],
                "additionalProperties": False,
            },
        },
    }


def build_extraction_generator() -> OpenAIChatGenerator:
    """
    The extraction LLM: deterministic temperature, the extraction token
    budget, and the json_schema response_format (merged into the generator's
    generation_kwargs via the generation_kwargs extension in app.llm).
    """
    return get_chat_model(
        temperature=0.0,
        max_tokens=settings.extract_max_tokens,
        generation_kwargs={"response_format": _triplet_json_schema()},
    )


def _clean_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return normalize_entity_name(value)


def _clean_type(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def _parse_triplet(item: Any) -> KGRecord | None:
    # Uppercased-then-validated, like v1's SchemaLLMPathExtractor: the LLM is
    # grammar-constrained to uppercase, and occasional lowercase drifts must
    # not silently drop the triplet (or, worse, create new lowercase types).
    if not isinstance(item, dict):
        return None
    head = _clean_name(item.get("head"))
    tail = _clean_name(item.get("tail"))
    head_type = _clean_type(item.get("head_type"))
    tail_type = _clean_type(item.get("tail_type"))
    relation = _clean_type(item.get("relation"))
    if not (head and tail and relation):
        return None
    if head_type not in ENTITY_TYPES or tail_type not in ENTITY_TYPES:
        return None
    if relation not in RELATION_TYPES:
        return None
    if (head_type, relation, tail_type) not in VALIDATION:
        return None
    return KGRecord(
        head=head,
        head_type=head_type,
        relation=relation,
        tail=tail,
        tail_type=tail_type,
    )


def _parse_triplets(payload: dict[str, Any]) -> list[KGRecord]:
    raw_triplets = payload.get("triplets")
    if not isinstance(raw_triplets, list):
        return []
    records: list[KGRecord] = []
    for item in raw_triplets:
        record = _parse_triplet(item)
        if record is not None:
            records.append(record)
    return records


@component
class TripletParser:
    """
    Turns the extraction generator's JSON reply into validated KGRecords.

    Invalid triplets are DISCARDED, not raised — a partial or off-schema
    1.5B reply must not poison the chunk, and one bad chunk must never abort
    a whole sync.
    Inputs: "replies" (list[ChatMessage]), the OpenAIChatGenerator output.
    Outputs: {"records": list[KGRecord]}.
    """

    @component.output_types(records=list[KGRecord])
    def run(self, replies: list[ChatMessage]) -> dict[str, list[KGRecord]]:
        if not replies:
            return {"records": []}
        text = replies[-1].text
        try:
            payload = json.loads(text) if text else None
        except json.JSONDecodeError:
            logger.warning("Extractor returned malformed JSON; no triplets for this chunk")
            return {"records": []}
        if not isinstance(payload, dict):
            return {"records": []}
        return {"records": _parse_triplets(payload)}


def build_extraction_pipeline() -> Pipeline:
    """
    Chunk text -> validated triplets: prompt_builder (rules + chunk template)
    -> generator (json_schema reply) -> parser (normalize + validate).

    Run contract:
        result = pipeline.run({"prompt_builder": {"text": chunk_text}})
        result["parser"]["records"]  # list[KGRecord]
    """
    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_system(EXTRACTION_SYSTEM_PROMPT),
            ChatMessage.from_user(EXTRACTION_USER_TEMPLATE),
        ]
    )
    pipeline = Pipeline()
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", build_extraction_generator())
    pipeline.add_component("parser", TripletParser())
    pipeline.connect("prompt_builder.prompt", "generator.messages")
    pipeline.connect("generator.replies", "parser.replies")
    return pipeline


@lru_cache
def get_driver() -> Driver:
    """
    Process-wide singleton of the raw neo4j driver (its internal connection
    pool): the graph branch speaks Cypher directly instead of using the
    neo4j-haystack integration (v1's Neo4jPropertyGraphStore analogue).
    """
    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    )


def query_cypher(query: str, **params: Any) -> list[dict[str, Any]]:
    """
    One auto-committing Neo4j query, rows as plain dicts.

    The neo4j driver performs its own socket I/O on internal threads, so the
    graph branch — like the rest of the Haystack pipeline — is synchronous
    and never touches asyncio.
    """
    records, _, _ = get_driver().execute_query(
        cast(LiteralString, query), parameters_=params
    )
    return [record.data() for record in records]


def ensure_entity_index(driver: Driver | None = None) -> None:
    """
    Creates the native vector index over Entity.embedding, idempotently.

    Neo4j 5.26 Community has no IF NOT EXISTS index DDL — the create call
    raises on a duplicate name, so existence is checked through SHOW INDEXES
    first (deterministic introspection, not exception-driven control flow).
    The DDL is a fixed statement of this module's schema constants, so it is
    embedded literally; every dynamic piece of Cypher elsewhere in the
    module goes through the validated-enum whitelist instead.
    """
    active = driver or get_driver()
    with active.session() as session:
        names = {record["name"] for record in session.run("SHOW INDEXES YIELD name RETURN name")}
        if ENTITY_EMBEDDING_INDEX in names:
            return
        session.run(
            cast(
                LiteralString,
                "CALL db.index.vector.createNodeIndex("
                f"'{ENTITY_EMBEDDING_INDEX}', '{ENTITY_LABEL}', 'embedding', "
                f"{settings.EMBEDDING_DIMENSION}, 'COSINE')",
            )
        )


@component
class Neo4jGraphWriter:
    """
    Writes extraction results into Neo4j with hand-rolled, idempotent Cypher:
    MERGE keeps re-runs from growing the graph (same records -> same counts).

    Node writes:
      Chunk  — MERGE on id, SET the provenance payload.
      Entity — MERGE on name, then the whitelisted per-type labels and, on
               first creation only, the vector property via the native
               db.create.setNodeVectorProperty call.
    Edge writes:
      MENTIONS      — Entity -> Chunk, per entity mention.
      typed relation — MERGE between entity endpoints.

    One run call = one extraction batch: every record is linked to every
    Chunk in `documents` (the sync passes one chunk per call).
    Inputs: "records" (list[KGRecord]), "documents" (list[Document]).
    Outputs: {"written": int} — number of Chunk nodes upserted.
    """

    def __init__(self, driver: Driver | None = None) -> None:
        self._driver = driver
        # Query-prefix embedder: entity-name vectors must live in the same
        # space as the query vector the retriever KNNs with. A fresh instance
        # — one component cannot be added to several pipelines, and the ONNX
        # session underneath stays process-wide via fastembed's backend
        # factory.
        self._embedder = make_text_embedder()
        self._embedding_cache: dict[str, list[float]] = {}

    @component.output_types(written=int)
    def run(
        self, records: list[KGRecord], documents: list[Document]
    ) -> dict[str, int]:
        entities = self._collect_entities(records)
        embeddings = self._embed_entities(entities)
        for document in documents:
            self._write_chunk(document)
        self._write_entities(entities, embeddings)
        for document in documents:
            for name in entities:
                self._merge_mentions(name, str(document.id))
        self._write_relations(records)
        return {"written": len(documents)}

    @staticmethod
    def _collect_entities(records: list[KGRecord]) -> dict[str, set[str]]:
        # name -> set of schema-valid type labels. The whitelist re-check here
        # is what keeps the f-string labels below injection-safe even if a
        # caller supplies records that bypassed the parser.
        entities: dict[str, set[str]] = {}
        for record in records:
            for name, entity_type in (
                (record.head, record.head_type),
                (record.tail, record.tail_type),
            ):
                if entity_type not in ENTITY_TYPES:
                    continue
                entities.setdefault(name, set()).add(entity_type)
        return entities

    def _embed_entities(
        self, entities: dict[str, set[str]]
    ) -> dict[str, list[float]]:
        # FastembedTextEmbedder.run takes a single string (its prefix logic
        # concatenates text, so batch lists are out of contract) — embed one
        # name per call, memoized per writer: entity names recur across the
        # 52 chunks, and the ONNX session underneath is process-wide anyway.
        missing = [name for name in entities if name not in self._embedding_cache]
        for name in missing:
            self._embedding_cache[name] = self._embedder.run(text=name)["embedding"]
        return {name: self._embedding_cache[name] for name in entities}

    def _write_chunk(self, document: Document) -> None:
        meta = document.meta
        query_cypher(
            (
                f"MERGE (c:{CHUNK_LABEL} {{id: $chunk_id}}) "
                "SET c.content = $content, c.service_name = $service_name, "
                "c.service_id = $service_id, c.doc_file_path = $doc_file_path, "
                "c.chunk_index = $chunk_index"
            ),
            chunk_id=document.id,
            content=document.content,
            service_name=meta.get("service_name", ""),
            service_id=meta.get("service_id", ""),
            doc_file_path=meta.get("doc_file_path", ""),
            chunk_index=meta.get("chunk_index", -1),
        )

    def _write_entities(
        self,
        entities: dict[str, set[str]],
        embeddings: dict[str, list[float]],
    ) -> None:
        for name, entity_types in entities.items():
            label_sets = " ".join(f"SET e:{t}" for t in sorted(entity_types))
            query_cypher(
                (
                    f"MERGE (e:{ENTITY_LABEL} {{name: $name}})\n"
                    f"{label_sets}\n"
                    "WITH e\n"
                    "WHERE e.embedding IS NULL\n"
                    "CALL db.create.setNodeVectorProperty(e, 'embedding', $embedding)"
                ),
                name=name,
                embedding=embeddings[name],
            )

    def _merge_mentions(self, entity_name: str, chunk_id: str) -> None:
        query_cypher(
            (
                f"MATCH (c:{CHUNK_LABEL} {{id: $chunk_id}}), "
                f"(e:{ENTITY_LABEL} {{name: $name}}) "
                f"MERGE (e)-[:{MENTIONS_REL}]->(c)"
            ),
            chunk_id=chunk_id,
            name=entity_name,
        )

    def _write_relations(self, records: list[KGRecord]) -> None:
        for record in records:
            if record.relation not in RELATION_TYPES:
                continue
            query_cypher(
                (
                    f"MATCH (h:{ENTITY_LABEL} {{name: $head}}), "
                    f"(t:{ENTITY_LABEL} {{name: $tail}}) "
                    f"MERGE (h)-[:{record.relation}]->(t)"
                ),
                head=record.head,
                tail=record.tail,
            )


@component
class EntityGraphRetriever:
    """
    Retrieval over the property graph — two independent signals over Entity
    nodes, fused by chunk:

      1. Vector KNN: db.index.vector.queryNodes over the 'entity-embeddings'
         index with the query embedding, $k = max(20, top_k * 5) candidates.
      2. Keyword: entity names containing any query token — tokens are
         [a-z0-9.]+ runs of length >= 3 from the lowercased query; exact
         containment scores 1.0.

    A chunk's score = max over the scores of its MENTIONed entities, so a
    chunk matching several entities is not over-weighted. A graph without
    entities (or without an index) yields [].
    Inputs: "query_text" (str), "query_embedding" (list[float]) — the latter
    is fed by the query embedder upstream in the pipeline.
    Outputs: {"documents": list[Document]} — content + provenance meta +
    score, top_k by score.
    """

    def __init__(self, driver: Driver, top_k: int = 4) -> None:
        self._driver = driver
        self._top_k = top_k

    @component.output_types(documents=list[Document])
    def run(
        self, query_text: str, query_embedding: list[float]
    ) -> dict[str, list[Document]]:
        if not query_text.strip():
            return {"documents": []}
        entity_scores: dict[str, float] = {}
        self._merge_scores(entity_scores, self._keyword_lookup(query_text))
        self._merge_scores(entity_scores, self._vector_lookup(query_embedding))
        if not entity_scores:
            return {"documents": []}
        return {"documents": self._chunk_documents(entity_scores)}

    @staticmethod
    def _merge_scores(
        total: dict[str, float], fresh: dict[str, float]
    ) -> None:
        for name, score in fresh.items():
            if score > total.get(name, 0.0):
                total[name] = score

    def _keyword_lookup(self, query_text: str) -> dict[str, float]:
        tokens = {
            token
            for token in re.findall(r"[a-z0-9.]+", query_text.lower())
            if len(token) >= 3
        }
        if not tokens:
            return {}
        rows = query_cypher(
            (
                f"MATCH (e:{ENTITY_LABEL}) "
                "WHERE any(t IN $tokens WHERE e.name CONTAINS t) "
                "RETURN e.name AS name"
            ),
            tokens=list(tokens),
        )
        return {row["name"]: 1.0 for row in rows}

    def _vector_lookup(self, query_embedding: list[float]) -> dict[str, float]:
        if not query_embedding:
            return {}
        try:
            rows = query_cypher(
                (
                    "CALL db.index.vector.queryNodes("
                    f"'{ENTITY_EMBEDDING_INDEX}', $k, $vector) "
                    "YIELD node, score "
                    "RETURN node.name AS name, score"
                ),
                k=max(20, self._top_k * 5),
                vector=list(query_embedding),
            )
        except Neo4jError as error:
            # A never-synced database has no vector index; keyword retrieval
            # still applies, so degrade instead of raising.
            logger.debug("entity-embeddings index unavailable: %s", error)
            return {}
        return {row["name"]: float(row["score"]) for row in rows}

    def _chunk_documents(
        self, entity_scores: dict[str, float]
    ) -> list[Document]:
        rows = query_cypher(
            (
                f"MATCH (e:{ENTITY_LABEL})-[:{MENTIONS_REL}]->"
                f"(c:{CHUNK_LABEL}) "
                "WHERE e.name IN $names "
                "RETURN c.id AS chunk_id, c.content AS content, "
                "c.service_name AS service_name, c.service_id AS service_id, "
                "c.doc_file_path AS doc_file_path, c.chunk_index AS chunk_index, "
                "e.name AS entity"
            ),
            names=list(entity_scores),
        )
        by_chunk: dict[str, dict[str, Any]] = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            slot = by_chunk.setdefault(
                chunk_id,
                {
                    "content": row["content"],
                    "service_name": row["service_name"],
                    "service_id": row["service_id"],
                    "doc_file_path": row["doc_file_path"],
                    "chunk_index": row["chunk_index"],
                    "score": 0.0,
                    "graph_entities": [],
                },
            )
            score = entity_scores[row["entity"]]
            slot["score"] = max(slot["score"], score)
            slot["graph_entities"].append(row["entity"])

        documents = [
            Document(
                id=chunk_id,
                content=slot["content"],
                meta={
                    "service_name": slot["service_name"],
                    "service_id": slot["service_id"],
                    "doc_file_path": slot["doc_file_path"],
                    "chunk_index": slot["chunk_index"],
                    "graph_entities": slot["graph_entities"],
                },
                score=slot["score"],
            )
            for chunk_id, slot in by_chunk.items()
        ]
        documents.sort(key=lambda doc: doc.score or 0.0, reverse=True)
        return documents[: self._top_k]


def build_graph_pipeline(top_k: int = 4) -> Pipeline:
    """
    Retrieval pipeline: query embedder -> native graph retriever.

    Run contract (Haystack connects only one-to-one output->input pairs, so
    the raw query text is supplied by the caller alongside the embedding):
        result = pipeline.run({
            "text_embedder": {"text": question},
            "graph_retriever": {"query_text": question},
        })
        result["graph_retriever"]["documents"]  # list[Document]
    """
    pipeline = Pipeline()
    pipeline.add_component("text_embedder", make_text_embedder())
    pipeline.add_component(
        "graph_retriever", EntityGraphRetriever(driver=get_driver(), top_k=top_k)
    )
    pipeline.connect("text_embedder.embedding", "graph_retriever.query_embedding")
    return pipeline


def _build_graph_documents(rows: list[Any]) -> list[Document]:
    """
    Chunk rows -> writer Documents. Mirrors the Qdrant-side document shape
    field-for-field (same provenance payload) so a Chunk node carries the
    identical metadata a Qdrant point does — the cross-store join contract.
    """
    return [
        Document(
            id=str(row["id"]),
            content=row["content"],
            meta={
                "service_name": row["name"],
                "service_id": str(row["service_id"]),
                "doc_file_path": row["doc_file_path"],
                "chunk_index": row["chunk_index"],
            },
        )
        for row in rows
    ]


def sync_from_postgres() -> dict[str, int]:
    """
    Full rebuild of the module-owned graph from Postgres chunks.

    Full rebuild instead of incremental upsert for the same reason as the
    vector branch: worker resyncs replace chunk rows with NEW uuids, and a
    wholesale wipe (MATCH (n) DETACH DELETE n) cannot leave dangling nodes.
    One bad chunk (LLM outage / malformed reply) is logged and stored
    without triplets — it never aborts the rebuild.
    """
    driver = get_driver()
    ensure_entity_index(driver)
    query_cypher("MATCH (n) DETACH DELETE n")

    rows = _load_chunk_rows()
    extraction = build_extraction_pipeline()
    writer = Neo4jGraphWriter(driver)

    for index, row in enumerate(rows, start=1):
        document = _build_graph_documents([row])[0]
        try:
            result = extraction.run({"prompt_builder": {"text": row["content"]}})
            records: list[KGRecord] = result["parser"]["records"]
        except Exception:
            logger.exception(
                "Extraction failed for chunk %s; storing it without triplets",
                row["id"],
            )
            records = []
        writer.run(records=records, documents=[document])
        if index % 10 == 0 or index == len(rows):
            logger.info("Graph chunks processed: %d/%d", index, len(rows))

    stats = {
        "postgres_rows": len(rows),
        "graph_chunks": query_cypher(
            f"MATCH (c:{CHUNK_LABEL}) RETURN count(c) AS n"
        )[0]["n"],
        "graph_entities": query_cypher(
            f"MATCH (e:{ENTITY_LABEL}) RETURN count(e) AS n"
        )[0]["n"],
        "graph_relations": query_cypher(
            f"MATCH ()-[r]->() WHERE type(r) <> '{MENTIONS_REL}' RETURN count(r) AS n"
        )[0]["n"],
    }
    logger.info("Graph rebuilt: %s", stats)
    return stats


def _cli() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "sync":
        stats = sync_from_postgres()
        print(f"synced: {stats}")
    else:
        print("usage: python -m app.graph_rag sync")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _cli()