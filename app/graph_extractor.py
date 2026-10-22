import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_chat_model

logger = logging.getLogger("graph_extractor")


class Entity(BaseModel):
    name: str = Field(description="Canonical entity name as mentioned in the text")
    type: str = Field(description="Entity category, e.g. service, endpoint, database, protocol, concept")


class Relation(BaseModel):
    source: str = Field(description="Name of the entity this relation starts from")
    target: str = Field(description="Name of the entity this relation points to")
    relation: str = Field(description="Short verb phrase describing the connection")


class GraphExtraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class GraphExtractor:
    """
    Turns a documentation chunk into entities/relations via the local LLM.

    The output is constrained by GraphExtraction's JSON Schema
    (method="json_schema"): the server grammar-validates decoding token by
    token, so the response is always schema-valid JSON. This is what replaces
    the old regex dance of scraping ```json fences out of free-form text.
    """

    EXTRACTION_PROMPT = """\
Extract entities and relationships from the documentation text below.
Entities are services, API endpoints, databases, queues, protocols or concepts.
Relations are directed edges between two entities, e.g. "sends requests to", "authenticates via", "stores data in".

Text:
{text}
"""

    def __init__(self):
        self.structured_llm = get_chat_model(temperature=0.1).with_structured_output(
            GraphExtraction,
            method="json_schema",
            include_raw=True,
        )

    async def extract(self, text: str) -> dict[str, Any]:
        """
        Extract the graph fragment for one chunk.

        Returns an empty fragment instead of raising when the LLM is offline
        or output validation fails: a documentation sync must not be lost
        because graph enrichment hiccuped, the vector chunks stay usable.
        """
        try:
            result = await self.structured_llm.ainvoke(
                [
                    SystemMessage(content="You extract knowledge graphs from technical documentation."),
                    HumanMessage(content=self.EXTRACTION_PROMPT.format(text=text)),
                ]
            )
        except Exception as exc:
            logger.warning(f"Graph extraction call failed: {exc}")
            return {"entities": [], "relations": []}

        extraction, parse_error = result["parsed"], result["parsing_error"]
        if extraction is None:
            logger.warning(f"Graph extraction output failed validation: {parse_error}")
            return {"entities": [], "relations": []}

        return extraction.model_dump()
