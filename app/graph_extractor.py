import logging
from typing import Any

from llama_index.core.prompts import PromptTemplate
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

    The output is constrained by GraphExtraction's JSON Schema, sent to the
    server as `response_format` by LlamaIndex's `astructured_predict()`
    (enabled by should_use_structured_outputs=True in app/llm.py): the server
    grammar-validates decoding token by token, so the response is always
    schema-valid JSON. This is what replaces the old regex dance of scraping
    ```json fences out of free-form text.
    """

    EXTRACTION_PROMPT = """\
Extract entities and relationships from the documentation text below.
Entities are services, API endpoints, databases, queues, protocols or concepts.
Relations are directed edges between two entities, e.g. "sends requests to", "authenticates via", "stores data in".

Text:
{text}
"""

    def __init__(self):
        self.llm = get_chat_model(
            temperature=0.1,
            system_prompt="You extract knowledge graphs from technical documentation.",
        )
        self.prompt = PromptTemplate(self.EXTRACTION_PROMPT)

    async def extract(self, text: str) -> dict[str, Any]:
        """
        Extract the graph fragment for one chunk.

        Returns an empty fragment instead of raising when the LLM is offline
        or output validation fails (astructured_predict raises ValidationError
        if the JSON still doesn't fit the schema after decoding): a
        documentation sync must not be lost because graph enrichment
        hiccuped, the vector chunks stay usable.
        """
        try:
            extraction = await self.llm.astructured_predict(
                GraphExtraction, self.prompt, text=text
            )
        except Exception as exc:
            logger.warning(f"Graph extraction failed: {exc}")
            return {"entities": [], "relations": []}

        return extraction.model_dump()
