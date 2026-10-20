import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentChunk:
    """Immutable data container representing a parsed chunk of documentation."""
    index: int
    content: str


def chunk_markdown_paragraphs(text: str, min_length: int = 10) -> list[DocumentChunk]:
    """
    Deterministic paragraph-based markdown chunker.
    
    Why paragraph chunking for technical microservice documentation?
    1. Technical markdown documents naturally group concepts into semantic blocks (endpoints,
       request/response payloads, configuration tables, prerequisites) separated by empty lines (`\\n\\n+`).
    2. Arbitrary token-window chunkers often fracture JSON payloads or YAML code blocks midway,
       destroying semantic context for LLMs and embedding models.
    3. Paragraph splitting preserves heading-context pairs cleanly and deterministically.
    
    Args:
        text: Raw markdown text content.
        min_length: Minimum character length for a paragraph block to avoid trivial noise (e.g. lone dashes).
        
    Returns:
        List of DocumentChunk instances with deterministic 0-based indices.
    """
    if not text or not text.strip():
        return []

    # Normalize Windows CRLF to standard LF
    normalized = text.replace("\r\n", "\n")

    # Split on two or more consecutive newlines (paragraph boundary)
    raw_blocks = re.split(r"\n\s*\n+", normalized)

    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for block in raw_blocks:
        cleaned = block.strip()
        # Filter out empty or trivially short blocks
        if len(cleaned) >= min_length:
            chunks.append(DocumentChunk(index=chunk_idx, content=cleaned))
            chunk_idx += 1

    return chunks
