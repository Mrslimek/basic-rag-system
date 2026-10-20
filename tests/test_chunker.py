from app.chunker import chunk_markdown_paragraphs


def test_chunk_markdown_paragraphs_basic():
    sample_md = """# Service Title

This is the introductory paragraph describing what this microservice does.

## Endpoints

`POST /api/v1/resource`
Creates a resource in the system.

`GET /api/v1/resource/{id}`
Retrieves a resource by identifier.
"""
    chunks = chunk_markdown_paragraphs(sample_md, min_length=5)
    
    # 5 distinct paragraph blocks separated by double newlines
    assert len(chunks) == 5
    assert chunks[0].index == 0
    assert chunks[0].content == "# Service Title"
    assert "introductory paragraph" in chunks[1].content
    assert chunks[2].content == "## Endpoints"
    assert "POST /api/v1/resource" in chunks[3].content
    assert "GET /api/v1/resource/{id}" in chunks[4].content


def test_chunk_markdown_paragraphs_filters_noise_and_empty():
    sample_md = "\n\n   \n\n- \n\nValid paragraph content here.\n\n\n\nAnother valid section.\n\n"
    chunks = chunk_markdown_paragraphs(sample_md, min_length=10)
    
    assert len(chunks) == 2
    assert chunks[0].index == 0
    assert chunks[0].content == "Valid paragraph content here."
    assert chunks[1].index == 1
    assert chunks[1].content == "Another valid section."


def test_chunk_markdown_empty_input():
    assert chunk_markdown_paragraphs("") == []
    assert chunk_markdown_paragraphs("   \n\n  \n") == []
