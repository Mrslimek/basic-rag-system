from functools import lru_cache

from llama_index.llms.openai_like import OpenAILike

from app.config import get_settings

# Single source of truth for the answer-policy prompt: the query engine in
# llama_rag.py gets it as system_prompt, so retrieval and synthesis can never
# drift apart via diverging prompts.
SYSTEM_INSTRUCTION = (
    "You are an expert microservices architecture assistant. "
    "Answer the user's question using ONLY the provided documentation context below. "
    "Always cite the microservice name and endpoint paths accurately. "
    "If the answer cannot be found in the context, explicitly state that."
)


@lru_cache
def get_chat_model(
    temperature: float = 0.2, system_prompt: str | None = None
) -> OpenAILike:
    """
    Shared OpenAILike client for every LLM call in the project.

    OpenAILike speaks the OpenAI wire protocol, which any local server
    (llama.cpp, Ollama, vLLM, LM Studio) exposes at LLM_API_BASE, so pointing
    it at the local endpoint is just a base_url swap. Instances are cached per
    (temperature, system_prompt) because constructing one opens an HTTP client
    pool under the hood, and we want a single pool reused by the API process
    and the worker.

    should_use_structured_outputs=True tells LlamaIndex that this server
    accepts `response_format={"type": "json_schema", ...}`, so
    `astructured_predict()` below the hood sends our Pydantic schema to the
    server and decoding becomes grammar-constrained. That is the flag replacing
    langchain's `with_structured_output(..., method="json_schema")`.
    """
    settings = get_settings()
    return OpenAILike(
        model=settings.LLM_MODEL,
        api_base=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        temperature=temperature,
        timeout=60.0,
        max_retries=1,
        is_chat_model=True,
        system_prompt=system_prompt,
        should_use_structured_outputs=True,
    )
