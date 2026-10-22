from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


@lru_cache
def get_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """
    Shared ChatOpenAI client for every LLM call in the project.

    ChatOpenAI speaks the OpenAI wire protocol, which any local server
    (llama.cpp, Ollama, vLLM, LM Studio) exposes at LLM_API_BASE, so pointing
    it at the local endpoint is just a base_url swap. Instances are cached per
    temperature because constructing one opens an HTTP client pool under the
    hood, and we want a single pool reused by the API process and the worker.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        base_url=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        temperature=temperature,
        timeout=60.0,
        max_retries=1,
    )
