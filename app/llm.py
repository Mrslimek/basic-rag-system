"""
The LLM half of the Haystack pipeline: one cached OpenAI-protocol chat
generator plus the prompt builder that owns the system instruction.

  Endpoint   = app.config LLM_API_BASE / LLM_MODEL / LLM_API_KEY. Any local
               OpenAI-compatible server (llama.cpp, Ollama, vLLM, LM Studio)
               is a base_url swap because OpenAIChatGenerator speaks the
               OpenAI wire protocol.
  Prompt     = SYSTEM_INSTRUCTION + the user template in build_chat_prompt.
               Haystack keeps the system prompt in the prompt TEMPLATE, not
               in the generator — unlike v1's get_chat_model(system_prompt=...).
  Instances  = get_chat_model is lru_cache-memoized per
               (temperature, max_tokens): constructing a generator opens an
               HTTP client pool, and the API process + worker must share one.

Owned by:
  app/query.py consumes both entry points (system prompt + chat client).
"""
from functools import lru_cache

from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from app.config import get_settings

# Single source of truth for the answer-policy text: build_chat_prompt embeds
# it as the system message, so retrieval and synthesis can never drift apart
# via diverging prompts.
SYSTEM_INSTRUCTION = (
    "You are an expert microservices architecture assistant. "
    "Answer the user's question using ONLY the provided documentation context below. "
    "Always cite the microservice name and endpoint paths accurately. "
    "If the answer cannot be found in the context, explicitly state that."
)


@lru_cache
def get_chat_model(
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> OpenAIChatGenerator:
    """
    Shared OpenAIChatGenerator for every LLM call in the project.

    OpenAIChatGenerator speaks the OpenAI wire protocol, which any local
    server (llama.cpp, Ollama, vLLM, LM Studio) exposes at LLM_API_BASE, so
    pointing it at the local endpoint is just a base_url swap. Instances are
    cached per (temperature, max_tokens) because constructing one opens an
    HTTP client pool under the hood, and the process wants a single reused
    pool.

    Conscious difference from v1: there is no `system_prompt` parameter —
    Haystack ChatMessage roles are part of the prompt (see build_chat_prompt),
    not of the generator. Structured outputs for the graph branch will be
    supplied per-call via generation_kwargs (e.g. response_format), the same
    opt-in "json_schema" contract v1 enabled for OpenAILike.
    """
    settings = get_settings()
    generation_kwargs: dict[str, float | int] = {"temperature": temperature}
    if max_tokens is not None:
        generation_kwargs["max_tokens"] = max_tokens

    return OpenAIChatGenerator(
        model=settings.LLM_MODEL,
        api_key=Secret.from_token(settings.LLM_API_KEY),
        api_base_url=settings.LLM_API_BASE,
        timeout=300.0,
        max_retries=1,
        generation_kwargs=generation_kwargs,
    )


def build_chat_prompt() -> ChatPromptBuilder:
    """
    Prompt builder for grounded answers: system instruction + a user template
    whose inputs are the retrieval context and the user question.

    Inputs/outputs are named by the template variables:
    run(query="...", context="...") -> {"prompt": [ChatMessage]}.
    """
    return ChatPromptBuilder(
        template=[
            ChatMessage.from_system(SYSTEM_INSTRUCTION),
            ChatMessage.from_user("{{context}}\n\nQuestion: {{query}}"),
        ]
    )