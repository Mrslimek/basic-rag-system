"""
LLM plumbing unit tests: prompt-template shape, lru-cache semantics and one
generation round through a stubbed OpenAI client.

No HTTP happens anywhere here: the generator's client is replaced with a
stub returning a real openai ChatCompletion pydantic object (the exact shape
Haystack's OpenAIChatGenerator parses), so the reply-extraction path is real.
"""
from __future__ import annotations

from typing import Any

from haystack.dataclasses import ChatMessage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from app.llm import SYSTEM_INSTRUCTION, build_chat_prompt, get_chat_model


def test_build_chat_prompt_produces_system_and_user_messages() -> None:
    prompt = build_chat_prompt().run(
        query="How do I retry a payment?",
        context="Billing docs context.",
    )["prompt"]
    assert len(prompt) == 2
    system, user = prompt[0], prompt[1]
    assert system.role.value == "system"
    assert system.text == SYSTEM_INSTRUCTION
    assert user.role.value == "user"
    user_text = user.text
    assert user_text is not None
    assert "Billing docs context." in user_text
    assert "How do I retry a payment?" in user_text
    assert "Question:" in user_text


def test_get_chat_model_same_kwargs_return_same_instance() -> None:
    assert get_chat_model(temperature=0.2) is get_chat_model(temperature=0.2)
    kwargs: dict[str, Any] = {"response_format": {"type": "json_schema"}}
    assert get_chat_model(generation_kwargs=kwargs) is get_chat_model(
        generation_kwargs=kwargs
    )


def test_get_chat_model_different_kwargs_return_new_instance() -> None:
    base = get_chat_model(temperature=0.2)
    cold = get_chat_model(temperature=0.0)
    assert base is not cold
    assert get_chat_model(generation_kwargs={"x": 1}) is not get_chat_model(
        generation_kwargs={"x": 2}
    )


class _StubCompletions:
    def __init__(self, owner: _StubClient) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> ChatCompletion:
        self._owner.calls.append(kwargs)
        return ChatCompletion(
            id="chatcmpl-ztest",
            created=0,
            model="ztest-model",
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    logprobs=None,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=self._owner.content,
                        tool_calls=None,
                        function_call=None,
                    ),
                )
            ],
        )


class _StubChat:
    def __init__(self, completions: _StubCompletions) -> None:
        self.completions = completions


class _StubClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []
        self.chat = _StubChat(_StubCompletions(self))


def test_generation_with_stub_client_extracts_reply_text() -> None:
    generator = get_chat_model()
    stub = _StubClient("mocked answer")
    generator.client = stub  # type: ignore
    replies = generator.run(messages=[ChatMessage.from_user("hi")])["replies"]
    assert len(replies) == 1
    assert replies[0].text == "mocked answer"
    assert replies[0].role.value == "assistant"
    assert "messages" in stub.calls[0]