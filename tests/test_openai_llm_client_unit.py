from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from aninamer.errors import OpenAIError
from aninamer.llm_client import ChatMessage
from aninamer.openai_llm_client import (
    HttpResponse,
    OpenAIConfig,
    OpenAIChatCompletionsLLM,
)


@dataclass
class FakeTransport:
    last_url: str | None = None
    last_body: bytes | None = None
    last_headers: dict[str, str] | None = None
    reply_status: int = 200
    reply_json: dict | None = None

    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        self.last_url = url
        self.last_body = body
        self.last_headers = headers

        payload = self.reply_json or {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"tmdb": 1}'},
                }
            ],
        }
        return HttpResponse(
            status=self.reply_status,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


def test_openai_llm_builds_chat_completions_request_and_extracts_text() -> None:
    transport = FakeTransport()
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com",
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIChatCompletionsLLM(cfg, transport=transport)

    out = llm.chat(
        [
            ChatMessage(role="system", content="Output JSON only."),
            ChatMessage(role="user", content='Return {"tmdb": 1}.'),
        ],
        max_output_tokens=64,
    )
    assert out.strip() == '{"tmdb": 1}'

    assert transport.last_url is not None
    assert transport.last_url.endswith("/v1/chat/completions")

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["model"] == "gpt-5.2"
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert isinstance(body["messages"], list)
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"

    assert transport.last_headers is not None
    assert "Authorization" in transport.last_headers
    assert transport.last_headers["Authorization"].startswith("Bearer ")


def test_openai_llm_base_url_with_v1_suffix() -> None:
    transport = FakeTransport()
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIChatCompletionsLLM(cfg, transport=transport)

    _ = llm.chat([ChatMessage(role="user", content="hi")], max_output_tokens=64)

    # base_url already had /v1
    assert transport.last_url is not None
    assert transport.last_url.endswith("/v1/chat/completions")


def test_openai_llm_includes_reasoning_effort() -> None:
    transport = FakeTransport()
    cfg = OpenAIConfig(
        api_key="KEY",
        model="o1-mini",
        base_url="https://api.openai.com/v1",
        reasoning_effort="high",
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIChatCompletionsLLM(cfg, transport=transport)

    _ = llm.chat([ChatMessage(role="user", content="hi")], max_output_tokens=64)

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["reasoning_effort"] == "high"


def test_openai_llm_raises_on_non_2xx() -> None:
    transport = FakeTransport(
        reply_status=401, reply_json={"error": {"message": "Invalid API key"}}
    )
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com",
        timeout=30.0,
        user_agent="x",
    )
    llm = OpenAIChatCompletionsLLM(cfg, transport=transport)

    with pytest.raises(OpenAIError) as exc:
        llm.chat([ChatMessage(role="user", content="hi")])

    assert "401" in str(exc.value)


def test_openai_llm_raises_if_no_choices_found() -> None:
    transport = FakeTransport(reply_json={"id": "chatcmpl", "choices": []})
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com",
        timeout=30.0,
        user_agent="x",
    )
    llm = OpenAIChatCompletionsLLM(cfg, transport=transport)

    with pytest.raises(OpenAIError, match="choices"):
        llm.chat([ChatMessage(role="user", content="hi")])
