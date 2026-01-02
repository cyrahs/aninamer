from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

import pytest

from aninamer.errors import OpenAIError
from aninamer.llm_client import ChatMessage
from aninamer.openai_llm_client import (
    HttpResponse,
    OpenAIConfig,
    OpenAIResponsesLLM,
)


@dataclass
class FakeTransport:
    last_url: str | None = None
    last_body: bytes | None = None
    last_headers: dict[str, str] | None = None
    reply_status: int = 200
    reply_json: dict | None = None

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> HttpResponse:
        self.last_url = url
        self.last_body = body
        self.last_headers = headers

        payload = self.reply_json or {
            "id": "resp_test",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"tmdb": 1}'}],
                }
            ],
        }
        return HttpResponse(
            status=self.reply_status,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )


def test_openai_llm_builds_responses_request_and_extracts_text() -> None:
    transport = FakeTransport()
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com",
        reasoning_effort=None,
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIResponsesLLM(cfg, transport=transport)

    out = llm.chat(
        [
            ChatMessage(role="system", content="Output JSON only."),
            ChatMessage(role="user", content="Return {\"tmdb\": 1}."),
        ],
        max_output_tokens=64,
    )
    assert out.strip() == '{"tmdb": 1}'

    assert transport.last_url is not None
    assert transport.last_url.endswith("/v1/responses")

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["model"] == "gpt-5.2"
    assert body["store"] is False
    assert body["max_output_tokens"] == 64
    assert body["instructions"] == "Output JSON only."
    assert isinstance(body["input"], list)
    assert body["input"][0]["role"] == "user"

    assert transport.last_headers is not None
    assert "Authorization" in transport.last_headers
    assert transport.last_headers["Authorization"].startswith("Bearer ")


def test_openai_llm_logs_reasoning_output(caplog: pytest.LogCaptureFixture) -> None:
    transport = FakeTransport(
        reply_json={
            "id": "resp_test",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Reasoning summary"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"tmdb": 1}'}],
                },
            ],
        }
    )
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com",
        reasoning_effort=None,
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIResponsesLLM(cfg, transport=transport)

    caplog.set_level(logging.DEBUG)

    out = llm.chat([ChatMessage(role="user", content="hi")], max_output_tokens=64)
    assert out.strip() == '{"tmdb": 1}'
    assert "openai: reasoning_output" in caplog.text
    assert "Reasoning summary" in caplog.text


def test_openai_llm_includes_reasoning_effort_and_bumps_max_output_tokens() -> None:
    transport = FakeTransport()
    cfg = OpenAIConfig(
        api_key="KEY",
        model="gpt-5.2",
        base_url="https://api.openai.com/v1",
        reasoning_effort="high",
        timeout=30.0,
        user_agent="aninamer-test/0.0",
    )
    llm = OpenAIResponsesLLM(cfg, transport=transport)

    _ = llm.chat([ChatMessage(role="user", content="hi")], max_output_tokens=64)

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["reasoning"]["effort"] == "high"
    # bumped
    assert body["max_output_tokens"] >= 256
    # base_url already had /v1
    assert transport.last_url is not None
    assert transport.last_url.endswith("/v1/responses")


def test_openai_llm_raises_on_non_2xx() -> None:
    transport = FakeTransport(reply_status=401, reply_json={"error": {"message": "Invalid API key"}})
    cfg = OpenAIConfig(api_key="KEY", model="gpt-5.2", base_url="https://api.openai.com", reasoning_effort=None, timeout=30.0, user_agent="x")
    llm = OpenAIResponsesLLM(cfg, transport=transport)

    with pytest.raises(OpenAIError) as exc:
        llm.chat([ChatMessage(role="user", content="hi")])

    assert "401" in str(exc.value)


def test_openai_llm_raises_if_no_output_text_found() -> None:
    transport = FakeTransport(reply_json={"id": "resp", "output": [{"type": "reasoning", "content": []}]})
    cfg = OpenAIConfig(api_key="KEY", model="gpt-5.2", base_url="https://api.openai.com", reasoning_effort=None, timeout=30.0, user_agent="x")
    llm = OpenAIResponsesLLM(cfg, transport=transport)

    with pytest.raises(OpenAIError):
        llm.chat([ChatMessage(role="user", content="hi")])
