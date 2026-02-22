from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from aninamer.llm_client import ChatMessage
from aninamer.openai_llm_client import (
    HttpResponse,
    openai_llm_for_tmdb_id_from_env,
    openai_llm_from_env,
)


@dataclass
class CaptureTransport:
    last_url: str | None = None
    last_body: bytes | None = None
    last_headers: dict[str, str] | None = None

    def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> HttpResponse:
        self.last_url = url
        self.last_body = body
        self.last_headers = headers

        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": '{"tmdb": 1}'},
                }
            ],
        }
        return HttpResponse(status=200, body=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})


def test_openai_llm_for_tmdb_id_uses_tmdb_reasoning_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_CHORE", "high")

    transport = CaptureTransport()
    llm = openai_llm_for_tmdb_id_from_env(transport=transport)

    _ = llm.chat([ChatMessage(role="user", content='Return {"tmdb": 1} only.')], max_output_tokens=512)

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 512


def test_openai_llm_from_env_prefers_mapping_reasoning_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "KEY")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.2")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_MAPPING", "high")

    transport = CaptureTransport()
    llm = openai_llm_from_env(transport=transport)

    _ = llm.chat([ChatMessage(role="user", content="hi")], max_output_tokens=64)

    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 64
