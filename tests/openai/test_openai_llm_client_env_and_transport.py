from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from aninamer.errors import OpenAIError
from aninamer.llm_client import ChatMessage
from aninamer.openai_llm_client import (
    HttpResponse,
    OpenAIConfig,
    OpenAIChatCompletionsLLM,
    load_openai_config_from_env,
    openai_llm_for_tmdb_id_from_env,
)


@dataclass
class CaptureTransport:
    response: HttpResponse
    calls: int = 0
    last_url: str | None = None
    last_body: bytes | None = None
    last_headers: dict[str, str] | None = None
    last_timeout: float | None = None

    def __call__(
        self, url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        self.calls += 1
        self.last_url = url
        self.last_body = body
        self.last_headers = headers
        self.last_timeout = timeout
        return self.response


def _response_with_text(text: str) -> HttpResponse:
    payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
            }
        ]
    }
    return HttpResponse(
        status=200, body=json.dumps(payload).encode("utf-8"), headers={}
    )


def test_load_openai_config_from_env_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        load_openai_config_from_env()


def test_load_openai_config_from_env_requires_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        load_openai_config_from_env()


def test_load_openai_config_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    config = load_openai_config_from_env()
    assert config.base_url == "https://api.openai.com"
    assert config.reasoning_effort is None


def test_load_openai_config_from_env_with_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_MAPPING", "medium")

    config = load_openai_config_from_env(
        reasoning_effort_env_vars=("OPENAI_REASONING_EFFORT_MAPPING",)
    )
    assert config.reasoning_effort == "medium"


def test_openai_llm_for_tmdb_id_from_env_prefers_tmdb_reasoning_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_CHORE", "high")

    transport = CaptureTransport(response=_response_with_text("ok"))
    client = openai_llm_for_tmdb_id_from_env(transport=transport)

    client.chat([ChatMessage(role="user", content="hello")], max_output_tokens=512)

    body = json.loads(transport.last_body.decode("utf-8"))
    assert body["max_tokens"] == 512
    assert body["reasoning_effort"] == "high"


def test_openai_llm_for_tmdb_id_from_env_ignores_legacy_reasoning_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_CHORE", raising=False)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "medium")

    transport = CaptureTransport(response=_response_with_text("ok"))
    client = openai_llm_for_tmdb_id_from_env(transport=transport)

    client.chat([ChatMessage(role="user", content="hello")], max_output_tokens=512)

    body = json.loads(transport.last_body.decode("utf-8"))
    assert body["reasoning_effort"] == "low"


def test_openai_llm_for_tmdb_id_from_env_defaults_low_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_CHORE", raising=False)

    transport = CaptureTransport(response=_response_with_text("ok"))
    client = openai_llm_for_tmdb_id_from_env(transport=transport)

    client.chat([ChatMessage(role="user", content="hello")], max_output_tokens=512)

    body = json.loads(transport.last_body.decode("utf-8"))
    assert body["reasoning_effort"] == "low"


def test_chat_builds_request_and_parses_output() -> None:
    transport = CaptureTransport(response=_response_with_text("ok"))
    config = OpenAIConfig(
        api_key="key", model="gpt-test", base_url="https://api.openai.com"
    )
    client = OpenAIChatCompletionsLLM(config, transport=transport)

    result = client.chat(
        [
            ChatMessage(role="system", content="sys1"),
            ChatMessage(role="system", content="sys2"),
            ChatMessage(role="user", content="hello"),
        ],
        max_output_tokens=10,
    )

    assert result == "ok"
    assert transport.last_url == "https://api.openai.com/v1/chat/completions"
    assert transport.last_headers is not None
    assert transport.last_headers["Authorization"] == "Bearer key"
    assert transport.last_headers["User-Agent"] == "aninamer/0.1"

    body = json.loads(transport.last_body.decode("utf-8"))
    assert body["model"] == "gpt-test"
    assert body["max_tokens"] == 10
    assert body["temperature"] == 0.0
    assert body["messages"] == [
        {"role": "system", "content": "sys1"},
        {"role": "system", "content": "sys2"},
        {"role": "user", "content": "hello"},
    ]


def test_chat_base_url_with_v1_suffix() -> None:
    transport = CaptureTransport(response=_response_with_text("ok"))
    config = OpenAIConfig(
        api_key="key", model="gpt-test", base_url="https://api.example.com/v1"
    )
    client = OpenAIChatCompletionsLLM(config, transport=transport)

    client.chat([ChatMessage(role="user", content="hello")])
    assert transport.last_url == "https://api.example.com/v1/chat/completions"


def test_chat_includes_reasoning_effort_when_configured() -> None:
    transport = CaptureTransport(response=_response_with_text("ok"))
    config = OpenAIConfig(
        api_key="key",
        model="o1-mini",
        base_url="https://api.openai.com",
        reasoning_effort="medium",
    )
    client = OpenAIChatCompletionsLLM(config, transport=transport)

    client.chat([ChatMessage(role="user", content="hello")], max_output_tokens=10)

    body = json.loads(transport.last_body.decode("utf-8"))
    assert body["reasoning_effort"] == "medium"


def test_chat_omits_reasoning_effort_when_not_configured() -> None:
    transport = CaptureTransport(response=_response_with_text("ok"))
    config = OpenAIConfig(
        api_key="key", model="gpt-test", base_url="https://api.openai.com"
    )
    client = OpenAIChatCompletionsLLM(config, transport=transport)

    client.chat([ChatMessage(role="user", content="hello")])

    body = json.loads(transport.last_body.decode("utf-8"))
    assert "reasoning_effort" not in body


def test_chat_raises_on_non_2xx() -> None:
    error_payload = {"error": {"message": "bad key"}}
    transport = CaptureTransport(
        response=HttpResponse(
            status=401,
            body=json.dumps(error_payload).encode("utf-8"),
            headers={},
        )
    )
    client = OpenAIChatCompletionsLLM(
        OpenAIConfig(
            api_key="key", model="gpt-test", base_url="https://api.openai.com"
        ),
        transport=transport,
    )

    with pytest.raises(OpenAIError, match="401"):
        client.chat([ChatMessage(role="user", content="hello")])


def test_chat_raises_when_no_choices() -> None:
    payload = {"choices": []}
    transport = CaptureTransport(
        response=HttpResponse(
            status=200, body=json.dumps(payload).encode("utf-8"), headers={}
        )
    )
    client = OpenAIChatCompletionsLLM(
        OpenAIConfig(
            api_key="key", model="gpt-test", base_url="https://api.openai.com"
        ),
        transport=transport,
    )

    with pytest.raises(OpenAIError, match="choices"):
        client.chat([ChatMessage(role="user", content="hello")])


def test_chat_raises_when_empty_content() -> None:
    payload = {"choices": [{"message": {"content": ""}}]}
    transport = CaptureTransport(
        response=HttpResponse(
            status=200, body=json.dumps(payload).encode("utf-8"), headers={}
        )
    )
    client = OpenAIChatCompletionsLLM(
        OpenAIConfig(
            api_key="key", model="gpt-test", base_url="https://api.openai.com"
        ),
        transport=transport,
    )

    with pytest.raises(OpenAIError, match="empty content"):
        client.chat([ChatMessage(role="user", content="hello")])
