from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Callable, Sequence
from urllib import error as url_error
from urllib import request

from aninamer.errors import OpenAIError
from aninamer.llm_client import ChatMessage, LLMClient


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com"
    reasoning_effort: str | None = None
    timeout: float = 60.0
    user_agent: str = "aninamer/0.1"


def load_openai_config_from_env() -> OpenAIConfig:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL")
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    if not model:
        raise ValueError("OPENAI_MODEL is required")

    if base_url is None or base_url.strip() == "":
        base_url = "https://api.openai.com"
    else:
        base_url = base_url.strip()

    if reasoning_effort is not None:
        reasoning_effort = reasoning_effort.strip()
        if reasoning_effort == "":
            reasoning_effort = None

    return OpenAIConfig(
        api_key=api_key,
        model=model,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]


Transport = Callable[[str, bytes, dict[str, str], float], HttpResponse]


def _default_transport(
    url: str, body: bytes, headers: dict[str, str], timeout: float
) -> HttpResponse:
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            resp_headers = {key: value for key, value in resp.headers.items()}
            return HttpResponse(status=resp.status, body=resp_body, headers=resp_headers)
    except url_error.HTTPError as exc:
        resp_body = exc.read()
        resp_headers = {key: value for key, value in exc.headers.items()}
        return HttpResponse(status=exc.code, body=resp_body, headers=resp_headers)
    except Exception as exc:  # pragma: no cover - guardrail for unexpected transport errors
        raise OpenAIError(f"transport error: {exc}") from exc


def _endpoint_for_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def _parse_response_json(body: bytes) -> dict[str, object]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"invalid JSON response: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise OpenAIError("invalid JSON response: expected object")
    return payload


def _extract_error_message(body: bytes) -> str:
    try:
        payload = _parse_response_json(body)
    except OpenAIError:
        text = body.decode("utf-8", errors="replace").strip()
        return text or "unknown error"

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return "unknown error"


class OpenAIResponsesLLM(LLMClient):
    def __init__(self, config: OpenAIConfig, *, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _default_transport

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        del temperature

        system_messages = [msg.content for msg in messages if msg.role == "system"]
        instructions = "\n\n".join(system_messages)
        has_instructions = any(content.strip() for content in system_messages)

        input_items = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"
        ]

        reasoning_effort = self._config.reasoning_effort
        if reasoning_effort is not None:
            reasoning_effort = reasoning_effort.strip()
            if reasoning_effort == "":
                reasoning_effort = None

        effective_max_output_tokens = max_output_tokens
        if reasoning_effort is not None and reasoning_effort.lower() != "none":
            effective_max_output_tokens = max(max_output_tokens, 256)

        payload: dict[str, object] = {
            "model": self._config.model,
            "input": input_items,
            "max_output_tokens": effective_max_output_tokens,
            "store": False,
        }
        if has_instructions:
            payload["instructions"] = instructions
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._config.user_agent,
        }
        endpoint = _endpoint_for_base_url(self._config.base_url)
        response = self._transport(endpoint, body, headers, self._config.timeout)

        if response.status < 200 or response.status >= 300:
            message = _extract_error_message(response.body)
            raise OpenAIError(f"OpenAI API error {response.status}: {message}")

        payload_obj = _parse_response_json(response.body)
        output = payload_obj.get("output")
        if not isinstance(output, list):
            raise OpenAIError("missing or invalid output field in response")

        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        texts.append(text)

        result = "\n".join(texts).strip()
        if not result:
            raise OpenAIError("no output_text found in response")
        return result


def openai_llm_from_env(*, transport: Transport | None = None) -> OpenAIResponsesLLM:
    config = load_openai_config_from_env()
    return OpenAIResponsesLLM(config, transport=transport)


def openai_llm_for_tmdb_id_from_env(
    *, transport: Transport | None = None
) -> OpenAIResponsesLLM:
    config = load_openai_config_from_env()
    config = replace(config, reasoning_effort="none")
    return OpenAIResponsesLLM(config, transport=transport)
