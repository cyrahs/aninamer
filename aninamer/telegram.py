from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Callable
from urllib import error as url_error
from urllib import request

from aninamer.errors import TelegramError


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    base_url: str = "https://api.telegram.org"
    timeout: float = 20.0
    user_agent: str = "aninamer/0.1"


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]


Transport = Callable[[str, bytes, dict[str, str], float], HttpResponse]


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned


def load_telegram_config(
    *,
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> TelegramConfig | None:
    token = _clean_optional(bot_token)
    chat = _clean_optional(chat_id)

    if token is None:
        token = _clean_optional(os.getenv("ANINAMER_TELEGRAM_BOT_TOKEN"))
    if chat is None:
        chat = _clean_optional(os.getenv("ANINAMER_TELEGRAM_CHAT_ID"))

    if token is None and chat is None:
        return None
    if token is None:
        raise ValueError("Telegram bot token is required when chat id is configured")
    if chat is None:
        raise ValueError("Telegram chat id is required when bot token is configured")
    return TelegramConfig(bot_token=token, chat_id=chat)


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
        raise TelegramError(f"transport error: {exc}") from exc


def _parse_response_json(body: bytes) -> dict[str, object]:
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TelegramError(f"invalid JSON response: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TelegramError("invalid JSON response: expected object")
    return payload


def _extract_error_message(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    try:
        payload = _parse_response_json(body)
    except TelegramError:
        return text or "unknown error"
    desc = payload.get("description")
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return text or "unknown error"


def _endpoint_for_method(config: TelegramConfig, method: str) -> str:
    return f"{config.base_url.rstrip('/')}/bot{config.bot_token}/{method}"


class TelegramNotifier:
    def __init__(
        self, config: TelegramConfig, *, transport: Transport | None = None
    ) -> None:
        self._config = config
        self._transport = transport or _default_transport

    def send_message(self, text: str, *, parse_mode: str | None = None) -> None:
        content = text.strip()
        if not content:
            raise TelegramError("message text is empty")

        payload = {
            "chat_id": self._config.chat_id,
            "text": content,
            "disable_web_page_preview": True,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        self._post_json("sendMessage", payload)

    def send_photo(
        self,
        photo_url: str,
        *,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> None:
        photo = photo_url.strip()
        if not photo:
            raise TelegramError("photo url is empty")

        payload: dict[str, object] = {
            "chat_id": self._config.chat_id,
            "photo": photo,
        }
        if caption is not None:
            caption_text = caption.strip()
            if caption_text:
                payload["caption"] = caption_text
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        self._post_json("sendPhoto", payload)

    def _post_json(self, method: str, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        endpoint = _endpoint_for_method(self._config, method)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._config.user_agent,
        }
        response = self._transport(endpoint, body, headers, self._config.timeout)
        if response.status < 200 or response.status >= 300:
            message = _extract_error_message(response.body)
            raise TelegramError(f"Telegram API error {response.status}: {message}")
        payload_obj = _parse_response_json(response.body)
        ok = payload_obj.get("ok")
        if ok is not True:
            message = _extract_error_message(response.body)
            raise TelegramError(f"Telegram API error: {message}")
