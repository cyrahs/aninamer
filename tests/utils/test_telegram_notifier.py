from __future__ import annotations

import json

import pytest

from aninamer.errors import TelegramError
from aninamer.telegram import HttpResponse, TelegramConfig, TelegramNotifier, load_telegram_config


def test_load_telegram_config_returns_none_when_not_configured(monkeypatch) -> None:
    monkeypatch.delenv("ANINAMER_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANINAMER_TELEGRAM_CHAT_ID", raising=False)

    assert load_telegram_config() is None


def test_load_telegram_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("ANINAMER_TELEGRAM_BOT_TOKEN", " bot-token ")
    monkeypatch.setenv("ANINAMER_TELEGRAM_CHAT_ID", " 12345 ")

    cfg = load_telegram_config()

    assert cfg is not None
    assert cfg.bot_token == "bot-token"
    assert cfg.chat_id == "12345"


def test_load_telegram_config_rejects_partial(monkeypatch) -> None:
    monkeypatch.setenv("ANINAMER_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.delenv("ANINAMER_TELEGRAM_CHAT_ID", raising=False)

    with pytest.raises(ValueError, match="chat id"):
        load_telegram_config()


def test_telegram_notifier_send_message_success() -> None:
    captured: dict[str, object] = {}

    def _transport(
        url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout
        return HttpResponse(
            status=200,
            body=b'{"ok":true,"result":{"message_id":1}}',
            headers={"content-type": "application/json"},
        )

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="abc123", chat_id="-1001"),
        transport=_transport,
    )
    notifier.send_message("archive summary")

    assert captured["url"] == "https://api.telegram.org/botabc123/sendMessage"
    payload = json.loads(captured["body"].decode("utf-8"))  # type: ignore[index]
    assert payload["chat_id"] == "-1001"
    assert payload["text"] == "archive summary"
    assert payload["disable_web_page_preview"] is True


def test_telegram_notifier_send_message_with_parse_mode() -> None:
    captured: dict[str, object] = {}

    def _transport(
        url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        captured["body"] = body
        return HttpResponse(
            status=200,
            body=b'{"ok":true,"result":{"message_id":1}}',
            headers={"content-type": "application/json"},
        )

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="abc123", chat_id="-1001"),
        transport=_transport,
    )
    notifier.send_message("msg", parse_mode="HTML")

    payload = json.loads(captured["body"].decode("utf-8"))  # type: ignore[index]
    assert payload["parse_mode"] == "HTML"


def test_telegram_notifier_send_photo_success() -> None:
    captured: dict[str, object] = {}

    def _transport(
        url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        captured["url"] = url
        captured["body"] = body
        return HttpResponse(
            status=200,
            body=b'{"ok":true,"result":{"message_id":1}}',
            headers={"content-type": "application/json"},
        )

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="abc123", chat_id="-1001"),
        transport=_transport,
    )
    notifier.send_photo(
        "https://image.tmdb.org/t/p/w500/cover.jpg",
        caption="cap",
        parse_mode="MarkdownV2",
    )

    assert captured["url"] == "https://api.telegram.org/botabc123/sendPhoto"
    payload = json.loads(captured["body"].decode("utf-8"))  # type: ignore[index]
    assert payload["photo"] == "https://image.tmdb.org/t/p/w500/cover.jpg"
    assert payload["caption"] == "cap"
    assert payload["parse_mode"] == "MarkdownV2"


def test_telegram_notifier_http_error_raises() -> None:
    def _transport(
        url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        return HttpResponse(
            status=400,
            body=b'{"ok":false,"description":"chat not found"}',
            headers={"content-type": "application/json"},
        )

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="abc123", chat_id="-1001"),
        transport=_transport,
    )

    with pytest.raises(TelegramError, match="chat not found"):
        notifier.send_message("failed")


def test_telegram_notifier_rejects_ok_false_payload() -> None:
    def _transport(
        url: str, body: bytes, headers: dict[str, str], timeout: float
    ) -> HttpResponse:
        return HttpResponse(
            status=200,
            body=b'{"ok":false,"description":"forbidden"}',
            headers={"content-type": "application/json"},
        )

    notifier = TelegramNotifier(
        TelegramConfig(bot_token="abc123", chat_id="-1001"),
        transport=_transport,
    )

    with pytest.raises(TelegramError, match="forbidden"):
        notifier.send_message("failed")


def test_telegram_notifier_rejects_empty_message() -> None:
    notifier = TelegramNotifier(TelegramConfig(bot_token="abc123", chat_id="-1001"))

    with pytest.raises(TelegramError, match="empty"):
        notifier.send_message("   ")
