from __future__ import annotations

import argparse

import pytest

from aninamer.cli import _telegram_notifier_from_args


def test_telegram_notifier_from_args_returns_none_when_not_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANINAMER_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANINAMER_TELEGRAM_CHAT_ID", raising=False)
    args = argparse.Namespace()

    notifier = _telegram_notifier_from_args(args)

    assert notifier is None


def test_telegram_notifier_from_args_rejects_partial_configuration(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ANINAMER_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANINAMER_TELEGRAM_CHAT_ID", raising=False)
    args = argparse.Namespace(
        telegram_bot_token="abc123",
        telegram_chat_id=None,
    )

    with pytest.raises(ValueError, match="chat id"):
        _telegram_notifier_from_args(args)


def test_telegram_notifier_from_args_accepts_cli_values(monkeypatch) -> None:
    monkeypatch.delenv("ANINAMER_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("ANINAMER_TELEGRAM_CHAT_ID", raising=False)
    args = argparse.Namespace(
        telegram_bot_token="abc123",
        telegram_chat_id="-1001",
    )

    notifier = _telegram_notifier_from_args(args)

    assert notifier is not None
