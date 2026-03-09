from __future__ import annotations

import json

from aninamer.config import NotificationConfig
from aninamer.webhook_delivery import WebhookResponse, send_notification_webhook


class CaptureTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.last_url: str | None = None
        self.last_body: bytes | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_timeout: float | None = None

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout: float,
    ) -> WebhookResponse:
        self.calls += 1
        self.last_url = url
        self.last_body = body
        self.last_headers = headers
        self.last_timeout = timeout
        return WebhookResponse(status=200, body=b"ok", headers={})


def test_send_notification_webhook_uses_exact_contract() -> None:
    transport = CaptureTransport()
    config = NotificationConfig(
        base_url="https://notify.example.test/",
        bearer_token="notify-token",
        timeout_seconds=12.0,
    )

    response = send_notification_webhook(
        config,
        markdown="# 处理完成",
        disable_web_page_preview=True,
        disable_notification=False,
        transport=transport,
    )

    assert response.status == 200
    assert transport.calls == 1
    assert transport.last_url == "https://notify.example.test/api/v2/notifications/webhook"
    assert transport.last_headers is not None
    assert transport.last_headers["Authorization"] == "Bearer notify-token"
    assert transport.last_headers["Content-Type"] == "application/json"
    assert transport.last_timeout == 12.0
    assert json.loads((transport.last_body or b"{}").decode("utf-8")) == {
        "markdown": "# 处理完成",
        "disable_web_page_preview": True,
        "disable_notification": False,
    }
