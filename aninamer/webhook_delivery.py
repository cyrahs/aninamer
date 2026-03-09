from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable
from urllib import error as url_error
from urllib import request

from aninamer.config import NotificationConfig
from aninamer.errors import NotificationDeliveryError


@dataclass(frozen=True)
class WebhookResponse:
    status: int
    body: bytes
    headers: dict[str, str]


WebhookTransport = Callable[[str, bytes, dict[str, str], float], WebhookResponse]


def webhook_url(config: NotificationConfig) -> str:
    return f"{config.base_url.rstrip('/')}/api/v2/notifications/webhook"


def default_webhook_transport(
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: float,
) -> WebhookResponse:
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
            resp_headers = {key: value for key, value in resp.headers.items()}
            return WebhookResponse(
                status=resp.status,
                body=resp_body,
                headers=resp_headers,
            )
    except url_error.HTTPError as exc:
        resp_body = exc.read()
        resp_headers = {key: value for key, value in exc.headers.items()}
        return WebhookResponse(status=exc.code, body=resp_body, headers=resp_headers)
    except Exception as exc:
        raise NotificationDeliveryError(f"webhook transport error: {exc}") from exc


def send_notification_webhook(
    config: NotificationConfig,
    *,
    markdown: str,
    disable_web_page_preview: bool,
    disable_notification: bool,
    transport: WebhookTransport | None = None,
) -> WebhookResponse:
    payload = {
        "markdown": markdown,
        "disable_web_page_preview": disable_web_page_preview,
        "disable_notification": disable_notification,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.bearer_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "aninamer/0.1",
    }
    sender = transport or default_webhook_transport
    return sender(webhook_url(config), body, headers, config.timeout_seconds)


def response_error_text(response: WebhookResponse) -> str:
    text = response.body.decode("utf-8", errors="replace").strip()
    if text:
        return text
    return f"HTTP {response.status}"
