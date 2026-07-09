from __future__ import annotations

from pathlib import Path

from starlette.responses import Response

from aninamer.api import create_app
from aninamer.config import (
    ApiConfig,
    AppConfig,
    DatabaseConfig,
    OpenAISettings,
    TmdbConfig,
    WatchRootConfig,
    WorkerConfig,
)
from aninamer.worker import WorkerHealthStatus, WorkerLivenessStatus


class DummyWorker:
    def health_status(self) -> WorkerHealthStatus:
        return WorkerHealthStatus(
            healthy=False,
            reason="worker scan is stale",
            running=True,
            started_at="2026-04-13T07:39:01+00:00",
            stopped_at=None,
            current_scan_started_at=None,
            last_scan_at="2026-04-13T07:40:01+00:00",
            last_success_at=None,
            last_error_at="2026-04-13T07:49:01+00:00",
            last_error_message="/mnt/cd2/115/emby_in/anime",
            consecutive_failures=3,
            stale_after_seconds=300,
            unavailable_watch_root_keys=("anime",),
        )

    def liveness_status(self) -> WorkerLivenessStatus:
        return WorkerLivenessStatus(
            live=False,
            reason="storage unavailable",
            running=True,
            unavailable_watch_root_keys=("anime", "hanime"),
        )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        database=DatabaseConfig(postgres_dsn="postgresql://unused"),
        tmdb=TmdbConfig(api_key="tmdb-key"),
        openai=OpenAISettings(api_key="openai-key", model="gpt-test"),
        notifications=None,
        notifications_warning=None,
        api=ApiConfig(token="secret"),
        worker=WorkerConfig(),
        watch_roots=(
            WatchRootConfig(
                key="anime",
                input_root=tmp_path / "input",
                output_root=tmp_path / "output",
            ),
        ),
    )


def test_public_healthz_unhealthy_payload_omits_raw_worker_error(tmp_path: Path) -> None:
    app = create_app(  # type: ignore[arg-type]
        _config(tmp_path),
        store=object(),
        worker=DummyWorker(),
    )

    route = next(route for route in app.routes if getattr(route, "path", None) == "/healthz")
    response = Response()
    payload_model = route.endpoint(response)
    payload = payload_model.model_dump(mode="json", exclude_none=True)

    assert response.status_code == 503
    assert payload == {
        "status": "unhealthy",
        "reason": "worker scan is stale",
        "worker": {
            "running": True,
            "last_scan_at": "2026-04-13T07:40:01+00:00",
            "consecutive_failures": 3,
            "stale_after_seconds": 300,
            "unavailable_watch_root_count": 1,
        },
    }
    assert "last_error_message" not in str(payload)
    assert "/mnt/cd2" not in str(payload)


def test_healthz_openapi_documents_unhealthy_response(tmp_path: Path) -> None:
    app = create_app(  # type: ignore[arg-type]
        _config(tmp_path),
        store=object(),
        worker=DummyWorker(),
    )

    schema = app.openapi()

    health_responses = schema["paths"]["/healthz"]["get"]["responses"]
    assert "503" in health_responses
    assert (
        health_responses["503"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/HealthResponse"
    )


def test_public_livez_reports_storage_unavailable_without_raw_paths(
    tmp_path: Path,
) -> None:
    app = create_app(  # type: ignore[arg-type]
        _config(tmp_path),
        store=object(),
        worker=DummyWorker(),
    )

    route = next(route for route in app.routes if getattr(route, "path", None) == "/livez")
    response = Response()
    payload_model = route.endpoint(response)
    payload = payload_model.model_dump(mode="json", exclude_none=True)

    assert response.status_code == 503
    assert payload == {
        "status": "unhealthy",
        "reason": "storage unavailable",
        "worker_running": True,
        "unavailable_watch_root_count": 2,
    }
    assert "/mnt/cd2" not in str(payload)


def test_livez_openapi_documents_unhealthy_response(tmp_path: Path) -> None:
    app = create_app(  # type: ignore[arg-type]
        _config(tmp_path),
        store=object(),
        worker=DummyWorker(),
    )

    schema = app.openapi()

    live_responses = schema["paths"]["/livez"]["get"]["responses"]
    assert "503" in live_responses
    assert (
        live_responses["503"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/LivenessResponse"
    )
