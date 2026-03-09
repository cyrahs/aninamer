from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from aninamer.artifacts import rename_plan_from_payload
from aninamer.llm_client import ChatMessage
from aninamer.store import RuntimeStore
from aninamer.tmdb_client import SeasonDetails, SeasonSummary, TvDetails
from aninamer.worker import AninamerWorker
from aninamer.webhook_delivery import WebhookResponse


class FakeTMDBClient:
    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = TvDetails(
            id=tv_id,
            name="测试动画",
            original_name=None,
            first_air_date="2020-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
            poster_path="/poster.jpg",
        )
        return details.name, details

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        return SeasonDetails(id=None, season_number=season_number, episodes=[])


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        return self.reply


class CaptureWebhookTransport:
    def __init__(self, response: WebhookResponse | None = None, *, error: Exception | None = None) -> None:
        self.response = response or WebhookResponse(status=200, body=b"ok", headers={})
        self.error = error
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
        if self.error is not None:
            raise self.error
        return self.response


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_worker_scan_creates_planned_job_and_persists_plan_artifact(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )

    worker.scan_once()

    jobs = runtime_store.list_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == "planned"
    assert job.tmdb_id == 123
    assert job.video_moves_count == 1
    assert job.subtitle_moves_count == 1
    assert runtime_store.load_artifact(job.id, "plan") is not None
    assert runtime_store.load_artifact(job.id, "result") is None
    assert runtime_store.load_artifact(job.id, "rollback") is None
    assert runtime_store.list_notifications_after(0) == []


def test_worker_apply_request_writes_result_and_rollback_artifacts(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )

    worker.scan_once()
    job = runtime_store.list_jobs()[0]
    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)

    worker.scan_once()

    updated_job = runtime_store.get_job(job.id)
    updated_request = runtime_store.get_job_request(request.id)
    assert updated_job is not None
    assert updated_request is not None
    assert updated_job.status == "succeeded"
    assert updated_request.status == "succeeded"
    assert runtime_store.load_artifact(updated_job.id, "plan") is not None
    result_payload = runtime_store.load_artifact(updated_job.id, "result")
    rollback_payload = runtime_store.load_artifact(updated_job.id, "rollback")
    assert result_payload is not None
    assert rollback_payload is not None
    assert result_payload["version"] == 1
    assert result_payload["finalize"]["status"] == "deleted"
    rollback_plan = rename_plan_from_payload(rollback_payload)
    assert len(rollback_plan.moves) == 2
    assert not series_dir.exists()
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_apply_succeeded"]
    assert notifications[0].payload == {"finalize_status": "deleted"}
    assert notifications[0].delivery_status == "disabled"


def test_worker_failed_job_is_persisted_for_restart(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")

    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":9,"e2":9,"u":[]}]}'
        ),
    )

    worker.scan_once()

    job = runtime_store.list_jobs()[0]
    assert job.status == "failed"
    assert job.error_stage == "plan"
    assert job.fail_path is not None
    assert Path(job.fail_path).exists()

    reloaded = RuntimeStore(app_config.database.postgres_dsn)
    reloaded_job = reloaded.get_job(job.id)
    assert reloaded_job is not None
    assert reloaded_job.status == "failed"
    assert reloaded_job.error_stage == "plan"
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_plan_failed"]
    assert notifications[0].payload["error_stage"] == "plan"
    assert notifications[0].delivery_status == "disabled"
    assert "fail" not in notifications[0].message.casefold()


def test_worker_rejected_request_creates_notification(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)
    worker = AninamerWorker(app_config, runtime_store)

    worker.scan_once()

    updated_request = runtime_store.get_job_request(request.id)
    assert updated_request is not None
    assert updated_request.status == "rejected"
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_request_rejected"]
    assert notifications[0].job_request_id == request.id
    assert notifications[0].payload == {
        "request_action": "apply_job",
        "job_id": job.id,
    }
    assert notifications[0].delivery_status == "disabled"


def test_worker_request_failure_creates_notification(
    app_config,
    runtime_store: RuntimeStore,
    monkeypatch,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)
    worker = AninamerWorker(app_config, runtime_store)

    def _boom(_job_id: int):
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_store, "get_job", _boom)

    worker.scan_once()

    updated_request = runtime_store.get_job_request(request.id)
    assert updated_request is not None
    assert updated_request.status == "failed"
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_request_failed"]
    assert notifications[0].job_request_id == request.id
    assert notifications[0].delivery_status == "disabled"


def test_worker_recover_emits_apply_failed_notification(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(job.id, status="applying")
    worker = AninamerWorker(app_config, runtime_store)

    worker.recover()

    recovered_job = runtime_store.get_job(job.id)
    assert recovered_job is not None
    assert recovered_job.status == "failed"
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_apply_failed"]
    assert notifications[0].payload["error_stage"] == "apply"
    assert notifications[0].delivery_status == "disabled"


def test_worker_apply_failure_creates_notification(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )

    worker.scan_once()
    job = runtime_store.list_jobs()[0]
    plan_payload = runtime_store.load_artifact(job.id, "plan")
    assert plan_payload is not None
    plan = rename_plan_from_payload(plan_payload)
    plan.moves[0].dst.parent.mkdir(parents=True, exist_ok=True)
    plan.moves[0].dst.write_bytes(b"existing")
    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)

    worker.scan_once()

    updated_job = runtime_store.get_job(job.id)
    updated_request = runtime_store.get_job_request(request.id)
    assert updated_job is not None
    assert updated_request is not None
    assert updated_job.status == "failed"
    assert updated_request.status == "succeeded"
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_apply_failed"]
    assert notifications[0].payload["error_stage"] == "apply"
    assert notifications[0].delivery_status == "disabled"


def test_worker_delivers_webhook_notifications_when_enabled(
    app_config_with_notifications,
    runtime_store: RuntimeStore,
) -> None:
    transport = CaptureWebhookTransport()
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=app_config_with_notifications.watch_roots[0].input_root / "ShowA",
        output_root=app_config_with_notifications.watch_roots[0].output_root,
    )
    runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)
    worker = AninamerWorker(
        app_config_with_notifications,
        runtime_store,
        webhook_transport=transport,
    )

    worker.scan_once()
    worker.scan_once()

    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_request_rejected"]
    delivered = notifications[0]
    assert delivered.delivery_status == "delivered"
    assert delivered.attempt_count == 1
    assert delivered.delivered_at is not None
    assert transport.calls == 1
    assert transport.last_url == "https://notify.example.test/api/v2/notifications/webhook"
    assert transport.last_headers is not None
    assert transport.last_headers["Authorization"] == "Bearer notify-token"
    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body == {
        "markdown": delivered.markdown,
        "image_url": "",
        "disable_web_page_preview": True,
        "disable_notification": False,
    }


def test_worker_retries_failed_webhook_delivery(
    app_config_with_notifications,
    runtime_store: RuntimeStore,
) -> None:
    transport = CaptureWebhookTransport(
        response=WebhookResponse(status=500, body=b"upstream error", headers={})
    )
    series_dir = app_config_with_notifications.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    worker = AninamerWorker(
        app_config_with_notifications,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":9,"e2":9,"u":[]}]}'
        ),
        webhook_transport=transport,
    )

    worker.scan_once()
    worker.scan_once()

    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_plan_failed"]
    retried = notifications[0]
    assert retried.image_url == "https://image.tmdb.org/t/p/original/poster.jpg"
    assert retried.delivery_status == "retry"
    assert retried.attempt_count == 1
    assert retried.last_error is not None
    assert "500" in retried.last_error
    assert retried.next_attempt_at is not None
    body = json.loads((transport.last_body or b"{}").decode("utf-8"))
    assert body["image_url"] == "https://image.tmdb.org/t/p/original/poster.jpg"
