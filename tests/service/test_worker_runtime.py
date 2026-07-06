from __future__ import annotations

from dataclasses import replace
import errno
import json
from pathlib import Path
import threading
from typing import Sequence

from aninamer.config import WatchRootConfig, WorkerConfig
from aninamer.llm_client import ChatMessage
from aninamer.monitoring import SeriesDiscoveryResult
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
        self.bodies: list[bytes] = []
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
        self.bodies.append(body)
        self.last_headers = headers
        self.last_timeout = timeout
        if self.error is not None:
            raise self.error
        return self.response


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_worker_discovery_is_isolated_per_watch_root(
    app_config,
    runtime_store: RuntimeStore,
    monkeypatch,
) -> None:
    first_root = app_config.watch_roots[0]
    second_root = WatchRootConfig(
        key="hanime",
        input_root=first_root.input_root.parent / "hanime_in",
        output_root=first_root.output_root.parent / "hanime_out",
    )
    config = replace(app_config, watch_roots=(first_root, second_root))
    discovered = second_root.input_root / "ShowB"

    def fake_discover(input_root: Path) -> SeriesDiscoveryResult:
        if input_root == first_root.input_root:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return SeriesDiscoveryResult(series_dirs=[discovered])

    monkeypatch.setattr("aninamer.worker.discover_series_dirs_status", fake_discover)

    worker = AninamerWorker(config, runtime_store)

    worker.scan_once()

    jobs = runtime_store.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].watch_root_key == "hanime"
    assert jobs[0].series_dir == str(discovered)


def test_worker_run_forever_continues_after_scan_exception(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    shutdown = threading.Event()
    config = replace(
        app_config,
        worker=WorkerConfig(
            settle_seconds=0,
            scan_interval_seconds=0,
            health_stale_after_seconds=1,
        ),
    )

    class FlakyWorker(AninamerWorker):
        def __init__(self) -> None:
            super().__init__(config, runtime_store)
            self.calls = 0

        def recover(self) -> None:
            return None

        def scan_once(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            shutdown.set()

    worker = FlakyWorker()

    worker.run_forever(shutdown.is_set)

    assert worker.calls == 2
    health = worker.health_status()
    assert health.running is False
    assert health.last_error_message is not None
    assert "Transport endpoint is not connected" in health.last_error_message


def test_worker_transient_plan_error_is_retried_not_failed(
    app_config,
    runtime_store: RuntimeStore,
    monkeypatch,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=app_config.watch_roots[0].output_root,
    )

    def broken_build(**_kwargs):  # noqa: ANN202
        raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

    monkeypatch.setattr(
        "aninamer.worker.build_rename_plan_result_for_series",
        broken_build,
    )

    worker = AninamerWorker(app_config, runtime_store)

    worker.scan_once()

    updated = runtime_store.get_job(job.id)
    assert updated is not None
    assert updated.status == "pending"
    assert updated.error_stage is None
    assert updated.fail_path is None
    assert runtime_store.list_notifications_after(0) == []


def test_worker_legacy_apply_is_rejected_before_execute_apply(
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
            '{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}'
        ),
    )
    worker.scan_once()
    job = runtime_store.list_jobs()[0]
    runtime_store.update_job(job.id, status="apply_requested")

    worker.scan_once()

    updated = runtime_store.get_job(job.id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_stage == "apply"
    assert updated.error_message == (
        "legacy apply is disabled until Operation/PathEvent/"
        "DestinationReservation safety chain is implemented"
    )
    assert updated.fail_path is None
    assert series_dir.exists()


def test_worker_scan_creates_planned_job_and_persists_move_items(
    app_config,
    runtime_store: RuntimeStore,
    monkeypatch,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    _write(series_dir / "notes.txt", b"notes")
    _write(series_dir / ".env", b"secret")
    _write(series_dir / ".env.local.mkv", b"secret-video")
    _write(series_dir / ".envrc.mkv", b"secret-video")
    _write(series_dir / ".envlocal", b"secret")
    _write(series_dir / ".env.d" / "ep2.mkv", b"secret-dir-video")
    _write(series_dir / ".envrc.d" / "ep3.mkv", b"secret-dir-video")
    original_stat = Path.stat

    def guarded_stat(self: Path, *args, **kwargs):  # noqa: ANN202
        if self.name.startswith(".env"):
            raise AssertionError(f"stat called for sensitive env file: {self}")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

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
    assert job.series_name == "测试动画"
    assert job.tmdb_id == 123
    assert job.video_moves_count == 1
    assert job.subtitle_moves_count == 1
    move_items = runtime_store.list_job_moves(job.id)
    assert [(item.kind, item.src_id, item.status) for item in move_items] == [
        ("subtitle", 2, "proposed"),
        ("video", 1, "proposed"),
    ]
    assert runtime_store.load_artifact(job.id, "result") is None
    snapshot = runtime_store.get_review_snapshot(job.id)
    assert snapshot is not None
    assert snapshot.selected_tmdb_info["selected_tmdb_id"] == 123
    assert snapshot.selected_tmdb_info["seasons"] == [
        {"season_number": 1, "episode_count": 1}
    ]
    assert [
        (item["kind"], item["final_disposition"])
        for item in snapshot.source_files
    ] == [
        ("subtitle", "planned_output"),
        ("video", "planned_output"),
        ("ignored", "original"),
    ]
    assert snapshot.source_files[2]["ignored_reason"] == "unsupported_extension"
    rel_paths = [
        item["original_path"]["rel_path"]
        for item in snapshot.source_files
    ]
    assert all(not path.startswith(".env") for path in rel_paths)
    assert snapshot.episode_targets[0]["video_source_id"] == 1
    assert snapshot.episode_targets[0]["subtitle_link_ids"] == [1]
    assert snapshot.subtitle_links[0]["subtitle_source_id"] == 2
    assert snapshot.subtitle_links[0]["variant_detected"] == "chs"
    assert snapshot.mapping_attempts[0]["mapped_video_count"] == 1
    assert snapshot.mapping_evidence
    assert runtime_store.list_notifications_after(0) == []


def test_worker_empty_plan_fails_without_archive_when_auto_apply_enabled(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    config = replace(
        app_config,
        worker=replace(app_config.worker, auto_apply=True),
    )
    series_dir = config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")

    worker = AninamerWorker(
        config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM('{"tmdb":123,"eps":[]}'),
    )

    worker.scan_once()

    job = runtime_store.list_jobs()[0]
    assert job.status == "failed"
    assert job.error_stage == "plan"
    assert job.error_message is not None
    assert "rename plan contains no moves" in job.error_message
    assert job.fail_path is not None
    fail_path = Path(job.fail_path)
    assert fail_path.parent.name == "fail"
    assert (fail_path / "ep1.mkv").exists()
    assert not series_dir.exists()
    assert not (
        config.watch_roots[0].input_root / "archive" / "ShowA {tmdb-123}"
    ).exists()

    assert runtime_store.list_job_moves(job.id) == []
    assert runtime_store.get_review_snapshot(job.id) is None
    assert runtime_store.load_artifact(job.id, "result") is None

    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_plan_failed"]
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: 测试动画"
    assert notifications[0].message == "生成计划失败：归档计划为空"


def test_worker_rejects_missing_move_items_at_apply_boundary(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA {tmdb-123}"
    _write(series_dir / "ep1.mkv", b"video")
    job = runtime_store.create_job(
        series_name="测试动画",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=app_config.watch_roots[0].output_root,
    )
    runtime_store.update_job(
        job.id,
        status="apply_requested",
        series_name="测试动画",
        tmdb_id=123,
    )

    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeTMDBClient(),
    )

    worker.scan_once()

    updated = runtime_store.get_job(job.id)
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error_stage == "apply"
    assert updated.error_message is not None
    assert updated.error_message == (
        "legacy apply is disabled until Operation/PathEvent/"
        "DestinationReservation safety chain is implemented"
    )
    assert updated.fail_path is None
    assert series_dir.exists()
    assert not (
        app_config.watch_roots[0].input_root / "archive" / "ShowA {tmdb-123}"
    ).exists()
    assert runtime_store.load_artifact(job.id, "result") is None

    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_apply_failed"]
    assert notifications[0].message == "归档失败：legacy apply is disabled until Operation/PathEvent/DestinationReservation safety chain is implemented"


def test_worker_apply_request_is_rejected_until_safe_apply_chain_exists(
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
    assert updated_job.status == "planned"
    assert updated_job.error_stage is None
    assert updated_job.error_message is None
    assert updated_request.status == "rejected"
    assert updated_request.error_message == (
        "legacy apply is disabled until Operation/PathEvent/"
        "DestinationReservation safety chain is implemented"
    )
    result_payload = runtime_store.load_artifact(updated_job.id, "result")
    assert result_payload is None
    move_items = runtime_store.list_job_moves(updated_job.id)
    assert [item.status for item in move_items] == ["proposed", "proposed"]
    assert all(item.applied_at is None for item in move_items)
    assert series_dir.exists()
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_request_rejected"]
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: 测试动画"
    assert notifications[0].message == "请求被拒绝：legacy apply is disabled until Operation/PathEvent/DestinationReservation safety chain is implemented"
    notification_lines = notifications[0].markdown.splitlines()
    assert notification_lines[:2] == [
        r"*Aninamer: 测试动画*",
        r"请求被拒绝：legacy apply is disabled until Operation/PathEvent/DestinationReservation safety chain is implemented",
    ]
    assert notifications[0].payload == {
        "request_action": "apply_job",
        "job_id": job.id,
        "error_message": (
            "legacy apply is disabled until Operation/PathEvent/"
            "DestinationReservation safety chain is implemented"
        ),
    }
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
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: ShowA {tmdb-123}"
    assert notifications[0].message == "生成计划失败：LLM 映射集数超出 TMDB 范围"
    assert notifications[0].payload["error_stage"] == "plan"
    assert notifications[0].delivery_status == "disabled"
    assert "fail" not in notifications[0].message.casefold()
    llm_failures = runtime_store.list_llm_failures(job.id)
    assert [item.stage for item in llm_failures] == [
        "episode_mapping",
        "episode_mapping",
        "episode_mapping",
    ]
    assert [item.attempt for item in llm_failures] == [1, 2, 3]
    assert all("e1" in item.raw_reply for item in llm_failures)
    assert all("exceeds season 1 count 1" in item.error_message for item in llm_failures)


def test_worker_clears_failed_job_after_fail_path_is_removed(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    series_dir = app_config.watch_roots[0].input_root / "ShowA"
    fail_path = app_config.watch_roots[0].input_root / "fail" / "ShowA"
    fail_path.mkdir(parents=True)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=app_config.watch_roots[0].output_root,
    )
    runtime_store.update_job(
        job.id,
        status="failed",
        error_stage="plan",
        error_message="LLMOutputError: invalid json",
        fail_path=str(fail_path),
    )
    fail_path.rmdir()

    worker = AninamerWorker(app_config, runtime_store)
    worker.scan_once()

    updated = runtime_store.get_job(job.id)
    assert updated is not None
    assert updated.status == "cleared"
    assert updated.fail_path is None
    assert updated.error_stage == "plan"
    assert updated.error_message == "LLMOutputError: invalid json"


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
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: ShowA"
    assert notifications[0].message == "请求被拒绝：任务状态不允许归档"
    assert notifications[0].markdown == "*Aninamer: ShowA*\n请求被拒绝：任务状态不允许归档"
    assert notifications[0].job_request_id == request.id
    assert notifications[0].payload == {
        "request_action": "apply_job",
        "job_id": job.id,
        "error_message": "job is not in planned status",
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
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: 归档任务"
    assert notifications[0].message == "请求失败：boom"
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
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: ShowA"
    assert notifications[0].message == "归档失败：Worker 重启，归档中断"
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
    move_items = runtime_store.list_job_moves(job.id)
    assert move_items
    destination = Path(move_items[0].dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"existing")
    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)

    worker.scan_once()

    updated_job = runtime_store.get_job(job.id)
    updated_request = runtime_store.get_job_request(request.id)
    assert updated_job is not None
    assert updated_request is not None
    assert updated_job.status == "planned"
    assert updated_request.status == "rejected"
    failed_moves = runtime_store.list_job_moves(job.id)
    assert [item.status for item in failed_moves] == ["proposed", "proposed"]
    notifications = runtime_store.list_notifications_after(0)
    assert [item.event_kind for item in notifications] == ["job_request_rejected"]
    assert notifications[0].severity == "error"
    assert notifications[0].title == "Aninamer: 测试动画"
    assert notifications[0].message == "请求被拒绝：legacy apply is disabled until Operation/PathEvent/DestinationReservation safety chain is implemented"
    assert notifications[0].payload["request_action"] == "apply_job"
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
        "pin": True,
    }


def test_worker_does_not_pin_non_error_webhook_notifications(
    app_config_with_notifications,
    runtime_store: RuntimeStore,
) -> None:
    transport = CaptureWebhookTransport()
    runtime_store.create_notification(
        event_kind="job_apply_succeeded",
        severity="success",
        title="处理完成",
        message="ShowA 已处理完成",
        markdown="# 处理完成",
    )
    runtime_store.create_notification(
        event_kind="job_apply_succeeded",
        severity="warning",
        title="处理完成",
        message="ShowB 已处理完成",
        markdown="# 处理完成",
    )
    worker = AninamerWorker(
        app_config_with_notifications,
        runtime_store,
        webhook_transport=transport,
    )

    worker.scan_once()

    assert transport.calls == 2
    bodies = [json.loads(body.decode("utf-8")) for body in transport.bodies]
    assert [body["pin"] for body in bodies] == [False, False]


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
    assert body["pin"] is True
