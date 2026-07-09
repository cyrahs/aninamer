from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time
from typing import Sequence

from fastapi.testclient import TestClient

from aninamer.api import create_app
from aninamer.config import WorkerConfig
from aninamer.llm_client import ChatMessage
from aninamer.plan import PlannedMove
from aninamer.store import RuntimeStore
from aninamer.tmdb_client import Episode, SeasonDetails, SeasonSummary, TvDetails
from aninamer.worker import AninamerWorker


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret"}


def _write(path, data: bytes) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class FakeRematchTMDBClient:
    def resolve_series_title(
        self,
        tv_id: int,
        *,
        country_codes: tuple[str, ...] = (),
    ) -> tuple[str, TvDetails]:
        title = "新动画" if tv_id == 456 else "旧动画"
        details = TvDetails(
            id=tv_id,
            name=title,
            original_name=f"Original {tv_id}",
            first_air_date="2021-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )
        return title, details

    def get_season(
        self,
        tv_id: int,
        season_number: int,
        *,
        language: str = "zh-CN",
    ) -> SeasonDetails:
        name = "开始" if language == "zh-CN" else "Start"
        return SeasonDetails(
            id=tv_id * 10 + season_number,
            season_number=season_number,
            episodes=[Episode(episode_number=1, name=name, overview="", air_date=None)],
        )


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.messages: list[ChatMessage] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.messages = list(messages)
        return self.reply


def _seed_rematch_job(runtime_store: RuntimeStore, tmp_path: Path):  # noqa: ANN202
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    job = runtime_store.create_job(
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(
        job.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
    )
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.mkv",
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=series_dir / "ep1.ass",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.chs.ass",
                kind="subtitle",
                src_id=2,
            ),
        ),
    )
    return job, series_dir, output_root, moves


def _seed_reopen_job(runtime_store: RuntimeStore, tmp_path: Path):  # noqa: ANN202
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    review_file = tmp_path / "input" / "review" / "job-1" / "ShowA S01E01.mkv"
    _write(review_file, b"video")
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(job.id, status="succeeded", tmdb_id=123)
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root
                / "ShowA (2020) {tmdb-123}"
                / "S01"
                / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.update_job_move(
        moves[0].id,
        status="withdrawn",
        applied_at="2026-07-06T00:00:03+00:00",
    )
    withdraw_operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="withdraw",
        idempotency_key="withdraw-seed",
        request_payload_hash="hash-withdraw-seed",
        status="succeeded",
    )
    runtime_store.create_operation_item_result(
        operation_id=withdraw_operation.id,
        subject_kind="move_item",
        subject_id=moves[0].id,
        status="succeeded",
        from_path={"display_path": str(moves[0].dst)},
        to_path={"display_path": str(review_file)},
        identity_check="matched",
    )
    return job, runtime_store.list_job_moves(job.id)[0], review_file


def _snapshot_path_ref(path, root_kind: str, root) -> dict[str, object]:  # noqa: ANN001
    normalized = str(path.resolve(strict=False))
    return {
        "root_kind": root_kind,
        "root_id": f"{root_kind}:default:1",
        "rel_path": path.relative_to(root).as_posix(),
        "display_path": str(path),
        "normalized_abs_path": normalized,
        "realpath": normalized,
        "casefold_key": normalized.casefold(),
        "redacted": False,
    }


def _snapshot_source_file(
    source_id: int,
    kind: str,
    path,
    series_dir,
    *,
    final_disposition: str,
    ignored_reason: str | None = None,
) -> dict[str, object]:  # noqa: ANN001
    try:
        stat_result = path.stat()
        size_bytes: int | None = stat_result.st_size
        mtime_ns: int | None = stat_result.st_mtime_ns
    except OSError:
        size_bytes = None
        mtime_ns = None
    return {
        "id": source_id,
        "kind": kind,
        "original_path": _snapshot_path_ref(path, "source", series_dir),
        "current_path": _snapshot_path_ref(path, "source", series_dir),
        "current_path_kind": "source",
        "ext": path.suffix,
        "file_identity": {
            "identity_strength": "weak" if size_bytes is not None else "missing",
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "content_hash": None,
        },
        "detected_subtitle_variant": "unknown" if kind == "subtitle" else None,
        "final_disposition": final_disposition,
        "ignored_reason": ignored_reason,
    }


def test_api_requires_bearer_token_except_healthz(app_config, runtime_store: RuntimeStore) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 503
        assert health.json()["status"] == "unhealthy"

        missing = client.get("/api/v1/jobs")
        assert missing.status_code == 401

        invalid = client.get(
            "/api/v1/jobs",
            headers={"Authorization": "Bearer wrong"},
        )
        assert invalid.status_code == 403


def test_ui_static_assets_are_served_without_api_token(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)

    with TestClient(app) as client:
        index_response = client.get("/ui")
        script_response = client.get("/ui/app.js")

    assert index_response.status_code == 200
    assert "aninamer review" in index_response.text
    assert script_response.status_code == 200
    assert "loadRuntimeAndJobs" in script_response.text


def test_healthz_reports_ok_when_worker_thread_is_running(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)
    shutdown = threading.Event()
    thread = threading.Thread(
        target=worker.run_forever,
        args=(shutdown.is_set,),
        daemon=True,
    )
    thread.start()
    try:
        for _ in range(50):
            if worker.health_status().running and runtime_store.snapshot().last_scan_at:
                break
            time.sleep(0.02)

        with TestClient(app) as client:
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
    finally:
        shutdown.set()
        thread.join(timeout=2.0)


def test_healthz_reports_unhealthy_when_worker_scan_is_stale(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    config = replace(
        app_config,
        worker=WorkerConfig(
            settle_seconds=0,
            scan_interval_seconds=1,
            health_stale_after_seconds=1,
        ),
    )
    worker = AninamerWorker(config, runtime_store)
    app = create_app(config, store=runtime_store, worker=worker)
    old_timestamp = "2026-04-13T07:40:01+00:00"
    runtime_store.set_last_scan_at(old_timestamp)
    worker._mark_worker_started()
    with worker._state_lock:
        worker._started_at = "2026-04-13T07:39:01+00:00"

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 503
        payload = health.json()
        assert payload["status"] == "unhealthy"
        assert payload["reason"] == "worker scan is stale"
        assert payload["worker"]["last_scan_at"] == old_timestamp


def test_jobs_endpoint_hides_internal_artifacts_and_status_endpoint_aggregates(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)

    job_a = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(
        job_a.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
    )

    job_b = runtime_store.create_job(
        series_name="ShowB",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowB",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(
        job_b.id,
        status="failed",
        error_stage="plan",
        error_message="LLMOutputError: invalid json",
        fail_path="/internal/fail-b",
    )

    job_c = runtime_store.create_job(
        series_name="ShowC",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowC",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(
        job_c.id,
        status="succeeded",
    )
    runtime_store.save_artifact(job_c.id, "result", {"version": 1})

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        jobs_response = client.get("/api/v1/jobs", headers=_auth_headers())
        assert jobs_response.status_code == 200
        payload = jobs_response.json()
        assert set(payload.keys()) == {"items", "total"}
        assert payload["total"] == 3
        first = payload["items"][0]
        assert set(first.keys()) == {
            "id",
            "series_name",
            "watch_root_key",
            "source_kind",
            "status",
            "tmdb_id",
            "video_moves_count",
            "subtitle_moves_count",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "error_stage",
            "error_message",
        }
        assert "fail_path" not in first
        assert "archive_path" not in first

        job_response = client.get(f"/api/v1/jobs/{job_a.id}", headers=_auth_headers())
        assert job_response.status_code == 200
        assert "plan_path" not in job_response.json()
        assert "result_path" not in job_response.json()
        assert "rollback_path" not in job_response.json()

        status_response = client.get("/api/v1/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert set(status_payload.keys()) == {"summary", "pending_items", "failed_items"}
        assert status_payload["summary"] == {
            "pending_count": 0,
            "planning_count": 0,
            "planned_count": 1,
            "apply_requested_count": 0,
            "applying_count": 0,
            "failed_count": 1,
        }
        assert [item["series_name"] for item in status_payload["pending_items"]] == [
            "ShowA"
        ]
        assert [item["series_name"] for item in status_payload["failed_items"]] == [
            "ShowB"
        ]
        assert status_payload["failed_items"][0]["error_stage"] == "plan"
        assert status_payload["failed_items"][0]["error_message"] == (
            "LLMOutputError: invalid json"
        )


def test_status_endpoint_ignores_cleared_jobs(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(
        job.id,
        status="cleared",
        error_stage="plan",
        error_message="LLMOutputError: invalid json",
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        jobs_response = client.get("/api/v1/jobs", headers=_auth_headers())
        assert jobs_response.status_code == 200
        assert jobs_response.json()["items"][0]["status"] == "cleared"

        status_response = client.get("/api/v1/status", headers=_auth_headers())
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["summary"] == {
            "pending_count": 0,
            "planning_count": 0,
            "planned_count": 0,
            "apply_requested_count": 0,
            "applying_count": 0,
            "failed_count": 0,
        }
        assert status_payload["pending_items"] == []
        assert status_payload["failed_items"] == []


def test_job_requests_round_trip_through_api(app_config, runtime_store: RuntimeStore, tmp_path) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(job.id, status="planned")

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        create_response = client.post(
            "/api/v1/job-requests",
            headers=_auth_headers(),
            json={"action": "apply_job", "job_id": job.id},
        )
        assert create_response.status_code == 202
        created = create_response.json()
        assert created["action"] == "apply_job"
        assert created["job_id"] == job.id
        assert created["status"] == "rejected"
        assert created["error_message"] == (
            "legacy apply is disabled until Operation/PathEvent/"
            "DestinationReservation safety chain is implemented"
        )

        request_id = created["id"]
        read_response = client.get(
            f"/api/v1/job-requests/{request_id}",
            headers=_auth_headers(),
        )
        assert read_response.status_code == 200
        assert read_response.json()["id"] == request_id


def test_apply_job_request_for_missing_job_returns_404(
    app_config,
    runtime_store: RuntimeStore,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/job-requests",
            headers=_auth_headers(),
            json={"action": "apply_job", "job_id": 999_999},
        )

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "job not found"


def test_review_payload_does_not_eagerly_check_rollback_eligibility(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.update_job_move(moves[0].id, status="moved")

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rollback_capability_hints"]["has_moved_items"] is True
    assert payload["source_files"]
    assert payload["episode_targets"]
    assert payload["episode_targets"][0]["season"] == 1
    assert payload["episode_targets"][0]["episode_start"] == 1
    assert payload["episode_targets"][0]["episode_end"] == 1
    assert payload["episode_targets"][0]["subtitle_link_ids"] == []
    assert "selected_tmdb_id" in payload["selected_tmdb_info"]
    assert payload["last_rollback_check_summary"] == {
        "status": "unchecked",
        "last_checked_at": None,
        "mode": None,
    }
    assert "check_token" not in response.text
    assert payload["move_items"][0]["current_path"]["root_kind"] == "output"


def test_rollback_eligibility_is_explicit_idempotent_and_not_persisted_in_review(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.update_job_move(moves[0].id, status="moved")

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = initial_review.json()["versions"]
        request_payload = {
            "mode": "withdraw_to_review",
            "scope": {"kind": "all"},
            "base_review_revision": versions["review_revision"],
            "base_move_state_revision": versions["move_state_revision"],
        }
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers=_auth_headers(),
            json=request_payload,
        )
        assert missing_key.status_code == 422

        first = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "rollback-1"},
            json=request_payload,
        )
        runtime_store.update_job_move(moves[0].id, status="failed")
        second = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "rollback-1"},
            json=request_payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "rollback-1"},
            json={**request_payload, "mode": "restore_moved_to_source"},
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "rollback-stale"},
            json=request_payload,
        )
        review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )
        safety = client.get(
            f"/api/v1/jobs/{job.id}/safety",
            headers=_auth_headers(),
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert first.json()["check_token"] is None
    assert second.json()["check_token"] is None
    assert first.json()["token_status"] == "not_current"
    assert first.json()["status"] == "ineligible"
    assert first.json()["affected_move_items"] == [moves[0].id]
    assert f"review/{job.id}-" in first.text
    assert first.json()["identity_mismatches"] == [
        {
            "move_item_id": moves[0].id,
            "reason": "strong_file_identity_not_recorded",
        }
    ]
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"
    assert "current_versions" in stale.json()["error"]["details"]

    review_payload = review.json()
    assert review_payload["last_rollback_check_summary"]["status"] == "checked_no_token"
    assert "check_token" not in review.text

    operations_payload = operations.json()
    assert operations_payload["total"] == 1
    assert operations_payload["items"][0]["type"] == "rollback_eligibility"
    assert operations_payload["items"][0]["result_summary"]["token_persisted"] is False
    assert operations_payload["items"][0]["result_summary"]["eligible_move_item_count"] == 0
    assert "rollback-check" not in operations.text

    safety_payload = safety.json()
    assert safety_payload["review_exclusion_proof"]["status"] == "unknown"
    assert safety_payload["review_exclusion_proof"]["provisional"] is True
    assert safety_payload["safety_token_summary"] == {
        "review_safety_check_token": "not_returned_by_get",
        "source_gate_check_token": "not_returned_by_get",
    }
    assert "rollback-check" not in safety.text


def test_rollback_eligibility_rejects_invalid_move_item_selection(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = review.json()["versions"]
        response = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "bad-move-selection"},
            json={
                "mode": "withdraw_to_review",
                "scope": {"kind": "move_items", "ids": [moves[0].id, 999_999]},
                "base_review_revision": versions["review_revision"],
                "base_move_state_revision": versions["move_state_revision"],
            },
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_rollback_selection"
    assert payload["error"]["details"]["unknown_move_item_ids"] == [999_999]
    assert payload["error"]["details"]["not_moved_move_item_ids"] == [moves[0].id]


def test_rematch_preview_with_explicit_tmdb_id_is_idempotent_and_does_not_replace_moves(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    mapping_llm = FakeLLM(
        '{"tmdb":456,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
    )
    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeRematchTMDBClient(),
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    job = runtime_store.create_job(
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(
        job.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
    )
    existing_moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.mkv",
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=series_dir / "ep1.ass",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.chs.ass",
                kind="subtitle",
                src_id=2,
            ),
        ),
    )
    runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            _snapshot_source_file(
                1,
                "video",
                series_dir / "ep1.mkv",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                2,
                "subtitle",
                series_dir / "ep1.ass",
                series_dir,
                final_disposition="planned_output",
            ),
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "旧动画",
            "original_name": "Old Anime",
            "year": 2020,
            "candidates": [],
            "seasons": [{"season_number": 1, "episode_count": 1}],
            "specials": [],
        },
        episode_targets=[
            {
                "id": 1,
                "video_source_id": 1,
                "season": 1,
                "episode_start": 1,
                "episode_end": 1,
                "status": "proposed",
                "manual": False,
                "created_by": "llm_mapping",
                "move_item_ids": [existing_moves[0].id],
                "subtitle_link_ids": [1],
            }
        ],
        subtitle_links=[
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": existing_moves[1].id,
            }
        ],
        mapping_attempts=[
            {
                "id": 1,
                "stage": "episode_mapping",
                "created_by": "worker",
                "tmdb_id": 123,
                "status": "succeeded",
            }
        ],
        mapping_evidence=[
            {
                "id": 1,
                "episode_target_id": 1,
                "source_file_id": 1,
                "evidence_kind": "llm_episode_mapping",
                "summary": "Existing snapshot evidence.",
                "confidence": "unknown",
            }
        ],
        path_event_summary={"status": "not_started", "pending_count": 0},
        validation_summary={"status": "planned"},
    )
    existing_snapshot = runtime_store.get_review_snapshot(job.id)
    assert existing_snapshot is not None

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = review.json()["versions"]
        request_payload = {
            "tmdb_id": 456,
            "base_review_revision": versions["review_revision"],
            "additional_instructions": "  集数整体偏移 +1\n只匹配正片  ",
        }

        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers=_auth_headers(),
            json=request_payload,
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-456"},
            json=request_payload,
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-456"},
            json=request_payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-456"},
            json={**request_payload, "tmdb_id": 789},
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-stale"},
            json=request_payload,
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )

    assert missing_key.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    payload = first.json()
    assert payload["status"] == "preview_ready"
    assert payload["tmdb_id"] == 456
    assert payload["base_review_revision"] == versions["review_revision"]
    assert payload["additional_instructions_present"] is True
    assert payload["additional_instructions_length"] == len("集数整体偏移 +1\n只匹配正片")
    assert isinstance(payload["additional_instructions_digest"], str)
    assert "集数整体偏移" not in first.text
    assert payload["selected_tmdb_info"]["selected_tmdb_id"] == 456
    assert payload["selected_tmdb_info"]["title_zh_cn"] == "新动画"
    assert payload["episode_targets"][0]["video_source_id"] == 1
    assert payload["subtitle_links"][0]["subtitle_source_id"] == 2
    assert payload["diff_summary"] == {
        "added": 0,
        "changed": 2,
        "removed": 0,
        "unchanged": 0,
        "total": 2,
    }
    assert {item["change_type"] for item in payload["move_items"]} == {"changed"}
    assert "新动画 (2021) {tmdb-456}" in first.text
    assert "旧动画 (2020) {tmdb-123}" in first.text
    assert payload["validation_summary"]["status"] == "preview_only"
    assert payload["mapping_evidence"]
    assert "user rematch instructions:" in mapping_llm.messages[1].content
    assert "集数整体偏移 +1 只匹配正片" in mapping_llm.messages[1].content
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"

    assert runtime_store.list_job_moves(job.id) == existing_moves
    assert runtime_store.get_review_snapshot(job.id) == existing_snapshot
    assert (series_dir / "ep1.mkv").exists()
    assert (series_dir / "ep1.ass").exists()
    assert not (
        output_root
        / "新动画 (2021) {tmdb-456}"
        / "S01"
        / "新动画 S01E01.mkv"
    ).exists()
    operations_payload = operations.json()
    assert operations_payload["total"] == 1
    operation = operations_payload["items"][0]
    assert operation["type"] == "rematch_preview"
    assert operation["status"] == "succeeded"
    assert operation["result_summary"]["tmdb_id"] == 456
    assert operation["result_summary"]["status"] == "preview_ready"
    assert operation["result_summary"]["additional_instructions_present"] is True
    assert operation["result_summary"]["additional_instructions_digest"] == payload[
        "additional_instructions_digest"
    ]
    assert operation["result_summary"]["response_payload"] == payload


def test_rematch_commit_replaces_plan_rows_and_snapshot_without_moving_files(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeRematchTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":456,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )
    job = runtime_store.create_job(
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(
        job.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
    )
    old_moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.mkv",
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=series_dir / "ep1.ass",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.chs.ass",
                kind="subtitle",
                src_id=2,
            ),
        ),
    )
    runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            _snapshot_source_file(
                1,
                "video",
                series_dir / "ep1.mkv",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                2,
                "subtitle",
                series_dir / "ep1.ass",
                series_dir,
                final_disposition="planned_output",
            ),
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "旧动画",
            "original_name": "Old Anime",
            "year": 2020,
            "candidates": [],
            "seasons": [{"season_number": 1, "episode_count": 1}],
            "specials": [],
        },
        episode_targets=[
            {
                "id": 1,
                "video_source_id": 1,
                "season": 1,
                "episode_start": 1,
                "episode_end": 1,
                "status": "proposed",
                "manual": False,
                "created_by": "llm_mapping",
                "move_item_ids": [old_moves[0].id],
                "subtitle_link_ids": [1],
            }
        ],
        subtitle_links=[
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": old_moves[1].id,
            }
        ],
        mapping_attempts=[{"id": 1, "stage": "episode_mapping", "status": "succeeded"}],
        mapping_evidence=[
            {
                "id": 1,
                "episode_target_id": 1,
                "source_file_id": 1,
                "evidence_kind": "llm_episode_mapping",
                "summary": "Existing snapshot evidence.",
                "confidence": "unknown",
            }
        ],
        path_event_summary={"status": "not_started", "pending_count": 0},
        validation_summary={"status": "planned"},
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        preview = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-preview-commit"},
            json={
                "tmdb_id": 456,
                "base_review_revision": initial_review.json()["review_revision"],
                "additional_instructions": "优先按文件名标题匹配",
            },
        )
        review_after_preview = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        commit_payload = {
            "preview_operation_id": preview.json()["operation_id"],
            "tmdb_id": 456,
            "base_review_revision": review_after_preview.json()["review_revision"],
        }
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers=_auth_headers(),
            json=commit_payload,
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-commit-1"},
            json=commit_payload,
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-commit-1"},
            json=commit_payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-commit-1"},
            json={**commit_payload, "tmdb_id": 789},
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "rematch-commit-stale"},
            json=commit_payload,
        )
        review_after_commit = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )

    assert preview.status_code == 200
    assert missing_key.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert first.json()["status"] == "committed"
    assert first.json()["preview_operation_id"] == preview.json()["operation_id"]
    assert first.json()["tmdb_id"] == 456
    assert first.json()["committed_move_item_count"] == 2
    assert first.json()["additional_instructions_present"] is True
    assert first.json()["additional_instructions_digest"] == preview.json()[
        "additional_instructions_digest"
    ]
    assert first.json()["additional_instructions_length"] == len("优先按文件名标题匹配")
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"

    updated_job = runtime_store.get_job(job.id)
    assert updated_job is not None
    assert updated_job.status == "planned"
    assert updated_job.series_name == "新动画"
    assert updated_job.tmdb_id == 456
    assert updated_job.video_moves_count == 1
    assert updated_job.subtitle_moves_count == 1
    committed_moves = runtime_store.list_job_moves(job.id)
    assert [move.status for move in committed_moves] == ["proposed", "proposed"]
    assert [move.src_id for move in committed_moves] == [2, 1]
    assert all("新动画 (2021) {tmdb-456}" in move.dst for move in committed_moves)
    assert [move.id for move in committed_moves] != [move.id for move in old_moves]
    assert (series_dir / "ep1.mkv").exists()
    assert (series_dir / "ep1.ass").exists()
    assert not (
        output_root
        / "新动画 (2021) {tmdb-456}"
        / "S01"
        / "新动画 S01E01.mkv"
    ).exists()

    snapshot = runtime_store.get_review_snapshot(job.id)
    assert snapshot is not None
    assert snapshot.selected_tmdb_info["selected_tmdb_id"] == 456
    assert snapshot.selected_tmdb_info["title_zh_cn"] == "新动画"
    assert snapshot.episode_targets[0]["created_by"] == "rematch"
    assert snapshot.episode_targets[0]["move_item_ids"] == [
        next(move.id for move in committed_moves if move.kind == "video")
    ]
    assert snapshot.subtitle_links[0]["created_by"] == "rematch"
    assert snapshot.subtitle_links[0]["move_item_id"] == next(
        move.id for move in committed_moves if move.kind == "subtitle"
    )
    assert snapshot.mapping_attempts[-1]["stage"] == "rematch"
    assert snapshot.mapping_attempts[-1]["mapped_video_count"] == 1
    assert snapshot.mapping_attempts[-1]["mapped_subtitle_count"] == 1
    assert snapshot.mapping_attempts[-1]["additional_instructions_present"] is True
    assert snapshot.mapping_attempts[-1]["additional_instructions_digest"] == preview.json()[
        "additional_instructions_digest"
    ]
    assert snapshot.mapping_evidence[-1]["evidence_kind"].startswith("committed_")
    assert snapshot.validation_summary["status"] == "planned"
    assert snapshot.validation_summary["rematched_from_operation_id"] == preview.json()[
        "operation_id"
    ]

    committed_review = review_after_commit.json()
    assert committed_review["selected_tmdb_info"]["selected_tmdb_id"] == 456
    assert committed_review["move_items"][0]["status"] == "proposed"
    assert committed_review["review_revision"] != review_after_preview.json()[
        "review_revision"
    ]
    operations_payload = operations.json()
    assert operations_payload["total"] == 2
    assert [item["type"] for item in operations_payload["items"]] == [
        "rematch_commit",
        "rematch_preview",
    ]


def test_corrections_commit_replaces_plan_rows_snapshot_and_rejects_invalid_states(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(
        job.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
    )
    old_moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.mkv",
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=series_dir / "ep1.ass",
                dst=output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.chs.ass",
                kind="subtitle",
                src_id=2,
            ),
        ),
    )
    runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            _snapshot_source_file(
                1,
                "video",
                series_dir / "ep1.mkv",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                2,
                "subtitle",
                series_dir / "ep1.ass",
                series_dir,
                final_disposition="planned_output",
            ),
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "旧动画",
            "original_name": "Old Anime",
            "year": 2020,
            "candidates": [],
            "seasons": [{"season_number": 1, "episode_count": 12}],
            "specials": [],
        },
        episode_targets=[
            {
                "id": 1,
                "video_source_id": 1,
                "season": 1,
                "episode_start": 1,
                "episode_end": 1,
                "status": "proposed",
                "manual": False,
                "created_by": "llm_mapping",
                "move_item_ids": [old_moves[0].id],
                "subtitle_link_ids": [1],
            }
        ],
        subtitle_links=[
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": old_moves[1].id,
            }
        ],
        mapping_attempts=[{"id": 1, "stage": "episode_mapping", "status": "succeeded"}],
        mapping_evidence=[
            {
                "id": 1,
                "episode_target_id": 1,
                "source_file_id": 1,
                "evidence_kind": "llm_episode_mapping",
                "summary": "Existing snapshot evidence.",
                "confidence": "unknown",
            }
        ],
        path_event_summary={"status": "not_started", "pending_count": 0},
        validation_summary={"status": "planned"},
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        correction_payload = {
            "base_review_revision": initial_review.json()["review_revision"],
            "atomic": True,
            "ops": [
                {
                    "op": "update_episode",
                    "episode_target_id": 1,
                    "season": 1,
                    "episode_start": 2,
                    "episode_end": 2,
                },
                {
                    "op": "override_subtitle_variant",
                    "subtitle_link_id": 1,
                    "variant": "cht",
                },
            ],
        }
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers=_auth_headers(),
            json=correction_payload,
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers={**_auth_headers(), "Idempotency-Key": "correction-1"},
            json=correction_payload,
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers={**_auth_headers(), "Idempotency-Key": "correction-1"},
            json=correction_payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers={**_auth_headers(), "Idempotency-Key": "correction-1"},
            json={
                **correction_payload,
                "ops": [
                    {
                        "op": "update_episode",
                        "episode_target_id": 1,
                        "season": 1,
                        "episode_start": 3,
                        "episode_end": 3,
                    }
                ],
            },
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers={**_auth_headers(), "Idempotency-Key": "correction-stale"},
            json=correction_payload,
        )
        review_after_correction = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )

        committed_moves = runtime_store.list_job_moves(job.id)
        runtime_store.update_job_move(committed_moves[0].id, status="moved")
        moved_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        non_proposed = client.post(
            f"/api/v1/jobs/{job.id}/corrections",
            headers={**_auth_headers(), "Idempotency-Key": "correction-moved"},
            json={
                **correction_payload,
                "base_review_revision": moved_review.json()["review_revision"],
            },
        )

    assert missing_key.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert first.json()["status"] == "committed"
    assert first.json()["base_review_revision"] == initial_review.json()[
        "review_revision"
    ]
    assert first.json()["committed_move_item_count"] == 2
    assert [item["op"] for item in first.json()["op_results"]] == [
        "update_episode",
        "override_subtitle_variant",
    ]
    assert first.json()["diff_summary"]["changed"] == 2
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"
    assert non_proposed.status_code == 422
    assert non_proposed.json()["error"]["code"] == "invalid_correction_state"

    updated_job = runtime_store.get_job(job.id)
    assert updated_job is not None
    assert updated_job.status == "planned"
    assert updated_job.series_name == "旧动画"
    assert updated_job.tmdb_id == 123
    committed_moves = runtime_store.list_job_moves(job.id)
    assert [move.src_id for move in committed_moves] == [2, 1]
    assert [move.id for move in committed_moves] != [move.id for move in old_moves]
    assert "S01E02.cht.ass" in committed_moves[0].dst
    assert "S01E02.mkv" in committed_moves[1].dst
    assert (series_dir / "ep1.mkv").exists()
    assert (series_dir / "ep1.ass").exists()
    assert not (
        output_root
        / "旧动画 (2020) {tmdb-123}"
        / "S01"
        / "旧动画 S01E02.mkv"
    ).exists()

    snapshot = runtime_store.get_review_snapshot(job.id)
    assert snapshot is not None
    assert snapshot.episode_targets[0]["manual"] is True
    assert snapshot.episode_targets[0]["created_by"] == "user"
    assert snapshot.episode_targets[0]["season"] == 1
    assert snapshot.episode_targets[0]["episode_start"] == 2
    assert snapshot.episode_targets[0]["move_item_ids"] == [
        next(move.id for move in committed_moves if move.kind == "video")
    ]
    assert snapshot.subtitle_links[0]["created_by"] == "user"
    assert snapshot.subtitle_links[0]["variant_override"] == "cht"
    assert snapshot.subtitle_links[0]["move_item_id"] == next(
        move.id for move in committed_moves if move.kind == "subtitle"
    )
    assert snapshot.mapping_attempts[-1]["stage"] == "correction"
    assert snapshot.mapping_evidence[-1]["evidence_kind"] == "manual_override"
    assert snapshot.validation_summary["corrected_by_operation_id"] == first.json()[
        "operation_id"
    ]
    assert review_after_correction.json()["review_revision"] != initial_review.json()[
        "review_revision"
    ]
    operations_payload = operations.json()
    assert operations_payload["total"] == 1
    assert operations_payload["items"][0]["type"] == "corrections"
    assert operations_payload["items"][0]["result_summary"]["response_payload"] == first.json()


def test_apply_validation_records_operation_without_moving_files(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    _write(series_dir / "ep1.mkv", b"video")
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(job.id, status="planned", tmdb_id=123)
    runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=output_root / "ShowA (2020) {tmdb-123}" / "S01" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )

    invalid_series_dir = tmp_path / "input" / "ShowB"
    invalid_output_root = tmp_path / "output-invalid"
    _write(invalid_series_dir / "ep1.mkv", b"video")
    existing_dest = (
        invalid_output_root
        / "ShowB (2020) {tmdb-456}"
        / "S01"
        / "ShowB S01E01.mkv"
    )
    _write(existing_dest, b"existing")
    invalid_job = runtime_store.create_job(
        series_name="ShowB",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=invalid_series_dir,
        output_root=invalid_output_root,
    )
    runtime_store.update_job(invalid_job.id, status="planned", tmdb_id=456)
    runtime_store.replace_job_moves(
        invalid_job.id,
        (
            PlannedMove(
                src=invalid_series_dir / "ep1.mkv",
                dst=existing_dest,
                kind="video",
                src_id=1,
            ),
        ),
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = initial_review.json()["versions"]
        payload = {
            "scope": "all_for_apply",
            "base_mapping_version": versions["mapping_version"],
            "base_move_plan_version": versions["move_plan_version"],
            "base_move_state_revision": versions["move_state_revision"],
            "base_review_revision": versions["review_revision"],
        }
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers=_auth_headers(),
            json=payload,
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "validate-1"},
            json=payload,
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "validate-1"},
            json=payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "validate-1"},
            json={**payload, "scope": "selected_check", "selection": {"kind": "all"}},
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "validate-stale"},
            json=payload,
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )

        invalid_review = client.get(
            f"/api/v1/jobs/{invalid_job.id}/review",
            headers=_auth_headers(),
        )
        invalid_versions = invalid_review.json()["versions"]
        invalid_response = client.post(
            f"/api/v1/jobs/{invalid_job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "validate-invalid"},
            json={
                "scope": "all_for_apply",
                "base_mapping_version": invalid_versions["mapping_version"],
                "base_move_plan_version": invalid_versions["move_plan_version"],
                "base_move_state_revision": invalid_versions["move_state_revision"],
                "base_review_revision": invalid_versions["review_revision"],
            },
        )

    assert missing_key.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"
    payload = first.json()
    assert payload["status"] == "valid"
    assert payload["validation_id"] == payload["operation_id"]
    assert payload["scope"] == "all_for_apply"
    assert payload["validated_mapping_version"] == versions["mapping_version"]
    assert payload["checked_move_state_revision"] == versions["move_state_revision"]
    assert payload["item_result_summary"] == {
        "checked": 1,
        "valid": 1,
        "invalid": 0,
        "warning": 0,
    }
    assert payload["blocking_items"] == []
    assert payload["affected_move_items"]
    assert (series_dir / "ep1.mkv").exists()
    assert not (
        output_root / "ShowA (2020) {tmdb-123}" / "S01" / "ShowA S01E01.mkv"
    ).exists()
    operations_payload = operations.json()
    assert operations_payload["total"] == 1
    assert operations_payload["items"][0]["type"] == "validate"
    assert operations_payload["items"][0]["status"] == "succeeded"
    assert operations_payload["items"][0]["result_summary"]["response_payload"] == payload

    invalid_payload = invalid_response.json()
    assert invalid_response.status_code == 200
    assert invalid_payload["status"] == "invalid"
    assert invalid_payload["item_result_summary"]["invalid"] == 1
    assert invalid_payload["blocking_items"][0]["code"] == "destination_exists"


def test_apply_endpoint_moves_validated_plan_and_records_item_results(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    destination = (
        output_root / "ShowA (2020) {tmdb-123}" / "S01" / "ShowA S01E01.mkv"
    )
    _write(source, b"video")
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=output_root,
    )
    runtime_store.update_job(job.id, status="planned", tmdb_id=123)
    runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=source,
                dst=destination,
                kind="video",
                src_id=1,
            ),
        ),
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = initial_review.json()["versions"]
        validation = client.post(
            f"/api/v1/jobs/{job.id}/validate",
            headers={**_auth_headers(), "Idempotency-Key": "apply-validate"},
            json={
                "scope": "all_for_apply",
                "base_mapping_version": versions["mapping_version"],
                "base_move_plan_version": versions["move_plan_version"],
                "base_move_state_revision": versions["move_state_revision"],
                "base_review_revision": versions["review_revision"],
            },
        )
        review_after_validation = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        apply_versions = review_after_validation.json()["versions"]
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/apply",
            headers=_auth_headers(),
            json={
                "scope": "all_validated",
                "validation_id": validation.json()["validation_id"],
                "base_mapping_version": apply_versions["mapping_version"],
                "base_move_plan_version": apply_versions["move_plan_version"],
                "base_move_state_revision": apply_versions["move_state_revision"],
                "base_review_revision": apply_versions["review_revision"],
            },
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/apply",
            headers={**_auth_headers(), "Idempotency-Key": "apply-1"},
            json={
                "scope": "all_validated",
                "validation_id": validation.json()["validation_id"],
                "base_mapping_version": apply_versions["mapping_version"],
                "base_move_plan_version": apply_versions["move_plan_version"],
                "base_move_state_revision": apply_versions["move_state_revision"],
                "base_review_revision": apply_versions["review_revision"],
            },
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/apply",
            headers={**_auth_headers(), "Idempotency-Key": "apply-1"},
            json={
                "scope": "all_validated",
                "validation_id": validation.json()["validation_id"],
                "base_mapping_version": apply_versions["mapping_version"],
                "base_move_plan_version": apply_versions["move_plan_version"],
                "base_move_state_revision": apply_versions["move_state_revision"],
                "base_review_revision": apply_versions["review_revision"],
            },
        )
        operation_items = client.get(
            f"/api/v1/jobs/{job.id}/operations/{first.json()['operation_id']}/items",
            headers=_auth_headers(),
        )

    assert validation.status_code == 200
    assert validation.json()["status"] == "valid"
    assert missing_key.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    payload = first.json()
    assert payload["status"] == "succeeded"
    assert payload["validation_id"] == validation.json()["validation_id"]
    assert payload["applied_move_item_count"] == 1
    assert not source.exists()
    assert destination.read_bytes() == b"video"
    updated_job = runtime_store.get_job(job.id)
    assert updated_job is not None
    assert updated_job.status == "succeeded"
    moved = runtime_store.list_job_moves(job.id)
    assert moved[0].status == "moved"
    assert moved[0].applied_at is not None
    items_payload = operation_items.json()
    assert items_payload["total"] == 1
    assert items_payload["items"][0]["status"] == "succeeded"
    assert items_payload["items"][0]["path_event_id"] is not None
    assert items_payload["items"][0]["reservation_id"] is not None


def test_reopen_from_review_endpoint_reopens_withdrawn_items_and_is_idempotent(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job, move, review_file = _seed_reopen_job(runtime_store, tmp_path)

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions = initial_review.json()["versions"]
        safety_payload = {"base_review_revision": versions["review_revision"]}
        safety_missing_key = client.post(
            f"/api/v1/jobs/{job.id}/review-safety-check",
            headers=_auth_headers(),
            json=safety_payload,
        )
        safety_first = client.post(
            f"/api/v1/jobs/{job.id}/review-safety-check",
            headers={**_auth_headers(), "Idempotency-Key": "review-safety-1"},
            json=safety_payload,
        )
        safety_second = client.post(
            f"/api/v1/jobs/{job.id}/review-safety-check",
            headers={**_auth_headers(), "Idempotency-Key": "review-safety-1"},
            json=safety_payload,
        )
        safety_conflict = client.post(
            f"/api/v1/jobs/{job.id}/review-safety-check",
            headers={**_auth_headers(), "Idempotency-Key": "review-safety-1"},
            json={"base_review_revision": versions["review_revision"] + 1},
        )
        safety_stale = client.post(
            f"/api/v1/jobs/{job.id}/review-safety-check",
            headers={**_auth_headers(), "Idempotency-Key": "review-safety-stale"},
            json=safety_payload,
        )
        safety = client.get(
            f"/api/v1/jobs/{job.id}/safety",
            headers=_auth_headers(),
        )
        operations_after_safety = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )
        review_after_safety = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        versions_after_safety = review_after_safety.json()["versions"]
        safety_token = safety_first.json()["review_safety_check_token"]
        payload = {
            "scope": {"kind": "all"},
            "base_review_revision": versions_after_safety["review_revision"],
            "base_move_state_revision": versions_after_safety["move_state_revision"],
            "review_safety_check_token": safety_token,
        }
        missing_key = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers=_auth_headers(),
            json=payload,
        )
        invalid_scope = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-invalid-scope"},
            json={**payload, "scope": {"kind": "move_items", "ids": [move.id]}},
        )
        first = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-1"},
            json=payload,
        )
        second = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-1"},
            json=payload,
        )
        conflict = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-1"},
            json={
                **payload,
                "base_move_state_revision": versions_after_safety[
                    "move_state_revision"
                ]
                + 1,
            },
        )
        stale = client.post(
            f"/api/v1/jobs/{job.id}/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-stale"},
            json=payload,
        )
        missing_job = client.post(
            "/api/v1/jobs/999999/reopen-from-review",
            headers={**_auth_headers(), "Idempotency-Key": "reopen-missing-job"},
            json=payload,
        )
        review_after_reopen = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        operation_items = client.get(
            f"/api/v1/jobs/{job.id}/operations/{first.json()['operation_id']}/items",
            headers=_auth_headers(),
        )

    assert safety_missing_key.status_code == 422
    assert safety_first.status_code == 200
    assert safety_second.status_code == 200
    assert safety_second.json() == safety_first.json()
    assert safety_conflict.status_code == 409
    assert safety_stale.status_code == 412
    assert safety_stale.json()["error"]["code"] == "stale_versions"
    safety_payload = safety_first.json()
    assert safety_payload["status"] == "verified"
    assert safety_payload["token_status"] == "current"
    assert safety_payload["review_safety_check_token"]
    assert safety_payload["review_exclusion_proof"]["status"] == "verified"
    assert safety_payload["review_exclusion_proof"]["covered_consumers"]

    assert safety.json()["review_exclusion_proof"]["status"] == "verified"
    assert "review_safety_check_token" in safety.json()["safety_token_summary"]
    assert safety_payload["review_safety_check_token"] not in safety.text

    operations_payload = operations_after_safety.json()
    assert operations_payload["total"] == 1
    assert operations_payload["items"][0]["type"] == "review_safety_check"
    assert (
        operations_payload["items"][0]["result_summary"]["response_payload"][
            "review_safety_check_token"
        ]
        is None
    )
    assert safety_payload["review_safety_check_token"] not in operations_after_safety.text

    assert missing_key.status_code == 422
    assert invalid_scope.status_code == 422
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()
    assert conflict.status_code == 409
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "stale_versions"
    assert missing_job.status_code == 404

    payload = first.json()
    assert payload["status"] == "succeeded"
    assert payload["reopened_move_item_count"] == 1
    assert payload["item_result_summary"] == {
        "succeeded": 1,
        "failed": 0,
        "total": 1,
    }
    assert payload["reopened_items"][0]["move_item_id"] == move.id
    assert review_file.read_bytes() == b"video"

    updated_job = runtime_store.get_job(job.id)
    assert updated_job is not None
    assert updated_job.status == "planned"
    updated_moves = runtime_store.list_job_moves(job.id)
    assert updated_moves[0].status == "proposed"
    assert updated_moves[0].src == str(review_file)
    assert updated_moves[0].applied_at is None

    review_payload = review_after_reopen.json()
    assert review_payload["rollback_capability_hints"]["has_review_records"] is False
    assert review_payload["move_items"][0]["status"] == "proposed"
    assert review_payload["move_items"][0]["source_path"]["display_path"] == str(
        review_file
    )

    items_payload = operation_items.json()
    assert items_payload["total"] == 1
    assert items_payload["items"][0]["status"] == "succeeded"
    assert items_payload["items"][0]["path_event_id"] is None
    assert items_payload["items"][0]["from_path"]["display_path"] == str(review_file)
    assert items_payload["items"][0]["to_path"]["display_path"] == move.dst
    assert (
        items_payload["items"][0]["from_path"]["original_source_path"][
            "display_path"
        ]
        == move.src
    )


def test_rematch_commit_rejects_non_proposed_move_items_before_creating_operation(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job, _series_dir, _output_root, moves = _seed_rematch_job(runtime_store, tmp_path)
    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeRematchTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":456,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )
    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        preview = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "non-proposed-preview"},
            json={
                "tmdb_id": 456,
                "base_review_revision": initial_review.json()["review_revision"],
            },
        )
        runtime_store.update_job_move(moves[0].id, status="moved")
        review_after_move = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        response = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "non-proposed-commit"},
            json={
                "preview_operation_id": preview.json()["operation_id"],
                "tmdb_id": 456,
                "base_review_revision": review_after_move.json()["review_revision"],
            },
        )
        operations = client.get(
            f"/api/v1/jobs/{job.id}/operations",
            headers=_auth_headers(),
        )

    assert preview.status_code == 200
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_rematch_commit_state"
    assert response.json()["error"]["details"] == {
        "non_proposed_move_item_ids": [moves[0].id]
    }
    assert runtime_store.list_job_moves(job.id)[0].status == "moved"
    assert operations.json()["total"] == 1
    assert operations.json()["items"][0]["type"] == "rematch_preview"


def test_rematch_commit_rejects_missing_or_mismatched_preview_operation(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job, _series_dir, _output_root, _moves = _seed_rematch_job(
        runtime_store,
        tmp_path,
    )
    worker = AninamerWorker(
        app_config,
        runtime_store,
        tmdb_client_factory=lambda: FakeRematchTMDBClient(),
        llm_for_mapping_factory=lambda: FakeLLM(
            '{"tmdb":456,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
        ),
    )
    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        initial_review = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        missing = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "missing-preview"},
            json={
                "preview_operation_id": 999_999,
                "tmdb_id": 456,
                "base_review_revision": initial_review.json()["review_revision"],
            },
        )
        preview = client.post(
            f"/api/v1/jobs/{job.id}/rematch-preview",
            headers={**_auth_headers(), "Idempotency-Key": "mismatch-preview"},
            json={
                "tmdb_id": 456,
                "base_review_revision": initial_review.json()["review_revision"],
            },
        )
        review_after_preview = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
        mismatch = client.post(
            f"/api/v1/jobs/{job.id}/rematch-commit",
            headers={**_auth_headers(), "Idempotency-Key": "mismatch-commit"},
            json={
                "preview_operation_id": preview.json()["operation_id"],
                "tmdb_id": 789,
                "base_review_revision": review_after_preview.json()[
                    "review_revision"
                ],
            },
        )

    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "invalid_rematch_preview"
    assert preview.status_code == 200
    assert mismatch.status_code == 422
    assert mismatch.json()["error"]["code"] == "invalid_rematch_preview"
    assert mismatch.json()["error"]["details"]["preview_tmdb_id"] == 456
    assert mismatch.json()["error"]["details"]["requested_tmdb_id"] == 789


def test_review_payload_parses_episode_targets_and_subtitle_links(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.ass",
                dst=tmp_path / "output" / "ShowA S01E01.chs.ass",
                kind="subtitle",
                src_id=2,
            ),
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep2.mkv",
                dst=tmp_path / "output" / "ShowA S01E02.mkv",
                kind="video",
                src_id=3,
            ),
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep2.ass",
                dst=tmp_path / "output" / "ShowA S01E02.cht.ass",
                kind="subtitle",
                src_id=4,
            ),
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep2.alt.ass",
                dst=tmp_path / "output" / "ShowA S01E02.cht.1.ass",
                kind="subtitle",
                src_id=5,
            ),
        ),
    )
    _write(tmp_path / "input" / "ShowA" / "unmapped.mkv", b"extra")
    _write(tmp_path / "input" / "ShowA" / "orphan.ass", b"sub")
    _write(tmp_path / "input" / "ShowA" / "notes.txt", b"notes")
    series_dir = tmp_path / "input" / "ShowA"
    runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            _snapshot_source_file(
                1,
                "video",
                series_dir / "ep1.mkv",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                2,
                "subtitle",
                series_dir / "ep1.ass",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                3,
                "video",
                series_dir / "ep2.mkv",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                4,
                "subtitle",
                series_dir / "ep2.ass",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                5,
                "subtitle",
                series_dir / "ep2.alt.ass",
                series_dir,
                final_disposition="planned_output",
            ),
            _snapshot_source_file(
                1_000_000_001,
                "video",
                series_dir / "unmapped.mkv",
                series_dir,
                final_disposition="original",
            ),
            _snapshot_source_file(
                1_000_000_002,
                "subtitle",
                series_dir / "orphan.ass",
                series_dir,
                final_disposition="original",
            ),
            _snapshot_source_file(
                1_000_000_003,
                "ignored",
                series_dir / "notes.txt",
                series_dir,
                final_disposition="original",
                ignored_reason="unsupported_extension",
            ),
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "ShowA",
            "original_name": "Show A",
            "year": 2026,
            "candidates": [
                {
                    "id": 123,
                    "name": "ShowA",
                    "original_name": "Show A",
                    "first_air_date": "2026-01-01",
                    "year": 2026,
                }
            ],
            "seasons": [
                {"season_number": 1, "episode_count": 12},
                {"season_number": 0, "episode_count": 1},
            ],
            "specials": [
                {
                    "season_number": 0,
                    "episode_number": 1,
                    "name": "OVA",
                    "overview": "OVA episode",
                }
            ],
        },
        episode_targets=[
            {
                "id": 1,
                "video_source_id": 1,
                "season": 1,
                "episode_start": 1,
                "episode_end": 1,
                "status": "proposed",
                "manual": False,
                "created_by": "llm_mapping",
                "move_item_ids": [moves[0].id],
                "subtitle_link_ids": [1],
            },
            {
                "id": 2,
                "video_source_id": 3,
                "season": 1,
                "episode_start": 2,
                "episode_end": 2,
                "status": "proposed",
                "manual": False,
                "created_by": "llm_mapping",
                "move_item_ids": [moves[2].id],
                "subtitle_link_ids": [2, 3],
            },
        ],
        subtitle_links=[
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": moves[1].id,
            },
            {
                "id": 2,
                "episode_target_id": 2,
                "subtitle_source_id": 4,
                "variant_detected": "cht",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": moves[3].id,
            },
            {
                "id": 3,
                "episode_target_id": 2,
                "subtitle_source_id": 5,
                "variant_detected": "cht",
                "variant_override": None,
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": moves[4].id,
            },
        ],
        mapping_attempts=[
            {
                "id": 1,
                "stage": "episode_mapping",
                "created_by": "worker",
                "tmdb_id": 123,
                "status": "succeeded",
            }
        ],
        mapping_evidence=[
            {
                "id": 1,
                "episode_target_id": 1,
                "source_file_id": 1,
                "evidence_kind": "llm_episode_mapping",
                "summary": "LLM mapped ep1.mkv to S01E01.",
                "confidence": "unknown",
            }
        ],
        path_event_summary={"status": "not_started", "pending_count": 0},
        validation_summary={"status": "planned"},
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["review_revision"] == payload["versions"]["review_revision"]
    assert payload["versions"]["mapping_version"] > 1
    assert payload["versions"]["move_plan_version"] > 1
    assert payload["versions"]["move_state_revision"] > 1
    assert payload["selected_tmdb_info"]["candidates"]
    assert payload["selected_tmdb_info"]["seasons"]
    assert payload["selected_tmdb_info"]["specials"]
    assert payload["mapping_attempts"]
    assert payload["mapping_evidence"]
    assert payload["validation_summary"]["status"] == "planned"
    assert [
        (item["season"], item["episode_start"], item["episode_end"])
        for item in payload["episode_targets"]
    ] == [(1, 1, 1), (1, 2, 2)]
    assert [
        item["subtitle_link_ids"] for item in payload["episode_targets"]
    ] == [[1], [2, 3]]
    assert [
        (item["episode_target_id"], item["subtitle_source_id"], item["variant_detected"])
        for item in payload["subtitle_links"]
    ] == [(1, 2, "chs"), (2, 4, "cht"), (2, 5, "cht")]
    source_files = {
        item["original_path"]["rel_path"]: item
        for item in payload["source_files"]
    }
    assert source_files["ep1.mkv"]["final_disposition"] == "planned_output"
    assert source_files["ep1.ass"]["final_disposition"] == "planned_output"
    assert source_files["unmapped.mkv"]["kind"] == "video"
    assert source_files["unmapped.mkv"]["final_disposition"] == "original"
    assert source_files["orphan.ass"]["kind"] == "subtitle"
    assert source_files["orphan.ass"]["final_disposition"] == "original"
    assert source_files["notes.txt"]["kind"] == "ignored"
    assert source_files["notes.txt"]["ignored_reason"] == "unsupported_extension"
    assert payload["readiness_summary"]["status"] == "blocked"
    assert payload["readiness_summary"]["mapped_video_count"] == 2
    assert payload["readiness_summary"]["unmapped_video_count"] == 1
    assert payload["readiness_summary"]["unassociated_subtitle_count"] == 1
    assert payload["readiness_summary"]["ignored_file_count"] == 1
    assert payload["readiness_summary"]["blocking_count"] == 1
    assert payload["readiness_summary"]["warning_count"] == 2
    assert payload["readiness_summary"]["affected_refs"]["unmapped_videos"][0][
        "source_file_id"
    ] == source_files["unmapped.mkv"]["id"]
    assert payload["readiness_summary"]["affected_refs"]["unassociated_subtitles"][0][
        "source_file_id"
    ] == source_files["orphan.ass"]["id"]

    _write(tmp_path / "input" / "ShowA" / "another-extra.mkv", b"extra")
    with TestClient(app) as client:
        changed_response = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
    changed_payload = changed_response.json()
    changed_source_files = {
        item["original_path"]["rel_path"]: item
        for item in changed_payload["source_files"]
    }
    assert changed_payload["review_revision"] == payload["review_revision"]
    assert "another-extra.mkv" not in changed_source_files
    assert changed_source_files["unmapped.mkv"]["id"] == source_files["unmapped.mkv"]["id"]

    runtime_store.save_review_snapshot(
        job.id,
        source_files=payload["source_files"],
        selected_tmdb_info=payload["selected_tmdb_info"],
        episode_targets=payload["episode_targets"],
        subtitle_links=payload["subtitle_links"],
        mapping_attempts=payload["mapping_attempts"],
        mapping_evidence=[
            {
                "id": 1,
                "episode_target_id": 1,
                "source_file_id": 1,
                "evidence_kind": "llm_episode_mapping",
                "summary": "Updated evidence.",
                "confidence": "unknown",
            }
        ],
        path_event_summary=payload["path_event_summary"],
        validation_summary=payload["validation_summary"],
    )
    with TestClient(app) as client:
        evidence_changed_response = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )
    assert evidence_changed_response.json()["review_revision"] != payload["review_revision"]


def test_review_payload_derives_targets_for_legacy_snapshot_without_targets(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    series_dir = tmp_path / "input" / "ShowA"
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=series_dir,
        output_root=tmp_path / "output",
    )
    runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=series_dir / "ep1.mkv",
                dst=tmp_path / "output" / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            _snapshot_source_file(
                1,
                "video",
                series_dir / "ep1.mkv",
                series_dir,
                final_disposition="planned_output",
            )
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "ShowA",
            "candidates": [],
            "seasons": [],
            "specials": [],
        },
        validation_summary={"status": "planned"},
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/jobs/{job.id}/review",
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    payload = response.json()
    assert [
        (item["season"], item["episode_start"], item["episode_end"])
        for item in payload["episode_targets"]
    ] == [(1, 1, 1)]
    assert payload["validation_summary"]["target_link_status"] == (
        "derived_from_moves_for_legacy_snapshot"
    )


def test_rollback_eligibility_rejects_unsupported_scope(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/jobs/{job.id}/rollback-eligibility",
            headers={**_auth_headers(), "Idempotency-Key": "bad-scope"},
            json={
                "mode": "withdraw_to_review",
                "scope": {"kind": "episode_targets", "ids": [1]},
                "base_review_revision": 1,
                "base_move_state_revision": 1,
            },
        )

    assert response.status_code == 422


def test_withdraw_endpoint_requires_review_safety_token(
    app_config,
    runtime_store: RuntimeStore,
    tmp_path,
) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )

    app = create_app(app_config, store=runtime_store, worker=worker)
    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/jobs/{job.id}/withdraw",
            headers={**_auth_headers(), "Idempotency-Key": "missing-safety-token"},
            json={
                "mode": "withdraw_to_review",
                "scope": {"kind": "all"},
                "base_review_revision": 1,
                "base_move_state_revision": 1,
                "check_token": "rollback-token",
            },
        )

    assert response.status_code == 422
    assert "review_safety_check_token" in response.text


def test_notification_stream_route_is_removed(app_config, runtime_store: RuntimeStore) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/notifications/stream",
            headers=_auth_headers(),
        )
        assert response.status_code == 404
