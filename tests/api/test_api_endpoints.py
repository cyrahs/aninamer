from __future__ import annotations

from dataclasses import replace
import threading
import time

from fastapi.testclient import TestClient

from aninamer.api import create_app
from aninamer.config import WorkerConfig
from aninamer.store import RuntimeStore
from aninamer.worker import AninamerWorker


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer secret"}


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
    runtime_store.save_artifact(job_a.id, "plan", {"version": 1})

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
    runtime_store.save_artifact(job_c.id, "rollback", {"version": 1})

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
        assert created["status"] == "pending"

        request_id = created["id"]
        read_response = client.get(
            f"/api/v1/job-requests/{request_id}",
            headers=_auth_headers(),
        )
        assert read_response.status_code == 200
        assert read_response.json()["id"] == request_id


def test_notification_stream_route_is_removed(app_config, runtime_store: RuntimeStore) -> None:
    worker = AninamerWorker(app_config, runtime_store)
    app = create_app(app_config, store=runtime_store, worker=worker)

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/notifications/stream",
            headers=_auth_headers(),
        )
        assert response.status_code == 404
