from __future__ import annotations

from pathlib import Path

from aninamer.store import RuntimeStore


def test_runtime_store_job_and_request_round_trip(
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
    updated = runtime_store.update_job(
        job.id,
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=2,
    )

    assert updated.status == "planned"
    assert updated.tmdb_id == 123
    assert updated.video_moves_count == 1
    assert updated.subtitle_moves_count == 2
    assert runtime_store.get_job(job.id) == updated

    request = runtime_store.create_job_request(kind="apply_job", target_job_id=job.id)
    updated_request = runtime_store.update_job_request(
        request.id,
        status="running",
        error_message=None,
    )
    assert updated_request.status == "running"
    assert runtime_store.get_job_request(request.id) == updated_request


def test_runtime_store_artifact_round_trip(runtime_store: RuntimeStore, tmp_path: Path) -> None:
    job = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )

    runtime_store.save_artifact(job.id, "plan", {"version": 1, "moves": []})
    runtime_store.save_artifact(job.id, "result", {"version": 1, "finalize": {"status": "deleted"}})
    runtime_store.save_artifact(job.id, "rollback", {"version": 1, "moves": []})

    assert runtime_store.load_artifact(job.id, "plan") == {"version": 1, "moves": []}
    assert runtime_store.load_artifact(job.id, "result") == {
        "version": 1,
        "finalize": {"status": "deleted"},
    }
    assert runtime_store.load_artifact(job.id, "rollback") == {"version": 1, "moves": []}


def test_runtime_store_recovers_incomplete_jobs(runtime_store: RuntimeStore, tmp_path: Path) -> None:
    planning = runtime_store.create_job(
        series_name="Planning",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "Planning",
        output_root=tmp_path / "output",
    )
    applying = runtime_store.create_job(
        series_name="Applying",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "Applying",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(planning.id, status="planning", error_message="stale")
    runtime_store.update_job(applying.id, status="applying")

    recovered = runtime_store.recover_incomplete_jobs()

    recovered_planning = runtime_store.get_job(planning.id)
    recovered_applying = runtime_store.get_job(applying.id)
    assert recovered_planning is not None
    assert recovered_applying is not None
    assert recovered_planning.status == "pending"
    assert recovered_planning.error_message is None
    assert recovered_applying.status == "failed"
    assert recovered_applying.error_stage == "apply"
    assert recovered_applying.error_message == "worker restarted during apply"
    assert [job.id for job in recovered] == [applying.id]


def test_runtime_store_notification_round_trip(
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

    first = runtime_store.create_notification(
        event_kind="job_apply_succeeded",
        severity="success",
        title="处理完成",
        message="ShowA 已处理完成",
        markdown="# 处理完成",
        job_id=job.id,
        payload={"finalize_status": "deleted"},
    )
    second = runtime_store.create_notification(
        event_kind="job_request_rejected",
        severity="warning",
        title="请求被拒绝",
        message="apply_job 请求被拒绝",
        markdown="# 请求被拒绝",
        job_id=job.id,
        job_request_id=request.id,
        payload={"request_action": "apply_job", "job_id": job.id},
        delivery_status="disabled",
    )

    assert runtime_store.latest_notification_id() == second.id
    assert runtime_store.list_notifications_after(0) == [first, second]
    assert runtime_store.list_notifications_after(first.id) == [second]
    due = runtime_store.list_due_notifications()
    assert due == [first]
    assert first.markdown == "# 处理完成"
    assert first.disable_web_page_preview is True
    assert first.disable_notification is False
    assert first.delivery_status == "pending"
    assert first.attempt_count == 0
    assert first.next_attempt_at is not None
    assert second.delivery_status == "disabled"


def test_runtime_store_notification_delivery_transitions(
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
    notification = runtime_store.create_notification(
        event_kind="job_apply_failed",
        severity="error",
        title="应用失败",
        message="ShowA 应用失败",
        markdown="# 应用失败",
        job_id=job.id,
        payload={"error_stage": "apply"},
    )

    retried = runtime_store.mark_notification_retry(
        notification.id,
        attempt_count=1,
        next_attempt_at="2026-03-09T00:00:10+00:00",
        last_error="HTTP 500",
    )
    assert retried.delivery_status == "retry"
    assert retried.attempt_count == 1
    assert retried.last_error == "HTTP 500"
    assert retried.next_attempt_at == "2026-03-09T00:00:10+00:00"
    assert retried.last_attempt_at is not None

    delivered = runtime_store.mark_notification_delivered(
        notification.id,
        attempt_count=2,
    )
    assert delivered.delivery_status == "delivered"
    assert delivered.attempt_count == 2
    assert delivered.delivered_at is not None
    assert delivered.last_error is None
    assert delivered.next_attempt_at is None
