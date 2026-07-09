from __future__ import annotations

from pathlib import Path

import pytest
import psycopg
from psycopg.types.json import Jsonb

from aninamer.plan import PlannedMove
from aninamer.store import (
    DestinationReservationConflictError,
    DestinationReservationSpec,
    ReopenMoveItemUpdate,
    RuntimeStore,
)


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

    runtime_store.save_artifact(job.id, "result", {"version": 1, "finalize": {"status": "deleted"}})

    assert runtime_store.load_artifact(job.id, "result") == {
        "version": 1,
        "finalize": {"status": "deleted"},
    }


def test_runtime_store_notification_dedupe_key_is_idempotent(
    runtime_store: RuntimeStore,
) -> None:
    first = runtime_store.create_notification(
        event_kind="job_apply_succeeded",
        severity="success",
        title="处理完成",
        message="ShowA 已处理完成",
        markdown="# 处理完成",
        payload={"finalize_status": "archived"},
        dedupe_key="job_apply_succeeded:1",
        delivery_status="disabled",
    )
    second = runtime_store.create_notification(
        event_kind="job_apply_succeeded",
        severity="success",
        title="处理完成",
        message="ShowA 已处理完成",
        markdown="# 处理完成",
        payload={"finalize_status": "archived"},
        dedupe_key="job_apply_succeeded:1",
        delivery_status="disabled",
    )

    notifications = runtime_store.list_notifications_after(0)
    assert first.id == second.id
    assert [item.id for item in notifications] == [first.id]


def test_runtime_store_preserves_legacy_plan_and_rollback_artifacts(
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
    with psycopg.connect(runtime_store.postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO job_artifacts (job_id, artifact_kind, payload)
                VALUES (%s, 'plan', %s), (%s, 'rollback', %s)
                """,
                (
                    job.id,
                    Jsonb({"legacy": "plan"}),
                    job.id,
                    Jsonb({"legacy": "rollback"}),
                ),
            )

    runtime_store.bootstrap()

    with psycopg.connect(runtime_store.postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT artifact_kind, payload
                FROM job_artifacts
                WHERE job_id = %s
                ORDER BY artifact_kind
                """,
                (job.id,),
            )
            rows = cur.fetchall()
    assert rows == [
        ("plan", {"legacy": "plan"}),
        ("rollback", {"legacy": "rollback"}),
    ]


def test_runtime_store_job_move_items_round_trip(
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
    moves = (
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
    )

    inserted = runtime_store.replace_job_moves(job.id, moves)

    assert [item.src_id for item in inserted] == [1, 2]
    assert [item.status for item in inserted] == ["proposed", "proposed"]
    assert [item.kind for item in runtime_store.list_job_moves(job.id)] == [
        "video",
        "subtitle",
    ]

    updated = runtime_store.update_job_moves_for_job(
        job.id,
        status="moved",
        error_message=None,
        applied_at="2026-07-05T00:00:00+00:00",
    )

    assert [item.status for item in updated] == ["moved", "moved"]
    assert all(item.applied_at == "2026-07-05T00:00:00+00:00" for item in updated)

    replacement = runtime_store.replace_job_moves(job.id, moves[:1])
    assert len(replacement) == 1
    assert len(runtime_store.list_job_moves(job.id)) == 1


def test_runtime_store_refuses_to_replace_moves_after_path_events_exist(
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
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="apply",
        idempotency_key="apply-1",
        request_payload_hash="hash-1",
    )
    with psycopg.connect(runtime_store.postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO path_events (
                    event_sequence,
                    operation_id,
                    job_id,
                    subject_type,
                    move_item_id,
                    from_path,
                    to_path
                )
                VALUES (1, %s, %s, 'move_item', %s, %s, %s)
                """,
                (
                    operation.id,
                    job.id,
                    moves[0].id,
                    Jsonb({"display_path": "from"}),
                    Jsonb({"display_path": "to"}),
                ),
            )

    with pytest.raises(RuntimeError, match="cannot replace job moves"):
        runtime_store.replace_job_moves(job.id, ())


def test_runtime_store_llm_failure_round_trip(
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

    failure = runtime_store.record_llm_failure(
        job_id=job.id,
        stage="episode_mapping",
        attempt=2,
        raw_reply='{"tmdb": 123, "eps": [{"v": 999}]}',
        error_message="video id 999 not in video ids",
    )

    assert failure.job_id == job.id
    assert failure.stage == "episode_mapping"
    assert runtime_store.list_llm_failures(job.id) == [failure]


def test_runtime_store_review_snapshot_round_trip(
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

    snapshot = runtime_store.save_review_snapshot(
        job.id,
        source_files=[
            {
                "id": 1,
                "kind": "video",
                "original_path": {"rel_path": "ep1.mkv"},
                "current_path": {"rel_path": "ep1.mkv"},
                "current_path_kind": "source",
                "ext": ".mkv",
                "file_identity": {"identity_strength": "weak"},
                "final_disposition": "planned_output",
            }
        ],
        selected_tmdb_info={
            "selected_tmdb_id": 123,
            "title_zh_cn": "ShowA",
            "candidates": [{"id": 123, "name": "ShowA"}],
            "seasons": [{"season_number": 1, "episode_count": 12}],
            "specials": [],
        },
        mapping_attempts=[{"id": 1, "status": "succeeded"}],
        mapping_evidence=[{"id": 1, "summary": "mapped"}],
        path_event_summary={"status": "not_started"},
        validation_summary={"status": "planned"},
    )

    assert snapshot.job_id == job.id
    read_back = runtime_store.get_review_snapshot(job.id)
    assert read_back is not None
    assert read_back.selected_tmdb_info["selected_tmdb_id"] == 123
    assert read_back.source_files[0]["final_disposition"] == "planned_output"
    assert read_back.episode_targets == []
    assert read_back.subtitle_links == []
    assert read_back.mapping_attempts == [{"id": 1, "status": "succeeded"}]
    assert read_back.mapping_evidence == [{"id": 1, "summary": "mapped"}]


def test_runtime_store_operation_timestamps_reflect_status(
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

    queued = runtime_store.create_operation(
        job_id=job.id,
        operation_type="check",
        idempotency_key="queued",
        request_payload_hash="hash-queued",
        status="queued",
    )
    running = runtime_store.create_operation(
        job_id=job.id,
        operation_type="check",
        idempotency_key="running",
        request_payload_hash="hash-running",
        status="running",
    )
    succeeded = runtime_store.create_operation(
        job_id=job.id,
        operation_type="check",
        idempotency_key="succeeded",
        request_payload_hash="hash-succeeded",
        status="succeeded",
    )

    assert queued.started_at is None
    assert queued.finished_at is None
    assert running.started_at is not None
    assert running.finished_at is None
    assert succeeded.started_at is not None
    assert succeeded.finished_at is not None


def test_runtime_store_rematch_commit_rolls_back_on_snapshot_failure(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="Old",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(job.id, status="planned", tmdb_id=123)
    old_moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "Old S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="rematch_commit",
        idempotency_key="rematch-1",
        request_payload_hash="hash-rematch-1",
        status="running",
    )

    def broken_snapshot_payload(_moves):  # noqa: ANN001, ANN202
        raise RuntimeError("snapshot failed")

    with pytest.raises(RuntimeError, match="snapshot failed"):
        runtime_store.commit_rematch_plan(
            job_id=job.id,
            operation_id=operation.id,
            series_name="New",
            tmdb_id=456,
            moves=(
                PlannedMove(
                    src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                    dst=tmp_path / "output" / "New S01E01.mkv",
                    kind="video",
                    src_id=1,
                ),
            ),
            snapshot_payload_factory=broken_snapshot_payload,
            operation_result_summary_factory=lambda _moves: {"status": "committed"},
        )

    assert runtime_store.list_job_moves(job.id) == old_moves
    unchanged_job = runtime_store.get_job(job.id)
    assert unchanged_job is not None
    assert unchanged_job.series_name == "Old"
    assert unchanged_job.tmdb_id == 123
    unchanged_operation = runtime_store.get_operation(job.id, operation.id)
    assert unchanged_operation is not None
    assert unchanged_operation.status == "running"
    assert unchanged_operation.finished_at is None


def test_runtime_store_rematch_commit_refuses_non_proposed_moves_in_transaction(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="Old",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(job.id, status="planned", tmdb_id=123)
    old_moves = runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "Old S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.update_job_move(old_moves[0].id, status="moved")
    moved_rows = runtime_store.list_job_moves(job.id)
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="rematch_commit",
        idempotency_key="rematch-non-proposed",
        request_payload_hash="hash-rematch-non-proposed",
        status="running",
    )

    with pytest.raises(RuntimeError, match="non-proposed"):
        runtime_store.commit_rematch_plan(
            job_id=job.id,
            operation_id=operation.id,
            series_name="New",
            tmdb_id=456,
            moves=(
                PlannedMove(
                    src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                    dst=tmp_path / "output" / "New S01E01.mkv",
                    kind="video",
                    src_id=1,
                ),
            ),
            snapshot_payload_factory=lambda _moves: {
                "source_files": [],
                "selected_tmdb_info": {},
            },
            operation_result_summary_factory=lambda _moves: {"status": "committed"},
        )

    assert runtime_store.list_job_moves(job.id) == moved_rows
    unchanged_job = runtime_store.get_job(job.id)
    assert unchanged_job is not None
    assert unchanged_job.series_name == "Old"
    assert unchanged_job.tmdb_id == 123
    unchanged_operation = runtime_store.get_operation(job.id, operation.id)
    assert unchanged_operation is not None
    assert unchanged_operation.status == "running"
    assert unchanged_operation.finished_at is None


def test_runtime_store_rematch_commit_replays_succeeded_operation_without_reinserting_moves(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job = runtime_store.create_job(
        series_name="Old",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    runtime_store.update_job(job.id, status="planned", tmdb_id=123)
    runtime_store.replace_job_moves(
        job.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "Old S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="rematch_commit",
        idempotency_key="rematch-replay",
        request_payload_hash="hash-rematch-replay",
        status="running",
    )

    def snapshot_payload(_moves):  # noqa: ANN001, ANN202
        return {
            "source_files": [],
            "selected_tmdb_info": {"selected_tmdb_id": 456},
            "episode_targets": [],
            "subtitle_links": [],
            "mapping_attempts": [],
            "mapping_evidence": [],
            "path_event_summary": {"status": "not_started"},
            "validation_summary": {"status": "planned"},
        }

    def result_summary(_moves):  # noqa: ANN001, ANN202
        return {
            "status": "committed",
            "response_payload": {"status": "committed", "marker": "first"},
        }

    first = runtime_store.commit_rematch_plan(
        job_id=job.id,
        operation_id=operation.id,
        series_name="New",
        tmdb_id=456,
        moves=(
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "New S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
        snapshot_payload_factory=snapshot_payload,
        operation_result_summary_factory=result_summary,
    )
    first_move_ids = [move.id for move in first[1]]

    second = runtime_store.commit_rematch_plan(
        job_id=job.id,
        operation_id=operation.id,
        series_name="Unexpected",
        tmdb_id=789,
        moves=(
            PlannedMove(
                src=tmp_path / "input" / "ShowA" / "ep1.mkv",
                dst=tmp_path / "output" / "Unexpected S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
        snapshot_payload_factory=lambda _moves: {
            "source_files": [],
            "selected_tmdb_info": {"selected_tmdb_id": 789},
        },
        operation_result_summary_factory=lambda _moves: {
            "status": "committed",
            "response_payload": {"status": "committed", "marker": "second"},
        },
    )

    assert [move.id for move in second[1]] == first_move_ids
    assert [move.dst for move in second[1]] == [move.dst for move in first[1]]
    assert second[0].series_name == "New"
    assert second[0].tmdb_id == 456
    assert second[3].result_summary["response_payload"]["marker"] == "first"
    assert runtime_store.list_job_moves(job.id) == list(first[1])


def test_runtime_store_complete_reopen_from_review_updates_move_and_ledger(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
    review_file = tmp_path / "input" / "review" / "job-1" / "ep1.mkv"
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
                dst=output_root / "ShowA S01E01.mkv",
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
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="reopen_from_review",
        idempotency_key="reopen-1",
        request_payload_hash="hash-reopen-1",
        status="running",
    )

    updated_operation = runtime_store.complete_reopen_from_review_operation_success(
        operation_id=operation.id,
        job_id=job.id,
        updates=(
            ReopenMoveItemUpdate(
                move_item_id=moves[0].id,
                source_path=str(review_file),
                from_path={
                    "display_path": str(review_file),
                    "original_source_path": {"display_path": str(moves[0].src)},
                },
                to_path={"display_path": str(moves[0].dst)},
            ),
        ),
        result_summary={
            "status": "succeeded",
            "response_payload": {"reopened_move_item_count": 1},
        },
    )

    assert updated_operation.status == "succeeded"
    assert updated_operation.result_summary["response_payload"] == {
        "reopened_move_item_count": 1
    }
    updated_job = runtime_store.get_job(job.id)
    assert updated_job is not None
    assert updated_job.status == "planned"
    assert updated_job.finished_at is None
    updated_moves = runtime_store.list_job_moves(job.id)
    assert len(updated_moves) == 1
    assert updated_moves[0].status == "proposed"
    assert updated_moves[0].src == str(review_file)
    assert updated_moves[0].applied_at is None
    item_results = runtime_store.list_operation_item_results(operation.id)
    assert len(item_results) == 1
    assert item_results[0].subject_id == moves[0].id
    assert item_results[0].status == "succeeded"
    assert item_results[0].from_path == {
        "display_path": str(review_file),
        "original_source_path": {"display_path": str(moves[0].src)},
    }
    assert item_results[0].to_path == {"display_path": str(moves[0].dst)}


def test_runtime_store_reopen_from_review_rolls_back_when_move_not_withdrawn(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "input" / "ShowA"
    output_root = tmp_path / "output"
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
                dst=output_root / "ShowA S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="reopen_from_review",
        idempotency_key="reopen-not-withdrawn",
        request_payload_hash="hash-reopen-not-withdrawn",
        status="running",
    )

    with pytest.raises(RuntimeError, match="not withdrawn"):
        runtime_store.complete_reopen_from_review_operation_success(
            operation_id=operation.id,
            job_id=job.id,
            updates=(
                ReopenMoveItemUpdate(
                    move_item_id=moves[0].id,
                    source_path=str(tmp_path / "input" / "review" / "ep1.mkv"),
                    from_path={"display_path": "review/ep1.mkv"},
                    to_path={"display_path": str(moves[0].dst)},
                ),
            ),
            result_summary={"status": "succeeded"},
        )

    assert runtime_store.list_job_moves(job.id) == moves
    unchanged_job = runtime_store.get_job(job.id)
    assert unchanged_job is not None
    assert unchanged_job.status == "succeeded"
    unchanged_operation = runtime_store.get_operation(job.id, operation.id)
    assert unchanged_operation is not None
    assert unchanged_operation.status == "running"
    assert runtime_store.list_operation_item_results(operation.id) == []


def test_runtime_store_reopen_from_review_rolls_back_when_move_missing(
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
    runtime_store.update_job(job.id, status="succeeded", tmdb_id=123)
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="reopen_from_review",
        idempotency_key="reopen-missing-move",
        request_payload_hash="hash-reopen-missing-move",
        status="running",
    )

    with pytest.raises(RuntimeError, match="missing"):
        runtime_store.complete_reopen_from_review_operation_success(
            operation_id=operation.id,
            job_id=job.id,
            updates=(
                ReopenMoveItemUpdate(
                    move_item_id=999_999,
                    source_path=str(tmp_path / "input" / "review" / "ep1.mkv"),
                    from_path={"display_path": "review/ep1.mkv"},
                    to_path={"display_path": "output/ep1.mkv"},
                ),
            ),
            result_summary={"status": "succeeded"},
        )

    unchanged_job = runtime_store.get_job(job.id)
    assert unchanged_job is not None
    assert unchanged_job.status == "succeeded"
    unchanged_operation = runtime_store.get_operation(job.id, operation.id)
    assert unchanged_operation is not None
    assert unchanged_operation.status == "running"
    assert runtime_store.list_operation_item_results(operation.id) == []


def test_destination_reservations_conflict_across_root_aliases(
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
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="apply",
        idempotency_key="apply-1",
        request_payload_hash="hash-1",
    )
    first = _reservation_spec(
        root_id="output-a",
        root_alias_group_id="real-output",
        canonical_destination_key="/real/output/show/ep1.mkv",
        canonical_parent_key="/real/output/show",
        ancestor_keys=("/", "/real", "/real/output", "/real/output/show"),
    )
    second = _reservation_spec(
        root_id="output-b",
        root_alias_group_id="real-output",
        canonical_destination_key="/real/output/show/ep1.mkv",
        canonical_parent_key="/real/output/show",
        ancestor_keys=("/", "/real", "/real/output", "/real/output/show"),
    )

    inserted = runtime_store.reserve_destinations(
        operation_id=operation.id,
        job_id=job.id,
        reservations=(first,),
    )

    assert inserted[0].canonical_destination_key == "/real/output/show/ep1.mkv"
    with pytest.raises(DestinationReservationConflictError) as exc_info:
        runtime_store.reserve_destinations(
            operation_id=operation.id,
            job_id=job.id,
            reservations=(second,),
        )
    assert exc_info.value.conflict_code == "destination_reserved"


def test_destination_reservations_conflict_on_parent_child_overlap(
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
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="restore",
        idempotency_key="restore-1",
        request_payload_hash="hash-1",
    )
    parent = _reservation_spec(
        root_id="review",
        root_alias_group_id="review-root",
        canonical_destination_key="/real/review/job-1",
        canonical_parent_key="/real/review",
        ancestor_keys=("/", "/real", "/real/review"),
    )
    child = _reservation_spec(
        root_id="review-child",
        root_alias_group_id="review-root",
        canonical_destination_key="/real/review/job-1/ep1.mkv",
        canonical_parent_key="/real/review/job-1",
        ancestor_keys=("/", "/real", "/real/review", "/real/review/job-1"),
    )

    with pytest.raises(DestinationReservationConflictError):
        runtime_store.reserve_destinations(
            operation_id=operation.id,
            job_id=job.id,
            reservations=(parent, child),
        )


def test_destination_reservations_conflict_on_casefold_key(
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
    operation = runtime_store.create_operation(
        job_id=job.id,
        operation_type="apply",
        idempotency_key="apply-casefold",
        request_payload_hash="hash-1",
    )
    first = _reservation_spec(
        root_id="output",
        root_alias_group_id="real-output",
        canonical_destination_key="/real/output/Show/Ep1.mkv",
        canonical_parent_key="/real/output/Show",
        ancestor_keys=("/", "/real", "/real/output", "/real/output/Show"),
    )
    second = _reservation_spec(
        root_id="output",
        root_alias_group_id="real-output",
        canonical_destination_key="/real/output/show/ep1.mkv",
        canonical_parent_key="/real/output/show",
        ancestor_keys=("/", "/real", "/real/output", "/real/output/show"),
    )

    runtime_store.reserve_destinations(
        operation_id=operation.id,
        job_id=job.id,
        reservations=(first,),
    )
    with pytest.raises(DestinationReservationConflictError):
        runtime_store.reserve_destinations(
            operation_id=operation.id,
            job_id=job.id,
            reservations=(second,),
        )


def test_destination_reservations_validate_operation_job_match(
    runtime_store: RuntimeStore,
    tmp_path: Path,
) -> None:
    job_a = runtime_store.create_job(
        series_name="ShowA",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowA",
        output_root=tmp_path / "output",
    )
    job_b = runtime_store.create_job(
        series_name="ShowB",
        watch_root_key="downloads",
        source_kind="monitor",
        series_dir=tmp_path / "input" / "ShowB",
        output_root=tmp_path / "output",
    )
    operation = runtime_store.create_operation(
        job_id=job_a.id,
        operation_type="apply",
        idempotency_key="apply-job-match",
        request_payload_hash="hash-1",
    )

    with pytest.raises(ValueError, match="does not belong"):
        runtime_store.reserve_destinations(
            operation_id=operation.id,
            job_id=job_b.id,
            reservations=(
                _reservation_spec(
                    root_id="output",
                    root_alias_group_id="real-output",
                    canonical_destination_key="/real/output/show/ep1.mkv",
                    canonical_parent_key="/real/output/show",
                    ancestor_keys=("/", "/real", "/real/output", "/real/output/show"),
                ),
            ),
        )


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
    applying_moves = runtime_store.replace_job_moves(
        applying.id,
        (
            PlannedMove(
                src=tmp_path / "input" / "Applying" / "ep1.mkv",
                dst=tmp_path / "output" / "Applying S01E01.mkv",
                kind="video",
                src_id=1,
            ),
        ),
    )
    runtime_store.update_job_move(applying_moves[0].id, status="applying")
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
    recovered_moves = runtime_store.list_job_moves(applying.id)
    assert recovered_moves[0].status == "failed"
    assert recovered_moves[0].error_message == "worker restarted during apply"
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


def test_runtime_store_releases_notification_dedupe_key_for_next_outage(
    runtime_store: RuntimeStore,
) -> None:
    dedupe_key = "worker_storage_unavailable:active"
    first = runtime_store.create_notification(
        event_kind="worker_storage_unavailable",
        severity="warning",
        title="存储不可用",
        message="监控目录不可用",
        markdown="# 存储不可用",
        dedupe_key=dedupe_key,
    )

    assert runtime_store.get_notification_by_dedupe_key(dedupe_key) == first
    suppressed = runtime_store.suppress_notification_delivery(
        first.id,
        reason="storage recovered before delivery",
    )
    assert suppressed is not None
    assert suppressed.delivery_status == "disabled"
    assert suppressed.next_attempt_at is None
    assert suppressed.last_error == "storage recovered before delivery"
    assert runtime_store.release_notification_dedupe_key(
        first.id,
        dedupe_key=dedupe_key,
    )
    assert runtime_store.get_notification_by_dedupe_key(dedupe_key) is None
    assert not runtime_store.release_notification_dedupe_key(
        first.id,
        dedupe_key=dedupe_key,
    )

    second = runtime_store.create_notification(
        event_kind="worker_storage_unavailable",
        severity="warning",
        title="存储不可用",
        message="监控目录再次不可用",
        markdown="# 存储再次不可用",
        dedupe_key=dedupe_key,
    )
    assert second.id != first.id


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


def _reservation_spec(
    *,
    root_id: str,
    root_alias_group_id: str,
    canonical_destination_key: str,
    canonical_parent_key: str,
    ancestor_keys: tuple[str, ...],
) -> DestinationReservationSpec:
    return DestinationReservationSpec(
        root_id=root_id,
        root_generation=1,
        root_alias_group_id=root_alias_group_id,
        path={
            "root_kind": "output",
            "root_id": root_id,
            "display_path": canonical_destination_key,
        },
        normalized_path_key=canonical_destination_key,
        casefold_path_key=canonical_destination_key.casefold(),
        canonical_destination_key=canonical_destination_key,
        canonical_parent_key=canonical_parent_key,
        ancestor_keys=ancestor_keys,
        descendant_prefix_key=f"{canonical_destination_key}/",
    )
