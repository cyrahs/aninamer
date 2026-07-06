from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from aninamer.api.schemas import (
    ApplyRequest,
    ApplyValidationRequest,
    CorrectionRequest,
    PathRef,
    ReopenFromReviewRequest,
    RematchPreviewRequest,
    RollbackEligibilityRequest,
    ReviewSourceFile,
    SelectionScope,
    TmdbReviewInfo,
    WithdrawRequest,
)
from aninamer.api.service import (
    ApiService,
    ApiServiceError,
    _apply_correction_ops,
    _apply_validation_response_payload,
    _assert_review_revision_current,
    _committed_source_files,
    _committed_targets_and_links,
    _correction_diff_summary,
    _correction_snapshot_payload,
    _destination_reservation_spec,
    _execute_apply_moves,
    _execute_reopen_from_review,
    _execute_withdraw_moves,
    _job_versions,
    _now_iso,
    _planned_moves_from_corrected_snapshot,
    _planned_moves_from_preview_payload,
    _payload_hash,
    _consumer_chain_digest,
    _review_exclusion_proof,
    _resolve_apply_validation_scope_or_raise,
    _validated_withdraw_check_or_raise,
    _validation_source_files,
    _withdraw_destination_reservation_spec,
)
from aninamer.store import (
    DestinationReservationConflictError,
    IdempotencyConflictError,
    JobMoveRecord,
    JobRecord,
    OperationRecord,
    OperationItemResultRecord,
)


def _path_ref(path: str) -> PathRef:
    return PathRef(
        root_kind="source",
        root_id="source:default:1",
        rel_path=path,
        display_path=f"/input/Show/{path}",
        normalized_abs_path=f"/input/Show/{path}",
        realpath=f"/input/Show/{path}",
        casefold_key=f"/input/show/{path.casefold()}",
        redacted=False,
    )


def _source_file(source_id: int, kind: str, rel_path: str) -> ReviewSourceFile:
    return ReviewSourceFile(
        id=source_id,
        kind=kind,
        original_path=_path_ref(rel_path),
        current_path=_path_ref(rel_path),
        current_path_kind="source",
        ext="." + rel_path.rsplit(".", 1)[-1],
        file_identity={"identity_strength": "weak"},
        detected_subtitle_variant="unknown" if kind == "subtitle" else None,
        final_disposition="planned_output",
        ignored_reason=None,
    )


def _move(
    move_id: int,
    source_id: int,
    kind: str,
    src: str,
    dst: str,
    status: str = "proposed",
) -> JobMoveRecord:
    return JobMoveRecord(
        id=move_id,
        job_id=1,
        src_id=source_id,
        kind=kind,  # type: ignore[arg-type]
        src=src,
        dst=dst,
        status=status,  # type: ignore[arg-type]
        error_message=None,
        applied_at=None,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
    )


def _job(output_root: str = "/output") -> JobRecord:
    return JobRecord(
        id=1,
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        error_stage=None,
        error_message=None,
        series_dir="/input/Show",
        output_root=output_root,
        archive_path=None,
        fail_path=None,
    )


def _operation(operation_id: int, operation_type: str) -> OperationRecord:
    return OperationRecord(
        id=operation_id,
        job_id=1,
        operation_type=operation_type,
        status="succeeded",
        idempotency_key=f"op-{operation_id}",
        request_payload_hash=f"hash-{operation_id}",
        base_versions={},
        progress={},
        result_summary={"status": "done"},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )


def test_rematch_commit_helpers_convert_preview_to_committed_models() -> None:
    preview_payload = {
        "episode_targets": [
            {
                "id": 1,
                "video_source_id": 1,
                "season": 1,
                "episode_start": 1,
                "episode_end": 1,
            }
        ],
        "subtitle_links": [
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": None,
            }
        ],
        "move_items": [
            {
                "source_id": 2,
                "kind": "subtitle",
                "source_path": {"display_path": "/input/Show/ep1.ass"},
                "preview_destination_path": {
                    "display_path": "/output/New/S01/New S01E01.chs.ass"
                },
            },
            {
                "source_id": 1,
                "kind": "video",
                "source_path": {"display_path": "/input/Show/ep1.mkv"},
                "preview_destination_path": {
                    "display_path": "/output/New/S01/New S01E01.mkv"
                },
            },
            {
                "source_id": 3,
                "kind": "video",
                "source_path": {"display_path": "/input/Show/old-extra.mkv"},
                "preview_destination_path": None,
            },
        ],
    }

    planned_moves = _planned_moves_from_preview_payload(preview_payload)

    assert [(move.src_id, move.kind) for move in planned_moves] == [
        (2, "subtitle"),
        (1, "video"),
    ]
    assert [str(move.dst) for move in planned_moves] == [
        "/output/New/S01/New S01E01.chs.ass",
        "/output/New/S01/New S01E01.mkv",
    ]

    committed_moves = [
        _move(10, 2, "subtitle", "/input/Show/ep1.ass", str(planned_moves[0].dst)),
        _move(11, 1, "video", "/input/Show/ep1.mkv", str(planned_moves[1].dst)),
    ]
    targets, links = _committed_targets_and_links(preview_payload, committed_moves)

    assert targets == [
        {
            "id": 1,
            "video_source_id": 1,
            "season": 1,
            "episode_start": 1,
            "episode_end": 1,
            "status": "proposed",
            "manual": False,
            "created_by": "rematch",
            "move_item_ids": [11],
            "subtitle_link_ids": [1],
        }
    ]
    assert links[0]["created_by"] == "rematch"
    assert links[0]["move_item_id"] == 10

    source_files = _committed_source_files(
        [
            _source_file(1, "video", "ep1.mkv"),
            _source_file(2, "subtitle", "ep1.ass"),
            _source_file(3, "video", "old-extra.mkv"),
        ],
        committed_moves,
    )

    assert [
        (item.id, item.final_disposition, item.current_path_kind)
        for item in source_files
    ] == [
        (1, "planned_output", "source"),
        (2, "planned_output", "source"),
        (3, "original", "source"),
    ]


def test_correction_helpers_rebuild_plan_snapshot_and_diff() -> None:
    job = JobRecord(
        id=1,
        series_name="旧动画",
        watch_root_key="downloads",
        source_kind="monitor",
        status="planned",
        tmdb_id=123,
        video_moves_count=1,
        subtitle_moves_count=1,
        created_at="2026-07-06T00:00:00+00:00",
        updated_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        error_stage=None,
        error_message=None,
        series_dir="/input/Show",
        output_root="/output",
        archive_path=None,
        fail_path=None,
    )
    selected_info = TmdbReviewInfo(
        selected_tmdb_id=123,
        title_zh_cn="旧动画",
        original_name="Old Anime",
        year=2020,
        candidates=[],
        seasons=[{"season_number": 1, "episode_count": 12}],
        specials=[],
    )
    current_moves = [
        _move(
            10,
            1,
            "video",
            "/input/Show/ep1.mkv",
            "/output/旧动画 (2020) {tmdb-123}/S01/旧动画 S01E01.mkv",
        ),
        _move(
            11,
            2,
            "subtitle",
            "/input/Show/ep1.ass",
            "/output/旧动画 (2020) {tmdb-123}/S01/旧动画 S01E01.chs.ass",
        ),
    ]
    payload = CorrectionRequest.model_validate(
        {
            "base_review_revision": 7,
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
    )

    corrected_targets, corrected_links, op_results = _apply_correction_ops(
        payload,
        selected_info=selected_info,
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
                "move_item_ids": [10],
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
                "move_item_id": 11,
            }
        ],
    )
    planned_moves = _planned_moves_from_corrected_snapshot(
        job,
        selected_info=selected_info,
        episode_targets=corrected_targets,
        subtitle_links=corrected_links,
        current_moves=current_moves,
    )

    assert op_results == [
        {
            "index": 1,
            "op": "update_episode",
            "status": "applied",
            "episode_target_id": 1,
            "before": {"season": 1, "episode_start": 1, "episode_end": 1},
            "after": {"season": 1, "episode_start": 2, "episode_end": 2},
        },
        {
            "index": 2,
            "op": "override_subtitle_variant",
            "status": "applied",
            "subtitle_link_id": 1,
            "before": {"variant": "chs"},
            "after": {"variant": "cht", "variant_override": "cht"},
        },
    ]
    assert [(move.src_id, move.kind) for move in planned_moves] == [
        (2, "subtitle"),
        (1, "video"),
    ]
    assert [str(move.dst) for move in planned_moves] == [
        "/output/旧动画 (2020) {tmdb-123}/S01/旧动画 S01E02.cht.ass",
        "/output/旧动画 (2020) {tmdb-123}/S01/旧动画 S01E02.mkv",
    ]
    committed_moves = [
        _move(20, 2, "subtitle", str(planned_moves[0].src), str(planned_moves[0].dst)),
        _move(21, 1, "video", str(planned_moves[1].src), str(planned_moves[1].dst)),
    ]
    snapshot = _correction_snapshot_payload(
        source_files=[
            _source_file(1, "video", "ep1.mkv"),
            _source_file(2, "subtitle", "ep1.ass"),
        ],
        committed_moves=committed_moves,
        selected_info=selected_info,
        corrected_targets=corrected_targets,
        corrected_links=corrected_links,
        existing_mapping_attempts=[],
        existing_mapping_evidence=[],
        op_results=op_results,
        operation_id=99,
    )

    assert snapshot["episode_targets"][0]["manual"] is True
    assert snapshot["episode_targets"][0]["created_by"] == "user"
    assert snapshot["episode_targets"][0]["move_item_ids"] == [21]
    assert snapshot["subtitle_links"][0]["variant_override"] == "cht"
    assert snapshot["subtitle_links"][0]["move_item_id"] == 20
    assert snapshot["mapping_attempts"][-1]["stage"] == "correction"
    assert snapshot["mapping_evidence"][0]["evidence_kind"] == "manual_override"
    assert snapshot["validation_summary"]["corrected_by_operation_id"] == 99
    assert _correction_diff_summary(current_moves, committed_moves) == {
        "added": 0,
        "changed": 2,
        "removed": 0,
        "unchanged": 0,
        "total": 2,
    }


def test_correction_helpers_validate_tmdb_bounds() -> None:
    selected_info = TmdbReviewInfo(
        selected_tmdb_id=123,
        title_zh_cn="旧动画",
        original_name="Old Anime",
        year=2020,
        candidates=[],
        seasons=[{"season_number": 1, "episode_count": 1}],
        specials=[{"season_number": 0, "episode_number": 2, "name": "OVA"}],
    )
    base_target = {
        "id": 1,
        "video_source_id": 1,
        "season": 1,
        "episode_start": 1,
        "episode_end": 1,
        "status": "proposed",
        "manual": False,
        "created_by": "llm_mapping",
        "move_item_ids": [10],
        "subtitle_link_ids": [],
    }

    out_of_range = CorrectionRequest.model_validate(
        {
            "base_review_revision": 7,
            "ops": [
                {
                    "op": "update_episode",
                    "episode_target_id": 1,
                    "season": 1,
                    "episode_start": 2,
                    "episode_end": 2,
                }
            ],
        }
    )
    with pytest.raises(ApiServiceError) as exc_info:
        _apply_correction_ops(
            out_of_range,
            selected_info=selected_info,
            episode_targets=[base_target],
            subtitle_links=[],
        )
    assert exc_info.value.code == "invalid_correction"
    assert exc_info.value.details == {
        "season": 1,
        "episode_start": 2,
        "episode_end": 2,
        "max_episode": 1,
    }

    missing_season = CorrectionRequest.model_validate(
        {
            "base_review_revision": 7,
            "ops": [
                {
                    "op": "update_episode",
                    "episode_target_id": 1,
                    "season": 2,
                    "episode_start": 1,
                    "episode_end": 1,
                }
            ],
        }
    )
    with pytest.raises(ApiServiceError) as missing_exc:
        _apply_correction_ops(
            missing_season,
            selected_info=selected_info,
            episode_targets=[base_target],
            subtitle_links=[],
        )
    assert missing_exc.value.code == "invalid_correction"
    assert missing_exc.value.details == {
        "season": 2,
        "available_seasons": [0, 1],
    }

    special = CorrectionRequest.model_validate(
        {
            "base_review_revision": 7,
            "ops": [
                {
                    "op": "update_episode",
                    "episode_target_id": 1,
                    "season": 0,
                    "episode_start": 2,
                    "episode_end": 2,
                }
            ],
        }
    )
    corrected_targets, _links, _op_results = _apply_correction_ops(
        special,
        selected_info=selected_info,
        episode_targets=[base_target],
        subtitle_links=[],
    )
    assert corrected_targets[0]["season"] == 0
    assert corrected_targets[0]["episode_start"] == 2


def test_correction_helpers_clear_subtitle_override_to_detected_variant() -> None:
    selected_info = TmdbReviewInfo(
        selected_tmdb_id=123,
        title_zh_cn="旧动画",
        original_name="Old Anime",
        year=2020,
        candidates=[],
        seasons=[{"season_number": 1, "episode_count": 12}],
        specials=[],
    )
    payload = CorrectionRequest.model_validate(
        {
            "base_review_revision": 7,
            "ops": [
                {
                    "op": "override_subtitle_variant",
                    "subtitle_link_id": 1,
                    "variant": "detected",
                }
            ],
        }
    )

    _targets, corrected_links, op_results = _apply_correction_ops(
        payload,
        selected_info=selected_info,
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
                "move_item_ids": [10],
                "subtitle_link_ids": [1],
            }
        ],
        subtitle_links=[
            {
                "id": 1,
                "episode_target_id": 1,
                "subtitle_source_id": 2,
                "variant_detected": "chs",
                "variant_override": "cht",
                "status": "proposed",
                "created_by": "llm_mapping",
                "move_item_id": 11,
            }
        ],
    )

    assert corrected_links[0]["variant_override"] is None
    assert corrected_links[0]["created_by"] == "user"
    assert op_results == [
        {
            "index": 1,
            "op": "override_subtitle_variant",
            "status": "applied",
            "subtitle_link_id": 1,
            "before": {"variant": "cht"},
            "after": {"variant": "chs", "variant_override": None},
        }
    ]


def test_review_revision_guard_excludes_current_and_preview_operations() -> None:
    job = _job()
    moves = [
        _move(
            10,
            1,
            "video",
            "/input/Show/ep1.mkv",
            "/output/旧动画 (2020) {tmdb-123}/S01/旧动画 S01E01.mkv",
        )
    ]
    expected_revision = _job_versions(
        job,
        moves,
        [],
        ApiService.model_review_source_files(job, moves, None),
        snapshot={},
    )["review_revision"]

    _assert_review_revision_current(
        expected_revision=expected_revision,
        job=job,
        moves=moves,
        operations=[_operation(1, "rematch_preview"), _operation(2, "rematch_commit")],
        snapshot=None,
        excluded_operation_ids={1, 2},
        message="stale",
    )
    with pytest.raises(ApiServiceError) as exc_info:
        _assert_review_revision_current(
            expected_revision=expected_revision,
            job=job,
            moves=moves,
            operations=[
                _operation(1, "rematch_preview"),
                _operation(2, "rematch_commit"),
                _operation(3, "corrections"),
            ],
            snapshot=None,
            excluded_operation_ids={1, 2},
            message="stale",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_versions"


def test_correction_helpers_reject_existing_destinations(tmp_path) -> None:  # noqa: ANN001
    output_root = tmp_path / "output"
    input_root = tmp_path / "input" / "Show"
    input_root.mkdir(parents=True)
    expected_video = (
        output_root
        / "旧动画 (2020) {tmdb-123}"
        / "S01"
        / "旧动画 S01E02.mkv"
    )
    expected_video.parent.mkdir(parents=True)
    expected_video.write_bytes(b"existing")
    job = _job(str(output_root))
    current_moves = [
        _move(
            10,
            1,
            "video",
            str(input_root / "ep1.mkv"),
            str(
                output_root
                / "旧动画 (2020) {tmdb-123}"
                / "S01"
                / "旧动画 S01E01.mkv"
            ),
        )
    ]
    selected_info = TmdbReviewInfo(
        selected_tmdb_id=123,
        title_zh_cn="旧动画",
        original_name="Old Anime",
        year=2020,
        candidates=[],
        seasons=[{"season_number": 1, "episode_count": 12}],
        specials=[],
    )

    with pytest.raises(ApiServiceError) as exc_info:
        _planned_moves_from_corrected_snapshot(
            job,
            selected_info=selected_info,
            episode_targets=[
                {
                    "id": 1,
                    "video_source_id": 1,
                    "season": 1,
                    "episode_start": 2,
                    "episode_end": 2,
                    "status": "proposed",
                    "manual": True,
                    "created_by": "user",
                    "move_item_ids": [10],
                    "subtitle_link_ids": [],
                }
            ],
            subtitle_links=[],
            current_moves=current_moves,
        )

    assert exc_info.value.code == "destination_conflict"
    assert exc_info.value.details == {"destination": str(expected_video)}


def test_correction_helpers_reject_symlink_escape_from_output_root(tmp_path) -> None:  # noqa: ANN001
    output_root = tmp_path / "output"
    outside_root = tmp_path / "outside"
    input_root = tmp_path / "input" / "Show"
    output_root.mkdir()
    outside_root.mkdir()
    input_root.mkdir(parents=True)
    (output_root / "旧动画 (2020) {tmdb-123}").symlink_to(
        outside_root,
        target_is_directory=True,
    )
    job = _job(str(output_root))
    current_moves = [
        _move(
            10,
            1,
            "video",
            str(input_root / "ep1.mkv"),
            str(
                output_root
                / "旧动画 (2020) {tmdb-123}"
                / "S01"
                / "旧动画 S01E01.mkv"
            ),
        )
    ]
    selected_info = TmdbReviewInfo(
        selected_tmdb_id=123,
        title_zh_cn="旧动画",
        original_name="Old Anime",
        year=2020,
        candidates=[],
        seasons=[{"season_number": 1, "episode_count": 12}],
        specials=[],
    )

    with pytest.raises(ApiServiceError) as exc_info:
        _planned_moves_from_corrected_snapshot(
            job,
            selected_info=selected_info,
            episode_targets=[
                {
                    "id": 1,
                    "video_source_id": 1,
                    "season": 1,
                    "episode_start": 2,
                    "episode_end": 2,
                    "status": "proposed",
                    "manual": True,
                    "created_by": "user",
                    "move_item_ids": [10],
                    "subtitle_link_ids": [],
                }
            ],
            subtitle_links=[],
            current_moves=current_moves,
        )

    assert exc_info.value.code == "invalid_correction_destination"
    assert exc_info.value.details == {
        "destination": str(outside_root / "S01" / "旧动画 S01E02.mkv"),
        "output_root": str(output_root),
    }


def test_apply_validation_response_reports_ready_and_blocking_items(tmp_path) -> None:  # noqa: ANN001
    input_root = tmp_path / "input" / "Show"
    output_root = tmp_path / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir()
    source = input_root / "ep1.mkv"
    source.write_bytes(b"video")
    job = _job(str(output_root))
    versions = {
        "mapping_version": 11,
        "move_plan_version": 12,
        "move_state_revision": 13,
        "review_revision": 14,
    }

    ready_move = _move(
        10,
        1,
        "video",
        str(source),
        str(output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E01.mkv"),
    )
    ready = _apply_validation_response_payload(
        operation=_operation(90, "validate"),
        job=job,
        moves=[ready_move],
        scope="all_for_apply",
        versions=versions,
    )

    assert ready["status"] == "valid"
    assert ready["validation_id"] == 90
    assert ready["item_result_summary"] == {
        "checked": 1,
        "valid": 1,
        "invalid": 0,
        "warning": 0,
    }
    assert ready["blocking_items"] == []
    assert ready["affected_move_items"] == [10]

    existing_dest = (
        output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E02.mkv"
    )
    existing_dest.parent.mkdir(parents=True)
    existing_dest.write_bytes(b"existing")
    missing_source_move = _move(
        11,
        2,
        "video",
        str(input_root / "missing.mkv"),
        str(output_root / "旧动画 (2020) {tmdb-123}" / "S01" / "旧动画 S01E03.mkv"),
    )
    existing_dest_move = _move(
        12,
        3,
        "video",
        str(source),
        str(existing_dest),
    )
    invalid = _apply_validation_response_payload(
        operation=_operation(91, "validate"),
        job=job,
        moves=[missing_source_move, existing_dest_move],
        scope="all_for_apply",
        versions=versions,
    )

    assert invalid["status"] == "invalid"
    assert invalid["item_result_summary"] == {
        "checked": 2,
        "valid": 0,
        "invalid": 2,
        "warning": 0,
    }
    assert [item["code"] for item in invalid["blocking_items"]] == [
        "source_missing",
        "destination_exists",
    ]


def test_apply_validation_blocks_env_paths_without_resolved_pathrefs(tmp_path) -> None:  # noqa: ANN001
    input_root = tmp_path / "input" / "Show"
    output_root = tmp_path / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir()
    job = _job(str(output_root))
    env_move = _move(
        10,
        1,
        "video",
        str(input_root / ".env.local.mkv"),
        str(output_root / "Show" / "S01" / "Show S01E01.mkv"),
    )

    result = _apply_validation_response_payload(
        operation=_operation(92, "validate"),
        job=job,
        moves=[env_move],
        scope="all_for_apply",
        versions={
            "mapping_version": 11,
            "move_plan_version": 12,
            "move_state_revision": 13,
            "review_revision": 14,
        },
    )

    assert result["status"] == "invalid"
    assert result["blocking_items"] == [
        {
            "move_item_id": 10,
            "source_id": 1,
            "kind": "video",
            "code": "sensitive_env_path",
            "message": "sensitive .env paths cannot be validated",
            "destination": str(output_root / "Show" / "S01" / "Show S01E01.mkv"),
        }
    ]
    assert result["destination_preview"][0]["from"]["redacted"] is True
    assert result["destination_preview"][0]["from"]["normalized_abs_path"] is None


def test_apply_validation_selected_check_rejects_non_proposed_before_path_checks() -> None:
    payload = ApplyValidationRequest.model_validate(
        {
            "scope": "selected_check",
            "base_mapping_version": 1,
            "base_move_plan_version": 1,
            "base_move_state_revision": 1,
            "base_review_revision": 1,
            "selection": {"kind": "move_items", "ids": [10]},
        }
    )
    moved = _move(
        10,
        1,
        "video",
        "/input/Show/.env.local.mkv",
        "/output/Show/S01/Show S01E01.mkv",
        status="moved",
    )

    with pytest.raises(ApiServiceError) as exc_info:
        _resolve_apply_validation_scope_or_raise(payload, [moved])

    assert exc_info.value.code == "invalid_validation_selection"
    assert exc_info.value.details == {"not_proposed_move_item_ids": [10]}


def test_validate_apply_does_not_resolve_env_move_paths(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    output_root = tmp_path / "output"
    output_root.mkdir()
    job = _job(str(output_root))
    env_move = _move(
        10,
        1,
        "video",
        str(tmp_path / "input" / "Show" / ".env.local.mkv"),
        str(output_root / "Show" / "S01" / "Show S01E01.mkv"),
    )
    versions = _job_versions(
        job,
        [env_move],
        [],
        _validation_source_files(job, [env_move], None),
        snapshot={},
    )
    operation = _operation(100, "validate")

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            return operation

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return OperationRecord(
                id=operation.id,
                job_id=operation.job_id,
                operation_type=operation.operation_type,
                status=changes["status"],  # type: ignore[arg-type]
                idempotency_key=operation.idempotency_key,
                request_payload_hash=operation.request_payload_hash,
                base_versions=operation.base_versions,
                progress=operation.progress,
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=operation.error,
                poll_after_ms=operation.poll_after_ms,
                created_at=operation.created_at,
                started_at=operation.started_at,
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    response = ApiService(store=FakeStore(), worker=object()).validate_apply(  # type: ignore[arg-type]
        job.id,
        ApplyValidationRequest(
            scope="all_for_apply",
            base_mapping_version=versions["mapping_version"],
            base_move_plan_version=versions["move_plan_version"],
            base_move_state_revision=versions["move_state_revision"],
            base_review_revision=versions["review_revision"],
        ),
        idempotency_key="validate-env",
    )

    assert response is not None
    assert response.status == "invalid"
    assert response.blocking_items[0]["code"] == "sensitive_env_path"
    assert response.destination_preview[0]["from"]["redacted"] is True


def _apply_context(tmp_path):  # noqa: ANN001, ANN202
    series_dir = tmp_path / "input" / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    destination = output_root / "Show (2020) {tmdb-123}" / "S01" / "Show S01E01.mkv"
    source.parent.mkdir(parents=True)
    output_root.mkdir()
    source.write_bytes(b"video")
    job = _job(str(output_root))
    move = _move(10, 1, "video", str(source), str(destination))
    source_files = _validation_source_files(job, [move], None)
    before_validation_versions = _job_versions(
        job,
        [move],
        [],
        source_files,
        snapshot={},
    )
    validation_payload = _apply_validation_response_payload(
        operation=_operation(90, "validate"),
        job=job,
        moves=[move],
        scope="all_for_apply",
        versions=before_validation_versions,
    )
    validation_operation = OperationRecord(
        id=90,
        job_id=job.id,
        operation_type="validate",
        status="succeeded",
        idempotency_key="validate-1",
        request_payload_hash="hash-validate",
        base_versions={},
        progress={},
        result_summary={"response_payload": validation_payload},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )
    current_versions = _job_versions(
        job,
        [move],
        [validation_operation],
        source_files,
        snapshot={},
    )
    return SimpleNamespace(
        job=job,
        move=move,
        source=source,
        destination=destination,
        validation_operation=validation_operation,
        current_versions=current_versions,
    )


def _withdraw_context(tmp_path):  # noqa: ANN001, ANN202
    series_dir = tmp_path / "input" / "Show"
    output_root = tmp_path / "output"
    output_file = output_root / "Show (2020) {tmdb-123}" / "S01" / "Show S01E01.mkv"
    source = series_dir / "ep1.mkv"
    source.parent.mkdir(parents=True)
    output_file.parent.mkdir(parents=True)
    output_file.write_bytes(b"video")
    job = replace(
        _job(str(output_root)),
        status="succeeded",
        series_dir=str(series_dir),
        fail_path=str(tmp_path / "input" / "fail" / "Show"),
    )
    move = _move(10, 1, "video", str(source), str(output_file), status="moved")
    source_files = ApiService(store=object(), worker=object()).model_review_source_files(  # type: ignore[arg-type]
        job,
        [move],
        None,
    )
    current_versions = _job_versions(
        job,
        [move],
        [],
        source_files,
        snapshot={},
    )
    return SimpleNamespace(
        job=job,
        move=move,
        source=source,
        output_file=output_file,
        current_versions=current_versions,
    )


def _withdraw_check_operation(
    context,  # noqa: ANN001
    *,
    check_token: str = "check-token",
    checked_at: str | None = None,
    status: str = "eligible",
    token_status: str = "current",
    operation_id: int = 301,
) -> OperationRecord:
    checked = checked_at or _now_iso()
    check_payload = {
        "mode": "withdraw_to_review",
        "status": status,
        "checked_at": checked,
        "ttl_seconds": 300,
        "check_token": check_token if status == "eligible" else None,
        "checked_move_state_revision": context.current_versions[
            "move_state_revision"
        ],
        "token_status": token_status,
        "destination_preview": [],
        "item_result_summary": {
            "checked": 1,
            "eligible": 1 if status == "eligible" else 0,
            "ineligible": 0 if status == "eligible" else 1,
        },
        "missing_files": [],
        "incomplete_records": [],
        "target_conflicts": [],
        "identity_mismatches": [],
        "retryable_items": (
            [{"move_item_id": context.move.id}] if status == "eligible" else []
        ),
        "non_retryable_items": [],
        "affected_move_items": [context.move.id],
        "archive_restore_summary": {"archive_path": None, "has_archive_records": False},
    }
    return OperationRecord(
        id=operation_id,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-check",
        request_payload_hash="hash-check",
        base_versions={
            "review_revision": context.current_versions["review_revision"],
            "move_state_revision": context.current_versions["move_state_revision"],
        },
        progress={},
        result_summary={"response_payload": check_payload},
        error=None,
        poll_after_ms=1000,
        created_at=checked,
        started_at=None,
        finished_at=checked,
    )


def _fake_worker(job):  # noqa: ANN001, ANN202
    return SimpleNamespace(
        config=SimpleNamespace(
            watch_roots=(
                SimpleNamespace(
                    key=job.watch_root_key,
                    input_root=Path(job.series_dir).parent,
                    output_root=Path(job.output_root),
                ),
            )
        )
    )


def _review_safety_operation(
    context,  # noqa: ANN001
    *,
    token: str = "safety-token",
    checked_at: str | None = None,
    operation_id: int = 302,
) -> OperationRecord:
    checked = checked_at or _now_iso()
    worker = _fake_worker(context.job)
    digest = _consumer_chain_digest(worker)
    proof = _review_exclusion_proof(
        job=context.job,
        worker=worker,
        review_root=Path(context.job.fail_path).parent.parent / "review",
        consumer_chain_digest=digest,
        checked_at=checked,
    )
    payload = {
        "status": "verified",
        "checked_at": checked,
        "ttl_seconds": 300,
        "review_safety_check_token": token,
        "token_status": "current",
        "consumer_chain_digest": digest,
        "review_exclusion_proof": proof,
    }
    return OperationRecord(
        id=operation_id,
        job_id=context.job.id,
        operation_type="review_safety_check",
        status="succeeded",
        idempotency_key="review-safety-check",
        request_payload_hash="hash-safety",
        base_versions={},
        progress={},
        result_summary={"response_payload": payload},
        error=None,
        poll_after_ms=1000,
        created_at=checked,
        started_at=None,
        finished_at=checked,
    )


def _review_safety_operation_with_payload(
    operation: OperationRecord,
    payload: dict[str, object],
) -> OperationRecord:
    return replace(
        operation,
        result_summary={
            **operation.result_summary,
            "response_payload": payload,
        },
    )


def _stale_digest_review_safety_operation(
    context,  # noqa: ANN001
    *,
    token: str = "stale-safety-token",
) -> OperationRecord:
    operation = _review_safety_operation(context, token=token)
    payload = dict(operation.result_summary["response_payload"])
    payload["consumer_chain_digest"] = "stale-digest"
    return _review_safety_operation_with_payload(operation, payload)


def _withdraw_versions_after_check(
    context,  # noqa: ANN001
    *operations,  # noqa: ANN002
):  # noqa: ANN202
    source_files = ApiService(store=object(), worker=object()).model_review_source_files(  # type: ignore[arg-type]
        context.job,
        [context.move],
        None,
    )
    return _job_versions(
        context.job,
        [context.move],
        list(operations),
        source_files,
        snapshot={},
    )


def _reopen_context(tmp_path):  # noqa: ANN001, ANN202
    context = _withdraw_context(tmp_path)
    review_file = tmp_path / "input" / "review" / "1-401" / context.output_file.name
    review_file.parent.mkdir(parents=True)
    review_file.write_bytes(b"video")
    context.output_file.unlink()
    withdrawn_move = replace(
        context.move,
        status="withdrawn",
        applied_at="2026-07-06T00:00:03+00:00",
    )
    source_files = ApiService(store=object(), worker=object()).model_review_source_files(  # type: ignore[arg-type]
        context.job,
        [withdrawn_move],
        None,
    )
    current_versions = _job_versions(
        context.job,
        [withdrawn_move],
        [],
        source_files,
        snapshot={},
    )
    withdraw_result = OperationItemResultRecord(
        id=901,
        operation_id=401,
        subject_kind="move_item",
        subject_id=withdrawn_move.id,
        status="succeeded",
        path_event_id=801,
        reservation_id=501,
        from_path={"display_path": str(context.output_file)},
        to_path={"display_path": str(review_file)},
        identity_check="matched",
        error_code=None,
        error_message=None,
        retryable=False,
        retry_block_reason=None,
        result_revision=1,
        created_at="2026-07-06T00:00:03+00:00",
    )
    return SimpleNamespace(
        job=context.job,
        move=withdrawn_move,
        original_source=context.source,
        output_file=context.output_file,
        review_file=review_file,
        current_versions=current_versions,
        withdraw_result=withdraw_result,
    )


def test_apply_job_moves_files_and_records_operation_lifecycle(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    job = context.job
    move = context.move
    source = context.source
    destination = context.destination
    validation_operation = context.validation_operation
    current_versions = context.current_versions
    apply_operation = OperationRecord(
        id=91,
        job_id=job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-1",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.move = move
            self.job = job
            self.path_events: list[dict[str, object]] = []
            self.item_results: list[dict[str, object]] = []
            self.reservation_state: str | None = None

        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == self.job.id
            return self.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == self.job.id
            return [self.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == self.job.id
            return [validation_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == self.job.id
            return None

        def get_operation(self, job_id, operation_id):  # noqa: ANN001, ANN201
            assert job_id == self.job.id
            assert operation_id == validation_operation.id
            return validation_operation

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return OperationRecord(
                id=apply_operation.id,
                job_id=apply_operation.job_id,
                operation_type=apply_operation.operation_type,
                status=apply_operation.status,
                idempotency_key=apply_operation.idempotency_key,
                request_payload_hash=apply_operation.request_payload_hash,
                base_versions=kwargs["base_versions"],
                progress=apply_operation.progress,
                result_summary=kwargs["result_summary"],
                error=apply_operation.error,
                poll_after_ms=apply_operation.poll_after_ms,
                created_at=apply_operation.created_at,
                started_at=apply_operation.started_at,
                finished_at=apply_operation.finished_at,
            )

        def reserve_destinations(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["operation_id"] == apply_operation.id
            return [SimpleNamespace(id=501)]

        def mark_apply_items_applying(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["job_id"] == self.job.id
            assert kwargs["move_item_ids"] == [self.move.id]
            self.job = JobRecord(
                id=self.job.id,
                series_name=self.job.series_name,
                watch_root_key=self.job.watch_root_key,
                source_kind=self.job.source_kind,
                status="applying",
                tmdb_id=self.job.tmdb_id,
                video_moves_count=self.job.video_moves_count,
                subtitle_moves_count=self.job.subtitle_moves_count,
                created_at=self.job.created_at,
                updated_at=self.job.updated_at,
                started_at=self.job.started_at,
                finished_at=None,
                error_stage=None,
                error_message=None,
                series_dir=self.job.series_dir,
                output_root=self.job.output_root,
                archive_path=self.job.archive_path,
                fail_path=self.job.fail_path,
            )
            return [self.move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 701, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 701
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def update_job_move(self, move_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == self.move.id
            self.move = JobMoveRecord(
                id=self.move.id,
                job_id=self.move.job_id,
                src_id=self.move.src_id,
                kind=self.move.kind,
                src=self.move.src,
                dst=self.move.dst,
                status=changes.get("status", self.move.status),  # type: ignore[arg-type]
                error_message=changes.get("error_message", self.move.error_message),  # type: ignore[arg-type]
                applied_at=changes.get("applied_at", self.move.applied_at),  # type: ignore[arg-type]
                created_at=self.move.created_at,
                updated_at=self.move.updated_at,
            )
            return self.move

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            self.item_results.append(kwargs)
            return SimpleNamespace(**kwargs)

        def update_destination_reservations_state(self, **kwargs):  # noqa: ANN003, ANN201
            self.reservation_state = kwargs["state"]
            return []

        def complete_apply_operation_success(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["operation_id"] == apply_operation.id
            assert kwargs["job_id"] == self.job.id
            self.reservation_state = "committed"
            self.job = JobRecord(
                id=self.job.id,
                series_name=self.job.series_name,
                watch_root_key=self.job.watch_root_key,
                source_kind=self.job.source_kind,
                status="succeeded",
                tmdb_id=self.job.tmdb_id,
                video_moves_count=self.job.video_moves_count,
                subtitle_moves_count=self.job.subtitle_moves_count,
                created_at=self.job.created_at,
                updated_at=self.job.updated_at,
                started_at=self.job.started_at,
                finished_at="2026-07-06T00:00:03+00:00",
                error_stage=None,
                error_message=None,
                series_dir=self.job.series_dir,
                output_root=self.job.output_root,
                archive_path=self.job.archive_path,
                fail_path=self.job.fail_path,
            )
            return OperationRecord(
                id=apply_operation.id,
                job_id=apply_operation.job_id,
                operation_type=apply_operation.operation_type,
                status="succeeded",
                idempotency_key=apply_operation.idempotency_key,
                request_payload_hash=apply_operation.request_payload_hash,
                base_versions=apply_operation.base_versions,
                progress=apply_operation.progress,
                result_summary=kwargs["result_summary"],
                error=None,
                poll_after_ms=apply_operation.poll_after_ms,
                created_at=apply_operation.created_at,
                started_at=apply_operation.started_at,
                finished_at="2026-07-06T00:00:03+00:00",
            )

        def update_job(self, job_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert job_id == self.job.id
            self.job = JobRecord(
                id=self.job.id,
                series_name=self.job.series_name,
                watch_root_key=self.job.watch_root_key,
                source_kind=self.job.source_kind,
                status=changes.get("status", self.job.status),  # type: ignore[arg-type]
                tmdb_id=self.job.tmdb_id,
                video_moves_count=self.job.video_moves_count,
                subtitle_moves_count=self.job.subtitle_moves_count,
                created_at=self.job.created_at,
                updated_at=self.job.updated_at,
                started_at=self.job.started_at,
                finished_at=changes.get("finished_at", self.job.finished_at),  # type: ignore[arg-type]
                error_stage=changes.get("error_stage", self.job.error_stage),  # type: ignore[arg-type]
                error_message=changes.get("error_message", self.job.error_message),  # type: ignore[arg-type]
                series_dir=self.job.series_dir,
                output_root=self.job.output_root,
                archive_path=self.job.archive_path,
                fail_path=self.job.fail_path,
            )
            return self.job

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == apply_operation.id
            return OperationRecord(
                id=apply_operation.id,
                job_id=apply_operation.job_id,
                operation_type=apply_operation.operation_type,
                status=changes["status"],  # type: ignore[arg-type]
                idempotency_key=apply_operation.idempotency_key,
                request_payload_hash=apply_operation.request_payload_hash,
                base_versions=apply_operation.base_versions,
                progress=apply_operation.progress,
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                poll_after_ms=apply_operation.poll_after_ms,
                created_at=apply_operation.created_at,
                started_at=apply_operation.started_at,
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    store = FakeStore()
    response = ApiService(store=store, worker=object()).apply_job(  # type: ignore[arg-type]
        job.id,
        ApplyRequest(
            scope="all_validated",
            validation_id=validation_operation.id,
            base_mapping_version=current_versions["mapping_version"],
            base_move_plan_version=current_versions["move_plan_version"],
            base_move_state_revision=current_versions["move_state_revision"],
            base_review_revision=current_versions["review_revision"],
        ),
        idempotency_key="apply-1",
    )

    assert response is not None
    assert response.status == "succeeded"
    assert response.applied_move_item_count == 1
    assert not source.exists()
    assert destination.read_bytes() == b"video"
    assert store.move.status == "moved"
    assert store.job.status == "succeeded"
    assert store.reservation_state == "committed"
    assert store.path_events[0]["result"] == "succeeded"
    assert store.item_results[0]["status"] == "succeeded"
    assert store.item_results[0]["path_event_id"] == 701


def test_execute_apply_moves_rejects_reservation_count_mismatch(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    operation = OperationRecord(
        id=91,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-1",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    with pytest.raises(ApiServiceError) as exc_info:
        _execute_apply_moves(
            store=object(),  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[context.move],
            reservations=[],
            validation_id=context.validation_operation.id,
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.code == "apply_failed"
    assert "reservation count" in exc_info.value.message
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_execute_apply_moves_claim_failure_does_not_reset_unowned_moves(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    operation = OperationRecord(
        id=91,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-1",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def mark_apply_items_applying(self, **_kwargs):  # noqa: ANN003, ANN201
            raise ValueError("move item is already moved")

        def update_job_move(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("claim failure must not reset move ownership")

        def update_job(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("claim failure must not fail an unowned job")

        def create_path_event(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("claim failure must not create path events")

    with pytest.raises(ApiServiceError) as exc_info:
        _execute_apply_moves(
            store=FakeStore(),  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[context.move],
            reservations=[SimpleNamespace(id=501)],
            validation_id=context.validation_operation.id,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "apply_move_not_ready"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_execute_apply_moves_marks_current_path_event_failed_when_move_not_ready(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    context.source.unlink()
    operation = OperationRecord(
        id=91,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-1",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.path_events: list[dict[str, object]] = []
            self.item_results: list[dict[str, object]] = []
            self.move_updates: list[dict[str, object]] = []
            self.job_updates: list[dict[str, object]] = []

        def mark_apply_items_applying(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["job_id"] == context.job.id
            assert kwargs["move_item_ids"] == [context.move.id]
            return [context.move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 701, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 701
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            self.item_results.append(kwargs)
            return SimpleNamespace(**kwargs)

        def update_job_move(self, move_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == context.move.id
            self.move_updates.append(changes)
            return context.move

        def update_job(self, job_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert job_id == context.job.id
            self.job_updates.append(changes)
            return context.job

    store = FakeStore()
    with pytest.raises(ApiServiceError) as exc_info:
        _execute_apply_moves(
            store=store,  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[context.move],
            reservations=[SimpleNamespace(id=501)],
            validation_id=context.validation_operation.id,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "apply_move_not_ready"
    assert store.path_events[0]["result"] == "failed"
    assert store.item_results[0]["status"] == "failed"
    assert store.item_results[0]["error_code"] == "apply_move_not_ready"
    assert store.move_updates[-1]["status"] == "proposed"
    assert store.job_updates[-1]["status"] == "failed"
    assert not context.destination.exists()


def test_execute_apply_moves_rejects_env_path_before_stat(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    env_source = tmp_path / "input" / "Show" / ".env.secret"
    env_destination = (
        tmp_path
        / "output"
        / "Show (2020) {tmdb-123}"
        / "S01"
        / "Show S01E01.mkv"
    )
    env_move = _move(10, 1, "video", str(env_source), str(env_destination))
    operation = OperationRecord(
        id=91,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-1",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )
    original_stat = Path.stat

    def guarded_stat(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"stat called for sensitive path: {self}")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    class FakeStore:
        def __init__(self) -> None:
            self.path_events: list[dict[str, object]] = []
            self.item_results: list[dict[str, object]] = []

        def mark_apply_items_applying(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["move_item_ids"] == [env_move.id]
            return [env_move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 701, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 701
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            self.item_results.append(kwargs)
            return SimpleNamespace(**kwargs)

        def update_job_move(self, move_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == env_move.id
            return env_move

        def update_job(self, job_id, **_changes):  # noqa: ANN001, ANN003, ANN201
            assert job_id == context.job.id
            return context.job

    store = FakeStore()
    with pytest.raises(ApiServiceError) as exc_info:
        _execute_apply_moves(
            store=store,  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[env_move],
            reservations=[SimpleNamespace(id=501)],
            validation_id=context.validation_operation.id,
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "apply_move_not_ready"
    assert store.path_events[0]["pre_identity"]["identity_strength"] == "not_checked"
    assert store.path_events[0]["result"] == "failed"
    assert store.item_results[0]["error_code"] == "apply_move_not_ready"


def test_destination_reservation_spec_rejects_env_path_before_resolve(
    tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    context = _apply_context(tmp_path)
    env_move = _move(
        10,
        1,
        "video",
        str(tmp_path / "input" / "Show" / ".env.secret"),
        str(context.destination),
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    with pytest.raises(ApiServiceError) as exc_info:
        _destination_reservation_spec(context.job, env_move)

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "sensitive_env_path"


def test_apply_job_final_validation_failure_does_not_move_files(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    context.destination.parent.mkdir(parents=True)
    context.destination.write_bytes(b"existing")

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.validation_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_llm_failures(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_operation(self, job_id, operation_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            assert operation_id == context.validation_operation.id
            return context.validation_operation

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("apply operation should not be created")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            ApplyRequest(
                scope="all_validated",
                validation_id=context.validation_operation.id,
                base_mapping_version=context.current_versions["mapping_version"],
                base_move_plan_version=context.current_versions["move_plan_version"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                base_review_revision=context.current_versions["review_revision"],
            ),
            idempotency_key="apply-final-invalid",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "apply_final_validation_failed"
    assert context.source.read_bytes() == b"video"
    assert context.destination.read_bytes() == b"existing"


def test_apply_job_stale_versions_do_not_move_files(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.validation_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def get_operation(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("validation should not be read after stale versions")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            ApplyRequest(
                scope="all_validated",
                validation_id=context.validation_operation.id,
                base_mapping_version=context.current_versions["mapping_version"],
                base_move_plan_version=context.current_versions["move_plan_version"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                base_review_revision=context.current_versions["review_revision"] + 1,
            ),
            idempotency_key="apply-stale",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_versions"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_apply_job_idempotency_conflict_does_not_move_files(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    existing = OperationRecord(
        id=200,
        job_id=context.job.id,
        operation_type="apply",
        status="succeeded",
        idempotency_key="apply-conflict",
        request_payload_hash="different",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("moves should not be read after idempotency conflict")

    with pytest.raises(IdempotencyConflictError):
        ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            ApplyRequest(
                scope="all_validated",
                validation_id=context.validation_operation.id,
                base_mapping_version=context.current_versions["mapping_version"],
                base_move_plan_version=context.current_versions["move_plan_version"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                base_review_revision=context.current_versions["review_revision"],
            ),
            idempotency_key="apply-conflict",
        )

    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_apply_job_replays_existing_success_without_moving_files(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    payload = ApplyRequest(
        scope="all_validated",
        validation_id=context.validation_operation.id,
        base_mapping_version=context.current_versions["mapping_version"],
        base_move_plan_version=context.current_versions["move_plan_version"],
        base_move_state_revision=context.current_versions["move_state_revision"],
        base_review_revision=context.current_versions["review_revision"],
    )
    response_payload = {
        "operation_id": 200,
        "status": "succeeded",
        "validation_id": context.validation_operation.id,
        "applied_move_item_count": 1,
        "item_result_summary": {"succeeded": 1, "failed": 0, "total": 1},
        "moved_items": [{"move_item_id": context.move.id}],
    }
    existing = OperationRecord(
        id=200,
        job_id=context.job.id,
        operation_type="apply",
        status="succeeded",
        idempotency_key="apply-replay",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={"response_payload": response_payload},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("replay should not read moves")

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("replay should not create an operation")

    response = ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
        context.job.id,
        payload,
        idempotency_key="apply-replay",
    )

    assert response is not None
    assert response.operation_id == 200
    assert response.status == "succeeded"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_apply_job_running_idempotency_key_does_not_execute(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    payload = ApplyRequest(
        scope="all_validated",
        validation_id=context.validation_operation.id,
        base_mapping_version=context.current_versions["mapping_version"],
        base_move_plan_version=context.current_versions["move_plan_version"],
        base_move_state_revision=context.current_versions["move_state_revision"],
        base_review_revision=context.current_versions["review_revision"],
    )
    existing = OperationRecord(
        id=200,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-running",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("running replay should not read moves")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="apply-running",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_apply_job_missing_validation_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.validation_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def get_operation(self, job_id, operation_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            assert operation_id == 999
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("invalid validation should not create operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            ApplyRequest(
                scope="all_validated",
                validation_id=999,
                base_mapping_version=context.current_versions["mapping_version"],
                base_move_plan_version=context.current_versions["move_plan_version"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                base_review_revision=context.current_versions["review_revision"],
            ),
            idempotency_key="apply-missing-validation",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "invalid_apply_validation"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_apply_job_destination_reservation_conflict_is_409(tmp_path) -> None:  # noqa: ANN001
    context = _apply_context(tmp_path)
    apply_operation = OperationRecord(
        id=91,
        job_id=context.job.id,
        operation_type="apply",
        status="running",
        idempotency_key="apply-reserved",
        request_payload_hash="hash-apply",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.updated_operation: OperationRecord | None = None

        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.validation_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def get_operation(self, job_id, operation_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            assert operation_id == context.validation_operation.id
            return context.validation_operation

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return OperationRecord(
                id=apply_operation.id,
                job_id=apply_operation.job_id,
                operation_type=apply_operation.operation_type,
                status=apply_operation.status,
                idempotency_key=apply_operation.idempotency_key,
                request_payload_hash=apply_operation.request_payload_hash,
                base_versions=kwargs["base_versions"],
                progress=apply_operation.progress,
                result_summary=kwargs["result_summary"],
                error=apply_operation.error,
                poll_after_ms=apply_operation.poll_after_ms,
                created_at=apply_operation.created_at,
                started_at=apply_operation.started_at,
                finished_at=apply_operation.finished_at,
            )

        def reserve_destinations(self, **_kwargs):  # noqa: ANN003, ANN201
            raise DestinationReservationConflictError(
                "destination_reserved",
                str(context.destination),
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == apply_operation.id
            self.updated_operation = OperationRecord(
                id=apply_operation.id,
                job_id=apply_operation.job_id,
                operation_type=apply_operation.operation_type,
                status=changes["status"],  # type: ignore[arg-type]
                idempotency_key=apply_operation.idempotency_key,
                request_payload_hash=apply_operation.request_payload_hash,
                base_versions=apply_operation.base_versions,
                progress=apply_operation.progress,
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                poll_after_ms=apply_operation.poll_after_ms,
                created_at=apply_operation.created_at,
                started_at=apply_operation.started_at,
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )
            return self.updated_operation

    store = FakeStore()
    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=store, worker=object()).apply_job(  # type: ignore[arg-type]
            context.job.id,
            ApplyRequest(
                scope="all_validated",
                validation_id=context.validation_operation.id,
                base_mapping_version=context.current_versions["mapping_version"],
                base_move_plan_version=context.current_versions["move_plan_version"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                base_review_revision=context.current_versions["review_revision"],
            ),
            idempotency_key="apply-reserved",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "destination_reserved"
    assert store.updated_operation is not None
    assert store.updated_operation.status == "failed"
    assert store.updated_operation.result_summary["response_payload"]["status"] == "failed"
    assert context.source.read_bytes() == b"video"
    assert not context.destination.exists()


def test_rollback_eligibility_returns_withdraw_token_for_moved_items(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-check",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return replace(
                operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    response = ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
        context.job.id,
        RollbackEligibilityRequest(
            mode="withdraw_to_review",
            scope=SelectionScope(kind="all"),
            base_review_revision=context.current_versions["review_revision"],
            base_move_state_revision=context.current_versions["move_state_revision"],
        ),
        idempotency_key="rollback-check",
    )

    assert response is not None
    assert response.status == "eligible"
    assert response.token_status == "current"
    assert response.check_token
    assert response.affected_move_items == [context.move.id]
    assert "/review/" in response.destination_preview[0]["to"]["display_path"]


def test_rollback_eligibility_running_idempotency_key_does_not_recompute(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    payload = RollbackEligibilityRequest(
        mode="withdraw_to_review",
        scope=SelectionScope(kind="all"),
        base_review_revision=context.current_versions["review_revision"],
        base_move_state_revision=context.current_versions["move_state_revision"],
    )
    existing = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="running",
        idempotency_key="rollback-running",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("running eligibility replay should not read moves")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="rollback-running",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"


def test_rollback_eligibility_create_conflict_running_does_not_recompute(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    payload = RollbackEligibilityRequest(
        mode="withdraw_to_review",
        scope=SelectionScope(kind="all"),
        base_review_revision=context.current_versions["review_revision"],
        base_move_state_revision=context.current_versions["move_state_revision"],
    )
    existing = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="running",
        idempotency_key="rollback-race",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def update_operation(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("non-owner rollback check must not update operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="rollback-race",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"


def test_operation_response_redacts_check_and_claim_tokens() -> None:
    operation = OperationRecord(
        id=500,
        job_id=1,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="op",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={
            "claim_token": "claim-secret",
            "response_payload": {
                "check_token": "check-secret",
                "review_safety_check_token": "review-safety-secret",
                "nested": {"source_gate_check_token": "source-secret"},
            },
        },
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    response = ApiService.model_operation(operation)

    assert response.result_summary["claim_token"] is None
    payload = response.result_summary["response_payload"]
    assert payload["check_token"] is None
    assert payload["review_safety_check_token"] is None
    assert payload["nested"]["source_gate_check_token"] is None


def test_execute_withdraw_moves_moves_output_to_review_and_records_lifecycle(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    destination = tmp_path / "input" / "review" / "1-301" / context.output_file.name
    operation = OperationRecord(
        id=401,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-1",
        request_payload_hash="hash-withdraw",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.move = context.move
            self.path_events: list[dict[str, object]] = []
            self.item_results: list[dict[str, object]] = []
            self.completed = False

        def mark_withdraw_items_applying(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["job_id"] == context.job.id
            assert kwargs["move_item_ids"] == [context.move.id]
            return [context.move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 801, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 801
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def update_job_move(self, move_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == context.move.id
            self.move = replace(
                self.move,
                status=changes.get("status", self.move.status),  # type: ignore[arg-type]
                error_message=changes.get("error_message", self.move.error_message),  # type: ignore[arg-type]
                applied_at=changes.get("applied_at", self.move.applied_at),  # type: ignore[arg-type]
            )
            return self.move

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            self.item_results.append(kwargs)
            return SimpleNamespace(**kwargs)

        def complete_withdraw_operation_success(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["operation_id"] == operation.id
            assert kwargs["job_id"] == context.job.id
            self.completed = True
            return replace(
                operation,
                status="succeeded",
                result_summary=kwargs["result_summary"],
                finished_at="2026-07-06T00:00:03+00:00",
            )

    store = FakeStore()
    response = _execute_withdraw_moves(
        store=store,  # type: ignore[arg-type]
        job=context.job,
        operation=operation,
        moves=[context.move],
        destinations={context.move.id: destination},
        reservations=[SimpleNamespace(id=701)],
    )

    assert response["status"] == "succeeded"
    assert response["withdrawn_move_item_count"] == 1
    assert store.move.status == "withdrawn"
    assert store.completed is True
    assert not context.output_file.exists()
    assert destination.read_bytes() == b"video"
    assert store.path_events[0]["result"] == "succeeded"
    assert store.item_results[0]["status"] == "succeeded"


def test_execute_withdraw_moves_claim_failure_does_not_reset_unowned_moves(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    destination = tmp_path / "input" / "review" / "1-301" / context.output_file.name
    operation = OperationRecord(
        id=401,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-1",
        request_payload_hash="hash-withdraw",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def mark_withdraw_items_applying(self, **_kwargs):  # noqa: ANN003, ANN201
            raise ValueError("move item is already withdrawn")

        def update_job_move(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("claim failure must not reset unowned move")

        def update_job(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("claim failure must not fail unowned job")

        def create_path_event(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("claim failure must not create path event")

    with pytest.raises(ApiServiceError) as exc_info:
        _execute_withdraw_moves(
            store=FakeStore(),  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[context.move],
            destinations={context.move.id: destination},
            reservations=[SimpleNamespace(id=701)],
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "withdraw_move_not_ready"
    assert context.output_file.read_bytes() == b"video"
    assert not destination.exists()


def test_execute_withdraw_moves_marks_current_event_failed_when_not_ready(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    context.output_file.unlink()
    destination = tmp_path / "input" / "review" / "1-301" / context.output_file.name
    operation = OperationRecord(
        id=401,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-1",
        request_payload_hash="hash-withdraw",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.path_events: list[dict[str, object]] = []
            self.item_results: list[dict[str, object]] = []
            self.move_updates: list[dict[str, object]] = []

        def mark_withdraw_items_applying(self, **_kwargs):  # noqa: ANN003, ANN201
            return [context.move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 801, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 801
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            self.item_results.append(kwargs)
            return SimpleNamespace(**kwargs)

        def update_job_move(self, move_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == context.move.id
            self.move_updates.append(changes)
            return context.move

        def update_job(self, job_id, **_changes):  # noqa: ANN001, ANN003, ANN201
            assert job_id == context.job.id
            return context.job

    store = FakeStore()
    with pytest.raises(ApiServiceError) as exc_info:
        _execute_withdraw_moves(
            store=store,  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[context.move],
            destinations={context.move.id: destination},
            reservations=[SimpleNamespace(id=701)],
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "withdraw_move_not_ready"
    assert store.path_events[0]["result"] == "failed"
    assert store.path_events[0]["pre_identity"]["identity_strength"] == "not_checked"
    assert store.item_results[0]["status"] == "failed"
    assert store.move_updates[-1]["status"] == "moved"


def test_withdraw_env_path_rejected_before_stat_and_resolve(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    env_output = tmp_path / "output" / ".env.secret"
    env_move = replace(context.move, dst=str(env_output))
    destination = tmp_path / "input" / "review" / "1-301" / ".env.secret"
    operation = OperationRecord(
        id=401,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-1",
        request_payload_hash="hash-withdraw",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:02+00:00",
        started_at="2026-07-06T00:00:02+00:00",
        finished_at=None,
    )
    original_stat = Path.stat
    original_resolve = Path.resolve

    def guarded_stat(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"stat called for sensitive path: {self}")
        return original_stat(self, *args, **kwargs)

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    class FakeStore:
        def __init__(self) -> None:
            self.path_events: list[dict[str, object]] = []

        def mark_withdraw_items_applying(self, **_kwargs):  # noqa: ANN003, ANN201
            return [env_move]

        def create_path_event(self, **kwargs):  # noqa: ANN003, ANN201
            event = {"id": 801, **kwargs}
            self.path_events.append(event)
            return event

        def update_path_event(self, event_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert event_id == 801
            self.path_events[-1].update(changes)
            return self.path_events[-1]

        def create_operation_item_result(self, **kwargs):  # noqa: ANN003, ANN201
            return SimpleNamespace(**kwargs)

        def update_job_move(self, move_id, **_changes):  # noqa: ANN001, ANN003, ANN201
            assert move_id == env_move.id
            return env_move

        def update_job(self, job_id, **_changes):  # noqa: ANN001, ANN003, ANN201
            assert job_id == context.job.id
            return context.job

    with pytest.raises(ApiServiceError) as exc_info:
        _execute_withdraw_moves(
            store=FakeStore(),  # type: ignore[arg-type]
            job=context.job,
            operation=operation,
            moves=[env_move],
            destinations={env_move.id: destination},
            reservations=[SimpleNamespace(id=701)],
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "withdraw_move_not_ready"

    with pytest.raises(ApiServiceError) as reservation_exc:
        _withdraw_destination_reservation_spec(context.job, env_move, destination)

    assert reservation_exc.value.status_code == 422
    assert reservation_exc.value.code == "sensitive_env_path"


def test_rollback_eligibility_does_not_resolve_env_paths(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    env_move = replace(
        context.move,
        src=str(tmp_path / "input" / "Show" / ".env.secret"),
        dst=str(tmp_path / "output" / ".env.output"),
    )
    source_files = _validation_source_files(context.job, [env_move], None)
    versions = _job_versions(
        context.job,
        [env_move],
        [],
        source_files,
        snapshot={},
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-env",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return replace(
                operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    response = ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
        context.job.id,
        RollbackEligibilityRequest(
            mode="withdraw_to_review",
            scope=SelectionScope(kind="all"),
            base_review_revision=versions["review_revision"],
            base_move_state_revision=versions["move_state_revision"],
        ),
        idempotency_key="rollback-env",
    )

    assert response is not None
    assert response.status == "ineligible"
    assert response.non_retryable_items[0]["retry_block_reason"] == "sensitive_env_path"


def test_rollback_eligibility_does_not_resolve_env_review_root(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    env_job = replace(
        context.job,
        series_dir=str(tmp_path / ".envparent" / "Show"),
        fail_path=None,
    )
    source_files = _validation_source_files(env_job, [context.move], None)
    versions = _job_versions(
        env_job,
        [context.move],
        [],
        source_files,
        snapshot={},
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    operation = OperationRecord(
        id=301,
        job_id=env_job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-env-review-root",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == env_job.id
            return env_job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == env_job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == env_job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == env_job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return replace(
                operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    response = ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
        env_job.id,
        RollbackEligibilityRequest(
            mode="withdraw_to_review",
            scope=SelectionScope(kind="all"),
            base_review_revision=versions["review_revision"],
            base_move_state_revision=versions["move_state_revision"],
        ),
        idempotency_key="rollback-env-review-root",
    )

    assert response is not None
    assert response.status == "ineligible"
    assert response.non_retryable_items[0]["retry_block_reason"] == "sensitive_env_path"


def test_withdraw_job_does_not_resolve_env_paths_before_final_check(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    env_move = replace(
        context.move,
        src=str(tmp_path / "input" / "Show" / ".env.secret"),
        dst=str(tmp_path / "output" / ".env.output"),
    )
    source_files = _validation_source_files(context.job, [env_move], None)
    versions_before_check = _job_versions(
        context.job,
        [env_move],
        [],
        source_files,
        snapshot={},
    )
    check_token = "check-token"
    check_operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-check",
        request_payload_hash="hash-check",
        base_versions={
            "review_revision": versions_before_check["review_revision"],
            "move_state_revision": versions_before_check["move_state_revision"],
        },
        progress={},
        result_summary={
            "response_payload": {
                "mode": "withdraw_to_review",
                "status": "eligible",
                "checked_at": _now_iso(),
                "ttl_seconds": 300,
                "check_token": check_token,
                "checked_move_state_revision": versions_before_check[
                    "move_state_revision"
                ],
                "token_status": "current",
                "destination_preview": [],
                "item_result_summary": {"checked": 1, "eligible": 1, "ineligible": 0},
                "missing_files": [],
                "incomplete_records": [],
                "target_conflicts": [],
                "identity_mismatches": [],
                "retryable_items": [{"move_item_id": env_move.id}],
                "non_retryable_items": [],
                "affected_move_items": [env_move.id],
                "archive_restore_summary": {
                    "archive_path": None,
                    "has_archive_records": False,
                },
            }
        },
        error=None,
        poll_after_ms=1000,
        created_at=_now_iso(),
        started_at=None,
        finished_at=_now_iso(),
    )
    safety_operation = _review_safety_operation(context)
    versions_after_check = _job_versions(
        context.job,
        [env_move],
        [check_operation, safety_operation],
        source_files,
        snapshot={},
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [check_operation, safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("env final check should not create withdraw operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=versions_after_check["move_state_revision"],
                check_token=check_token,
                review_safety_check_token="safety-token",
            ),
            idempotency_key="withdraw-env",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "withdraw_final_check_failed"


def test_review_does_not_resolve_env_move_paths(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    env_move = replace(
        context.move,
        src=str(tmp_path / "input" / "Show" / ".env.secret"),
        dst=str(tmp_path / "output" / ".env.output"),
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_llm_failures(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

    response = ApiService(store=FakeStore(), worker=object()).review(context.job.id)  # type: ignore[arg-type]

    assert response is not None
    assert response.move_items[0].source_path.redacted is True
    assert response.move_items[0].current_path.redacted is True


def test_safety_does_not_resolve_env_job_roots(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    job = replace(
        _job(str(tmp_path / ".env.output")),
        series_dir=str(tmp_path / "input" / ".env.Show"),
        archive_path=str(tmp_path / ".env.archive" / "Show"),
        fail_path=str(tmp_path / "input" / "fail" / ".env.Show"),
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return job

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return []

    response = ApiService(store=FakeStore(), worker=_fake_worker(job)).safety(job.id)  # type: ignore[arg-type]

    assert response is not None
    assert response.roots["source"]["redacted"] is True
    assert response.roots["output"]["redacted"] is True
    assert response.roots["archive"]["redacted"] is True
    assert response.roots["fail"]["redacted"] is True
    assert response.review_exclusion_proof["status"] == "verified"


def test_safety_does_not_resolve_env_watch_roots(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    job = replace(
        _job(str(tmp_path / "output")),
        series_dir=str(tmp_path / "input" / "Show"),
    )
    worker = SimpleNamespace(
        config=SimpleNamespace(
            watch_roots=(
                SimpleNamespace(
                    key=job.watch_root_key,
                    input_root=tmp_path / ".env.input",
                    output_root=tmp_path / ".env.output",
                ),
            )
        )
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return job

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == job.id
            return []

    response = ApiService(store=FakeStore(), worker=worker).safety(job.id)  # type: ignore[arg-type]

    assert response is not None
    assert response.review_exclusion_proof["status"] == "failed"
    assert (
        response.review_exclusion_proof["reason"]
        == "review_root_not_sibling_of_watch_root"
    )


def test_unsupported_restore_eligibility_does_not_probe_env_paths(
    tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    context = _withdraw_context(tmp_path)
    env_move = replace(
        context.move,
        src=str(tmp_path / "input" / "Show" / ".env.secret"),
        dst=str(tmp_path / "output" / ".env.output"),
    )
    source_files = _validation_source_files(context.job, [env_move], None)
    versions = _job_versions(
        context.job,
        [env_move],
        [],
        source_files,
        snapshot={},
    )
    original_resolve = Path.resolve
    original_exists = Path.exists

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    def guarded_exists(self):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"exists called for sensitive path: {self}")
        return original_exists(self)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(Path, "exists", guarded_exists)

    operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="running",
        idempotency_key="rollback-restore-env",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return replace(
                operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    response = ApiService(store=FakeStore(), worker=object()).rollback_eligibility(  # type: ignore[arg-type]
        context.job.id,
        RollbackEligibilityRequest(
            mode="restore_moved_to_source",
            scope=SelectionScope(kind="all"),
            base_review_revision=versions["review_revision"],
            base_move_state_revision=versions["move_state_revision"],
        ),
        idempotency_key="rollback-restore-env",
    )

    assert response is not None
    assert response.status == "ineligible"
    assert response.missing_files[0]["reason"] == "sensitive_env_path"


def test_rematch_preview_does_not_resolve_env_current_move_paths(
    tmp_path, monkeypatch  # noqa: ANN001
) -> None:
    context = _withdraw_context(tmp_path)
    env_move = replace(
        context.move,
        src=str(tmp_path / "input" / "Show" / ".env.secret"),
        dst=str(tmp_path / "output" / ".env.output"),
        status="proposed",
    )
    source_files = _validation_source_files(context.job, [env_move], None)
    versions = _job_versions(
        context.job,
        [env_move],
        [],
        source_files,
        snapshot={},
    )
    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    preview_move = SimpleNamespace(
        src=Path(env_move.src),
        dst=tmp_path / "output" / "Show S01E01.mkv",
        kind="video",
        src_id=env_move.src_id,
    )
    preview = SimpleNamespace(
        plan=SimpleNamespace(
            tmdb_id=123,
            series_name_zh_cn="旧动画",
            moves=[preview_move],
        ),
        tmdb_details=SimpleNamespace(
            original_name="Show",
            year=2020,
            seasons=[],
        ),
        tmdb_candidates=(),
        specials_zh=None,
        season_episode_counts={1: 1},
        mapping=SimpleNamespace(
            items=[
                SimpleNamespace(
                    video_id=env_move.src_id,
                    season=1,
                    episode_start=1,
                    episode_end=1,
                    subtitle_ids=[],
                )
            ]
        ),
    )
    operation = OperationRecord(
        id=601,
        job_id=context.job.id,
        operation_type="rematch_preview",
        status="running",
        idempotency_key="rematch-env",
        request_payload_hash="hash",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [env_move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == operation.id
            return replace(
                operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )

    class FakeWorker:
        def build_rematch_preview(self, job, tmdb_id):  # noqa: ANN001, ANN201
            assert job.id == context.job.id
            assert tmdb_id == 123
            return preview

    response = ApiService(store=FakeStore(), worker=FakeWorker()).rematch_preview(  # type: ignore[arg-type]
        context.job.id,
        RematchPreviewRequest(
            tmdb_id=123,
            base_review_revision=versions["review_revision"],
        ),
        idempotency_key="rematch-env",
    )

    assert response is not None
    assert response.move_items[0]["source_path"]["redacted"] is True
    assert response.move_items[0]["current_planned_destination_path"]["redacted"] is True


def test_rematch_preview_running_idempotency_key_does_not_recompute(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    payload = RematchPreviewRequest(
        tmdb_id=123,
        base_review_revision=context.current_versions["review_revision"],
    )
    existing = OperationRecord(
        id=601,
        job_id=context.job.id,
        operation_type="rematch_preview",
        status="running",
        idempotency_key="rematch-running",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("running rematch replay should not read moves")

    class FakeWorker:
        def build_rematch_preview(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("running rematch replay should not call worker")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=FakeWorker()).rematch_preview(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="rematch-running",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"


def test_rematch_preview_create_conflict_running_does_not_call_worker(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    payload = RematchPreviewRequest(
        tmdb_id=123,
        base_review_revision=context.current_versions["review_revision"],
    )
    existing = OperationRecord(
        id=601,
        job_id=context.job.id,
        operation_type="rematch_preview",
        status="running",
        idempotency_key="rematch-race",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def update_operation(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("non-owner rematch preview must not update operation")

    class FakeWorker:
        def build_rematch_preview(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("non-owner rematch preview must not call worker")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=FakeWorker()).rematch_preview(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="rematch-race",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"


def test_withdraw_job_final_check_failure_does_not_move_files(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_token = "check-token"
    check_payload = {
        "mode": "withdraw_to_review",
        "status": "eligible",
        "checked_at": _now_iso(),
        "ttl_seconds": 300,
        "check_token": check_token,
        "checked_move_state_revision": context.current_versions[
            "move_state_revision"
        ],
        "token_status": "current",
        "destination_preview": [],
        "item_result_summary": {"checked": 1, "eligible": 1, "ineligible": 0},
        "missing_files": [],
        "incomplete_records": [],
        "target_conflicts": [],
        "identity_mismatches": [],
        "retryable_items": [{"move_item_id": context.move.id}],
        "non_retryable_items": [],
        "affected_move_items": [context.move.id],
        "archive_restore_summary": {"archive_path": None, "has_archive_records": False},
    }
    check_operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-check",
        request_payload_hash="hash-check",
        base_versions={
            "review_revision": context.current_versions["review_revision"],
            "move_state_revision": context.current_versions["move_state_revision"],
        },
        progress={},
        result_summary={"response_payload": check_payload},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )
    safety_operation = _review_safety_operation(context)
    source_files = ApiService(store=object(), worker=object()).model_review_source_files(  # type: ignore[arg-type]
        context.job,
        [context.move],
        None,
    )
    versions_after_check = _job_versions(
        context.job,
        [context.move],
        [check_operation, safety_operation],
        source_files,
        snapshot={},
    )
    review_destination = (
        tmp_path / "input" / "review" / "1-301" / context.output_file.name
    )
    review_destination.parent.mkdir(parents=True)
    review_destination.write_bytes(b"existing")

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [check_operation, safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("withdraw operation should not be created")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                check_token=check_token,
                review_safety_check_token="safety-token",
            ),
            idempotency_key="withdraw-final-invalid",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "withdraw_final_check_failed"
    assert context.output_file.read_bytes() == b"video"
    assert review_destination.read_bytes() == b"existing"


def test_withdraw_job_replays_existing_success_without_moving_files(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    versions_after_check = _withdraw_versions_after_check(context, check_operation)
    payload = WithdrawRequest(
        mode="withdraw_to_review",
        scope=SelectionScope(kind="all"),
        base_review_revision=versions_after_check["review_revision"],
        base_move_state_revision=context.current_versions["move_state_revision"],
        check_token="check-token",
        review_safety_check_token="safety-token",
    )
    response_payload = {
        "operation_id": 500,
        "status": "succeeded",
        "mode": "withdraw_to_review",
        "withdrawn_move_item_count": 1,
        "item_result_summary": {"succeeded": 1, "failed": 0, "total": 1},
        "withdrawn_items": [{"move_item_id": context.move.id}],
    }
    existing = OperationRecord(
        id=500,
        job_id=context.job.id,
        operation_type="withdraw",
        status="succeeded",
        idempotency_key="withdraw-replay",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={"response_payload": response_payload},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("withdraw replay should not read moves")

    response = ApiService(store=FakeStore(), worker=object()).withdraw_job(  # type: ignore[arg-type]
        context.job.id,
        payload,
        idempotency_key="withdraw-replay",
    )

    assert response is not None
    assert response.status == "succeeded"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_running_idempotency_key_does_not_execute(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    versions_after_check = _withdraw_versions_after_check(context, check_operation)
    payload = WithdrawRequest(
        mode="withdraw_to_review",
        scope=SelectionScope(kind="all"),
        base_review_revision=versions_after_check["review_revision"],
        base_move_state_revision=context.current_versions["move_state_revision"],
        check_token="check-token",
        review_safety_check_token="safety-token",
    )
    existing = OperationRecord(
        id=500,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-running",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("running withdraw replay should not read moves")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="withdraw-running",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_idempotency_conflict_does_not_execute(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    versions_after_check = _withdraw_versions_after_check(context, check_operation)
    existing = OperationRecord(
        id=500,
        job_id=context.job.id,
        operation_type="withdraw",
        status="succeeded",
        idempotency_key="withdraw-conflict",
        request_payload_hash="different",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at=None,
        finished_at="2026-07-06T00:00:01+00:00",
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("conflicting withdraw replay should not read moves")

    with pytest.raises(IdempotencyConflictError):
        ApiService(store=FakeStore(), worker=object()).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                check_token="check-token",
                review_safety_check_token="safety-token",
            ),
            idempotency_key="withdraw-conflict",
        )

    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_stale_versions_do_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("stale withdraw should not create operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=context.current_versions["review_revision"] + 1,
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                check_token="check-token",
                review_safety_check_token="safety-token",
            ),
            idempotency_key="withdraw-stale",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_versions"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_invalid_review_safety_token_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    safety_operation = _review_safety_operation(context)
    versions_after_check = _withdraw_versions_after_check(
        context,
        check_operation,
        safety_operation,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [check_operation, safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("invalid safety token should not create withdraw")

        def reserve_destinations(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("invalid safety token should not reserve destinations")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=versions_after_check["move_state_revision"],
                check_token="check-token",
                review_safety_check_token="wrong-safety-token",
            ),
            idempotency_key="withdraw-invalid-safety",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "invalid_review_safety_check"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_expired_review_safety_token_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    safety_operation = _review_safety_operation(
        context,
        token="expired-safety-token",
        checked_at="2000-01-01T00:00:00+00:00",
    )
    versions_after_check = _withdraw_versions_after_check(
        context,
        check_operation,
        safety_operation,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [check_operation, safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("expired safety token should not create withdraw")

        def reserve_destinations(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("expired safety token should not reserve destinations")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=versions_after_check["move_state_revision"],
                check_token="check-token",
                review_safety_check_token="expired-safety-token",
            ),
            idempotency_key="withdraw-expired-safety",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_review_safety_check"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_job_destination_reservation_conflict_is_409(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_operation = _withdraw_check_operation(context)
    safety_operation = _review_safety_operation(context)
    versions_after_check = _withdraw_versions_after_check(
        context,
        check_operation,
        safety_operation,
    )
    withdraw_operation = OperationRecord(
        id=500,
        job_id=context.job.id,
        operation_type="withdraw",
        status="running",
        idempotency_key="withdraw-reserved",
        request_payload_hash="hash-withdraw",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:00+00:00",
        started_at="2026-07-06T00:00:00+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.updated_operation: OperationRecord | None = None

        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [check_operation, safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                withdraw_operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def reserve_destinations(self, **_kwargs):  # noqa: ANN003, ANN201
            raise DestinationReservationConflictError(
                "destination_reserved",
                "review/path",
            )

        def update_operation(self, operation_id, **changes):  # noqa: ANN001, ANN003, ANN201
            assert operation_id == withdraw_operation.id
            self.updated_operation = replace(
                withdraw_operation,
                status=changes["status"],  # type: ignore[arg-type]
                result_summary=changes["result_summary"],  # type: ignore[arg-type]
                error=changes.get("error"),  # type: ignore[arg-type]
                finished_at=changes["finished_at"],  # type: ignore[arg-type]
            )
            return self.updated_operation

    store = FakeStore()
    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=store, worker=_fake_worker(context.job)).withdraw_job(  # type: ignore[arg-type]
            context.job.id,
            WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_check["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                check_token="check-token",
                review_safety_check_token="safety-token",
            ),
            idempotency_key="withdraw-reserved",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "destination_reserved"
    assert store.updated_operation is not None
    assert store.updated_operation.status == "failed"
    assert store.updated_operation.result_summary["response_payload"]["status"] == "failed"
    assert context.output_file.read_bytes() == b"video"


def test_withdraw_check_token_expiry_is_rejected(tmp_path) -> None:  # noqa: ANN001
    context = _withdraw_context(tmp_path)
    check_token = "expired-check-token"
    check_payload = {
        "mode": "withdraw_to_review",
        "status": "eligible",
        "checked_at": "2000-01-01T00:00:00+00:00",
        "ttl_seconds": 1,
        "check_token": check_token,
        "checked_move_state_revision": context.current_versions[
            "move_state_revision"
        ],
        "token_status": "current",
        "destination_preview": [],
        "item_result_summary": {"checked": 1, "eligible": 1, "ineligible": 0},
        "missing_files": [],
        "incomplete_records": [],
        "target_conflicts": [],
        "identity_mismatches": [],
        "retryable_items": [{"move_item_id": context.move.id}],
        "non_retryable_items": [],
        "affected_move_items": [context.move.id],
        "archive_restore_summary": {"archive_path": None, "has_archive_records": False},
    }
    check_operation = OperationRecord(
        id=301,
        job_id=context.job.id,
        operation_type="rollback_eligibility",
        status="succeeded",
        idempotency_key="rollback-check",
        request_payload_hash="hash-check",
        base_versions={},
        progress={},
        result_summary={"response_payload": check_payload},
        error=None,
        poll_after_ms=1000,
        created_at="2000-01-01T00:00:00+00:00",
        started_at=None,
        finished_at="2000-01-01T00:00:01+00:00",
    )

    with pytest.raises(ApiServiceError) as exc_info:
        _validated_withdraw_check_or_raise(
            [check_operation],
            payload=WithdrawRequest(
                mode="withdraw_to_review",
                scope=SelectionScope(kind="all"),
                base_review_revision=context.current_versions["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                check_token=check_token,
                review_safety_check_token="safety-token",
            ),
            current_versions=context.current_versions,
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_rollback_check"


def test_reopen_from_review_marks_withdrawn_move_proposed_from_review(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    safety_operation = _review_safety_operation(context)
    versions_after_safety = _job_versions(
        context.job,
        [context.move],
        [safety_operation],
        _validation_source_files(context.job, [context.move], None),
        snapshot={},
    )
    reopen_operation = OperationRecord(
        id=601,
        job_id=context.job.id,
        operation_type="reopen_from_review",
        status="running",
        idempotency_key="reopen-1",
        request_payload_hash="hash-reopen",
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:04+00:00",
        started_at="2026-07-06T00:00:04+00:00",
        finished_at=None,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.move = context.move
            self.completed_summary: dict[str, object] | None = None

        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [self.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_latest_successful_withdraw_item_results(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["job_id"] == context.job.id
            assert kwargs["move_item_ids"] == [context.move.id]
            return {context.move.id: context.withdraw_result}

        def create_operation(self, **kwargs):  # noqa: ANN003, ANN201
            return replace(
                reopen_operation,
                request_payload_hash=kwargs["request_payload_hash"],
                base_versions=kwargs["base_versions"],
                result_summary=kwargs["result_summary"],
            )

        def complete_reopen_from_review_operation_success(self, **kwargs):  # noqa: ANN003, ANN201
            assert kwargs["operation_id"] == reopen_operation.id
            assert kwargs["job_id"] == context.job.id
            updates = kwargs["updates"]
            assert len(updates) == 1
            update = updates[0]
            assert update.move_item_id == context.move.id
            assert update.source_path == str(context.review_file)
            assert update.from_path["original_source_path"]["display_path"] == str(
                context.original_source
            )
            self.move = replace(
                self.move,
                src=update.source_path,
                status="proposed",
                applied_at=None,
            )
            self.completed_summary = kwargs["result_summary"]
            return replace(
                reopen_operation,
                status="succeeded",
                result_summary=kwargs["result_summary"],
                finished_at="2026-07-06T00:00:05+00:00",
            )

        def update_operation(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            raise AssertionError("successful reopen should not fail operation")

    store = FakeStore()
    response = ApiService(store=store, worker=_fake_worker(context.job)).reopen_from_review(  # type: ignore[arg-type]
        context.job.id,
        ReopenFromReviewRequest(
            scope=SelectionScope(kind="all"),
            base_review_revision=versions_after_safety["review_revision"],
            base_move_state_revision=versions_after_safety["move_state_revision"],
            review_safety_check_token="safety-token",
        ),
        idempotency_key="reopen-1",
    )

    assert response is not None
    assert response.status == "succeeded"
    assert response.reopened_move_item_count == 1
    assert store.move.status == "proposed"
    assert store.move.src == str(context.review_file)
    assert context.review_file.read_bytes() == b"video"
    assert store.completed_summary is not None
    assert store.completed_summary["response_payload"]["reopened_move_item_count"] == 1


def test_reopen_from_review_invalid_review_safety_token_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    safety_operation = _review_safety_operation(context)
    versions_after_safety = _job_versions(
        context.job,
        [context.move],
        [safety_operation],
        _validation_source_files(context.job, [context.move], None),
        snapshot={},
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_latest_successful_withdraw_item_results(self, **_kwargs):  # noqa: ANN003, ANN201
            return {context.move.id: context.withdraw_result}

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("invalid safety token should not create reopen")

        def complete_reopen_from_review_operation_success(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("invalid safety token should not complete reopen")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            ReopenFromReviewRequest(
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_safety["review_revision"],
                base_move_state_revision=versions_after_safety["move_state_revision"],
                review_safety_check_token="wrong-safety-token",
            ),
            idempotency_key="reopen-invalid-safety",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "invalid_review_safety_check"
    assert context.move.status == "withdrawn"
    assert context.review_file.read_bytes() == b"video"


def test_reopen_from_review_stale_review_safety_proof_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    safety_operation = _stale_digest_review_safety_operation(context)
    versions_after_safety = _job_versions(
        context.job,
        [context.move],
        [safety_operation],
        _validation_source_files(context.job, [context.move], None),
        snapshot={},
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [safety_operation]

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_latest_successful_withdraw_item_results(self, **_kwargs):  # noqa: ANN003, ANN201
            return {context.move.id: context.withdraw_result}

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("stale safety proof should not create reopen")

        def complete_reopen_from_review_operation_success(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("stale safety proof should not complete reopen")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=_fake_worker(context.job)).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            ReopenFromReviewRequest(
                scope=SelectionScope(kind="all"),
                base_review_revision=versions_after_safety["review_revision"],
                base_move_state_revision=versions_after_safety["move_state_revision"],
                review_safety_check_token="stale-safety-token",
            ),
            idempotency_key="reopen-stale-safety",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_review_safety_check"
    assert context.move.status == "withdrawn"
    assert context.review_file.read_bytes() == b"video"


def test_reopen_from_review_missing_review_file_does_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    context.review_file.unlink()

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_latest_successful_withdraw_item_results(self, **_kwargs):  # noqa: ANN003, ANN201
            return {context.move.id: context.withdraw_result}

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("missing review file should not create operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            ReopenFromReviewRequest(
                scope=SelectionScope(kind="all"),
                base_review_revision=context.current_versions["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                review_safety_check_token="safety-token",
            ),
            idempotency_key="reopen-missing",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "reopen_from_review_not_ready"


def test_reopen_from_review_stale_versions_do_not_create_operation(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("stale reopen should not create operation")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            ReopenFromReviewRequest(
                scope=SelectionScope(kind="all"),
                base_review_revision=context.current_versions["review_revision"] + 1,
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                review_safety_check_token="safety-token",
            ),
            idempotency_key="reopen-stale",
        )

    assert exc_info.value.status_code == 412
    assert exc_info.value.code == "stale_versions"


def test_reopen_from_review_env_review_path_rejected_before_resolve(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    env_result = replace(
        context.withdraw_result,
        to_path={"display_path": str(context.review_file.parent / ".env.local.mkv")},
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return None

        def list_job_moves(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return [context.move]

        def list_operations(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return []

        def get_review_snapshot(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return None

        def list_latest_successful_withdraw_item_results(self, **_kwargs):  # noqa: ANN003, ANN201
            return {context.move.id: env_result}

        def create_operation(self, **_kwargs):  # noqa: ANN003, ANN201
            raise AssertionError("sensitive review path should not create operation")

    original_resolve = Path.resolve

    def guarded_resolve(self, strict=False):  # noqa: ANN001, ANN202
        if any(str(part).startswith(".env") for part in self.parts):
            raise AssertionError(f"resolve called for sensitive path: {self}")
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            ReopenFromReviewRequest(
                scope=SelectionScope(kind="all"),
                base_review_revision=context.current_versions["review_revision"],
                base_move_state_revision=context.current_versions[
                    "move_state_revision"
                ],
                review_safety_check_token="safety-token",
            ),
            idempotency_key="reopen-env",
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "reopen_from_review_not_ready"
    assert exc_info.value.details["problems"][0]["code"] == "sensitive_env_path"


def test_reopen_from_review_running_idempotency_key_does_not_read_moves(tmp_path) -> None:  # noqa: ANN001
    context = _reopen_context(tmp_path)
    payload = ReopenFromReviewRequest(
        scope=SelectionScope(kind="all"),
        base_review_revision=context.current_versions["review_revision"],
        base_move_state_revision=context.current_versions["move_state_revision"],
        review_safety_check_token="safety-token",
    )
    existing = OperationRecord(
        id=601,
        job_id=context.job.id,
        operation_type="reopen_from_review",
        status="running",
        idempotency_key="reopen-running",
        request_payload_hash=_payload_hash(payload.model_dump(mode="json")),
        base_versions={},
        progress={},
        result_summary={},
        error=None,
        poll_after_ms=1000,
        created_at="2026-07-06T00:00:04+00:00",
        started_at="2026-07-06T00:00:04+00:00",
        finished_at=None,
    )

    class FakeStore:
        def get_job(self, job_id):  # noqa: ANN001, ANN201
            assert job_id == context.job.id
            return context.job

        def get_operation_by_idempotency(self, **_kwargs):  # noqa: ANN003, ANN201
            return existing

        def list_job_moves(self, *_args):  # noqa: ANN002, ANN201
            raise AssertionError("running reopen replay should not read moves")

    with pytest.raises(ApiServiceError) as exc_info:
        ApiService(store=FakeStore(), worker=object()).reopen_from_review(  # type: ignore[arg-type]
            context.job.id,
            payload,
            idempotency_key="reopen-running",
        )

    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "operation_in_progress"


def test_reopen_from_review_request_rejects_partial_move_scope() -> None:
    with pytest.raises(ValueError, match="supports all scope"):
        ReopenFromReviewRequest(
            scope=SelectionScope(kind="move_items", ids=[10]),
            base_review_revision=1,
            base_move_state_revision=1,
            review_safety_check_token="safety-token",
        )


def test_frontend_preserves_rollback_and_safety_tokens_across_checks() -> None:
    source = Path("aninamer/api/static/app.js").read_text(encoding="utf-8")
    safety_start = source.index("async function checkReviewSafety()")
    rollback_start = source.index("async function checkRollbackEligibility()")
    validate_start = source.index("async function validateApply()")
    safety_body = source[safety_start:rollback_start]
    rollback_body = source[rollback_start:validate_start]

    assert "state.rollbackEligibility = null" not in safety_body
    assert "preserveRollback: true" in safety_body
    assert "preserveReviewSafety: true" in safety_body
    assert "preserveRollback: true" in rollback_body
    assert "preserveReviewSafety: true" in rollback_body
