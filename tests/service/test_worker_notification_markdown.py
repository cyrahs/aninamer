from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.plan import PlannedMove, RenamePlan
from aninamer.store import JobRecord, JobRequestRecord
from aninamer.worker import (
    _build_notification_presentation,
    _escape_telegram_markdown_v2,
    _notification_subject,
)


def _job_record(
    series_name: str = "Show (2026) #1!",
    *,
    error_message: str = "Bad #hash_(boom)! v0.1",
) -> JobRecord:
    return JobRecord(
        id=12,
        series_name=series_name,
        watch_root_key="downloads",
        source_kind="monitor",
        status="failed",
        tmdb_id=321,
        video_moves_count=0,
        subtitle_moves_count=0,
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        error_stage="apply_stage#1",
        error_message=error_message,
        series_dir="/tmp/input",
        output_root="/tmp/output",
        archive_path=None,
        fail_path=None,
    )


def _job_request_record(
    *,
    error_message: str = "Bad #hash_(boom)! v0.1",
) -> JobRequestRecord:
    return JobRequestRecord(
        id=34,
        kind="apply_job",
        status="rejected",
        target_job_id=12,
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        error_message=error_message,
    )


def _plan() -> RenamePlan:
    return RenamePlan(
        tmdb_id=321,
        series_name_zh_cn="测试动画",
        year=2026,
        series_dir=Path("/tmp/input/Show"),
        output_root=Path("/tmp/output"),
        moves=(
            PlannedMove(
                src=Path("/tmp/input/Show/01.mkv"),
                dst=Path("/tmp/output/测试动画/S02/测试动画 S02E01.mkv"),
                kind="video",
                src_id=1,
            ),
            PlannedMove(
                src=Path("/tmp/input/Show/02.mkv"),
                dst=Path("/tmp/output/测试动画/S02/测试动画 S02E02.mkv"),
                kind="video",
                src_id=2,
            ),
            PlannedMove(
                src=Path("/tmp/input/Show/04.mkv"),
                dst=Path("/tmp/output/测试动画/S02/测试动画 S02E04-E05.mkv"),
                kind="video",
                src_id=3,
            ),
            PlannedMove(
                src=Path("/tmp/input/Show/01.ass"),
                dst=Path("/tmp/output/测试动画/S02/测试动画 S02E01.chs.ass"),
                kind="subtitle",
                src_id=101,
            ),
        ),
    )


def _four_video_plan() -> RenamePlan:
    return RenamePlan(
        tmdb_id=321,
        series_name_zh_cn="测试动画",
        year=2026,
        series_dir=Path("/tmp/input/Show"),
        output_root=Path("/tmp/output"),
        moves=tuple(
            PlannedMove(
                src=Path(f"/tmp/input/Show/{episode:02d}.mkv"),
                dst=Path(
                    "/tmp/output/测试动画/S02/"
                    f"测试动画 S02E{episode:02d}.mkv"
                ),
                kind="video",
                src_id=episode,
            )
            for episode in range(1, 5)
        ),
    )


def test_escape_telegram_markdown_v2_escapes_reserved_characters() -> None:
    assert (
        _escape_telegram_markdown_v2(r"A_B#(C)! v0.1\path")
        == r"A\_B\#\(C\)\! v0\.1\\path"
    )


def test_notification_subject_falls_back_without_job() -> None:
    assert _notification_subject(None) == "归档任务"


@pytest.mark.parametrize(
    ("finalize_status", "expected_severity"),
    [
        ("archived", "success"),
        ("deleted", "warning"),
        ("skipped", "warning"),
    ],
)
def test_build_notification_presentation_uses_chinese_title_and_episode_summary(
    finalize_status: str,
    expected_severity: str,
) -> None:
    presentation = _build_notification_presentation(
        event_kind="job_apply_succeeded",
        job=_job_record(series_name="Raw Folder"),
        job_request=None,
        payload={"finalize_status": finalize_status},
        plan=_plan(),
    )

    assert presentation.severity == expected_severity
    assert presentation.title == "Aninamer: 测试动画"
    assert presentation.message == "S02E01-S02E02, S02E04-S02E05 | 视频: 3 | 字幕: 1"
    assert presentation.markdown.splitlines() == [
        "*Aninamer: 测试动画*",
        r"S02E01\-S02E02, S02E04\-S02E05 \| 视频: 3 \| 字幕: 1",
        "/tmp/output/测试动画",
    ]
    assert "事件" not in presentation.markdown
    assert "Job ID" not in presentation.markdown
    assert "TMDB" not in presentation.markdown
    assert "Job Request ID" not in presentation.markdown
    assert "请求动作" not in presentation.markdown
    assert "完成状态" not in presentation.markdown


def test_build_notification_presentation_merges_four_videos_without_subtitles() -> None:
    presentation = _build_notification_presentation(
        event_kind="job_apply_succeeded",
        job=_job_record(series_name="Raw Folder"),
        job_request=None,
        payload={"finalize_status": "archived"},
        plan=_four_video_plan(),
    )

    assert presentation.severity == "success"
    assert presentation.title == "Aninamer: 测试动画"
    assert presentation.message == "S02E01-S02E04 | 视频: 4 | 字幕: 0"
    assert presentation.markdown.splitlines() == [
        "*Aninamer: 测试动画*",
        r"S02E01\-S02E04 \| 视频: 4 \| 字幕: 0",
        "/tmp/output/测试动画",
    ]


@pytest.mark.parametrize(
    ("error_message", "expected_message"),
    [
        (
            "ApplyError: destination already exists: /tmp/out/测试动画.mkv",
            "归档失败：目标文件已存在",
        ),
        (
            "LLMOutputError: eps[1] episode range 9-9 exceeds season 1 count 1",
            "归档失败：LLM 映射集数超出 TMDB 范围",
        ),
        ("worker restarted during apply", "归档失败：Worker 重启，归档中断"),
        (
            "RuntimeError: Bad #hash_(boom)! v0.1",
            "归档失败：RuntimeError: Bad #hash_(boom)! v0.1",
        ),
    ],
)
def test_build_notification_presentation_compacts_failure_reasons(
    error_message: str,
    expected_message: str,
) -> None:
    presentation = _build_notification_presentation(
        event_kind="job_apply_failed",
        job=_job_record(series_name="Raw Folder", error_message=error_message),
        job_request=None,
        payload={"error_stage": "apply", "error_message": error_message},
        plan=_plan(),
    )

    assert presentation.severity == "error"
    assert presentation.title == "Aninamer: 测试动画"
    assert presentation.message == expected_message
    assert presentation.markdown.splitlines() == [
        "*Aninamer: 测试动画*",
        _escape_telegram_markdown_v2(expected_message),
    ]


def test_build_notification_presentation_uses_request_rejection_reason() -> None:
    presentation = _build_notification_presentation(
        event_kind="job_request_rejected",
        job=_job_record(series_name="Raw Folder"),
        job_request=_job_request_record(error_message="job is not in planned status"),
        payload={"request_action": "apply_job"},
        plan=None,
    )

    assert presentation.severity == "error"
    assert presentation.title == "Aninamer: Raw Folder"
    assert presentation.message == "请求被拒绝：任务状态不允许归档"
    assert presentation.markdown.splitlines() == [
        "*Aninamer: Raw Folder*",
        "请求被拒绝：任务状态不允许归档",
    ]


def test_build_notification_presentation_uses_generic_title_without_job() -> None:
    presentation = _build_notification_presentation(
        event_kind="job_request_rejected",
        job=None,
        job_request=_job_request_record(
            error_message="apply_job request requires target_job_id"
        ),
        payload={"request_action": "apply_job"},
    )

    assert presentation.severity == "error"
    assert presentation.title == "Aninamer: 归档任务"
    assert presentation.message == "请求被拒绝：请求缺少目标任务"
    assert presentation.markdown.splitlines() == [
        "*Aninamer: 归档任务*",
        "请求被拒绝：请求缺少目标任务",
    ]
