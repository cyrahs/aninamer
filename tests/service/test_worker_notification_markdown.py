from __future__ import annotations

import pytest

from aninamer.store import JobRecord, JobRequestRecord
from aninamer.worker import (
    _build_notification_presentation,
    _escape_telegram_markdown_v2,
    _notification_subject,
)


def _job_record(series_name: str = "Show (2026) #1!") -> JobRecord:
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
        error_message="Bad #hash_(boom)! v0.1",
        series_dir="/tmp/input",
        output_root="/tmp/output",
        archive_path=None,
        fail_path=None,
    )


def _job_request_record() -> JobRequestRecord:
    return JobRequestRecord(
        id=34,
        kind="apply_job",
        status="rejected",
        target_job_id=12,
        created_at="2026-03-10T00:00:00+00:00",
        updated_at="2026-03-10T00:00:00+00:00",
        started_at=None,
        finished_at=None,
        error_message="Bad #hash_(boom)! v0.1",
    )


def test_escape_telegram_markdown_v2_escapes_reserved_characters() -> None:
    assert (
        _escape_telegram_markdown_v2(r"A_B#(C)! v0.1\path")
        == r"A\_B\#\(C\)\! v0\.1\\path"
    )


def test_notification_subject_falls_back_without_job() -> None:
    assert _notification_subject(None) == "归档任务"


@pytest.mark.parametrize(
    ("event_kind", "payload", "expected_severity", "expected_title", "expected_message"),
    [
        (
            "job_apply_succeeded",
            {"finalize_status": "archived"},
            "success",
            "归档成功",
            "已归档完成",
        ),
        (
            "job_apply_succeeded",
            {"finalize_status": "deleted"},
            "warning",
            "处理完成",
            "已完成处理，源目录已清理",
        ),
        (
            "job_apply_succeeded",
            {"finalize_status": "skipped"},
            "warning",
            "处理完成",
            "已完成处理，未执行归档",
        ),
        (
            "job_plan_failed",
            {"error_stage": "plan", "error_message": "Bad #hash_(boom)! v0.1"},
            "error",
            "归档失败",
            "生成归档计划失败",
        ),
        (
            "job_apply_failed",
            {"error_stage": "apply", "error_message": "Bad #hash_(boom)! v0.1"},
            "error",
            "归档失败",
            "执行归档失败",
        ),
        (
            "job_request_rejected",
            {"request_action": "apply_job", "error_message": "Bad #hash_(boom)! v0.1"},
            "error",
            "归档失败",
            "归档请求被拒绝",
        ),
        (
            "job_request_failed",
            {"request_action": "apply_job", "error_message": "Bad #hash_(boom)! v0.1"},
            "error",
            "归档失败",
            "处理归档请求失败",
        ),
    ],
)
def test_build_notification_presentation_compacts_display(
    event_kind: str,
    payload: dict[str, object],
    expected_severity: str,
    expected_title: str,
    expected_message: str,
) -> None:
    presentation = _build_notification_presentation(
        event_kind=event_kind,
        job=_job_record(),
        job_request=_job_request_record(),
        payload=payload,
    )

    assert presentation.severity == expected_severity
    assert presentation.title == expected_title
    assert presentation.message == expected_message
    assert presentation.markdown.splitlines() == [
        _escape_telegram_markdown_v2(expected_title),
        "",
        r"《Show \(2026\) \#1\!》",
        _escape_telegram_markdown_v2(expected_message),
    ]
    assert "事件" not in presentation.markdown
    assert "Job ID" not in presentation.markdown
    assert "TMDB" not in presentation.markdown
    assert "Job Request ID" not in presentation.markdown
    assert "请求动作" not in presentation.markdown
    assert "完成状态" not in presentation.markdown
    assert "Bad \\#hash\\_\\(boom\\)\\! v0\\.1" not in presentation.markdown


def test_build_notification_presentation_uses_generic_subject_without_job() -> None:
    presentation = _build_notification_presentation(
        event_kind="job_request_rejected",
        job=None,
        job_request=_job_request_record(),
        payload={"request_action": "apply_job", "error_message": "Bad #hash_(boom)! v0.1"},
    )

    assert presentation.severity == "error"
    assert presentation.title == "归档失败"
    assert presentation.message == "归档请求被拒绝"
    assert presentation.markdown.splitlines() == [
        "归档失败",
        "",
        "归档任务",
        "归档请求被拒绝",
    ]
