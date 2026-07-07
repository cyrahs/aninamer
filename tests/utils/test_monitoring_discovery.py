from __future__ import annotations

import errno
from pathlib import Path

from aninamer.monitoring import (
    discover_series_dirs,
    discover_series_dirs_status,
    series_dir_contains_candidate_video,
)


def test_discover_series_dirs_returns_empty_when_mount_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def broken_iterdir(self: Path):  # noqa: ANN202
        if self == tmp_path:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return original_iterdir(self)

    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", broken_iterdir)

    assert discover_series_dirs(tmp_path) == []
    result = discover_series_dirs_status(tmp_path)
    assert result.series_dirs == []
    assert result.unavailable is True
    assert result.error_message is not None
    assert "Transport endpoint is not connected" in result.error_message


def test_discover_series_dirs_marks_root_unavailable_when_entries_cannot_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = tmp_path / "ShowA"
    child.mkdir()

    original_is_dir = Path.is_dir

    def broken_is_dir(self: Path) -> bool:
        if self == child:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", broken_is_dir)

    assert discover_series_dirs(tmp_path) == []
    result = discover_series_dirs_status(tmp_path)
    assert result.series_dirs == []
    assert result.unavailable is True
    assert result.error_message is not None
    assert "Transport endpoint is not connected" in result.error_message


def test_discover_series_dirs_excludes_archive_fail_and_review_roots(
    tmp_path: Path,
) -> None:
    (tmp_path / "archive").mkdir()
    (tmp_path / "fail").mkdir()
    (tmp_path / "review").mkdir()
    show = tmp_path / "ShowA"
    show.mkdir()

    assert discover_series_dirs(tmp_path) == [show]


def test_series_dir_contains_candidate_video_detects_nested_video(
    tmp_path: Path,
) -> None:
    show = tmp_path / "ShowA"
    (show / "Subs").mkdir(parents=True)
    (show / "Subs" / "ep1.ass").write_bytes(b"sub")
    (show / "Season 1").mkdir()
    (show / "Season 1" / "ep1.mkv").write_bytes(b"video")

    assert series_dir_contains_candidate_video(show) is True


def test_series_dir_contains_candidate_video_ignores_empty_and_unsupported_dirs(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "Empty"
    empty.mkdir()
    unsupported = tmp_path / "Unsupported"
    unsupported.mkdir()
    (unsupported / "poster.jpg").write_bytes(b"image")
    (unsupported / "notes.nfo").write_bytes(b"notes")
    (unsupported / "episode.ass").write_bytes(b"sub")
    ignored = tmp_path / "Ignored"
    (ignored / "trailer").mkdir(parents=True)
    (ignored / "trailer" / "preview.mkv").write_bytes(b"video")

    assert series_dir_contains_candidate_video(empty) is False
    assert series_dir_contains_candidate_video(unsupported) is False
    assert series_dir_contains_candidate_video(ignored) is False
