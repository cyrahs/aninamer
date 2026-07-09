from __future__ import annotations

import errno
from pathlib import Path

import aninamer.monitoring as monitoring
from aninamer.monitoring import (
    discover_series_dirs,
    discover_series_dirs_status,
    finalize_series_dir_after_apply,
    series_dir_contains_candidate_video,
    snapshot_series_files,
)
from aninamer.plan import PlannedMove, RenamePlan


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


def test_finalize_series_dir_after_apply_archives_remaining_source_tree(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "Season 1" / "ep1.mkv"
    note = series_dir / "notes.txt"
    destination = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    source.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    note.write_bytes(b"notes")
    before_files = snapshot_series_files(series_dir)
    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=source, dst=destination, kind="video", src_id=1),
        ),
    )

    source.unlink()
    destination.write_bytes(b"video")

    result = finalize_series_dir_after_apply(
        series_dir,
        plan,
        before_files=before_files,
    )

    assert result.status == "archived"
    assert result.archive_path == tmp_path / "archive" / "Show"
    assert not series_dir.exists()
    assert (result.archive_path / "notes.txt").read_bytes() == b"notes"


def test_finalize_series_dir_after_apply_skips_when_source_changed(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    destination = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    series_dir.mkdir()
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    before_files = snapshot_series_files(series_dir)
    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=source, dst=destination, kind="video", src_id=1),
        ),
    )

    source.unlink()
    (series_dir / "new-download.mkv").write_bytes(b"new")

    result = finalize_series_dir_after_apply(
        series_dir,
        plan,
        before_files=before_files,
    )

    assert result.status == "skipped"
    assert result.archive_path is None
    assert series_dir.exists()
    assert (series_dir / "new-download.mkv").read_bytes() == b"new"


def test_finalize_series_dir_after_apply_skips_when_env_entry_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    series_dir = tmp_path / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    destination = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    series_dir.mkdir()
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    before_files = snapshot_series_files(series_dir)
    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=source, dst=destination, kind="video", src_id=1),
        ),
    )

    source.unlink()
    destination.write_bytes(b"video")

    def fake_walk(path: Path, *_args, **_kwargs):  # noqa: ANN202
        assert Path(path) == series_dir
        yield str(series_dir), [], [".env.secret"]

    monkeypatch.setattr(monitoring.os, "walk", fake_walk)

    result = finalize_series_dir_after_apply(
        series_dir,
        plan,
        before_files=before_files,
    )

    assert result.status == "skipped"
    assert result.archive_path is None
    assert series_dir.exists()


def test_finalize_series_dir_after_apply_skips_nested_env_in_skipped_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    series_dir = tmp_path / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    destination = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    series_dir.mkdir()
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    before_files = snapshot_series_files(series_dir)
    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=source, dst=destination, kind="video", src_id=1),
        ),
    )

    source.unlink()
    destination.write_bytes(b"video")

    def fake_walk(path: Path, *_args, **_kwargs):  # noqa: ANN202
        assert Path(path) == series_dir
        yield str(series_dir), ["extras"], []
        yield str(series_dir / "extras"), [], [".env.local"]

    monkeypatch.setattr(monitoring.os, "walk", fake_walk)

    result = finalize_series_dir_after_apply(
        series_dir,
        plan,
        before_files=before_files,
    )

    assert result.status == "skipped"
    assert result.archive_path is None
    assert series_dir.exists()


def test_finalize_series_dir_after_apply_does_not_move_late_env_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    series_dir = tmp_path / "Show"
    output_root = tmp_path / "output"
    source = series_dir / "ep1.mkv"
    note = series_dir / "misc" / "notes.txt"
    destination = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    series_dir.mkdir()
    note.parent.mkdir(parents=True)
    destination.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    note.write_bytes(b"notes")
    before_files = snapshot_series_files(series_dir)
    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=source, dst=destination, kind="video", src_id=1),
        ),
    )
    source.unlink()
    destination.write_bytes(b"video")
    original_archive = monitoring.archive_series_dir_files

    def archive_with_late_env(path: Path, rel_paths: set[str]) -> Path:
        (series_dir / "misc" / ".env.local").write_text("secret", encoding="utf-8")
        return original_archive(path, rel_paths)

    monkeypatch.setattr(monitoring, "archive_series_dir_files", archive_with_late_env)

    result = finalize_series_dir_after_apply(
        series_dir,
        plan,
        before_files=before_files,
    )

    assert result.status == "archived"
    assert result.archive_path == tmp_path / "archive" / "Show"
    assert (result.archive_path / "misc" / "notes.txt").read_bytes() == b"notes"
    assert (series_dir / "misc" / ".env.local").read_text(encoding="utf-8") == "secret"
    assert not (result.archive_path / "misc" / ".env.local").exists()
