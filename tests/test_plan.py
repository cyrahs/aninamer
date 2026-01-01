from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.episode_mapping import EpisodeMapItem, EpisodeMappingResult
from aninamer.errors import PlanValidationError
from aninamer.plan import (
    build_rename_plan,
    format_episode_base,
    format_series_root_folder,
    sanitize_path_component,
)
from aninamer.scanner import FileCandidate, ScanResult


def _candidate(series_dir: Path, path: Path, file_id: int) -> FileCandidate:
    return FileCandidate(
        id=file_id,
        rel_path=path.relative_to(series_dir).as_posix(),
        ext=path.suffix.lower(),
        size_bytes=path.stat().st_size,
    )


def test_sanitize_path_component_normalizes() -> None:
    raw = '  My/Show<>:"\\|?*   Title..  '
    assert sanitize_path_component(raw) == "My Show Title"


def test_sanitize_path_component_dot_only_returns_unknown() -> None:
    assert sanitize_path_component("") == "Unknown"
    assert sanitize_path_component("   ") == "Unknown"
    assert sanitize_path_component(".") == "Unknown"
    assert sanitize_path_component("..") == "Unknown"
    assert sanitize_path_component("...") == "Unknown"


def test_format_series_root_folder_includes_tmdb_tag() -> None:
    assert format_series_root_folder("My/Show", 2024, 12) == "My Show (2024) {tmdb-12}"
    assert format_series_root_folder("My/Show", None, 12) == "My Show {tmdb-12}"


def test_format_episode_base_single_and_range() -> None:
    assert format_episode_base("My/Show", 1, 2, 2) == "My Show S01E02"
    assert format_episode_base("My/Show", 2, 3, 4) == "My Show S02E03-E04"


def test_build_rename_plan_basic_and_ignores_unmapped(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    video1 = series_dir / "ep1.mkv"
    video1.write_bytes(b"video1")
    video2 = series_dir / "ep2.mkv"
    video2.write_bytes(b"video2")
    subtitle1 = series_dir / "ep1.CHS.ass"
    subtitle1.write_text("text", encoding="utf-8")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[
            _candidate(series_dir, video1, 1),
            _candidate(series_dir, video2, 2),
        ],
        subtitles=[_candidate(series_dir, subtitle1, 3)],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(
            EpisodeMapItem(
                video_id=1,
                season=1,
                episode_start=1,
                episode_end=1,
                subtitle_ids=(3,),
            ),
        ),
    )

    output_root = tmp_path / "out"
    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="Series Name",
        year=2024,
        tmdb_id=123,
        output_root=output_root,
    )

    moves_by_id = {move.src_id: move for move in plan.moves}
    assert set(moves_by_id) == {1, 3}

    expected_root = (
        output_root.resolve(strict=False)
        / "Series Name (2024) {tmdb-123}"
        / "S01"
    )
    video_move = moves_by_id[1]
    assert video_move.kind == "video"
    assert video_move.src == video1.resolve(strict=False)
    assert video_move.dst == (expected_root / "Series Name S01E01.mkv").resolve(
        strict=False
    )

    subtitle_move = moves_by_id[3]
    assert subtitle_move.kind == "subtitle"
    assert subtitle_move.src == subtitle1.resolve(strict=False)
    assert subtitle_move.dst == (expected_root / "Series Name S01E01.chs.ass").resolve(
        strict=False
    )


def test_build_rename_plan_disambiguates_subtitles(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    video = series_dir / "ep1.mkv"
    video.write_bytes(b"video")
    subtitle1 = series_dir / "ep1.CHS.ass"
    subtitle1.write_text("text", encoding="utf-8")
    subtitle2 = series_dir / "ep1.chs.ass"
    subtitle2.write_text("more", encoding="utf-8")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[_candidate(series_dir, video, 1)],
        subtitles=[
            _candidate(series_dir, subtitle1, 2),
            _candidate(series_dir, subtitle2, 3),
        ],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=55,
        items=(
            EpisodeMapItem(
                video_id=1,
                season=1,
                episode_start=1,
                episode_end=1,
                subtitle_ids=(2, 3),
            ),
        ),
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="Series Name",
        year=None,
        tmdb_id=55,
        output_root=tmp_path / "out",
    )

    moves_by_id = {move.src_id: move for move in plan.moves}
    assert moves_by_id[2].dst.name == "Series Name S01E01.chs.ass"
    assert moves_by_id[3].dst.name == "Series Name S01E01.chs.1.ass"


def test_build_rename_plan_rejects_existing_dest(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    video = series_dir / "ep1.mkv"
    video.write_bytes(b"video")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[_candidate(series_dir, video, 1)],
        subtitles=[],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=1,
        items=(
            EpisodeMapItem(
                video_id=1,
                season=1,
                episode_start=1,
                episode_end=1,
                subtitle_ids=(),
            ),
        ),
    )

    output_root = tmp_path / "out"
    dest = (
        output_root
        / "Series Name {tmdb-1}"
        / "S01"
        / "Series Name S01E01.mkv"
    )
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing")

    with pytest.raises(PlanValidationError):
        build_rename_plan(
            scan=scan,
            mapping=mapping,
            series_name_zh_cn="Series Name",
            year=None,
            tmdb_id=1,
            output_root=output_root,
        )


def test_build_rename_plan_tmdb_mismatch(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    video = series_dir / "ep1.mkv"
    video.write_bytes(b"video")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[_candidate(series_dir, video, 1)],
        subtitles=[],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=2,
        items=(
            EpisodeMapItem(
                video_id=1,
                season=1,
                episode_start=1,
                episode_end=1,
                subtitle_ids=(),
            ),
        ),
    )

    with pytest.raises(PlanValidationError):
        build_rename_plan(
            scan=scan,
            mapping=mapping,
            series_name_zh_cn="Series Name",
            year=None,
            tmdb_id=1,
            output_root=tmp_path / "out",
        )


def test_build_rename_plan_missing_video_id(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    video = series_dir / "ep1.mkv"
    video.write_bytes(b"video")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[_candidate(series_dir, video, 1)],
        subtitles=[],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=1,
        items=(
            EpisodeMapItem(
                video_id=99,
                season=1,
                episode_start=1,
                episode_end=1,
                subtitle_ids=(),
            ),
        ),
    )

    with pytest.raises(PlanValidationError):
        build_rename_plan(
            scan=scan,
            mapping=mapping,
            series_name_zh_cn="Series Name",
            year=None,
            tmdb_id=1,
            output_root=tmp_path / "out",
        )
