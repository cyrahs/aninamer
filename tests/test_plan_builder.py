from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.errors import PlanValidationError
from aninamer.episode_mapping import EpisodeMapItem, EpisodeMappingResult
from aninamer.plan import build_rename_plan
from aninamer.scanner import FileCandidate, ScanResult


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_build_rename_plan_videos_and_subs_with_chs_cht(tmp_path: Path) -> None:
    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    # Create source files
    _write(series_dir / "ep1.mkv", b"video1")
    _write(series_dir / "ep2.mkv", b"video2")
    _write(series_dir / "ep1.ass", "国国国 后后后".encode("utf-8"))   # CHS
    _write(series_dir / "ep2.ass", "國國國 後後後".encode("utf-8"))   # CHT

    scan = ScanResult(
        series_dir=series_dir,
        videos=[
            FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=(series_dir / "ep1.mkv").stat().st_size),
            FileCandidate(id=2, rel_path="ep2.mkv", ext=".mkv", size_bytes=(series_dir / "ep2.mkv").stat().st_size),
        ],
        subtitles=[
            FileCandidate(id=3, rel_path="ep1.ass", ext=".ass", size_bytes=(series_dir / "ep1.ass").stat().st_size),
            FileCandidate(id=4, rel_path="ep2.ass", ext=".ass", size_bytes=(series_dir / "ep2.ass").stat().st_size),
        ],
    )

    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(
            EpisodeMapItem(video_id=1, season=1, episode_start=1, episode_end=1, subtitle_ids=(3,)),
            EpisodeMapItem(video_id=2, season=1, episode_start=2, episode_end=2, subtitle_ids=(4,)),
        ),
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="测试动画",
        year=2020,
        tmdb_id=123,
        output_root=out_root,
    )

    # Expect 4 moves total (2 videos + 2 subs)
    assert len(plan.moves) == 4

    # Build expected base paths
    series_folder = out_root / "测试动画 (2020) {tmdb-123}"
    s01 = series_folder / "S01"

    dsts = {m.src_id: m.dst for m in plan.moves}

    assert dsts[1] == s01 / "测试动画 S01E01.mkv"
    assert dsts[2] == s01 / "测试动画 S01E02.mkv"
    assert dsts[3] == s01 / "测试动画 S01E01.chs.ass"
    assert dsts[4] == s01 / "测试动画 S01E02.cht.ass"


def test_build_rename_plan_includes_s00_for_ova(tmp_path: Path) -> None:
    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    _write(series_dir / "ova.mkv", b"video")
    _write(series_dir / "ova.ass", "OVA 特别篇 国".encode("utf-8"))

    scan = ScanResult(
        series_dir=series_dir,
        videos=[FileCandidate(id=1, rel_path="ova.mkv", ext=".mkv", size_bytes=(series_dir / "ova.mkv").stat().st_size)],
        subtitles=[FileCandidate(id=2, rel_path="ova.ass", ext=".ass", size_bytes=(series_dir / "ova.ass").stat().st_size)],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(EpisodeMapItem(video_id=1, season=0, episode_start=1, episode_end=1, subtitle_ids=(2,)),),
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="测试动画",
        year=2020,
        tmdb_id=123,
        output_root=out_root,
    )

    series_folder = out_root / "测试动画 (2020) {tmdb-123}"
    s00 = series_folder / "S00"
    dsts = {m.src_id: m.dst for m in plan.moves}
    assert dsts[1] == s00 / "测试动画 S00E01.mkv"
    assert dsts[2] == s00 / "测试动画 S00E01.chs.ass"


def test_build_rename_plan_disambiguates_multiple_subs_same_variant_same_ext(tmp_path: Path) -> None:
    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "a.ass", "国国国".encode("utf-8"))  # CHS
    _write(series_dir / "b.ass", "国国国".encode("utf-8"))  # CHS

    scan = ScanResult(
        series_dir=series_dir,
        videos=[FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=(series_dir / "ep1.mkv").stat().st_size)],
        subtitles=[
            FileCandidate(id=2, rel_path="a.ass", ext=".ass", size_bytes=(series_dir / "a.ass").stat().st_size),
            FileCandidate(id=3, rel_path="b.ass", ext=".ass", size_bytes=(series_dir / "b.ass").stat().st_size),
        ],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(EpisodeMapItem(video_id=1, season=1, episode_start=1, episode_end=1, subtitle_ids=(2, 3)),),
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="测试动画",
        year=2020,
        tmdb_id=123,
        output_root=out_root,
    )

    # Expect both subtitles, one gets .chs.ass and the other gets .chs.1.ass
    dsts = sorted([m.dst.name for m in plan.moves if m.kind == "subtitle"])
    assert dsts == ["测试动画 S01E01.chs.1.ass", "测试动画 S01E01.chs.ass"]


def test_build_rename_plan_rejects_dst_outside_output_root_via_bad_title(tmp_path: Path) -> None:
    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    _write(series_dir / "ep1.mkv", b"video")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=(series_dir / "ep1.mkv").stat().st_size)],
        subtitles=[],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(EpisodeMapItem(video_id=1, season=1, episode_start=1, episode_end=1, subtitle_ids=()),),
    )

    # Title contains path separators and traversal-like parts; sanitization should neutralize it
    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="../Bad/Name..",
        year=2020,
        tmdb_id=123,
        output_root=out_root,
    )

    # Must still be inside output_root
    for m in plan.moves:
        assert str(m.dst.resolve()).startswith(str(out_root.resolve()))


def test_build_rename_plan_rejects_existing_destination_when_not_allowed(tmp_path: Path) -> None:
    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    _write(series_dir / "ep1.mkv", b"video")

    scan = ScanResult(
        series_dir=series_dir,
        videos=[FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=(series_dir / "ep1.mkv").stat().st_size)],
        subtitles=[],
    )
    mapping = EpisodeMappingResult(
        tmdb_id=123,
        items=(EpisodeMapItem(video_id=1, season=1, episode_start=1, episode_end=1, subtitle_ids=()),),
    )

    # Pre-create a conflicting dest file
    conflict = out_root / "测试动画 (2020) {tmdb-123}" / "S01" / "测试动画 S01E01.mkv"
    _write(conflict, b"existing")

    with pytest.raises(PlanValidationError):
        build_rename_plan(
            scan=scan,
            mapping=mapping,
            series_name_zh_cn="测试动画",
            year=2020,
            tmdb_id=123,
            output_root=out_root,
            allow_existing_dest=False,
        )
