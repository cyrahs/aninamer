from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.apply import apply_rename_plan
from aninamer.errors import ApplyError
from aninamer.plan import PlannedMove, RenamePlan


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_apply_dry_run_does_not_touch_disk(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    out_root = tmp_path / "out"

    src = series_dir / "ep1.mkv"
    dst = out_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    _write(src, b"video")

    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=out_root,
        moves=(PlannedMove(src=src, dst=dst, kind="video", src_id=1),),
    )

    res = apply_rename_plan(plan, dry_run=True)
    assert res.dry_run is True
    assert res.applied == ()
    assert src.exists()
    assert not dst.exists()


def test_apply_moves_files_and_can_rollback(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    out_root = tmp_path / "out"

    src_v = series_dir / "ep1.mkv"
    src_s = series_dir / "ep1.ass"
    dst_v = out_root / "测试动画 (2020) {tmdb-123}" / "S01" / "测试动画 S01E01.mkv"
    dst_s = out_root / "测试动画 (2020) {tmdb-123}" / "S01" / "测试动画 S01E01.chs.ass"

    _write(src_v, b"video")
    _write(src_s, "国国国".encode("utf-8"))

    plan = RenamePlan(
        tmdb_id=123,
        series_name_zh_cn="测试动画",
        year=2020,
        series_dir=series_dir,
        output_root=out_root,
        moves=(
            PlannedMove(src=src_v, dst=dst_v, kind="video", src_id=1),
            PlannedMove(src=src_s, dst=dst_s, kind="subtitle", src_id=2),
        ),
    )

    res = apply_rename_plan(plan, dry_run=False)
    assert res.dry_run is False
    assert res.temp_dir is None
    assert dst_v.exists()
    assert dst_s.exists()
    assert not src_v.exists()
    assert not src_s.exists()

    # Rollback by applying a new plan constructed from rollback_moves
    rollback_plan = RenamePlan(
        tmdb_id=plan.tmdb_id,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        series_dir=plan.series_dir,
        output_root=plan.output_root,
        moves=res.rollback_moves,
    )
    _ = apply_rename_plan(rollback_plan, dry_run=False)

    assert src_v.exists()
    assert src_s.exists()
    assert not dst_v.exists()
    assert not dst_s.exists()


def test_apply_handles_dependency_chain_single_stage(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    out_root = tmp_path / "out"

    a = series_dir / "a.txt"
    b = series_dir / "b.txt"
    c = series_dir / "c.txt"

    _write(a, b"A")
    _write(b, b"B")

    # Plan a -> b, b -> c (b exists and is also a source)
    plan = RenamePlan(
        tmdb_id=0,
        series_name_zh_cn="X",
        year=None,
        series_dir=series_dir,
        output_root=out_root,
        moves=(
            PlannedMove(src=a, dst=b, kind="video", src_id=1),
            PlannedMove(src=b, dst=c, kind="video", src_id=2),
        ),
    )

    _ = apply_rename_plan(plan, dry_run=False)

    assert not a.exists()
    assert b.read_bytes() == b"A"
    assert c.read_bytes() == b"B"


def test_apply_raises_if_destination_exists_and_is_not_a_source(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    out_root = tmp_path / "out"

    src = series_dir / "a.txt"
    dst = series_dir / "dest.txt"

    _write(src, b"A")
    _write(dst, b"EXISTING")

    plan = RenamePlan(
        tmdb_id=0,
        series_name_zh_cn="X",
        year=None,
        series_dir=series_dir,
        output_root=out_root,
        moves=(PlannedMove(src=src, dst=dst, kind="video", src_id=1),),
    )

    with pytest.raises(ApplyError):
        apply_rename_plan(plan, dry_run=False)

    # Ensure no change
    assert src.exists()
    assert dst.read_bytes() == b"EXISTING"
