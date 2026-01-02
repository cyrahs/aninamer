from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.apply import apply_rename_plan, build_rollback_moves
from aninamer.errors import ApplyError
from aninamer.plan import PlannedMove, RenamePlan


def _plan(
    series_dir: Path, output_root: Path, moves: tuple[PlannedMove, ...]
) -> RenamePlan:
    return RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Series",
        year=2024,
        series_dir=series_dir,
        output_root=output_root,
        moves=moves,
    )


def test_build_rollback_moves_preserves_order(tmp_path: Path) -> None:
    move1 = PlannedMove(
        src=tmp_path / "a.mkv",
        dst=tmp_path / "b.mkv",
        kind="video",
        src_id=1,
    )
    move2 = PlannedMove(
        src=tmp_path / "c.ass",
        dst=tmp_path / "d.ass",
        kind="subtitle",
        src_id=2,
    )
    plan = _plan(tmp_path, tmp_path, (move1, move2))

    rollback = build_rollback_moves(plan)

    assert rollback == (
        PlannedMove(
            src=move1.dst, dst=move1.src, kind=move1.kind, src_id=move1.src_id
        ),
        PlannedMove(
            src=move2.dst, dst=move2.src, kind=move2.kind, src_id=move2.src_id
        ),
    )


def test_apply_dry_run_does_not_touch_fs(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    src = series_dir / "ep1.mkv"
    src.write_bytes(b"video")

    output_root = tmp_path / "out"
    dest = (
        output_root
        / "Series {tmdb-1}"
        / "S01"
        / "Series S01E01.mkv"
    )
    plan = _plan(
        series_dir,
        output_root,
        (
            PlannedMove(
                src=src, dst=dest, kind="video", src_id=1
            ),
        ),
    )

    result = apply_rename_plan(plan, dry_run=True)

    assert result.dry_run is True
    assert result.applied == ()
    assert len(result.rollback_moves) == 1
    assert result.temp_dir is None
    assert src.exists()
    assert not dest.exists()
    assert not output_root.exists()


def test_apply_swaps_files_with_staging(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    file_a = root / "a.mkv"
    file_b = root / "b.mkv"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")

    plan = _plan(
        root,
        root,
        (
            PlannedMove(
                src=file_a, dst=file_b, kind="video", src_id=1
            ),
            PlannedMove(
                src=file_b, dst=file_a, kind="video", src_id=2
            ),
        ),
    )

    result = apply_rename_plan(plan, dry_run=False, two_stage=True)

    assert result.dry_run is False
    assert file_a.read_bytes() == b"b"
    assert file_b.read_bytes() == b"a"
    assert len(result.applied) == 2
    assert result.temp_dir is not None
    assert not result.temp_dir.exists()


def test_apply_raises_on_existing_destination_not_source(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    src = series_dir / "ep1.mkv"
    src.write_bytes(b"video")

    output_root = tmp_path / "out"
    dest = (
        output_root
        / "Series {tmdb-1}"
        / "S01"
        / "Series S01E01.mkv"
    )
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"existing")

    plan = _plan(
        series_dir,
        output_root,
        (
            PlannedMove(
                src=src, dst=dest, kind="video", src_id=1
            ),
        ),
    )

    with pytest.raises(ApplyError):
        apply_rename_plan(plan, dry_run=False)

    assert src.exists()
    assert dest.exists()


def test_apply_single_stage_cycle_requires_two_stage(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    file_a = root / "a.mkv"
    file_b = root / "b.mkv"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")

    plan = _plan(
        root,
        root,
        (
            PlannedMove(
                src=file_a, dst=file_b, kind="video", src_id=1
            ),
            PlannedMove(
                src=file_b, dst=file_a, kind="video", src_id=2
            ),
        ),
    )

    with pytest.raises(ApplyError) as excinfo:
        apply_rename_plan(plan, dry_run=False)

    assert "--two-stage" in str(excinfo.value)
    assert file_a.read_bytes() == b"a"
    assert file_b.read_bytes() == b"b"


def test_apply_rolls_back_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    src = root / "a.mkv"
    src.write_bytes(b"video")
    dest = root / "b.mkv"

    plan = _plan(
        root,
        root,
        (
            PlannedMove(
                src=src, dst=dest, kind="video", src_id=1
            ),
        ),
    )

    import aninamer.apply as apply_module

    real_move = apply_module.shutil.move
    call_count = {"count": 0}

    def flaky_move(src_path: str, dst_path: str, *args: object, **kwargs: object) -> str:
        call_count["count"] += 1
        if call_count["count"] == 2:
            raise OSError("boom")
        return real_move(src_path, dst_path, *args, **kwargs)

    monkeypatch.setattr(apply_module.shutil, "move", flaky_move)

    with pytest.raises(ApplyError):
        apply_module.apply_rename_plan(plan, dry_run=False, two_stage=True)

    assert src.read_bytes() == b"video"
    assert not dest.exists()

    tmp_dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith(".aninamer_tmp_")
    ]
    assert len(tmp_dirs) == 1


def test_apply_skips_noop_moves(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    src = root / "noop.mkv"
    src.write_bytes(b"video")

    plan = _plan(
        root,
        root,
        (
            PlannedMove(
                src=src, dst=src, kind="video", src_id=1
            ),
        ),
    )

    result = apply_rename_plan(plan, dry_run=False)

    assert result.applied == ()
    assert src.read_bytes() == b"video"
