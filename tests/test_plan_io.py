from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.errors import PlanValidationError
from aninamer.plan import PlannedMove, RenamePlan
from aninamer.plan_io import (
    read_rename_plan_json,
    rename_plan_from_dict,
    rename_plan_to_dict,
    write_rename_plan_json,
)


def _sample_plan(tmp_path: Path) -> RenamePlan:
    series_dir = tmp_path / "series"
    output_root = tmp_path / "out"
    src_video = series_dir / "ep1.mkv"
    dst_video = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    src_sub = series_dir / "ep1.ass"
    dst_sub = output_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.chs.ass"
    return RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=output_root,
        moves=(
            PlannedMove(src=src_video, dst=dst_video, kind="video", src_id=1),
            PlannedMove(src=src_sub, dst=dst_sub, kind="subtitle", src_id=2),
        ),
    )


def test_rename_plan_roundtrip_dict(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    payload = rename_plan_to_dict(plan)
    assert payload["version"] == 1
    restored = rename_plan_from_dict(payload)
    assert restored == plan


def test_rename_plan_from_dict_rejects_unknown_keys(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    payload = rename_plan_to_dict(plan)
    payload["extra"] = "nope"
    with pytest.raises(PlanValidationError):
        rename_plan_from_dict(payload)

    payload = rename_plan_to_dict(plan)
    payload["moves"][0]["extra"] = "nope"
    with pytest.raises(PlanValidationError):
        rename_plan_from_dict(payload)


def test_rename_plan_from_dict_rejects_bad_version(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    payload = rename_plan_to_dict(plan)
    payload["version"] = 2
    with pytest.raises(PlanValidationError):
        rename_plan_from_dict(payload)


def test_write_read_rename_plan_json(tmp_path: Path) -> None:
    plan = _sample_plan(tmp_path)
    plan_path = tmp_path / "plans" / "rename_plan.json"
    write_rename_plan_json(plan_path, plan)
    restored = read_rename_plan_json(plan_path)
    assert restored == plan
