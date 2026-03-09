from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from aninamer.errors import PlanValidationError
from aninamer.monitoring import MonitorFinalizeResult
from aninamer.plan import PlannedMove, RenamePlan

if TYPE_CHECKING:
    from aninamer.pipeline import ApplyExecutionResult


def _validate_keys(obj: dict[str, object], expected: set[str], label: str) -> None:
    keys = set(obj.keys())
    if keys == expected:
        return
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    raise PlanValidationError(
        f"{label} keys invalid; missing={missing or None} extra={extra or None}"
    )


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(f"{label} must be int")
    return value


def _require_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PlanValidationError(f"{label} must be str")
    return value


def rename_plan_to_payload(plan: RenamePlan) -> dict[str, Any]:
    return {
        "version": 1,
        "tmdb_id": plan.tmdb_id,
        "series_name_zh_cn": plan.series_name_zh_cn,
        "year": plan.year,
        "series_dir": str(plan.series_dir),
        "output_root": str(plan.output_root),
        "moves": [
            {
                "src_id": move.src_id,
                "kind": move.kind,
                "src": str(move.src),
                "dst": str(move.dst),
            }
            for move in plan.moves
        ],
    }


def rename_plan_from_payload(data: dict[str, Any]) -> RenamePlan:
    expected_keys = {
        "version",
        "tmdb_id",
        "series_name_zh_cn",
        "year",
        "series_dir",
        "output_root",
        "moves",
    }
    _validate_keys(data, expected_keys, "plan")

    version = data.get("version")
    if version != 1:
        raise PlanValidationError("version must be 1")

    tmdb_id = _require_int(data.get("tmdb_id"), "tmdb_id")
    series_name = _require_str(data.get("series_name_zh_cn"), "series_name_zh_cn")

    year_raw = data.get("year")
    if year_raw is None:
        year = None
    elif isinstance(year_raw, bool) or not isinstance(year_raw, int):
        raise PlanValidationError("year must be int or null")
    else:
        year = year_raw

    series_dir = Path(_require_str(data.get("series_dir"), "series_dir"))
    output_root = Path(_require_str(data.get("output_root"), "output_root"))

    moves_raw = data.get("moves")
    if not isinstance(moves_raw, list):
        raise PlanValidationError("moves must be list")

    move_keys = {"src_id", "kind", "src", "dst"}
    moves: list[PlannedMove] = []
    for idx, entry in enumerate(moves_raw):
        if not isinstance(entry, dict):
            raise PlanValidationError(f"moves[{idx}] must be object")
        _validate_keys(entry, move_keys, f"moves[{idx}]")
        src_id = _require_int(entry.get("src_id"), f"moves[{idx}].src_id")
        kind = entry.get("kind")
        if kind not in ("video", "subtitle"):
            raise PlanValidationError(f"moves[{idx}].kind must be video or subtitle")
        src = Path(_require_str(entry.get("src"), f"moves[{idx}].src"))
        dst = Path(_require_str(entry.get("dst"), f"moves[{idx}].dst"))
        moves.append(
            PlannedMove(
                src=src,
                dst=dst,
                kind=kind,
                src_id=src_id,
            )
        )

    return RenamePlan(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name,
        year=year,
        series_dir=series_dir,
        output_root=output_root,
        moves=tuple(moves),
    )


def rename_result_to_payload(
    *,
    execution: ApplyExecutionResult,
    finalize: MonitorFinalizeResult,
) -> dict[str, Any]:
    return {
        "version": 1,
        "dry_run": execution.apply_result.dry_run,
        "applied_count": execution.applied_count,
        "temp_dir": (
            str(execution.apply_result.temp_dir)
            if execution.apply_result.temp_dir is not None
            else None
        ),
        "finalize": {
            "status": finalize.status,
            "archive_path": (
                str(finalize.archive_path) if finalize.archive_path is not None else None
            ),
        },
    }
