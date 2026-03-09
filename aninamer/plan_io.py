from __future__ import annotations

import json
from pathlib import Path

from aninamer.artifacts import rename_plan_from_payload, rename_plan_to_payload
from aninamer.errors import PlanValidationError
from aninamer.plan import RenamePlan


def rename_plan_to_dict(plan: RenamePlan) -> dict:
    return rename_plan_to_payload(plan)


def rename_plan_from_dict(data: dict) -> RenamePlan:
    return rename_plan_from_payload(data)


def write_rename_plan_json(path: Path, plan: RenamePlan) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rename_plan_to_dict(plan)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def read_rename_plan_json(path: Path) -> RenamePlan:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PlanValidationError("plan json must be an object")
    return rename_plan_from_dict(data)
