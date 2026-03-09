from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from aninamer.artifacts import rename_result_to_payload
from aninamer.monitoring import MonitorFinalizeResult
from aninamer.pipeline import ApplyExecutionResult


def write_rename_result_json(
    path: Path,
    *,
    execution: ApplyExecutionResult,
    finalize: MonitorFinalizeResult,
) -> None:
    payload = rename_result_to_payload(execution=execution, finalize=finalize)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


FinalizeStatus = Literal["deleted", "archived", "skipped"]
