from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import shutil
from typing import Literal
import uuid

from aninamer.errors import ApplyError
from aninamer.plan import PlannedMove, RenamePlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppliedMove:
    src: Path
    dst: Path
    kind: Literal["video", "subtitle"]
    src_id: int


@dataclass(frozen=True)
class ApplyResult:
    dry_run: bool
    applied: tuple[AppliedMove, ...]
    rollback_moves: tuple[PlannedMove, ...]
    temp_dir: Path | None


def build_rollback_moves(plan: RenamePlan) -> tuple[PlannedMove, ...]:
    return tuple(
        PlannedMove(
            src=move.dst,
            dst=move.src,
            kind=move.kind,
            src_id=move.src_id,
        )
        for move in plan.moves
    )


def _resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _validate_parent_creatable(dst: Path) -> None:
    current = _resolve_path(dst.parent)
    while True:
        if current.exists() and not current.is_dir():
            raise ApplyError(f"destination parent {current} is not a directory")
        if current == current.parent:
            break
        current = current.parent


def _unique_temp_path(temp_dir: Path, base_name: str) -> Path:
    candidate = temp_dir / base_name
    if not candidate.exists():
        return candidate
    counter = 1
    while True:
        candidate = temp_dir / f"{base_name}.{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


def apply_rename_plan(plan: RenamePlan, *, dry_run: bool = True) -> ApplyResult:
    logger.info(
        "apply: start dry_run=%s move_count=%s", dry_run, len(plan.moves)
    )

    output_root = _resolve_path(plan.output_root)
    sources_set = {_resolve_path(move.src) for move in plan.moves}
    moves_to_apply: list[tuple[PlannedMove, Path, Path]] = []

    for move in plan.moves:
        src = _resolve_path(move.src)
        dst = _resolve_path(move.dst)
        if not src.exists() or not src.is_file():
            raise ApplyError(f"source {src} does not exist or is not a file")
        _validate_parent_creatable(dst)
        if dst.exists() and dst not in sources_set:
            raise ApplyError(f"destination already exists: {dst}")
        if src == dst:
            continue
        moves_to_apply.append((move, src, dst))

    if dry_run:
        logger.info("apply: done applied_count=%s", 0)
        return ApplyResult(
            dry_run=True,
            applied=(),
            rollback_moves=build_rollback_moves(plan),
            temp_dir=None,
        )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ApplyError(
            f"failed to create output_root {output_root}: {exc}"
        ) from exc

    temp_dir = output_root / f".aninamer_tmp_{uuid.uuid4().hex}"
    try:
        temp_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ApplyError(f"failed to create temp dir {temp_dir}: {exc}") from exc

    staged: list[tuple[PlannedMove, Path, Path, Path]] = []
    try:
        for move, src, dst in moves_to_apply:
            base_name = f"{move.kind}_{move.src_id}{src.suffix}"
            tmp_path = _unique_temp_path(temp_dir, base_name)
            if tmp_path.exists():
                raise ApplyError(f"temp path already exists: {tmp_path}")
            logger.debug("apply: stage_move src=%s tmp=%s", src, tmp_path)
            shutil.move(str(src), str(tmp_path))
            staged.append((move, src, dst, tmp_path))

        logger.info(
            "apply: staged staged_count=%s temp_dir=%s",
            len(staged),
            temp_dir,
        )

        for move, _src, dst, tmp_path in staged:
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("apply: final_move tmp=%s dst=%s", tmp_path, dst)
            shutil.move(str(tmp_path), str(dst))

        applied = tuple(
            AppliedMove(
                src=move.src,
                dst=move.dst,
                kind=move.kind,
                src_id=move.src_id,
            )
            for move, _src, _dst, _tmp in staged
        )

        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

        logger.info("apply: done applied_count=%s", len(applied))
        return ApplyResult(
            dry_run=False,
            applied=applied,
            rollback_moves=build_rollback_moves(plan),
            temp_dir=temp_dir,
        )
    except Exception as exc:
        logger.exception("apply: failed; attempting rollback")
        for move, src, dst, tmp_path in reversed(staged):
            try:
                if tmp_path.exists():
                    logger.debug(
                        "apply: rollback_move src=%s dst=%s", tmp_path, src
                    )
                    shutil.move(str(tmp_path), str(src))
                elif dst.exists():
                    logger.debug(
                        "apply: rollback_move src=%s dst=%s", dst, src
                    )
                    shutil.move(str(dst), str(src))
            except Exception:
                logger.debug(
                    "apply: rollback_move failed src=%s dst=%s",
                    tmp_path if tmp_path.exists() else dst,
                    src,
                    exc_info=True,
                )
        raise ApplyError(f"apply failed: {exc}") from exc
