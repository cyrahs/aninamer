from __future__ import annotations

from dataclasses import dataclass
import heapq
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


def _single_stage_order(
    moves_to_apply: list[tuple[PlannedMove, Path, Path]]
) -> list[int]:
    src_to_idx: dict[Path, int] = {}
    for idx, (_move, src, _dst) in enumerate(moves_to_apply):
        if src not in src_to_idx:
            src_to_idx[src] = idx

    adjacency: list[list[int]] = [[] for _ in range(len(moves_to_apply))]
    indegree = [0] * len(moves_to_apply)
    for idx, (_move, _src, dst) in enumerate(moves_to_apply):
        dep_idx = src_to_idx.get(dst)
        if dep_idx is None or dep_idx == idx:
            continue
        adjacency[dep_idx].append(idx)
        indegree[idx] += 1

    heap: list[int] = [idx for idx, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        idx = heapq.heappop(heap)
        order.append(idx)
        for neighbor in adjacency[idx]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, neighbor)

    if len(order) != len(moves_to_apply):
        raise ApplyError("cycle detected in rename plan; re-run with --two-stage")
    return order


def _apply_single_stage(
    plan: RenamePlan,
    moves_to_apply: list[tuple[PlannedMove, Path, Path]],
) -> ApplyResult:
    order = _single_stage_order(moves_to_apply)
    preview = [moves_to_apply[idx][2] for idx in order[:3]]
    logger.debug(
        "apply: single_stage_order move_count=%s dst_preview=%s",
        len(order),
        [str(path) for path in preview],
    )

    applied: list[tuple[PlannedMove, Path, Path]] = []
    try:
        for idx in order:
            move, src, dst = moves_to_apply[idx]
            dst.parent.mkdir(parents=True, exist_ok=True)
            logger.info("apply: single_move src=%s dst=%s", src, dst)
            shutil.move(str(src), str(dst))
            applied.append((move, src, dst))

        applied_moves = tuple(
            AppliedMove(
                src=move.src,
                dst=move.dst,
                kind=move.kind,
                src_id=move.src_id,
            )
            for move, _src, _dst in applied
        )
        logger.info("apply: done applied_count=%s", len(applied_moves))
        return ApplyResult(
            dry_run=False,
            applied=applied_moves,
            rollback_moves=build_rollback_moves(plan),
            temp_dir=None,
        )
    except Exception as exc:
        logger.exception("apply: failed; attempting rollback")
        for _move, src, dst in reversed(applied):
            try:
                if dst.exists():
                    logger.info("apply: rollback_move src=%s dst=%s", dst, src)
                    shutil.move(str(dst), str(src))
            except Exception:
                logger.info(
                    "apply: rollback_move failed src=%s dst=%s",
                    dst,
                    src,
                    exc_info=True,
                )
        raise ApplyError(f"apply failed: {exc}") from exc


def _apply_two_stage(
    plan: RenamePlan,
    moves_to_apply: list[tuple[PlannedMove, Path, Path]],
    output_root: Path,
) -> ApplyResult:
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
                    logger.info(
                        "apply: rollback_move src=%s dst=%s", tmp_path, src
                    )
                    shutil.move(str(tmp_path), str(src))
                elif dst.exists():
                    logger.info(
                        "apply: rollback_move src=%s dst=%s", dst, src
                    )
                    shutil.move(str(dst), str(src))
            except Exception:
                logger.info(
                    "apply: rollback_move failed src=%s dst=%s",
                    tmp_path if tmp_path.exists() else dst,
                    src,
                    exc_info=True,
                )
        raise ApplyError(f"apply failed: {exc}") from exc


def apply_rename_plan(
    plan: RenamePlan, *, dry_run: bool = True, two_stage: bool = False
) -> ApplyResult:
    logger.info(
        "apply: start dry_run=%s two_stage=%s move_count=%s",
        dry_run,
        two_stage,
        len(plan.moves),
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

    if not moves_to_apply:
        logger.info("apply: done applied_count=%s", 0)
        return ApplyResult(
            dry_run=False,
            applied=(),
            rollback_moves=build_rollback_moves(plan),
            temp_dir=None,
        )

    if two_stage:
        return _apply_two_stage(plan, moves_to_apply, output_root)
    return _apply_single_stage(plan, moves_to_apply)
