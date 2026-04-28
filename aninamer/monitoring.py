from __future__ import annotations

from dataclasses import dataclass
import errno
import logging
import os
from pathlib import Path
import time
from typing import Literal

from aninamer.plan import RenamePlan
from aninamer.scanner import SKIP_DIR_NAMES

logger = logging.getLogger(__name__)

MONITOR_ARCHIVE_DIR = "archive"
MONITOR_FAIL_DIR = "fail"
MONITOR_EXCLUDED_DIR_NAMES = frozenset({MONITOR_ARCHIVE_DIR, MONITOR_FAIL_DIR})
TRANSIENT_FILESYSTEM_ERRNOS = frozenset({errno.ENOTCONN, errno.EIO, errno.ESTALE})


@dataclass(frozen=True)
class MonitorTarget:
    key: str
    input_root: Path
    output_root: Path


@dataclass(frozen=True)
class MonitorFinalizeResult:
    status: Literal["deleted", "archived", "skipped"]
    archive_path: Path | None = None


@dataclass(frozen=True)
class SeriesDiscoveryResult:
    series_dirs: list[Path]
    unavailable: bool = False
    error_message: str | None = None


def is_within(child: Path, root: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        child_resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def max_tree_mtime(path: Path) -> float:
    now = time.time()
    latest = 0.0
    try:
        for root, dirs, files in os.walk(
            path,
            followlinks=False,
            onerror=_raise_walk_error,
        ):
            dirs[:] = [name for name in dirs if name.casefold() not in SKIP_DIR_NAMES]
            root_path = Path(root)
            try:
                latest = max(latest, root_path.stat().st_mtime)
            except OSError as exc:
                logger.warning(
                    "monitor: stat_unavailable path=%s error=%s",
                    root_path,
                    exc,
                )
                return now
            for name in files:
                file_path = root_path / name
                try:
                    latest = max(latest, file_path.stat().st_mtime)
                except OSError as exc:
                    logger.warning(
                        "monitor: stat_unavailable path=%s error=%s",
                        file_path,
                        exc,
                    )
                    return now
    except OSError as exc:
        logger.warning("monitor: walk_unavailable path=%s error=%s", path, exc)
        return now
    return latest


def is_settled(path: Path, settle_seconds: int, *, now: float | None = None) -> bool:
    if settle_seconds <= 0:
        return True
    if now is None:
        now = time.time()
    return (now - max_tree_mtime(path)) >= settle_seconds


def discover_series_dirs(input_root: Path) -> list[Path]:
    return discover_series_dirs_status(input_root).series_dirs


def discover_series_dirs_status(input_root: Path) -> SeriesDiscoveryResult:
    try:
        entries = list(input_root.iterdir())
    except FileNotFoundError:
        return SeriesDiscoveryResult(series_dirs=[])
    except OSError as exc:
        logger.warning(
            "monitor: discover_unavailable input_root=%s error=%s",
            input_root,
            exc,
        )
        return SeriesDiscoveryResult(
            series_dirs=[],
            unavailable=True,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    series_dirs: list[Path] = []
    unavailable_errors: list[str] = []
    for path in entries:
        if path.name.startswith("."):
            continue
        if path.name.casefold() in MONITOR_EXCLUDED_DIR_NAMES:
            continue
        try:
            if path.is_dir():
                series_dirs.append(path)
        except OSError as exc:
            logger.warning(
                "monitor: discover_entry_unavailable path=%s error=%s",
                path,
                exc,
            )
            if is_transient_filesystem_error(exc):
                unavailable_errors.append(f"{type(exc).__name__}: {exc}")
    series_dirs.sort(key=lambda path: path.name.casefold())
    if unavailable_errors and not series_dirs:
        return SeriesDiscoveryResult(
            series_dirs=[],
            unavailable=True,
            error_message=unavailable_errors[0],
        )
    return SeriesDiscoveryResult(series_dirs=series_dirs)


def snapshot_series_files(series_dir: Path) -> set[str]:
    try:
        if not series_dir.exists() or not series_dir.is_dir():
            return set()
    except OSError as exc:
        logger.warning(
            "monitor: snapshot_unavailable series_dir=%s error=%s",
            series_dir,
            exc,
        )
        raise
    files: set[str] = set()
    try:
        walker = os.walk(
            series_dir,
            followlinks=False,
            onerror=_raise_walk_error,
        )
        for root, _dirs, names in walker:
            root_path = Path(root)
            for name in names:
                file_path = root_path / name
                files.add(file_path.relative_to(series_dir).as_posix())
    except OSError as exc:
        logger.warning(
            "monitor: snapshot_unavailable series_dir=%s error=%s",
            series_dir,
            exc,
        )
        raise
    return files


def is_transient_filesystem_error(exc: OSError) -> bool:
    return exc.errno in TRANSIENT_FILESYSTEM_ERRNOS


def path_is_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except OSError as exc:
        logger.warning("monitor: path_unavailable path=%s error=%s", path, exc)
        raise


def _raise_walk_error(exc: OSError) -> None:
    raise exc


def plan_source_rel_paths(plan: RenamePlan, series_dir: Path) -> set[str]:
    source_paths: set[str] = set()
    resolved_series = series_dir.resolve(strict=False)
    for move in plan.moves:
        move_src = move.src.resolve(strict=False)
        try:
            rel = move_src.relative_to(resolved_series)
        except ValueError:
            continue
        source_paths.add(rel.as_posix())
    return source_paths


def prune_empty_tree(path: Path) -> bool:
    if not path.exists():
        return True
    for root, _dirs, files in os.walk(path, topdown=False, followlinks=False):
        if files:
            continue
        root_path = Path(root)
        try:
            root_path.rmdir()
        except OSError:
            continue
    return not path.exists()


def next_monitor_bucket_path(bucket_root: Path, series_name: str) -> Path:
    candidate = bucket_root / series_name
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        candidate = bucket_root / f"{series_name}.{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def move_series_dir_to_monitor_bucket(series_dir: Path, bucket_name: str) -> Path:
    bucket_root = series_dir.parent / bucket_name
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_path = next_monitor_bucket_path(bucket_root, series_dir.name)
    series_dir.rename(bucket_path)
    return bucket_path


def archive_series_dir(series_dir: Path) -> Path:
    return move_series_dir_to_monitor_bucket(series_dir, MONITOR_ARCHIVE_DIR)


def move_series_dir_to_fail(series_dir: Path) -> Path:
    return move_series_dir_to_monitor_bucket(series_dir, MONITOR_FAIL_DIR)


def finalize_series_dir_after_apply(
    series_dir: Path,
    plan: RenamePlan,
    *,
    before_files: set[str],
) -> MonitorFinalizeResult:
    after_files = snapshot_series_files(series_dir)
    expected_after = before_files - plan_source_rel_paths(plan, series_dir)
    if after_files != expected_after:
        logger.info(
            "monitor: source_changed_skip_finalize series_dir=%s expected_remaining=%s actual_remaining=%s",
            series_dir,
            len(expected_after),
            len(after_files),
        )
        return MonitorFinalizeResult(status="skipped")

    if prune_empty_tree(series_dir):
        logger.info("monitor: source_deleted series_dir=%s", series_dir)
        return MonitorFinalizeResult(status="deleted")

    archive_path = archive_series_dir(series_dir)
    logger.info(
        "monitor: source_archived series_dir=%s archive_path=%s",
        series_dir,
        archive_path,
    )
    return MonitorFinalizeResult(status="archived", archive_path=archive_path)
