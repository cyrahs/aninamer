from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import signal
import sys
from pathlib import Path
import time
from typing import Callable, Sequence

from aninamer import __version__
from aninamer.apply import apply_rename_plan
from aninamer.episode_mapping import map_episodes_with_llm
from aninamer.errors import (
    ApplyError,
    LLMOutputError,
    OpenAIError,
    PlanValidationError,
)
from aninamer.llm_client import LLMClient
from aninamer.logging_utils import configure_logging
from aninamer.name_clean import build_tmdb_query_variants, extract_tmdb_id_tag
from aninamer.openai_llm_client import (
    openai_llm_for_tmdb_id_from_env,
    openai_llm_from_env,
)
from aninamer.plan import (
    RenamePlan,
    build_rename_plan,
    format_season_folder,
    format_series_root_folder,
)
from aninamer.plan_io import read_rename_plan_json, write_rename_plan_json
from aninamer.scanner import SKIP_DIR_NAMES, scan_series_dir
from aninamer.tmdb_client import (
    TMDBClient,
    TMDBError,
    TvSearchResult,
)
from aninamer.tmdb_resolve import (
    resolve_tmdb_search_title_with_llm,
    resolve_tmdb_tv_id_with_llm,
)

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aninamer")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (repeatable).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Explicit log level override.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("./logs"),
        help="Directory for logs and operational artifacts.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Build a rename plan.")
    plan_parser.add_argument("series_dir", type=Path, help="Series directory.")
    plan_parser.add_argument("--out", type=Path, required=True, help="Output root.")
    plan_parser.add_argument("--tmdb", type=int, help="TMDB TV id.")
    plan_parser.add_argument("--plan-file", type=Path, help="Output plan file path.")
    plan_parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Max TMDB candidates for LLM selection.",
    )
    plan_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max output tokens for mapping LLM.",
    )
    plan_parser.add_argument(
        "--allow-existing-dest",
        action="store_true",
        help="Allow pre-existing destinations.",
    )

    run_parser = subparsers.add_parser("run", help="Plan and optionally apply renames.")
    run_parser.add_argument("series_dir", type=Path, help="Series directory.")
    run_parser.add_argument("--out", type=Path, required=True, help="Output root.")
    run_parser.add_argument("--tmdb", type=int, help="TMDB TV id.")
    run_parser.add_argument("--plan-file", type=Path, help="Output plan file path.")
    run_parser.add_argument(
        "--rollback-file",
        type=Path,
        help="Rollback plan output path.",
    )
    run_parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Max TMDB candidates for LLM selection.",
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max output tokens for mapping LLM.",
    )
    run_parser.add_argument(
        "--allow-existing-dest",
        action="store_true",
        help="Allow pre-existing destinations.",
    )
    run_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves (default is plan only).",
    )
    run_parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage moves with a staging temp dir.",
    )

    apply_parser = subparsers.add_parser("apply", help="Apply a rename plan.")
    apply_parser.add_argument("plan_json", type=Path, help="Rename plan JSON file.")
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without moving files.",
    )
    apply_parser.add_argument(
        "--rollback-file",
        type=Path,
        help="Rollback plan output path.",
    )
    apply_parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage moves with a staging temp dir.",
    )

    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor input root for series directories."
    )
    monitor_parser.add_argument("input_root", type=Path, help="Input root directory.")
    monitor_parser.add_argument("--out", type=Path, required=True, help="Output root.")
    monitor_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply moves (default is plan only).",
    )
    monitor_parser.add_argument(
        "--once",
        action="store_true",
        help="Run one iteration and exit.",
    )
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between iterations.",
    )
    monitor_parser.add_argument(
        "--settle-seconds",
        type=int,
        default=15,
        help="Require directory contents to be unchanged for N seconds.",
    )
    monitor_parser.add_argument(
        "--state-file",
        type=Path,
        help="Path to monitor state file.",
    )
    monitor_parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Process existing directories instead of only new arrivals.",
    )
    monitor_parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Use two-stage moves with a staging temp dir.",
    )
    monitor_parser.add_argument("--tmdb", type=int, help="TMDB TV id.")
    monitor_parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Max TMDB candidates for LLM selection.",
    )
    monitor_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Max output tokens for mapping LLM.",
    )
    monitor_parser.add_argument(
        "--allow-existing-dest",
        action="store_true",
        help="Allow pre-existing destinations.",
    )

    return parser


def _log_level_from_args(args: argparse.Namespace) -> str:
    if args.log_level:
        return args.log_level
    return "DEBUG" if args.verbose and args.verbose > 0 else "INFO"


def _tmdb_client_from_env() -> TMDBClient:
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TMDB_API_KEY is required")
    return TMDBClient(api_key=api_key, timeout=30.0)


def _resolve_series_name(details_name: str | None, original: str | None) -> str:
    if details_name is not None:
        cleaned = details_name.strip()
        if cleaned:
            return cleaned
    if original is not None:
        cleaned = original.strip()
        if cleaned:
            return cleaned
    return "Unknown"


def _list_existing_s00_files(
    output_root: Path,
    series_name_zh_cn: str,
    year: int | None,
    tmdb_id: int,
) -> list[str]:
    series_folder = format_series_root_folder(series_name_zh_cn, year, tmdb_id)
    s00_dir = output_root / series_folder / format_season_folder(0)
    if not s00_dir.exists() or not s00_dir.is_dir():
        return []
    names = [path.name for path in s00_dir.iterdir() if path.is_file()]
    names.sort(key=lambda value: value.casefold())
    return names


def _search_tmdb_candidates(
    tmdb: TMDBClient,
    name: str,
    *,
    llm_title_factory: Callable[[], LLMClient] | None = None,
) -> list[TvSearchResult]:
    queries = build_tmdb_query_variants(name)
    languages = ["zh-CN", "en-US", "ja-JP"]
    logger.info("tmdb_search: start name=%s variants=%s", name, len(queries))

    def _search_queries(query_list: Sequence[str]) -> list[TvSearchResult]:
        for query in query_list:
            candidates_by_id: dict[int, TvSearchResult] = {}
            first_language_with_results: str | None = None
            for language in languages:
                results = tmdb.search_tv(query, language=language, page=1)
                logger.info(
                    "tmdb_search: attempt query=%s language=%s results=%s",
                    query,
                    language,
                    len(results),
                )
                if results and first_language_with_results is None:
                    first_language_with_results = language
                for candidate in results:
                    if candidate.id not in candidates_by_id:
                        candidates_by_id[candidate.id] = candidate
            if candidates_by_id:
                logger.info(
                    "tmdb_search: success query=%s language=%s candidates=%s",
                    query,
                    first_language_with_results or languages[0],
                    len(candidates_by_id),
                )
                return list(candidates_by_id.values())
        return []

    candidates = _search_queries(queries)
    if candidates:
        return candidates

    if llm_title_factory is None:
        logger.error("tmdb_search: failed name=%s queries=%s", name, queries)
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        )

    logger.info("tmdb_search: llm_title_fallback name=%s", name)
    try:
        llm = llm_title_factory()
    except Exception as exc:
        logger.warning("tmdb_search: llm_title_unavailable error=%s", exc)
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        ) from exc

    try:
        llm_title = resolve_tmdb_search_title_with_llm(name, llm)
    except (OpenAIError, LLMOutputError) as exc:
        logger.warning("tmdb_search: llm_title_failed error=%s", exc)
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        ) from exc

    llm_queries = build_tmdb_query_variants(llm_title)
    attempted = {query.casefold() for query in queries}
    llm_queries = [query for query in llm_queries if query.casefold() not in attempted]
    if not llm_queries:
        logger.error(
            "tmdb_search: llm_title_no_new_queries name=%s llm_title=%s",
            name,
            llm_title,
        )
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        )

    candidates = _search_queries(llm_queries)
    if candidates:
        return candidates

    all_queries = queries + llm_queries
    logger.error(
        "tmdb_search: failed name=%s queries=%s llm_title=%s",
        name,
        all_queries,
        llm_title,
    )
    raise ValueError(
        f"no TMDB results for name '{name}' (attempted queries: {all_queries})"
    )


def _print_plan_summary(plan: RenamePlan, plan_file: Path) -> None:
    video_moves = sum(1 for move in plan.moves if move.kind == "video")
    subtitle_moves = sum(1 for move in plan.moves if move.kind == "subtitle")
    print(f"Plan file: {plan_file}")
    print(f"Moves: {video_moves} videos, {subtitle_moves} subtitles")
    for move in plan.moves[:5]:
        print(f"{move.src} -> {move.dst}")


def _print_apply_summary(
    dry_run: bool, applied_count: int, rollback_file: Path
) -> None:
    status = "dry-run" if dry_run else "applied"
    print(f"Apply {status}: {applied_count} moves")
    print(f"Rollback plan: {rollback_file}")


def _print_run_plan_summary(plan: RenamePlan, plan_file: Path) -> None:
    video_moves = sum(1 for move in plan.moves if move.kind == "video")
    subtitle_moves = sum(1 for move in plan.moves if move.kind == "subtitle")
    print(f"wrote plan to {plan_file}")
    print(f"moves: videos={video_moves} subtitles={subtitle_moves}")


def _print_run_apply_summary(applied_count: int, rollback_file: Path) -> None:
    print(f"applied moves: {applied_count}")
    print(f"wrote rollback to {rollback_file}")


_INVALID_FILENAME_CHARS = set('/\\:*?"<>|')


def _safe_filename_component(name: str, *, max_len: int = 80) -> str:
    cleaned = "".join("_" if ch in _INVALID_FILENAME_CHARS else ch for ch in name)
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        cleaned = "Unknown"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip()
        if not cleaned:
            cleaned = "Unknown"
    return cleaned


def _hash8_for_path(p: Path) -> str:
    resolved = p.resolve(strict=False)
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()
    return digest[:8]


def _default_plan_paths(log_path: Path, series_dir: Path) -> tuple[Path, Path]:
    plans_dir = log_path / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_filename_component(series_dir.name)
    hash8 = _hash8_for_path(series_dir)
    plan_path = plans_dir / f"{safe}_{hash8}.rename_plan.json"
    rollback_path = plans_dir / f"{safe}_{hash8}.rollback_plan.json"
    return plan_path, rollback_path


def _default_rollback_path_from_plan(log_path: Path, plan_path: Path) -> Path:
    plans_dir = log_path / "plans"
    if plan_path.name.endswith(".rename_plan.json"):
        base = plan_path.name[: -len(".rename_plan.json")]
        filename = f"{base}.rollback_plan.json"
    else:
        filename = f"{plan_path.stem}.rollback_plan.json"
    return plans_dir / filename


def _is_within(child: Path, root: Path) -> bool:
    child_resolved = child.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    try:
        child_resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def _ensure_not_within(
    artifact: Path,
    series_dir: Path,
    output_root: Path,
    label: str,
) -> None:
    if _is_within(artifact, series_dir) or _is_within(artifact, output_root):
        raise ValueError(
            f"Refusing to write {label} under series_dir/output_root. Use --log-path or choose a different path."
        )


@dataclass
class MonitorState:
    baseline: set[str]
    pending: set[str]
    planned: set[str]
    processed: set[str]
    failed: set[str]


def _empty_monitor_state() -> MonitorState:
    return MonitorState(
        baseline=set(),
        pending=set(),
        planned=set(),
        processed=set(),
        failed=set(),
    )


def _load_state_list(data: dict[str, object], field: str) -> set[str]:
    value = data.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"state {field} must be a list of strings")
    return set(value)


def _load_monitor_state(path: Path) -> tuple[MonitorState, str]:
    if not path.exists():
        return _empty_monitor_state(), "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("state must be an object")
        version = data.get("version")
        if version == 1:
            processed = _load_state_list(data, "processed")
            return (
                MonitorState(
                    baseline=set(),
                    pending=set(),
                    planned=set(),
                    processed=processed,
                    failed=set(),
                ),
                "v1",
            )
        if version == 2:
            baseline = _load_state_list(data, "baseline")
            pending = _load_state_list(data, "pending")
            planned = _load_state_list(data, "planned")
            processed = _load_state_list(data, "processed")
            return (
                MonitorState(
                    baseline=baseline,
                    pending=pending,
                    planned=planned,
                    processed=processed,
                    failed=set(),
                ),
                "v2",
            )
        if version == 3:
            baseline = _load_state_list(data, "baseline")
            pending = _load_state_list(data, "pending")
            planned = _load_state_list(data, "planned")
            processed = _load_state_list(data, "processed")
            failed = _load_state_list(data, "failed")
            return (
                MonitorState(
                    baseline=baseline,
                    pending=pending,
                    planned=planned,
                    processed=processed,
                    failed=failed,
                ),
                "v3",
            )
        raise ValueError("state version must be 1, 2, or 3")
    except Exception as exc:
        logger.warning(
            "monitor: state_read_failed path=%s error=%s",
            path,
            exc,
        )
        return _empty_monitor_state(), "invalid"


def _write_monitor_state(path: Path, state: MonitorState) -> None:
    payload = {
        "version": 3,
        "baseline": sorted(state.baseline),
        "pending": sorted(state.pending),
        "planned": sorted(state.planned),
        "processed": sorted(state.processed),
        "failed": sorted(state.failed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _max_tree_mtime(path: Path) -> float:
    now = time.time()
    latest = 0.0
    try:
        for root, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if name.casefold() not in SKIP_DIR_NAMES]
            root_path = Path(root)
            try:
                latest = max(latest, root_path.stat().st_mtime)
            except FileNotFoundError:
                return now
            for name in files:
                file_path = root_path / name
                try:
                    latest = max(latest, file_path.stat().st_mtime)
                except FileNotFoundError:
                    return now
    except FileNotFoundError:
        return now
    return latest


def _is_settled(path: Path, settle_seconds: int, *, now: float | None = None) -> bool:
    if settle_seconds <= 0:
        return True
    if now is None:
        now = time.time()
    return (now - _max_tree_mtime(path)) >= settle_seconds


def _discover_series_dirs(input_root: Path) -> list[Path]:
    try:
        entries = list(input_root.iterdir())
    except FileNotFoundError:
        return []
    series_dirs = [
        path for path in entries if path.is_dir() and not path.name.startswith(".")
    ]
    series_dirs.sort(key=lambda path: path.name.casefold())
    return series_dirs


def _build_plan_from_args(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> tuple[RenamePlan, Path]:
    series_dir = args.series_dir
    output_root = args.out

    scan = scan_series_dir(series_dir)

    tmdb = tmdb_client_factory() if tmdb_client_factory else _tmdb_client_from_env()

    tag_tmdb_id = extract_tmdb_id_tag(series_dir.name)
    if args.tmdb is not None:
        if tag_tmdb_id is not None and tag_tmdb_id != args.tmdb:
            raise ValueError(
                f"tmdb tag {tag_tmdb_id} does not match --tmdb {args.tmdb} for {series_dir.name}"
            )
        tmdb_id = args.tmdb
    elif tag_tmdb_id is not None:
        tmdb_id = tag_tmdb_id
        logger.info(
            "tmdb_resolve: using tag id=%s dirname=%s",
            tmdb_id,
            series_dir.name,
        )
    else:
        llm_title_factory = (
            llm_for_tmdb_id_factory
            if llm_for_tmdb_id_factory
            else openai_llm_for_tmdb_id_from_env
        )
        candidates = _search_tmdb_candidates(
            tmdb,
            series_dir.name,
            llm_title_factory=llm_title_factory,
        )
        if len(candidates) == 1:
            tmdb_id = candidates[0].id
        else:
            llm_for_id = (
                llm_for_tmdb_id_factory()
                if llm_for_tmdb_id_factory
                else openai_llm_for_tmdb_id_from_env()
            )
            tmdb_id = resolve_tmdb_tv_id_with_llm(
                dirname=series_dir.name,
                candidates=candidates,
                llm=llm_for_id,
                max_candidates=args.max_candidates,
            )

    series_name_zh_cn, details = tmdb.resolve_series_title(tmdb_id)
    year = details.year
    season_episode_counts = {
        season.season_number: season.episode_count for season in details.seasons
    }

    specials_count = season_episode_counts.get(0, 0)
    if specials_count > 0:
        specials_zh = tmdb.get_season(tmdb_id, 0, language="zh-CN")
        specials_en = tmdb.get_season(tmdb_id, 0, language="en-US")
    else:
        specials_zh = None
        specials_en = None

    llm_for_mapping = (
        llm_for_mapping_factory() if llm_for_mapping_factory else openai_llm_from_env()
    )
    existing_s00_files = _list_existing_s00_files(
        output_root,
        series_name_zh_cn,
        year,
        tmdb_id,
    )
    mapping = map_episodes_with_llm(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        season_episode_counts=season_episode_counts,
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
        existing_s00_files=existing_s00_files,
        llm=llm_for_mapping,
        max_output_tokens=args.max_output_tokens,
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        tmdb_id=tmdb_id,
        output_root=output_root,
        allow_existing_dest=args.allow_existing_dest,
    )

    plan_file = args.plan_file
    if plan_file is None:
        raise ValueError("plan_file is required")
    return plan, plan_file


def _do_plan(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
    summary: Callable[[RenamePlan, Path], None],
) -> tuple[RenamePlan, Path]:
    plan, plan_file = _build_plan_from_args(
        args,
        tmdb_client_factory=tmdb_client_factory,
        llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
        llm_for_mapping_factory=llm_for_mapping_factory,
    )
    write_rename_plan_json(plan_file, plan)
    summary(plan, plan_file)
    return plan, plan_file


def _do_apply_from_plan(
    plan: RenamePlan,
    *,
    rollback_file: Path,
    dry_run: bool,
    two_stage: bool,
) -> int:
    result = apply_rename_plan(plan, dry_run=dry_run, two_stage=two_stage)
    rollback_plan = RenamePlan(
        tmdb_id=plan.tmdb_id,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        series_dir=plan.series_dir,
        output_root=plan.output_root,
        moves=result.rollback_moves,
    )
    write_rename_plan_json(rollback_file, rollback_plan)
    return len(result.applied)


def _run_plan(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> int:
    log_path = Path(args.log_path)
    default_plan_path, _ = _default_plan_paths(log_path, args.series_dir)
    plan_file = args.plan_file or default_plan_path
    _ensure_not_within(plan_file, args.series_dir, args.out, "plan file")
    args.plan_file = plan_file
    _do_plan(
        args,
        tmdb_client_factory=tmdb_client_factory,
        llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
        llm_for_mapping_factory=llm_for_mapping_factory,
        summary=_print_plan_summary,
    )
    return 0


def _run_apply(
    args: argparse.Namespace,
    *,
    log_path: Path,
    plan: RenamePlan | None = None,
) -> int:
    plan_path = args.plan_json
    if plan is None:
        plan = read_rename_plan_json(plan_path)

    rollback_file = args.rollback_file or _default_rollback_path_from_plan(
        log_path, plan_path
    )
    _ensure_not_within(
        rollback_file, plan.series_dir, plan.output_root, "rollback file"
    )
    applied_count = _do_apply_from_plan(
        plan,
        rollback_file=rollback_file,
        dry_run=args.dry_run,
        two_stage=args.two_stage,
    )
    _print_apply_summary(args.dry_run, applied_count, rollback_file)
    return 0


def _run_run(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> int:
    log_path = Path(args.log_path)
    default_plan_path, default_rollback_path = _default_plan_paths(
        log_path, args.series_dir
    )
    plan_file = args.plan_file or default_plan_path
    _ensure_not_within(plan_file, args.series_dir, args.out, "plan file")
    args.plan_file = plan_file
    plan, plan_file = _do_plan(
        args,
        tmdb_client_factory=tmdb_client_factory,
        llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
        llm_for_mapping_factory=llm_for_mapping_factory,
        summary=_print_run_plan_summary,
    )
    if not args.apply:
        return 0

    rollback_file = args.rollback_file or default_rollback_path
    _ensure_not_within(rollback_file, args.series_dir, args.out, "rollback file")
    applied_count = _do_apply_from_plan(
        plan,
        rollback_file=rollback_file,
        dry_run=False,
        two_stage=args.two_stage,
    )
    _print_run_apply_summary(applied_count, rollback_file)
    return 0


def _run_monitor(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> int:
    input_root = args.input_root
    output_root = args.out
    log_path = Path(args.log_path)
    state_file = args.state_file or (log_path / "monitor_state.json")
    _ensure_not_within(state_file, input_root, output_root, "state file")

    logger.info(
        "monitor: start input_root=%s out_root=%s apply=%s once=%s interval=%s settle_seconds=%s",
        input_root,
        output_root,
        args.apply,
        args.once,
        args.interval,
        args.settle_seconds,
    )

    # Set up signal handling for graceful shutdown
    shutdown_requested = False

    def _handle_shutdown_signal(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.info("monitor: received signal=%s, shutting down gracefully", sig_name)
        shutdown_requested = True

    original_sigterm = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    original_sigint = signal.signal(signal.SIGINT, _handle_shutdown_signal)

    try:
        return _monitor_loop(
            args,
            input_root=input_root,
            output_root=output_root,
            log_path=log_path,
            state_file=state_file,
            tmdb_client_factory=tmdb_client_factory,
            llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
            llm_for_mapping_factory=llm_for_mapping_factory,
            shutdown_check=lambda: shutdown_requested,
        )
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)


def _monitor_loop(
    args: argparse.Namespace,
    *,
    input_root: Path,
    output_root: Path,
    log_path: Path,
    state_file: Path,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
    shutdown_check: Callable[[], bool],
) -> int:
    tmdb = tmdb_client_factory() if tmdb_client_factory else _tmdb_client_from_env()
    llm_for_id_factory: Callable[[], LLMClient] | None = None
    if args.tmdb is None:
        llm_for_id_factory = (
            llm_for_tmdb_id_factory
            if llm_for_tmdb_id_factory
            else openai_llm_for_tmdb_id_from_env
        )
    llm_for_mapping = (
        llm_for_mapping_factory() if llm_for_mapping_factory else openai_llm_from_env()
    )

    baseline_bootstrapped = False

    while not shutdown_check():
        state, state_origin = _load_monitor_state(state_file)
        series_dirs = _discover_series_dirs(input_root)
        logger.debug("monitor: discovered count=%s", len(series_dirs))
        resolved_map = {
            str(series_dir.resolve(strict=False)): series_dir
            for series_dir in series_dirs
        }
        current_dirs = set(resolved_map.keys())

        if (
            not args.include_existing
            and not baseline_bootstrapped
            and state_origin in ("missing", "v1", "invalid")
        ):
            baseline = current_dirs - state.processed
            state = MonitorState(
                baseline=baseline,
                pending=set(),
                planned=set(),
                processed=set(state.processed),
                failed=set(state.failed),
            )
            _write_monitor_state(state_file, state)
            logger.info(
                "monitor: baseline_bootstrap baseline_count=%s",
                len(baseline),
            )
            baseline_bootstrapped = True
            if args.once:
                break
            # Sleep in small increments to allow faster shutdown response
            sleep_remaining = args.interval
            while sleep_remaining > 0 and not shutdown_check():
                sleep_chunk = min(sleep_remaining, 1.0)
                time.sleep(sleep_chunk)
                sleep_remaining -= sleep_chunk
            continue

        # Clean up failed entries from pending/planned
        failed_cleanup_dirty = False
        if state.failed:
            if state.pending.intersection(state.failed):
                state.pending.difference_update(state.failed)
                failed_cleanup_dirty = True
            if state.planned.intersection(state.failed):
                state.planned.difference_update(state.failed)
                failed_cleanup_dirty = True
        if failed_cleanup_dirty:
            _write_monitor_state(state_file, state)

        # Discover new directories and add to pending
        new_pending_dirty = False
        for series_dir in series_dirs:
            resolved = str(series_dir.resolve(strict=False))
            if resolved in state.processed:
                logger.debug("monitor: skip_processed series_dir=%s", series_dir)
                continue
            if resolved in state.failed:
                logger.debug("monitor: skip_failed series_dir=%s", series_dir)
                continue
            if not args.include_existing and resolved in state.baseline:
                logger.debug("monitor: skip_baseline series_dir=%s", series_dir)
                continue
            if resolved in state.planned or resolved in state.pending:
                continue
            state.pending.add(resolved)
            new_pending_dirty = True
            logger.info("monitor: new_dir series_dir=%s", series_dir)
        if new_pending_dirty:
            _write_monitor_state(state_file, state)

        if args.apply:
            for resolved in sorted(state.planned):
                if resolved in state.failed:
                    state.planned.discard(resolved)
                    _write_monitor_state(state_file, state)
                    continue
                series_dir = resolved_map.get(resolved, Path(resolved))
                plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
                _ensure_not_within(plan_path, series_dir, output_root, "plan file")
                _ensure_not_within(
                    rollback_path, series_dir, output_root, "rollback file"
                )
                try:
                    plan = read_rename_plan_json(plan_path)
                    applied_count = _do_apply_from_plan(
                        plan,
                        rollback_file=rollback_path,
                        dry_run=False,
                        two_stage=args.two_stage,
                    )
                    logger.info(
                        "monitor: applied series_dir=%s applied_count=%s",
                        series_dir,
                        applied_count,
                    )
                    state.planned.discard(resolved)
                    state.processed.add(resolved)
                    _write_monitor_state(state_file, state)
                except Exception as exc:
                    logger.exception(
                        "monitor: failed series_dir=%s error=%s",
                        series_dir,
                        exc,
                    )
                    state.planned.discard(resolved)
                    state.failed.add(resolved)
                    _write_monitor_state(state_file, state)
                    continue

        for resolved in sorted(state.pending):
            if resolved in state.failed:
                state.pending.discard(resolved)
                _write_monitor_state(state_file, state)
                continue
            series_dir = resolved_map.get(resolved, Path(resolved))
            if resolved not in current_dirs:
                continue
            if args.settle_seconds > 0:
                now = time.time()
                max_mtime = _max_tree_mtime(series_dir)
                age_seconds = max(0.0, now - max_mtime)
                if age_seconds < args.settle_seconds:
                    logger.info(
                        "monitor: pending_not_settled series_dir=%s age_seconds=%.1f",
                        series_dir,
                        age_seconds,
                    )
                    continue

            plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
            _ensure_not_within(plan_path, series_dir, output_root, "plan file")
            if args.apply:
                _ensure_not_within(
                    rollback_path, series_dir, output_root, "rollback file"
                )
            plan_args = argparse.Namespace(
                series_dir=series_dir,
                out=output_root,
                tmdb=args.tmdb,
                plan_file=plan_path,
                max_candidates=args.max_candidates,
                max_output_tokens=args.max_output_tokens,
                allow_existing_dest=args.allow_existing_dest,
            )
            try:
                plan, _ = _build_plan_from_args(
                    plan_args,
                    tmdb_client_factory=lambda: tmdb,
                    llm_for_tmdb_id_factory=llm_for_id_factory,
                    llm_for_mapping_factory=lambda: llm_for_mapping,
                )
                write_rename_plan_json(plan_path, plan)
                logger.info(
                    "monitor: planned plan_path=%s moves_count=%s",
                    plan_path,
                    len(plan.moves),
                )

                # Move from pending to planned immediately after plan is written
                # This ensures recovery works if apply crashes
                state.pending.discard(resolved)
                state.planned.add(resolved)
                _write_monitor_state(state_file, state)

                if args.apply:
                    applied_count = _do_apply_from_plan(
                        plan,
                        rollback_file=rollback_path,
                        dry_run=False,
                        two_stage=args.two_stage,
                    )
                    logger.info(
                        "monitor: applied series_dir=%s applied_count=%s",
                        series_dir,
                        applied_count,
                    )
                    state.planned.discard(resolved)
                    state.processed.add(resolved)
                    _write_monitor_state(state_file, state)
            except Exception as exc:
                logger.exception(
                    "monitor: failed series_dir=%s error=%s",
                    series_dir,
                    exc,
                )
                # Remove from pending or planned (depending on where failure occurred)
                state.pending.discard(resolved)
                state.planned.discard(resolved)
                state.failed.add(resolved)
                _write_monitor_state(state_file, state)
                continue

        if args.once:
            break

        # Sleep in small increments to allow faster shutdown response
        sleep_remaining = args.interval
        while sleep_remaining > 0 and not shutdown_check():
            sleep_chunk = min(sleep_remaining, 1.0)
            time.sleep(sleep_chunk)
            sleep_remaining -= sleep_chunk

    if shutdown_check():
        logger.info("monitor: shutdown complete")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None = None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None = None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None = None,
) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    log_path = Path(args.log_path)
    log_level = _log_level_from_args(args)
    plan_for_apply: RenamePlan | None = None
    try:
        if args.command == "apply":
            plan_for_apply = read_rename_plan_json(args.plan_json)
            _ensure_not_within(
                log_path,
                plan_for_apply.series_dir,
                plan_for_apply.output_root,
                "log path",
            )
        elif args.command in ("plan", "run"):
            _ensure_not_within(log_path, args.series_dir, args.out, "log path")
        elif args.command == "monitor":
            _ensure_not_within(log_path, args.input_root, args.out, "log path")
    except (OSError, PlanValidationError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    configure_logging(log_level, log_path=log_path)
    logger.info("aninamer v%s", __version__)

    try:
        if args.command == "plan":
            return _run_plan(
                args,
                tmdb_client_factory=tmdb_client_factory,
                llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
                llm_for_mapping_factory=llm_for_mapping_factory,
            )
        if args.command == "run":
            return _run_run(
                args,
                tmdb_client_factory=tmdb_client_factory,
                llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
                llm_for_mapping_factory=llm_for_mapping_factory,
            )
        if args.command == "apply":
            return _run_apply(args, log_path=log_path, plan=plan_for_apply)
        if args.command == "monitor":
            return _run_monitor(
                args,
                tmdb_client_factory=tmdb_client_factory,
                llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
                llm_for_mapping_factory=llm_for_mapping_factory,
            )
        logger.error("unknown command: %s", args.command)
        return 1
    except (
        ValueError,
        TMDBError,
        OpenAIError,
        LLMOutputError,
        PlanValidationError,
        ApplyError,
    ) as exc:
        logger.exception("command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
