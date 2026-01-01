from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Sequence

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
from aninamer.openai_llm_client import (
    openai_llm_for_tmdb_id_from_env,
    openai_llm_from_env,
)
from aninamer.plan import RenamePlan, build_rename_plan
from aninamer.plan_io import read_rename_plan_json, write_rename_plan_json
from aninamer.scanner import scan_series_dir
from aninamer.tmdb_client import TMDBClient, TMDBError
from aninamer.tmdb_resolve import resolve_tmdb_tv_id_with_llm

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

    run_parser = subparsers.add_parser(
        "run", help="Plan and optionally apply renames."
    )
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
        "--yes",
        action="store_true",
        help="Apply moves (default is dry-run).",
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


def _print_plan_summary(plan: RenamePlan, plan_file: Path) -> None:
    video_moves = sum(1 for move in plan.moves if move.kind == "video")
    subtitle_moves = sum(1 for move in plan.moves if move.kind == "subtitle")
    print(f"Plan file: {plan_file}")
    print(f"Moves: {video_moves} videos, {subtitle_moves} subtitles")
    for move in plan.moves[:5]:
        print(f"{move.src} -> {move.dst}")


def _print_apply_summary(dry_run: bool, applied_count: int, rollback_file: Path) -> None:
    status = "dry-run" if dry_run else "applied"
    print(f"Apply {status}: {applied_count} moves")
    print(f"Rollback plan: {rollback_file}")


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

    if args.tmdb is not None:
        tmdb_id = args.tmdb
    else:
        query = series_dir.name
        candidates = tmdb.search_tv(query, language="zh-CN", page=1)
        if not candidates:
            raise ValueError(f"no TMDB results for query '{query}'")
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

    details = tmdb.get_tv_details(tmdb_id, language="zh-CN")
    series_name_zh_cn = _resolve_series_name(details.name, details.original_name)
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
        llm_for_mapping_factory()
        if llm_for_mapping_factory
        else openai_llm_from_env()
    )
    mapping = map_episodes_with_llm(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        season_episode_counts=season_episode_counts,
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
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

    plan_file = args.plan_file or Path("rename_plan.json")
    return plan, plan_file


def _run_plan(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> int:
    plan, plan_file = _build_plan_from_args(
        args,
        tmdb_client_factory=tmdb_client_factory,
        llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
        llm_for_mapping_factory=llm_for_mapping_factory,
    )
    write_rename_plan_json(plan_file, plan)
    _print_plan_summary(plan, plan_file)
    return 0


def _run_apply(args: argparse.Namespace) -> int:
    plan_path = args.plan_json
    plan = read_rename_plan_json(plan_path)
    result = apply_rename_plan(plan, dry_run=args.dry_run)

    rollback_file = args.rollback_file or plan_path.with_name("rollback_plan.json")
    rollback_plan = RenamePlan(
        tmdb_id=plan.tmdb_id,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        series_dir=plan.series_dir,
        output_root=plan.output_root,
        moves=result.rollback_moves,
    )
    write_rename_plan_json(rollback_file, rollback_plan)
    _print_apply_summary(result.dry_run, len(result.applied), rollback_file)
    return 0


def _run_run(
    args: argparse.Namespace,
    *,
    tmdb_client_factory: Callable[[], TMDBClient] | None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None,
) -> int:
    plan, plan_file = _build_plan_from_args(
        args,
        tmdb_client_factory=tmdb_client_factory,
        llm_for_tmdb_id_factory=llm_for_tmdb_id_factory,
        llm_for_mapping_factory=llm_for_mapping_factory,
    )
    write_rename_plan_json(plan_file, plan)
    _print_plan_summary(plan, plan_file)

    dry_run = not args.yes
    result = apply_rename_plan(plan, dry_run=dry_run)

    rollback_file = args.rollback_file or plan_file.with_name("rollback_plan.json")
    rollback_plan = RenamePlan(
        tmdb_id=plan.tmdb_id,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        series_dir=plan.series_dir,
        output_root=plan.output_root,
        moves=result.rollback_moves,
    )
    write_rename_plan_json(rollback_file, rollback_plan)
    _print_apply_summary(result.dry_run, len(result.applied), rollback_file)
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

    configure_logging(_log_level_from_args(args))

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
            return _run_apply(args)
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
