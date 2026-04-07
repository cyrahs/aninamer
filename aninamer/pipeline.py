from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Callable, Sequence

from aninamer.apply import ApplyResult, apply_rename_plan
from aninamer.config import TmdbConfig
from aninamer.episode_mapping import map_episodes_with_llm
from aninamer.errors import LLMOutputError, OpenAIError
from aninamer.llm_client import LLMClient
from aninamer.name_clean import build_tmdb_query_variants, extract_tmdb_id_tag
from aninamer.plan import (
    RenamePlan,
    build_rename_plan,
    format_season_folder,
)
from aninamer.scanner import VIDEO_EXTS, scan_series_dir
from aninamer.tmdb_client import SeasonDetails, TMDBClient, TvSearchResult
from aninamer.tmdb_resolve import (
    resolve_tmdb_search_title_with_llm,
    resolve_tmdb_tv_id_with_llm,
)

logger = logging.getLogger(__name__)
_EPISODE_PATTERN = re.compile(r"S(?P<season>\d{2})E(?P<e1>\d{2})(?:-E(?P<e2>\d{2}))?")


@dataclass(frozen=True)
class PlanBuildOptions:
    max_candidates: int = 5
    max_output_tokens: int = 2048
    allow_existing_dest: bool = False


@dataclass(frozen=True)
class ApplyExecutionResult:
    apply_result: ApplyResult
    rollback_plan: RenamePlan
    applied_count: int


@dataclass(frozen=True)
class ExistingEpisodeInventory:
    matched_series_dirs: tuple[Path, ...]
    occupied_episode_numbers_by_season: dict[int, tuple[int, ...]]
    existing_s00_files: tuple[str, ...]


def tmdb_client_from_env() -> TMDBClient:
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TMDB_API_KEY is required")
    return TMDBClient(api_key=api_key, timeout=30.0)


def tmdb_client_from_settings(settings: TmdbConfig) -> TMDBClient:
    return TMDBClient(api_key=settings.api_key, timeout=settings.timeout)


def _find_existing_series_dirs_by_tmdb_id(output_root: Path, tmdb_id: int) -> tuple[Path, ...]:
    if not output_root.exists() or not output_root.is_dir():
        return ()

    matches: list[Path] = []
    for path in output_root.iterdir():
        if not path.is_dir():
            continue
        try:
            path_tmdb_id = extract_tmdb_id_tag(path.name)
        except ValueError:
            logger.warning("inventory: skipping invalid tmdb tag dir=%s", path)
            continue
        if path_tmdb_id == tmdb_id:
            matches.append(path)

    matches.sort(key=lambda path: path.name.casefold())
    return tuple(matches)


def _parse_existing_episode_slot(file_name: str) -> tuple[int, tuple[int, ...]] | None:
    match = _EPISODE_PATTERN.search(file_name)
    if match is None:
        return None

    season = int(match.group("season"))
    episode_start = int(match.group("e1"))
    raw_episode_end = match.group("e2")
    episode_end = int(raw_episode_end) if raw_episode_end is not None else episode_start
    return season, tuple(range(episode_start, episode_end + 1))


def inspect_existing_episode_inventory(
    output_root: Path,
    tmdb_id: int,
) -> ExistingEpisodeInventory:
    matched_series_dirs = _find_existing_series_dirs_by_tmdb_id(output_root, tmdb_id)
    occupied_episode_numbers_by_season: dict[int, set[int]] = {}
    existing_s00_files: set[str] = set()

    for series_dir in matched_series_dirs:
        for path in series_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            parsed_slot = _parse_existing_episode_slot(path.name)
            if parsed_slot is None:
                continue

            season, episode_numbers = parsed_slot
            occupied_episode_numbers_by_season.setdefault(season, set()).update(
                episode_numbers
            )
            if season == 0:
                existing_s00_files.add(path.name)

    normalized_occupancy = {
        season: tuple(sorted(episode_numbers))
        for season, episode_numbers in occupied_episode_numbers_by_season.items()
    }
    sorted_s00_files = tuple(sorted(existing_s00_files, key=lambda value: value.casefold()))

    return ExistingEpisodeInventory(
        matched_series_dirs=matched_series_dirs,
        occupied_episode_numbers_by_season=normalized_occupancy,
        existing_s00_files=sorted_s00_files,
    )


def search_tmdb_candidates(
    tmdb: TMDBClient,
    name: str,
    *,
    llm_title_factory: Callable[[], LLMClient] | None = None,
) -> list[TvSearchResult]:
    queries = build_tmdb_query_variants(name)
    languages = ["zh-CN", "en-US", "ja-JP"]
    logger.info("tmdb_search: start name=%s variants=%s", name, len(queries))

    def _search_anime_queries(query_list: Sequence[str]) -> list[TvSearchResult]:
        for query in query_list:
            candidates_by_id: dict[int, TvSearchResult] = {}
            for language in languages:
                results = tmdb.search_tv_anime(query, language=language, max_pages=1)
                logger.info(
                    "tmdb_search: anime query=%s language=%s results=%s",
                    query,
                    language,
                    len(results),
                )
                for candidate in results:
                    candidates_by_id.setdefault(candidate.id, candidate)
            if candidates_by_id:
                return list(candidates_by_id.values())
        return []

    def _search_all_queries(query_list: Sequence[str]) -> list[TvSearchResult]:
        for query in query_list:
            candidates_by_id: dict[int, TvSearchResult] = {}
            for language in languages:
                results = tmdb.search_tv(query, language=language, page=1)
                logger.info(
                    "tmdb_search: fallback query=%s language=%s results=%s",
                    query,
                    language,
                    len(results),
                )
                for candidate in results:
                    candidates_by_id.setdefault(candidate.id, candidate)
            if candidates_by_id:
                return list(candidates_by_id.values())
        return []

    candidates = _search_anime_queries(queries)
    if candidates:
        return candidates

    logger.info("tmdb_search: no anime results, trying fallback search")
    candidates = _search_all_queries(queries)
    if candidates:
        return candidates

    if llm_title_factory is None:
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        )

    llm = llm_title_factory()
    try:
        llm_title = resolve_tmdb_search_title_with_llm(name, llm)
    except (OpenAIError, LLMOutputError) as exc:
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        ) from exc

    llm_queries = build_tmdb_query_variants(llm_title)
    attempted = {query.casefold() for query in queries}
    llm_queries = [query for query in llm_queries if query.casefold() not in attempted]
    if not llm_queries:
        raise ValueError(
            f"no TMDB results for name '{name}' (attempted queries: {queries})"
        )

    candidates = _search_anime_queries(llm_queries)
    if candidates:
        return candidates
    candidates = _search_all_queries(llm_queries)
    if candidates:
        return candidates

    raise ValueError(
        f"no TMDB results for name '{name}' (attempted queries: {queries + llm_queries})"
    )


def build_rename_plan_for_series(
    *,
    series_dir: Path,
    output_root: Path,
    options: PlanBuildOptions,
    tmdb_client_factory: Callable[[], TMDBClient] | None = None,
    llm_for_tmdb_id_factory: Callable[[], LLMClient] | None = None,
    llm_for_mapping_factory: Callable[[], LLMClient] | None = None,
) -> RenamePlan:
    scan = scan_series_dir(series_dir)
    if tmdb_client_factory is None:
        raise ValueError("tmdb_client_factory is required")
    tmdb = tmdb_client_factory()

    tag_tmdb_id = extract_tmdb_id_tag(series_dir.name)
    tmdb_id = None
    if tag_tmdb_id is not None:
        tmdb_id = tag_tmdb_id
        logger.info("tmdb_resolve: using tag id=%s dirname=%s", tmdb_id, series_dir.name)
    else:
        candidates = search_tmdb_candidates(
            tmdb,
            series_dir.name,
            llm_title_factory=llm_for_tmdb_id_factory,
        )
        if len(candidates) == 1:
            tmdb_id = candidates[0].id
        else:
            if llm_for_tmdb_id_factory is None:
                raise ValueError("llm_for_tmdb_id_factory is required")
            llm_for_id = llm_for_tmdb_id_factory()
            tmdb_id = resolve_tmdb_tv_id_with_llm(
                dirname=series_dir.name,
                candidates=candidates,
                llm=llm_for_id,
                max_candidates=options.max_candidates,
            )

    if tmdb_id is None:
        raise ValueError("tmdb_id could not be resolved")

    series_name_zh_cn, details = tmdb.resolve_series_title(tmdb_id)
    year = details.year
    season_episode_counts = {
        season.season_number: season.episode_count for season in details.seasons
    }
    regular_seasons_zh: dict[int, SeasonDetails] = {}
    regular_seasons_en: dict[int, SeasonDetails] = {}
    for season_number in sorted(season_episode_counts):
        if season_number <= 0:
            continue
        if season_episode_counts[season_number] <= 0:
            continue
        regular_seasons_zh[season_number] = tmdb.get_season(
            tmdb_id, season_number, language="zh-CN"
        )
        regular_seasons_en[season_number] = tmdb.get_season(
            tmdb_id, season_number, language="en-US"
        )

    specials_count = season_episode_counts.get(0, 0)
    if specials_count > 0:
        specials_zh = tmdb.get_season(tmdb_id, 0, language="zh-CN")
        specials_en = tmdb.get_season(tmdb_id, 0, language="en-US")
    else:
        specials_zh = None
        specials_en = None

    if llm_for_mapping_factory is None:
        raise ValueError("llm_for_mapping_factory is required")
    llm_for_mapping = llm_for_mapping_factory()
    existing_inventory = inspect_existing_episode_inventory(output_root, tmdb_id)
    mapping = map_episodes_with_llm(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        season_episode_counts=season_episode_counts,
        regular_seasons_zh=regular_seasons_zh,
        regular_seasons_en=regular_seasons_en,
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
        existing_episode_numbers_by_season=(
            existing_inventory.occupied_episode_numbers_by_season
        ),
        existing_s00_files=existing_inventory.existing_s00_files,
        llm=llm_for_mapping,
        max_output_tokens=options.max_output_tokens,
    )
    return build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        tmdb_id=tmdb_id,
        output_root=output_root,
        allow_existing_dest=options.allow_existing_dest,
    )


def execute_apply(
    plan: RenamePlan,
    *,
    dry_run: bool,
    two_stage: bool,
) -> ApplyExecutionResult:
    apply_result = apply_rename_plan(plan, dry_run=dry_run, two_stage=two_stage)
    rollback_plan = RenamePlan(
        tmdb_id=plan.tmdb_id,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        series_dir=plan.series_dir,
        output_root=plan.output_root,
        moves=apply_result.rollback_moves,
    )
    return ApplyExecutionResult(
        apply_result=apply_result,
        rollback_plan=rollback_plan,
        applied_count=len(apply_result.applied),
    )
