from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
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
    format_series_root_folder,
)
from aninamer.scanner import scan_series_dir
from aninamer.tmdb_client import TMDBClient, TvSearchResult
from aninamer.tmdb_resolve import (
    resolve_tmdb_search_title_with_llm,
    resolve_tmdb_tv_id_with_llm,
)

logger = logging.getLogger(__name__)


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


def tmdb_client_from_env() -> TMDBClient:
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TMDB_API_KEY is required")
    return TMDBClient(api_key=api_key, timeout=30.0)


def tmdb_client_from_settings(settings: TmdbConfig) -> TMDBClient:
    return TMDBClient(api_key=settings.api_key, timeout=settings.timeout)


def list_existing_s00_files(
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
    existing_s00_files = list_existing_s00_files(
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
