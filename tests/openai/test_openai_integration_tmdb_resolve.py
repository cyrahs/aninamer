from __future__ import annotations

import os

import pytest

from aninamer.openai_llm_client import openai_llm_for_tmdb_id_from_env
from aninamer.tmdb_client import TvSearchResult
from aninamer.tmdb_resolve import resolve_tmdb_tv_id_with_llm


def _env_present(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip() != ""


@pytest.mark.integration
def test_resolve_tmdb_id_with_real_llm_smoke() -> None:
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    # Uses the TMDB-id reasoning profile (OPENAI_REASONING_EFFORT_CHORE, default low).
    llm = openai_llm_for_tmdb_id_from_env()

    dirname = "完全匹配作品名"
    candidates = [
        TvSearchResult(
            id=111,
            name="完全匹配作品名",
            first_air_date="2020-01-01",
            original_name="Exact Match",
            popularity=1.0,
            vote_count=10,
        ),
        TvSearchResult(
            id=222,
            name="其他作品名",
            first_air_date="2020-01-01",
            original_name="Other",
            popularity=1000.0,
            vote_count=999999,
        ),
    ]

    chosen = resolve_tmdb_tv_id_with_llm(dirname, candidates, llm, max_candidates=2)
    assert chosen == 111


@pytest.mark.integration
def test_resolve_tmdb_id_charlotte() -> None:
    """
    Real case: keyword 'Charlotte' should resolve to TMDB ID 63145 (Charlotte anime, 2015).

    This tests the LLM's ability to select the correct anime from 7 animation results
    filtered by search_tv_anime (scraped from TMDB API zh-CN search).
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    llm = openai_llm_for_tmdb_id_from_env()

    dirname = "Charlotte"
    # Real TMDB search_tv_anime results for "Charlotte" (zh-CN) - filtered by Animation genre
    candidates = [
        TvSearchResult(
            id=205192,
            name="E.B. White's Charlotte's Web",
            first_air_date="2025-10-02",
            original_name="E.B. White's Charlotte's Web",
            popularity=1.5174,
            vote_count=6,
            genre_ids=(10751, 16, 18, 10762),
            origin_country=("US",),
        ),
        TvSearchResult(
            id=114379,
            name="Os Óculos Mágicos de Charlotte",
            first_air_date="2020-12-14",
            original_name="Os Óculos Mágicos de Charlotte",
            popularity=0.2788,
            vote_count=0,
            genre_ids=(16, 10762),
            origin_country=("BR",),
        ),
        # The correct anime - Charlotte (2015)
        TvSearchResult(
            id=63145,
            name="夏洛特",
            first_air_date="2015-07-05",
            original_name="シャーロット",
            popularity=10.9381,
            vote_count=627,
            genre_ids=(16, 18, 10765),
            origin_country=("JP",),
        ),
        TvSearchResult(
            id=33835,
            name="Strawberry Shortcake's Berry Bitty Adventures",
            first_air_date="2010-10-10",
            original_name="Strawberry Shortcake's Berry Bitty Adventures",
            popularity=5.2955,
            vote_count=28,
            genre_ids=(16, 10751, 10762, 35),
            origin_country=("FR", "CA", "US"),
        ),
        TvSearchResult(
            id=16033,
            name="Strawberry Shortcake",
            first_air_date="2003-03-11",
            original_name="Strawberry Shortcake",
            popularity=3.5494,
            vote_count=33,
            genre_ids=(16, 10762),
            origin_country=("US",),
        ),
        TvSearchResult(
            id=100682,
            name="女王陛下的侦探安洁",
            first_air_date="1977-12-13",
            original_name="女王陛下のプティアンジェ",
            popularity=3.7484,
            vote_count=2,
            genre_ids=(16, 9648),
            origin_country=("JP",),
        ),
        TvSearchResult(
            id=23591,
            name="若草夏洛特",
            first_air_date="1977-10-29",
            original_name="若草のシャルロット",
            popularity=2.1655,
            vote_count=4,
            genre_ids=(16, 18),
            origin_country=("JP",),
        ),
    ]

    chosen = resolve_tmdb_tv_id_with_llm(dirname, candidates, llm, max_candidates=10)
    assert chosen == 63145
