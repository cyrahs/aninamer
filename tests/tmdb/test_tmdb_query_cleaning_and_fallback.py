from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.llm_client import ChatMessage
from aninamer.name_clean import build_tmdb_query_variants, clean_tmdb_query
from aninamer.pipeline import PlanBuildOptions, build_rename_plan_for_series, search_tmdb_candidates
from aninamer.tmdb_client import (
    SeasonDetails,
    SeasonSummary,
    TvDetails,
    TvSearchResult,
)


def test_clean_tmdb_query_removes_brackets_quality_and_season_markers() -> None:
    raw = "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]"
    cleaned = clean_tmdb_query(raw)

    assert "Mahouka" in cleaned
    assert "Rettousei" in cleaned
    assert "DMG" not in cleaned
    assert "VCB" not in cleaned
    assert "1080p" not in cleaned.lower()
    assert "ma10p" not in cleaned.lower()
    assert "S3" not in cleaned
    assert "s3" not in cleaned.lower()


def test_build_tmdb_query_variants_includes_cleaned_and_shortened() -> None:
    raw = "[X] Mahouka Koukou no Rettousei S3 [1080p]"
    variants = build_tmdb_query_variants(raw, max_variants=6)

    assert variants
    assert variants[0].strip() == raw.strip()

    cleaned = clean_tmdb_query(raw)
    assert cleaned in variants

    words = cleaned.split()
    if len(words) >= 4:
        assert " ".join(words[:2]) in variants


@dataclass
class FakeLLM:
    reply: str
    calls: int = 0
    last_messages: list[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        self.last_messages = list(messages)
        return self.reply


class FakeTMDBClientForCleaning:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        self.calls.append((query, language))
        if "[" in query or "]" in query:
            return []
        if query.strip() == "Mahouka Koukou no Rettousei":
            return [
                TvSearchResult(
                    id=1000,
                    name="魔法科高校的劣等生",
                    first_air_date="2014-04-06",
                    original_name="魔法科高校の劣等生",
                    popularity=1.0,
                    vote_count=10,
                )
            ]
        return []

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ) -> list[TvSearchResult]:
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return TvDetails(
            id=tv_id,
            name="魔法科高校的劣等生",
            original_name="魔法科高校の劣等生",
            first_air_date="2014-04-06",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        return SeasonDetails(id=None, season_number=season_number, episodes=[])

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = self.get_tv_details(tv_id)
        return details.name, details


class FakeTMDBClientForLLMTitle:
    def __init__(self, target_query: str, tmdb_id: int) -> None:
        self.calls: list[tuple[str, str]] = []
        self.target_query = target_query
        self.tmdb_id = tmdb_id

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        self.calls.append((query, language))
        if query.strip() == self.target_query:
            return [
                TvSearchResult(
                    id=self.tmdb_id,
                    name="Found Title",
                    first_air_date="2022-01-01",
                    original_name="Found Title",
                    popularity=1.0,
                    vote_count=10,
                )
            ]
        return []

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ) -> list[TvSearchResult]:
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return TvDetails(
            id=tv_id,
            name="Found Title",
            original_name="Found Title",
            first_air_date="2022-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        return SeasonDetails(id=None, season_number=season_number, episodes=[])

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = self.get_tv_details(tv_id)
        return details.name, details


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_search_tmdb_candidates_falls_back_to_cleaned_query() -> None:
    tmdb = FakeTMDBClientForCleaning()

    results = search_tmdb_candidates(
        tmdb,
        "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]",
    )

    assert results[0].id == 1000
    assert ("Mahouka Koukou no Rettousei", "zh-CN") in tmdb.calls or (
        "Mahouka Koukou no Rettousei",
        "en-US",
    ) in tmdb.calls


def test_search_tmdb_candidates_falls_back_to_llm_title_when_search_empty() -> None:
    tmdb = FakeTMDBClientForLLMTitle(target_query="LLM Title", tmdb_id=4242)
    llm_title = FakeLLM(reply='{"title": "LLM Title"}')

    results = search_tmdb_candidates(
        tmdb,
        "[X] Unfindable Title [1080p]",
        llm_title_factory=lambda: llm_title,
    )

    assert results[0].id == 4242
    assert ("LLM Title", "zh-CN") in tmdb.calls or ("LLM Title", "en-US") in tmdb.calls
    assert llm_title.calls == 1


def test_search_tmdb_candidates_tries_title_without_the_animation_suffix() -> None:
    tmdb = FakeTMDBClientForLLMTitle(target_query="ながちち永井さん", tmdb_id=298953)

    results = search_tmdb_candidates(
        tmdb,
        "ながちち永井さん THE ANIMATION",
    )

    assert results[0].id == 298953
    queries = [query for query, _language in tmdb.calls]
    assert queries.index("ながちち永井さん THE ANIMATION") < queries.index(
        "ながちち永井さん"
    )


def test_search_tmdb_candidates_tries_traditional_chinese_query_variant() -> None:
    tmdb = FakeTMDBClientForLLMTitle(target_query="向日葵在夜晚綻放", tmdb_id=248253)

    results = search_tmdb_candidates(
        tmdb,
        "向日葵在夜晚绽放",
    )

    assert results[0].id == 248253
    queries = [query for query, _language in tmdb.calls]
    assert queries.index("向日葵在夜晚绽放") < queries.index("向日葵在夜晚綻放")


def test_search_tmdb_candidates_tries_japanese_quoted_title_variant() -> None:
    target = "オタクの僕が一軍ギャルと付き合えるまでの話"
    tmdb = FakeTMDBClientForLLMTitle(target_query=target, tmdb_id=321116)

    results = search_tmdb_candidates(
        tmdb,
        "アニメ版「オタクの仆が一军ギャルと付き合えるまでの话」",
    )

    assert results[0].id == 321116
    queries = [query for query, _language in tmdb.calls]
    assert target in queries


def test_build_rename_plan_error_includes_attempted_queries(tmp_path: Path) -> None:
    series_dir = tmp_path / "[X] TotallyUnfindableTitle [1080p]"
    out_root = tmp_path / "out"
    _write(series_dir / "ep1.mkv", b"video")

    class AlwaysEmptyTMDB(FakeTMDBClientForCleaning):
        def search_tv(
            self, query: str, *, language: str = "zh-CN", page: int = 1
        ) -> list[TvSearchResult]:
            self.calls.append((query, language))
            return []

    tmdb = AlwaysEmptyTMDB()
    llm_id = FakeLLM(reply='{"tmdb": 1}')
    llm_map = FakeLLM(reply='{"tmdb":1,"eps":[]}')

    with pytest.raises(ValueError, match="attempted queries"):
        build_rename_plan_for_series(
            series_dir=series_dir,
            output_root=out_root,
            options=PlanBuildOptions(),
            tmdb_client_factory=lambda: tmdb,
            llm_for_tmdb_id_factory=lambda: llm_id,
            llm_for_mapping_factory=lambda: llm_map,
        )
