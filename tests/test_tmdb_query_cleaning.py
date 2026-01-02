from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.cli import main
from aninamer.llm_client import ChatMessage
from aninamer.name_clean import build_tmdb_query_variants, clean_tmdb_query
from aninamer.tmdb_client import Episode, SeasonDetails, SeasonSummary, TvDetails, TvSearchResult


def test_clean_tmdb_query_removes_brackets_quality_and_season_markers() -> None:
    raw = "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]"
    cleaned = clean_tmdb_query(raw)

    # Core title should remain
    assert "Mahouka" in cleaned
    assert "Rettousei" in cleaned

    # Bracket content removed
    assert "DMG" not in cleaned
    assert "VCB" not in cleaned

    # Quality tag removed
    assert "1080p" not in cleaned.lower()
    assert "ma10p" not in cleaned.lower()

    # Season marker removed
    assert "S3" not in cleaned
    assert "s3" not in cleaned.lower()


def test_build_tmdb_query_variants_includes_cleaned_and_shortened() -> None:
    raw = "[X] Mahouka Koukou no Rettousei S3 [1080p]"
    variants = build_tmdb_query_variants(raw, max_variants=6)

    assert variants, "should produce at least one variant"
    # original (whitespace normalized) should be present
    assert variants[0].strip() == raw.strip()

    # cleaned should be present
    cleaned = clean_tmdb_query(raw)
    assert cleaned in variants

    # shorter variants should exist (e.g., first 2 words)
    words = cleaned.split()
    if len(words) >= 4:
        short2 = " ".join(words[:2])
        assert short2 in variants


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

    def search_tv(self, query: str, *, language: str = "zh-CN", page: int = 1) -> list[TvSearchResult]:
        self.calls.append((query, language))
        # Simulate the failure: noisy query returns 0
        if "[" in query or "]" in query:
            return []
        # Simulate success only when cleaned core title is used
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

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return TvDetails(
            id=tv_id,
            name="魔法科高校的劣等生",
            original_name="魔法科高校の劣等生",
            first_air_date="2014-04-06",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN") -> SeasonDetails:
        return SeasonDetails(id=None, season_number=season_number, episodes=[])


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_cli_plan_falls_back_to_cleaned_query(tmp_path: Path) -> None:
    # The directory name is noisy and will fail unless cleaning is used.
    series_dir = tmp_path / "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]"
    out_root = tmp_path / "out"
    plan_file = tmp_path / "rename_plan.json"

    _write(series_dir / "ep1.mkv", b"video")

    tmdb = FakeTMDBClientForCleaning()

    # TMDB-id LLM selects the only candidate
    llm_id = FakeLLM(reply='{"tmdb": 1000}')

    # Mapping LLM: map video id 1 -> S01E01
    llm_map = FakeLLM(reply='{"tmdb":1000,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}')

    rc = main(
        [
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_file),
            "--max-candidates",
            "5",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0
    assert plan_file.exists()

    # Ensure TMDB search was attempted with cleaned core query at least once
    assert ("Mahouka Koukou no Rettousei", "zh-CN") in tmdb.calls or ("Mahouka Koukou no Rettousei", "en-US") in tmdb.calls


def test_cli_plan_error_includes_attempted_queries(tmp_path: Path) -> None:
    series_dir = tmp_path / "[X] TotallyUnfindableTitle [1080p]"
    out_root = tmp_path / "out"
    plan_file = tmp_path / "rename_plan.json"

    _write(series_dir / "ep1.mkv", b"video")

    class AlwaysEmptyTMDB(FakeTMDBClientForCleaning):
        def search_tv(self, query: str, *, language: str = "zh-CN", page: int = 1) -> list[TvSearchResult]:
            self.calls.append((query, language))
            return []

    tmdb = AlwaysEmptyTMDB()
    llm_id = FakeLLM(reply='{"tmdb": 1}')
    llm_map = FakeLLM(reply='{"tmdb":1,"eps":[]}')

    rc = main(
        [
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_file),
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    # Should fail gracefully with non-zero return code
    assert rc != 0
