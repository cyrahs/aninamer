from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.episode_mapping import map_episodes_with_llm
from aninamer.llm_client import ChatMessage
from aninamer.plan import build_rename_plan
from aninamer.scanner import FileCandidate, ScanResult, scan_series_dir
from aninamer.tmdb_client import Episode, SeasonDetails, TvSearchResult
from aninamer.tmdb_resolve import resolve_tmdb_tv_id_with_llm


@dataclass
class FakeLLM:
    reply: str
    calls: int = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        return self.reply


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_scan_logs_start_and_done(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    series = tmp_path / "series"
    _write(series / "ep1.mkv", b"v")
    _write(series / "ep1.ass", b"s")
    _write(series / "font.ttf", b"x")  # ignored

    caplog.set_level(logging.INFO)

    _ = scan_series_dir(series)

    text = caplog.text
    assert "scan: start" in text
    assert "scan: done" in text
    assert "videos=1" in text
    assert "subtitles=1" in text


def test_tmdb_resolve_logs_raw_llm_output(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    llm = FakeLLM(reply='```json\n{"tmdb": 111}\n```')
    candidates = [
        TvSearchResult(id=111, name="A", first_air_date="2020-01-01", original_name=None, popularity=None, vote_count=None),
        TvSearchResult(id=222, name="B", first_air_date="2020-01-01", original_name=None, popularity=None, vote_count=None),
    ]
    chosen = resolve_tmdb_tv_id_with_llm("DirName", candidates, llm, max_candidates=2)
    assert chosen == 111

    text = caplog.text
    assert "tmdb_resolve: start" in text
    assert "tmdb_resolve: llm_call" in text
    assert "tmdb_resolve: llm_prompt" in text
    assert "dirname: DirName" in text
    assert "tmdb_resolve: raw_llm_output" in text
    assert '{"tmdb": 111}' in text
    assert "tmdb_resolve: selected" in text


def test_episode_mapping_and_plan_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    series_dir = tmp_path / "input"
    out_root = tmp_path / "out"

    # Source files
    _write(series_dir / "ep1.mkv", b"video1")
    _write(series_dir / "ep1.ass", "国国国 后后后".encode("utf-8"))

    scan = ScanResult(
        series_dir=series_dir,
        videos=[FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=(series_dir / "ep1.mkv").stat().st_size)],
        subtitles=[FileCandidate(id=2, rel_path="ep1.ass", ext=".ass", size_bytes=(series_dir / "ep1.ass").stat().st_size)],
    )

    # LLM output maps video 1 -> S01E01 and subtitle 2
    llm = FakeLLM(reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}')

    specials_zh = SeasonDetails(id=999, season_number=0, episodes=[Episode(episode_number=1, name="OVA", overview="OVA")])
    specials_en = SeasonDetails(id=999, season_number=0, episodes=[Episode(episode_number=1, name="OVA", overview="OVA")])

    mapping = map_episodes_with_llm(
        tmdb_id=123,
        series_name_zh_cn="测试动画",
        year=2020,
        season_episode_counts={0: 1, 1: 1},
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=512,
    )

    plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn="测试动画",
        year=2020,
        tmdb_id=123,
        output_root=out_root,
    )

    # Sanity
    assert len(plan.moves) == 2

    text = caplog.text
    assert "episode_map: start" in text
    assert "episode_map: llm_call" in text
    assert "episode_map: llm_prompt" in text
    assert "schema (no extra keys)" in text
    assert "episode_map: parsed" in text
    assert "plan: start" in text
    assert "plan: built" in text
