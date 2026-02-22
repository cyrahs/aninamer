from __future__ import annotations

import os
from pathlib import Path

import pytest

from aninamer.openai_llm_client import openai_llm_from_env
from aninamer.episode_mapping import map_episodes_with_llm
from aninamer.scanner import FileCandidate, ScanResult
from aninamer.tmdb_client import Episode, SeasonDetails


def _env_present(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip() != ""


@pytest.mark.integration
def test_episode_mapping_with_real_llm_smoke(tmp_path: Path) -> None:
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path,
        videos=[
            FileCandidate(id=1, rel_path="测试动画 S01E01.mkv", ext=".mkv", size_bytes=100_000_000),
            FileCandidate(id=2, rel_path="测试动画 S01E02.mkv", ext=".mkv", size_bytes=100_000_000),
            FileCandidate(id=3, rel_path="测试动画 OVA.mkv", ext=".mkv", size_bytes=200_000_000),
            FileCandidate(id=4, rel_path="测试动画 OP.mkv", ext=".mkv", size_bytes=10_000_000),
        ],
        subtitles=[
            FileCandidate(id=5, rel_path="测试动画 S01E01.ass", ext=".ass", size_bytes=200_000),
            FileCandidate(id=6, rel_path="测试动画 S01E02.ass", ext=".ass", size_bytes=200_000),
            FileCandidate(id=7, rel_path="测试动画 OVA.ass", ext=".ass", size_bytes=200_000),
            FileCandidate(id=8, rel_path="测试动画 OP.ass", ext=".ass", size_bytes=50_000),
        ],
    )

    # Minimal TMDB structure to constrain mapping
    season_episode_counts = {0: 1, 1: 2}

    specials_zh = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[Episode(episode_number=1, name="OVA", overview="OVA 特别篇")],
    )
    specials_en = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[Episode(episode_number=1, name="OVA", overview="OVA special")],
    )

    res = map_episodes_with_llm(
        tmdb_id=123,
        series_name_zh_cn="测试动画",
        year=2020,
        season_episode_counts=season_episode_counts,
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=1024,
    )
    # Must at least map the two explicit S01E01/S01E02, and map OVA to S00E01.
    by_vid = {it.video_id: it for it in res.items}

    assert 1 in by_vid and by_vid[1].season == 1 and by_vid[1].episode_start == 1
    assert 2 in by_vid and by_vid[2].season == 1 and by_vid[2].episode_start == 2
    assert 3 in by_vid and by_vid[3].season == 0 and by_vid[3].episode_start == 1

    # OP should be omitted (untouched)
    assert 4 not in by_vid
