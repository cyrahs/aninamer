"""
End-to-end LLM tests using real file structures from production rename plans.

These tests reconstruct the exact file lists from actual aninamer runs
and verify that the LLM produces correct episode mappings.

Requires OPENAI_API_KEY and OPENAI_MODEL environment variables.
Mark: pytest -m integration
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aninamer.episode_mapping import map_episodes_with_llm
from aninamer.openai_llm_client import openai_llm_from_env
from aninamer.scanner import FileCandidate, ScanResult
from aninamer.tmdb_client import Episode, SeasonDetails


def _env_present(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip() != ""


@dataclass
class ExpectedMapping:
    """Expected mapping result for a video file."""

    video_id: int
    season: int
    episode_start: int
    episode_end: int


@dataclass
class PlanTestCase:
    """A test case derived from a real rename plan."""

    name: str
    series_dir_name: str
    tmdb_id: int
    series_name_zh_cn: str
    year: int
    season_episode_counts: dict[int, int]
    specials_zh: SeasonDetails | None
    specials_en: SeasonDetails | None
    videos: list[FileCandidate]
    subtitles: list[FileCandidate]
    expected_mappings: list[ExpectedMapping]


def _extract_season_episode(dst_path: str) -> tuple[int, int, int]:
    """Extract season and episode range from destination path."""
    # Pattern: S01E01 or S01E01-E02
    match = re.search(r"S(\d+)E(\d+)(?:-E(\d+))?", dst_path)
    if not match:
        raise ValueError(f"Cannot extract season/episode from {dst_path}")
    season = int(match.group(1))
    ep_start = int(match.group(2))
    ep_end = int(match.group(3)) if match.group(3) else ep_start
    return season, ep_start, ep_end


def _build_file_candidate(
    file_id: int, src_path: str, series_dir: str
) -> FileCandidate:
    """Build FileCandidate from source path."""
    # Extract relative path from series_dir
    if series_dir in src_path:
        idx = src_path.index(series_dir)
        rel_path = src_path[idx + len(series_dir) + 1 :]  # +1 for /
    else:
        rel_path = Path(src_path).name
    ext = Path(src_path).suffix.lower()
    # Use fake size (not relevant for mapping)
    size = 100_000_000 if ext == ".mkv" else 200_000
    return FileCandidate(id=file_id, rel_path=rel_path, ext=ext, size_bytes=size)


# ==============================================================================
# Test Case: Mahouka S1 - 26 episodes, no subtitles
# ==============================================================================


def build_mahouka_s1_case() -> PlanTestCase:
    """
    魔法科高校的劣等生 Season 1 - 26 episodes
    From: [VCB-Studio] Mahouka Koukou no Rettousei_e0713705.rename_plan.json
    """
    series_dir = "[VCB-Studio] Mahouka Koukou no Rettousei"
    base_path = f"/mnt/cd2/115/emby_in/anime/{series_dir}/[VCB-Studio] Mahouka Koukou no Rettousei [Ma10p_1080p]"

    videos: list[FileCandidate] = []
    expected: list[ExpectedMapping] = []

    for ep in range(1, 27):
        vid_id = ep + 1  # IDs start at 2 in the plan
        filename = f"[VCB-Studio] Mahouka Koukou no Rettousei [{ep:02d}][Ma10p_1080p][x265_flac].mkv"
        videos.append(
            FileCandidate(
                id=vid_id,
                rel_path=f"[VCB-Studio] Mahouka Koukou no Rettousei [Ma10p_1080p]/{filename}",
                ext=".mkv",
                size_bytes=100_000_000,
            )
        )
        expected.append(
            ExpectedMapping(video_id=vid_id, season=1, episode_start=ep, episode_end=ep)
        )

    return PlanTestCase(
        name="Mahouka S1",
        series_dir_name=series_dir,
        tmdb_id=60833,
        series_name_zh_cn="魔法科高校的劣等生",
        year=2014,
        season_episode_counts={0: 10, 1: 26, 2: 13, 3: 13},  # Actual TMDB counts
        specials_zh=None,
        specials_en=None,
        videos=videos,
        subtitles=[],
        expected_mappings=expected,
    )


# ==============================================================================
# Test Case: Mahouka S2 (Raihousha Hen) - 13 episodes with chs/cht subtitles
# ==============================================================================


def build_mahouka_s2_case() -> PlanTestCase:
    """
    魔法科高校的劣等生 Season 2 - 13 episodes with subtitles
    From: [DMG&VCB-Studio] Mahouka Koukou no Rettousei - Raihousha Hen [Ma10p_1080p]_1e0e399c.rename_plan.json
    """
    series_dir = "[DMG&VCB-Studio] Mahouka Koukou no Rettousei - Raihousha Hen [Ma10p_1080p]"

    videos: list[FileCandidate] = []
    subtitles: list[FileCandidate] = []
    expected: list[ExpectedMapping] = []

    for ep in range(1, 14):
        vid_id = ep
        filename = f"[DMG&VCB-Studio] Mahouka Koukou no Rettousei - Raihousha Hen [{ep:02d}][Ma10p_1080p][x265_flac].mkv"
        videos.append(
            FileCandidate(
                id=vid_id, rel_path=filename, ext=".mkv", size_bytes=100_000_000
            )
        )
        expected.append(
            ExpectedMapping(video_id=vid_id, season=2, episode_start=ep, episode_end=ep)
        )

    # Subtitles start after videos
    sub_id = 14
    for ep in range(1, 14):
        for suffix in ["chs", "cht"]:
            filename = f"[DMG&VCB-Studio] Mahouka Koukou no Rettousei - Raihousha Hen [{ep:02d}][Ma10p_1080p][x265_flac].{suffix}.ass"
            subtitles.append(
                FileCandidate(id=sub_id, rel_path=filename, ext=".ass", size_bytes=200_000)
            )
            sub_id += 1

    return PlanTestCase(
        name="Mahouka S2 (Raihousha Hen)",
        series_dir_name=series_dir,
        tmdb_id=60833,
        series_name_zh_cn="魔法科高校的劣等生",
        year=2014,
        season_episode_counts={0: 10, 1: 26, 2: 13, 3: 13},
        specials_zh=None,
        specials_en=None,
        videos=videos,
        subtitles=subtitles,
        expected_mappings=expected,
    )


# ==============================================================================
# Test Case: Mahouka S3 - 13 episodes with chs/cht subtitles
# ==============================================================================


def build_mahouka_s3_case() -> PlanTestCase:
    """
    魔法科高校的劣等生 Season 3 - 13 episodes with subtitles
    From: [DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]_ef392741.rename_plan.json
    """
    series_dir = "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [Ma10p_1080p]"

    videos: list[FileCandidate] = []
    subtitles: list[FileCandidate] = []
    expected: list[ExpectedMapping] = []

    # Videos with varying codecs
    codec_map = {
        1: "x265_flac_aac",
        2: "x265_flac",
        3: "x265_flac",
        4: "x265_flac_aac",
        5: "x265_flac",
        6: "x265_flac_aac",
        7: "x265_flac_aac",
        8: "x265_flac",
        9: "x265_flac",
        10: "x265_flac_aac",
        11: "x265_flac",
        12: "x265_flac",
        13: "x265_flac_aac",
    }

    for ep in range(1, 14):
        vid_id = ep
        codec = codec_map[ep]
        filename = f"[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [{ep:02d}][Ma10p_1080p][{codec}].mkv"
        videos.append(
            FileCandidate(
                id=vid_id, rel_path=filename, ext=".mkv", size_bytes=100_000_000
            )
        )
        expected.append(
            ExpectedMapping(video_id=vid_id, season=3, episode_start=ep, episode_end=ep)
        )

    # Subtitles
    sub_id = 14
    for ep in range(1, 14):
        codec = codec_map[ep]
        for suffix in ["chs", "cht"]:
            filename = f"[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei S3 [{ep:02d}][Ma10p_1080p][{codec}].{suffix}.ass"
            subtitles.append(
                FileCandidate(id=sub_id, rel_path=filename, ext=".ass", size_bytes=200_000)
            )
            sub_id += 1

    return PlanTestCase(
        name="Mahouka S3",
        series_dir_name=series_dir,
        tmdb_id=60833,
        series_name_zh_cn="魔法科高校的劣等生",
        year=2014,
        season_episode_counts={0: 10, 1: 26, 2: 13, 3: 13},
        specials_zh=None,
        specials_en=None,
        videos=videos,
        subtitles=subtitles,
        expected_mappings=expected,
    )


# ==============================================================================
# Test Case: Mahouka OVA (Tsuioku Hen) - 1 special episode
# ==============================================================================


def build_mahouka_ova_case() -> PlanTestCase:
    """
    魔法科高校的劣等生 追忆篇 OVA - mapped to S00E08
    From: [DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei - Tsuioku Hen [Ma10p_1080p_2e4b6875.rename_plan.json
    """
    series_dir = "[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei - Tsuioku Hen [Ma10p_1080p]"

    videos = [
        FileCandidate(
            id=1,
            rel_path="[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei - Tsuioku Hen [Ma10p_1080p][x265_flac].mkv",
            ext=".mkv",
            size_bytes=100_000_000,
        )
    ]

    subtitles = [
        FileCandidate(
            id=2,
            rel_path="[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei - Tsuioku Hen [Ma10p_1080p][x265_flac].chs.ass",
            ext=".ass",
            size_bytes=200_000,
        ),
        FileCandidate(
            id=3,
            rel_path="[DMG&SumiSora&VCB-Studio] Mahouka Koukou no Rettousei - Tsuioku Hen [Ma10p_1080p][x265_flac].cht.ass",
            ext=".ass",
            size_bytes=200_000,
        ),
    ]

    # OVA maps to S00E08 based on the actual plan
    expected = [ExpectedMapping(video_id=1, season=0, episode_start=8, episode_end=8)]

    # Specials info to help LLM map correctly
    specials_zh = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="特番 1", overview=""),
            Episode(episode_number=2, name="特番 2", overview=""),
            Episode(episode_number=3, name="特番 3", overview=""),
            Episode(episode_number=4, name="特番 4", overview=""),
            Episode(episode_number=5, name="特番 5", overview=""),
            Episode(episode_number=6, name="特番 6", overview=""),
            Episode(episode_number=7, name="特番 7", overview=""),
            Episode(episode_number=8, name="追忆篇", overview="追忆篇 OVA"),
            Episode(episode_number=9, name="特番 9", overview=""),
            Episode(episode_number=10, name="特番 10", overview=""),
        ],
    )

    specials_en = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="Special 1", overview=""),
            Episode(episode_number=2, name="Special 2", overview=""),
            Episode(episode_number=3, name="Special 3", overview=""),
            Episode(episode_number=4, name="Special 4", overview=""),
            Episode(episode_number=5, name="Special 5", overview=""),
            Episode(episode_number=6, name="Special 6", overview=""),
            Episode(episode_number=7, name="Special 7", overview=""),
            Episode(episode_number=8, name="Tsuioku-hen", overview="Reminiscence Arc OVA"),
            Episode(episode_number=9, name="Special 9", overview=""),
            Episode(episode_number=10, name="Special 10", overview=""),
        ],
    )

    return PlanTestCase(
        name="Mahouka OVA (Tsuioku Hen)",
        series_dir_name=series_dir,
        tmdb_id=60833,
        series_name_zh_cn="魔法科高校的劣等生",
        year=2014,
        season_episode_counts={0: 10, 1: 26, 2: 13, 3: 13},
        specials_zh=specials_zh,
        specials_en=specials_en,
        videos=videos,
        subtitles=subtitles,
        expected_mappings=expected,
    )


# ==============================================================================
# Test Case: Attack on Titan - Multi-season complex case (subset)
# ==============================================================================


def build_aot_s3_case() -> PlanTestCase:
    """
    进击的巨人 Season 3 - Episodes 38-59 mapped to S03E01-22
    From: [VCB-Studio] Shingeki no Kyojin_d4aa381c.rename_plan.json (subset)
    """
    series_dir = "[VCB-Studio] Shingeki no Kyojin"

    videos: list[FileCandidate] = []
    subtitles: list[FileCandidate] = []
    expected: list[ExpectedMapping] = []

    # Season 3 has episodes 38-59 (original numbering) mapped to S03E01-22
    vid_id = 1
    for orig_ep, s3_ep in zip(range(38, 60), range(1, 23)):
        filename = f"[BeanSub&VCB-Studio] Shingeki no Kyojin Season 3 [Ma10p_1080p]/[BeanSub&VCB-Studio] Shingeki no Kyojin [{orig_ep}][Ma10p_1080p][x265_flac].mkv"
        videos.append(
            FileCandidate(
                id=vid_id, rel_path=filename, ext=".mkv", size_bytes=100_000_000
            )
        )
        expected.append(
            ExpectedMapping(
                video_id=vid_id, season=3, episode_start=s3_ep, episode_end=s3_ep
            )
        )
        vid_id += 1

    # Add corresponding subtitles
    sub_id = len(videos) + 1
    for orig_ep in range(38, 60):
        for suffix in ["chs", "cht"]:
            filename = f"[BeanSub&VCB-Studio] Shingeki no Kyojin Season 3 [Ma10p_1080p]/[BeanSub&VCB-Studio] Shingeki no Kyojin [{orig_ep}][Ma10p_1080p][x265_flac].{suffix}.ass"
            subtitles.append(
                FileCandidate(id=sub_id, rel_path=filename, ext=".ass", size_bytes=200_000)
            )
            sub_id += 1

    return PlanTestCase(
        name="Attack on Titan S3",
        series_dir_name=series_dir,
        tmdb_id=1429,
        series_name_zh_cn="进击的巨人",
        year=2013,
        season_episode_counts={0: 40, 1: 25, 2: 12, 3: 22, 4: 28},  # Approx TMDB counts
        specials_zh=None,
        specials_en=None,
        videos=videos,
        subtitles=subtitles,
        expected_mappings=expected,
    )


# ==============================================================================
# Parameterized tests
# ==============================================================================


ALL_TEST_CASES = [
    pytest.param(build_mahouka_s1_case, id="mahouka_s1"),
    pytest.param(build_mahouka_s2_case, id="mahouka_s2_raihousha"),
    pytest.param(build_mahouka_s3_case, id="mahouka_s3"),
    pytest.param(build_mahouka_ova_case, id="mahouka_ova_tsuioku"),
    pytest.param(build_aot_s3_case, id="aot_s3"),
]


@pytest.mark.integration
@pytest.mark.parametrize("build_case", ALL_TEST_CASES)
def test_llm_episode_mapping_from_plan(
    build_case: Any, tmp_path: Path
) -> None:
    """
    End-to-end test that verifies LLM produces correct episode mappings
    for real anime file structures extracted from production rename plans.
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    case: PlanTestCase = build_case()
    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path / case.series_dir_name,
        videos=case.videos,
        subtitles=case.subtitles,
    )

    result = map_episodes_with_llm(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        season_episode_counts=case.season_episode_counts,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=4096,
    )

    # Build lookup by video ID
    by_vid = {item.video_id: item for item in result.items}

    # Verify all expected mappings
    for exp in case.expected_mappings:
        assert exp.video_id in by_vid, (
            f"Video {exp.video_id} not in mapping result for {case.name}"
        )
        actual = by_vid[exp.video_id]
        assert actual.season == exp.season, (
            f"Video {exp.video_id}: expected season {exp.season}, got {actual.season}"
        )
        assert actual.episode_start == exp.episode_start, (
            f"Video {exp.video_id}: expected ep_start {exp.episode_start}, "
            f"got {actual.episode_start}"
        )
        assert actual.episode_end == exp.episode_end, (
            f"Video {exp.video_id}: expected ep_end {exp.episode_end}, "
            f"got {actual.episode_end}"
        )

    # Verify total count matches
    assert len(result.items) == len(case.expected_mappings), (
        f"{case.name}: expected {len(case.expected_mappings)} mappings, "
        f"got {len(result.items)}"
    )


# ==============================================================================
# Focused tests for specific challenging cases
# ==============================================================================


@pytest.mark.integration
def test_llm_handles_numbered_episodes_sequentially(tmp_path: Path) -> None:
    """
    Verify LLM correctly maps sequentially numbered episodes [01] through [26]
    to S01E01 through S01E26.
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    case = build_mahouka_s1_case()
    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path / case.series_dir_name,
        videos=case.videos,
        subtitles=case.subtitles,
    )

    result = map_episodes_with_llm(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        season_episode_counts=case.season_episode_counts,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=4096,
    )

    # All 26 episodes should be mapped
    assert len(result.items) == 26

    # Verify sequential mapping
    for item in result.items:
        # Video IDs are 2-27 for episodes 1-26
        expected_ep = item.video_id - 1
        assert item.season == 1
        assert item.episode_start == expected_ep
        assert item.episode_end == expected_ep


@pytest.mark.integration
def test_llm_maps_ova_to_correct_special_slot(tmp_path: Path) -> None:
    """
    Verify LLM correctly maps OVA (Tsuioku Hen) to S00E08 based on TMDB specials metadata.
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    case = build_mahouka_ova_case()
    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path / case.series_dir_name,
        videos=case.videos,
        subtitles=case.subtitles,
    )

    result = map_episodes_with_llm(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        season_episode_counts=case.season_episode_counts,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=1024,
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item.video_id == 1
    assert item.season == 0
    assert item.episode_start == 8, f"Expected S00E08, got S00E{item.episode_start:02d}"


@pytest.mark.integration
def test_llm_associates_subtitles_with_videos(tmp_path: Path) -> None:
    """
    Verify LLM correctly associates .chs.ass and .cht.ass subtitles with their videos.
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    case = build_mahouka_s2_case()
    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path / case.series_dir_name,
        videos=case.videos,
        subtitles=case.subtitles,
    )

    result = map_episodes_with_llm(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        season_episode_counts=case.season_episode_counts,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=4096,
    )

    # All 13 episodes should be mapped
    assert len(result.items) == 13

    # Each episode should have 2 subtitles (chs + cht)
    for item in result.items:
        assert len(item.subtitle_ids) == 2, (
            f"Episode {item.episode_start} should have 2 subtitles, "
            f"got {len(item.subtitle_ids)}"
        )


@pytest.mark.integration
def test_llm_handles_absolute_episode_numbers(tmp_path: Path) -> None:
    """
    Verify LLM correctly maps absolute episode numbers (38-59) to season-relative (S03E01-22).
    """
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    case = build_aot_s3_case()
    llm = openai_llm_from_env()

    scan = ScanResult(
        series_dir=tmp_path / case.series_dir_name,
        videos=case.videos,
        subtitles=case.subtitles,
    )

    result = map_episodes_with_llm(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        season_episode_counts=case.season_episode_counts,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=4096,
    )

    # All 22 Season 3 episodes should be mapped
    assert len(result.items) == 22

    # Verify mapping to S03
    for item in result.items:
        assert item.season == 3, f"Expected season 3, got {item.season}"
        assert 1 <= item.episode_start <= 22
        assert 1 <= item.episode_end <= 22

