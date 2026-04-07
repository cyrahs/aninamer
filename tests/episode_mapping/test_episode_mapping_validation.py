from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.errors import LLMOutputError
from aninamer.episode_mapping import parse_episode_mapping_output
from aninamer.prompts import build_episode_mapping_messages
from aninamer.scanner import FileCandidate, ScanResult
from aninamer.tmdb_client import Episode, SeasonDetails


def _fc(i: int, rel: str, ext: str = ".mkv", size: int = 123) -> FileCandidate:
    return FileCandidate(id=i, rel_path=rel, ext=ext, size_bytes=size)


def test_build_episode_mapping_messages_contains_schema_and_files_and_seasons() -> None:
    videos = [_fc(1, "Show - 01.mkv", ".mkv", 100), _fc(2, "Show - OP.mkv", ".mkv", 10)]
    subs = [_fc(3, "Show - 01.ass", ".ass", 50)]
    s01_zh = SeasonDetails(
        id=111,
        season_number=1,
        episodes=[
            Episode(episode_number=1, name="第一话", overview="普通集简介不应出现在提示词中"),
        ],
    )
    s01_en = SeasonDetails(
        id=112,
        season_number=1,
        episodes=[
            Episode(episode_number=1, name="Episode One", overview="regular overview should be omitted"),
        ],
    )
    s00_zh = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="OVA 1", overview="这是OVA第一话\n含换行"),
        ],
    )
    s00_en = SeasonDetails(
        id=999,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="OVA 1", overview="OVA episode 1"),
        ],
    )

    msgs = build_episode_mapping_messages(
        tmdb_id=123,
        series_name_zh_cn="测试动画",
        year=2020,
        series_dir="测试动画 S2",
        season_episode_counts={0: 1, 1: 12},
        regular_seasons_zh={1: s01_zh},
        regular_seasons_en={1: s01_en},
        specials_zh=s00_zh,
        specials_en=s00_en,
        videos=videos,
        subtitles=subs,
        existing_episode_numbers_by_season={0: (1,), 1: (1, 2)},
    )

    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"

    user = msgs[1].content
    assert '"tmdb"' in user and '"eps"' in user
    assert "tmdb_id: 123" in user
    assert "series_name_zh_cn: 测试动画" in user
    assert "S01=12" in user
    assert "S00=1" in user
    assert "series_dir: 测试动画 S2" in user
    assert "regular season episode names:" in user
    assert "S01|1|第一话|Episode One" in user
    assert "regular overview should be omitted" not in user
    assert "普通集简介不应出现在提示词中" not in user
    assert "existing destination episode inventory:" in user
    assert "S00|1" in user
    assert "S01|1,2" in user
    # file lines
    assert "1|Show - 01.mkv|100" in user
    assert "2|Show - OP.mkv|10" in user
    assert "3|Show - 01.ass|50" in user
    # specials line should be one-line (newline removed)
    assert "1|OVA 1|" in user
    assert "\n含换行" not in user


def test_parse_episode_mapping_output_accepts_valid_and_enforces_bounds_and_uniqueness() -> None:
    text = """
    ```json
    {"tmdb": 123, "eps": [
      {"v": 1, "s": 1, "e1": 1, "e2": 1, "u": [10]},
      {"v": 2, "s": 0, "e1": 1, "e2": 1, "u": [11, 12]}
    ]}
    ```
    """
    res = parse_episode_mapping_output(
        text,
        expected_tmdb_id=123,
        video_ids={1, 2, 3},
        subtitle_ids={10, 11, 12, 13},
        season_episode_counts={0: 2, 1: 12},
    )
    assert res.tmdb_id == 123
    assert len(res.items) == 2
    assert res.items[0].video_id == 1
    assert res.items[1].season == 0


def test_parse_episode_mapping_output_rejects_wrong_tmdb() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 999, "eps": []}',
            expected_tmdb_id=123,
            video_ids=set(),
            subtitle_ids=set(),
            season_episode_counts={1: 12},
        )


def test_parse_episode_mapping_output_rejects_extra_top_level_keys() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": [], "x": 1}',
            expected_tmdb_id=123,
            video_ids=set(),
            subtitle_ids=set(),
            season_episode_counts={1: 12},
        )


def test_parse_episode_mapping_output_rejects_id_not_allowed() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": [{"v": 999, "s": 1, "e1": 1, "e2": 1, "u": []}]}',
            expected_tmdb_id=123,
            video_ids={1, 2},
            subtitle_ids=set(),
            season_episode_counts={1: 12},
        )


def test_parse_episode_mapping_output_rejects_episode_out_of_bounds() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": [{"v": 1, "s": 1, "e1": 13, "e2": 13, "u": []}]}',
            expected_tmdb_id=123,
            video_ids={1},
            subtitle_ids=set(),
            season_episode_counts={1: 12},
        )


def test_parse_episode_mapping_output_rejects_overlapping_ranges() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": ['
            '{"v": 1, "s": 1, "e1": 1, "e2": 2, "u": []},'
            '{"v": 2, "s": 1, "e1": 2, "e2": 2, "u": []}'
            ']}',
            expected_tmdb_id=123,
            video_ids={1, 2},
            subtitle_ids=set(),
            season_episode_counts={1: 12},
        )


def test_parse_episode_mapping_output_rejects_existing_destination_inventory_overlap() -> None:
    with pytest.raises(LLMOutputError, match="already exists in destination inventory"):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": [{"v": 1, "s": 1, "e1": 2, "e2": 2, "u": []}]}',
            expected_tmdb_id=123,
            video_ids={1},
            subtitle_ids=set(),
            season_episode_counts={1: 12},
            existing_episode_numbers_by_season={1: (1, 2)},
        )


def test_parse_episode_mapping_output_rejects_subtitle_used_twice() -> None:
    with pytest.raises(LLMOutputError):
        parse_episode_mapping_output(
            '{"tmdb": 123, "eps": ['
            '{"v": 1, "s": 1, "e1": 1, "e2": 1, "u": [10]},'
            '{"v": 2, "s": 1, "e1": 2, "e2": 2, "u": [10]}'
            ']}',
            expected_tmdb_id=123,
            video_ids={1, 2},
            subtitle_ids={10},
            season_episode_counts={1: 12},
        )
