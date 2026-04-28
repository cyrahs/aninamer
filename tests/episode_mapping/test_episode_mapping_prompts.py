from __future__ import annotations

from aninamer.prompts import build_episode_mapping_messages
from aninamer.scanner import FileCandidate
from aninamer.tmdb_client import Episode, SeasonDetails


def test_build_episode_mapping_messages_includes_sections_and_sanitizes() -> None:
    specials_zh = SeasonDetails(
        id=1,
        season_number=0,
        episodes=[
            Episode(
                episode_number=1,
                name="OVA|Name\nZ",
                overview="ABC\nDEF|GHI",
            )
        ],
    )
    specials_en = SeasonDetails(
        id=2,
        season_number=0,
        episodes=[
            Episode(
                episode_number=1,
                name="OVA\tEN",
                overview="EN|OV\nERVIEW",
            )
        ],
    )
    regular_zh = {
        1: SeasonDetails(
            id=3,
            season_number=1,
            episodes=[
                Episode(
                    episode_number=1,
                    name="正片|标题\n一",
                    overview="普通季简介不应出现在提示词中",
                )
            ],
        )
    }
    regular_en = {
        1: SeasonDetails(
            id=4,
            season_number=1,
            episodes=[
                Episode(
                    episode_number=1,
                    name="Episode\tOne",
                    overview="Regular season overview should not appear",
                )
            ],
        )
    }
    videos = [
        FileCandidate(id=1, rel_path="dir|a\nb.mkv", ext=".mkv", size_bytes=123)
    ]
    subtitles = [
        FileCandidate(id=4, rel_path="subs\tfile|.srt", ext=".srt", size_bytes=456)
    ]
    existing_s00_files = ["Series S00E01.mkv", "OVA|02\nchs.ass"]

    messages = build_episode_mapping_messages(
        tmdb_id=123,
        series_name_zh_cn="Series|Name\nX",
        year=2020,
        series_dir="Series Name S2",
        season_episode_counts={0: 2, 1: 12},
        regular_seasons_zh=regular_zh,
        regular_seasons_en=regular_en,
        specials_zh=specials_zh,
        specials_en=specials_en,
        videos=videos,
        subtitles=subtitles,
        existing_episode_numbers_by_season={0: (1,), 1: (1, 2)},
        existing_s00_files=existing_s00_files,
        max_special_overview_chars=5,
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "Output ONLY valid JSON" in messages[0].content
    assert "season markers" in messages[0].content

    user = messages[1].content
    assert '{"tmdb": <int>, "eps":' in user
    assert "tmdb_id: 123" in user
    assert "series_name_zh_cn: Series Name X" in user
    assert "year: 2020" in user
    assert "series_dir: Series Name S2" in user
    assert "S00=2" in user
    assert "S01=12" in user
    assert "regular season episode names:" in user
    assert "S01|1|正片 标题 一|Episode One" in user
    assert "Regular season overview should not appear" not in user
    assert "普通季简介不应出现在提示词中" not in user
    assert "specials (season 0):" in user
    assert "1|OVA Name Z|ABC D|OVA EN|EN OV" in user
    assert "existing destination episode inventory:" in user
    assert "S00|1" in user
    assert "S01|1,2" in user
    assert "existing destination S00 files:" in user
    assert "Series S00E01.mkv" in user
    assert "OVA 02 chs.ass" in user
    assert "1|dir a b.mkv|123" in user
    assert "4|subs file .srt|456" in user


def test_build_episode_mapping_messages_guides_semantic_match_after_existing_inventory() -> None:
    s01_zh = SeasonDetails(
        id=30168601,
        season_number=1,
        episodes=[
            Episode(episode_number=1, name="出会い", overview=""),
            Episode(episode_number=2, name="崩れないオンナ", overview=""),
        ],
    )
    videos = [
        FileCandidate(
            id=1,
            rel_path="クール de M ～崩れないオンナ～ [中文字幕] [404988].mp4",
            ext=".mp4",
            size_bytes=100,
        )
    ]

    messages = build_episode_mapping_messages(
        tmdb_id=301686,
        series_name_zh_cn="高冷的M女",
        year=2025,
        series_dir="クール de M",
        season_episode_counts={1: 2},
        regular_seasons_zh={1: s01_zh},
        regular_seasons_en={},
        specials_zh=None,
        specials_en=None,
        videos=videos,
        subtitles=[],
        existing_episode_numbers_by_season={1: (1,)},
    )

    system = messages[0].content
    user = messages[1].content
    assert "file title semantically matches the TMDB episode title" in system
    assert "later unoccupied episodes" in user
    assert "existing destination episode inventory:" in user
    assert "S01|1" in user
    assert "S01|2|崩れないオンナ|" in user
    assert "1|クール de M ～崩れないオンナ～ [中文字幕] [404988].mp4|100" in user
