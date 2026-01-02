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
        season_episode_counts={0: 2, 1: 12},
        specials_zh=specials_zh,
        specials_en=specials_en,
        videos=videos,
        subtitles=subtitles,
        existing_s00_files=existing_s00_files,
        max_special_overview_chars=5,
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "Output ONLY valid JSON" in messages[0].content

    user = messages[1].content
    assert '{"tmdb": <int>, "eps":' in user
    assert "tmdb_id: 123" in user
    assert "series_name_zh_cn: Series Name X" in user
    assert "year: 2020" in user
    assert "S00=2" in user
    assert "S01=12" in user
    assert "specials (season 0):" in user
    assert "1|OVA Name Z|ABC D|OVA EN|EN OV" in user
    assert "existing destination S00 files:" in user
    assert "Series S00E01.mkv" in user
    assert "OVA 02 chs.ass" in user
    assert "1|dir a b.mkv|123" in user
    assert "4|subs file .srt|456" in user
