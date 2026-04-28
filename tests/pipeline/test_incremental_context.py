from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aninamer.llm_client import ChatMessage
from aninamer.pipeline import (
    PlanBuildOptions,
    build_rename_plan_for_series,
    inspect_existing_episode_inventory,
)
from aninamer.tmdb_client import Episode, SeasonDetails, SeasonSummary, TvDetails


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_inspect_existing_episode_inventory_matches_by_tmdb_id(tmp_path: Path) -> None:
    output_root = tmp_path / "library"

    matched_root = output_root / "旧标题 (2020) {tmdb-123}"
    _write(matched_root / "S01" / "旧标题 S01E01.mkv", b"video-1")
    _write(matched_root / "S01" / "旧标题 S01E03-E04.mkv", b"video-34")
    _write(matched_root / "S02" / "旧标题 S02E01.mkv", b"video-s02")
    _write(matched_root / "S00" / "旧标题 S00E01.mkv", b"ova")
    _write(matched_root / "S01" / "旧标题 S01E01.chs.ass", b"sub")

    unmatched_root = output_root / "别的标题 (2020) {tmdb-999}"
    _write(unmatched_root / "S01" / "别的标题 S01E01.mkv", b"other")

    inventory = inspect_existing_episode_inventory(output_root, 123)

    assert inventory.matched_series_dirs == (matched_root,)
    assert inventory.occupied_episode_numbers_by_season == {
        0: (1,),
        1: (1, 3, 4),
        2: (1,),
    }
    assert inventory.existing_s00_files == ("旧标题 S00E01.mkv",)


@dataclass
class FakeLLM:
    reply: str
    last_messages: list[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.last_messages = list(messages)
        return self.reply


class FakeTMDBClient:
    def __init__(self) -> None:
        self.get_season_calls: list[tuple[int, str]] = []

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        return (
            "测试动画",
            TvDetails(
                id=tv_id,
                name="测试动画",
                original_name="Test Anime",
                first_air_date="2020-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=3)],
            ),
        )

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        self.get_season_calls.append((season_number, language))
        if season_number != 1:
            return SeasonDetails(id=None, season_number=season_number, episodes=[])
        if language == "zh-CN":
            return SeasonDetails(
                id=1001,
                season_number=1,
                episodes=[
                    Episode(episode_number=1, name="开始", overview="不应进入普通季提示词"),
                    Episode(episode_number=2, name="追踪", overview="不应进入普通季提示词"),
                    Episode(episode_number=3, name="归来", overview="不应进入普通季提示词"),
                ],
            )
        return SeasonDetails(
            id=1002,
            season_number=1,
            episodes=[
                Episode(episode_number=1, name="Start", overview="regular overview omitted"),
                Episode(episode_number=2, name="Chase", overview="regular overview omitted"),
                Episode(episode_number=3, name="Return", overview="regular overview omitted"),
            ],
        )


class CoolDeMTMDBClient:
    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        return (
            "高冷的M女",
            TvDetails(
                id=tv_id,
                name="高冷的M女",
                original_name="クール de M",
                first_air_date="2025-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=2)],
            ),
        )

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        if season_number != 1:
            return SeasonDetails(id=None, season_number=season_number, episodes=[])
        if language == "zh-CN":
            return SeasonDetails(
                id=30168601,
                season_number=1,
                episodes=[
                    Episode(episode_number=1, name="出会い", overview=""),
                    Episode(episode_number=2, name="崩れないオンナ", overview=""),
                ],
            )
        return SeasonDetails(
            id=30168602,
            season_number=1,
            episodes=[
                Episode(episode_number=1, name="Encounter", overview=""),
                Episode(episode_number=2, name="The Unbreakable Woman", overview=""),
            ],
        )


def test_build_rename_plan_for_series_includes_incremental_inventory_and_regular_episode_names(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "归来篇 {tmdb-123}"
    output_root = tmp_path / "library"
    _write(series_dir / "归来.mkv", b"video")
    _write(series_dir / "归来.chs.ass", b"subtitle")

    existing_root = output_root / "旧标题 (2020) {tmdb-123}"
    _write(existing_root / "S01" / "旧标题 S01E01.mkv", b"old-1")
    _write(existing_root / "S01" / "旧标题 S01E02.mkv", b"old-2")

    llm = FakeLLM(
        reply='{"tmdb": 123, "eps": [{"v": 1, "s": 1, "e1": 3, "e2": 3, "u": [2]}]}'
    )
    tmdb = FakeTMDBClient()

    plan = build_rename_plan_for_series(
        series_dir=series_dir,
        output_root=output_root,
        options=PlanBuildOptions(max_output_tokens=512),
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm,
    )

    assert [move.dst.name for move in plan.moves] == [
        "测试动画 S01E03.chs.ass",
        "测试动画 S01E03.mkv",
    ]
    assert tmdb.get_season_calls == [(1, "zh-CN"), (1, "en-US")]

    assert llm.last_messages is not None
    prompt = llm.last_messages[1].content
    assert "regular season episode names:" in prompt
    assert "S01|3|归来|Return" in prompt
    assert "existing destination episode inventory:" in prompt
    assert "S01|1,2" in prompt
    assert "regular overview omitted" not in prompt
    assert "不应进入普通季提示词" not in prompt


def test_build_rename_plan_for_series_maps_semantic_title_after_existing_episode(
    tmp_path: Path,
) -> None:
    series_dir = tmp_path / "クール de M {tmdb-301686}"
    output_root = tmp_path / "library"
    video_name = "クール de M ～崩れないオンナ～ [中文字幕] [404988].mp4"
    _write(series_dir / video_name, b"video")

    existing_root = output_root / "高冷的M女 (2025) {tmdb-301686}"
    _write(existing_root / "S01" / "高冷的M女 S01E01.mp4", b"existing")

    llm = FakeLLM(
        reply='{"tmdb": 301686, "eps": [{"v": 1, "s": 1, "e1": 2, "e2": 2, "u": []}]}'
    )

    plan = build_rename_plan_for_series(
        series_dir=series_dir,
        output_root=output_root,
        options=PlanBuildOptions(max_output_tokens=512),
        tmdb_client_factory=CoolDeMTMDBClient,
        llm_for_mapping_factory=lambda: llm,
    )

    assert len(plan.moves) == 1
    move = plan.moves[0]
    assert move.kind == "video"
    assert move.src.name == video_name
    assert move.dst == (
        output_root
        / "高冷的M女 (2025) {tmdb-301686}"
        / "S01"
        / "高冷的M女 S01E02.mp4"
    ).resolve(strict=False)

    assert llm.last_messages is not None
    prompt = llm.last_messages[1].content
    assert "existing destination episode inventory:" in prompt
    assert "S01|1" in prompt
    assert "S01|2|崩れないオンナ|The Unbreakable Woman" in prompt
    assert f"1|{video_name}|" in prompt
