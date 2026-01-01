from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aninamer.episode_mapping import (
    EpisodeMappingResult,
    map_episodes_with_llm,
    parse_episode_mapping_output,
)
from aninamer.errors import LLMOutputError
from aninamer.llm_client import ChatMessage
from aninamer.scanner import FileCandidate, ScanResult


def test_parse_episode_mapping_output_accepts_valid() -> None:
    text = 'prefix {"tmdb": 1, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 2, "u": [3]}]} trailing'
    result = parse_episode_mapping_output(
        text,
        expected_tmdb_id=1,
        video_ids={1, 2},
        subtitle_ids={3, 4},
        season_episode_counts={0: 2, 1: 12},
    )

    assert isinstance(result, EpisodeMappingResult)
    assert result.tmdb_id == 1
    assert len(result.items) == 1
    item = result.items[0]
    assert item.video_id == 1
    assert item.season == 1
    assert item.episode_start == 1
    assert item.episode_end == 2
    assert item.subtitle_ids == (3,)


@pytest.mark.parametrize(
    "text, match",
    [
        (
            '{"tmdb": 1, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 1, "u": []}, {"v": 1, "s": 1, "e1": 2, "e2": 2, "u": []}]}',
            "video id 1 appears more than once",
        ),
        (
            '{"tmdb": 1, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 1, "u": [3]}, {"v": 2, "s": 1, "e1": 2, "e2": 2, "u": [3]}]}',
            "subtitle id 3 used by multiple items",
        ),
        (
            '{"tmdb": 1, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 2, "u": []}, {"v": 2, "s": 1, "e1": 2, "e2": 3, "u": []}]}',
            "episode overlap",
        ),
        (
            '{"tmdb": 1, "eps": [{"v": 1, "s": 2, "e1": 1, "e2": 1, "u": []}]}',
            "season 2 not in season_episode_counts",
        ),
        (
            '{"tmdb": 1, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 99, "u": []}]}',
            "exceeds season 1 count",
        ),
    ],
)
def test_parse_episode_mapping_output_rejects_invalid(text: str, match: str) -> None:
    with pytest.raises(LLMOutputError, match=match):
        parse_episode_mapping_output(
            text,
            expected_tmdb_id=1,
            video_ids={1, 2},
            subtitle_ids={3, 4},
            season_episode_counts={0: 2, 1: 12},
        )


def test_parse_episode_mapping_output_rejects_missing_json() -> None:
    with pytest.raises(LLMOutputError, match="invalid json"):
        parse_episode_mapping_output(
            "not json",
            expected_tmdb_id=1,
            video_ids={1},
            subtitle_ids=set(),
            season_episode_counts={1: 1},
        )


@dataclass
class FakeLLM:
    reply: str
    calls: list[tuple[list[ChatMessage], float, int]] = field(default_factory=list)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls.append((list(messages), temperature, max_output_tokens))
        return self.reply


def test_map_episodes_with_llm_calls_llm_and_parses() -> None:
    scan = ScanResult(
        series_dir=Path("series"),
        videos=[FileCandidate(id=1, rel_path="ep1.mkv", ext=".mkv", size_bytes=100)],
        subtitles=[FileCandidate(id=2, rel_path="ep1.srt", ext=".srt", size_bytes=10)],
    )
    llm = FakeLLM(
        reply='{"tmdb": 99, "eps": [{"v": 1, "s": 1, "e1": 1, "e2": 1, "u": [2]}]}'
    )

    result = map_episodes_with_llm(
        tmdb_id=99,
        series_name_zh_cn="Series Name",
        year=None,
        season_episode_counts={1: 1},
        specials_zh=None,
        specials_en=None,
        scan=scan,
        llm=llm,
        max_output_tokens=123,
    )

    assert result.tmdb_id == 99
    assert result.items[0].video_id == 1
    assert result.items[0].subtitle_ids == (2,)
    assert len(llm.calls) == 1
    _messages, temperature, max_output_tokens = llm.calls[0]
    assert temperature == 0.0
    assert max_output_tokens == 123
