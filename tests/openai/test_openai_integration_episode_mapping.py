from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import NoReturn

import pytest

from aninamer.config import OpenAISettings
from aninamer.episode_mapping import (
    EpisodeMappingResult,
    parse_episode_mapping_output,
)
from aninamer.errors import LLMOutputError
from aninamer.openai_llm_client import openai_llm_from_settings
from aninamer.prompts import build_episode_mapping_messages
from aninamer.scanner import FileCandidate, ScanResult
from aninamer.tmdb_client import Episode, SeasonDetails


@dataclass(frozen=True)
class EpisodeMappingEvalCase:
    name: str
    series_dir: str
    videos: tuple[FileCandidate, ...]
    subtitles: tuple[FileCandidate, ...]
    season_episode_counts: dict[int, int]
    required_mappings: dict[int, tuple[int, int, int]]
    expected_subtitles: dict[int, set[int]]
    forbidden_video_ids: set[int]
    tmdb_id: int = 123
    series_name_zh_cn: str = "测试动画"
    year: int | None = 2020
    forbidden_subtitle_ids: set[int] = field(default_factory=set)
    specials_zh: SeasonDetails | None = None
    specials_en: SeasonDetails | None = None
    regular_seasons_zh: dict[int, SeasonDetails] = field(default_factory=dict)
    regular_seasons_en: dict[int, SeasonDetails] = field(default_factory=dict)
    existing_episode_numbers_by_season: dict[int, tuple[int, ...]] | None = None


def _video(i: int, rel_path: str, *, size: int = 100_000_000) -> FileCandidate:
    ext = Path(rel_path).suffix or ".mkv"
    return FileCandidate(id=i, rel_path=rel_path, ext=ext, size_bytes=size)


def _subtitle(i: int, rel_path: str, *, size: int = 200_000) -> FileCandidate:
    ext = Path(rel_path).suffix
    return FileCandidate(id=i, rel_path=rel_path, ext=ext, size_bytes=size)


def _regular_season(season_number: int, count: int, *, lang: str) -> SeasonDetails:
    if lang == "zh":
        episodes = [
            Episode(episode_number=i, name=f"第{i}话", overview="")
            for i in range(1, count + 1)
        ]
    else:
        episodes = [
            Episode(episode_number=i, name=f"Episode {i}", overview="")
            for i in range(1, count + 1)
        ]
    return SeasonDetails(id=season_number, season_number=season_number, episodes=episodes)


def _case_debug_header(
    case: EpisodeMappingEvalCase,
    settings: OpenAISettings,
    *,
    raw_output_path: Path,
) -> str:
    return (
        f"case={case.name}\n"
        f"model={settings.model}\n"
        f"base_url={settings.base_url}\n"
        f"reasoning_effort_mapping={settings.reasoning_effort_mapping}\n"
        "temperature=0.0\n"
        "max_output_tokens=4096\n"
        f"raw_output_path={raw_output_path}\n"
    )


def _write_raw_output(
    path: Path,
    *,
    case: EpisodeMappingEvalCase,
    settings: OpenAISettings,
    raw_output: str,
    parsed: EpisodeMappingResult | None = None,
) -> None:
    payload: dict[str, object] = {
        "case": case.name,
        "model": settings.model,
        "base_url": settings.base_url,
        "reasoning_effort_mapping": settings.reasoning_effort_mapping,
        "temperature": 0.0,
        "max_output_tokens": 4096,
        "raw_output": raw_output,
    }
    if parsed is not None:
        payload["parsed"] = asdict(parsed)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fail_eval(
    case: EpisodeMappingEvalCase,
    settings: OpenAISettings,
    *,
    raw_output_path: Path,
    raw_output: str,
    failures: list[str],
    parsed: EpisodeMappingResult | None = None,
) -> NoReturn:
    _write_raw_output(
        raw_output_path,
        case=case,
        settings=settings,
        raw_output=raw_output,
        parsed=parsed,
    )
    parsed_text = ""
    if parsed is not None:
        parsed_text = "\nparsed:\n" + json.dumps(
            asdict(parsed), ensure_ascii=False, indent=2
        )
    pytest.fail(
        _case_debug_header(case, settings, raw_output_path=raw_output_path)
        + "\n".join(failures)
        + "\nraw output:\n"
        + raw_output
        + parsed_text
    )


def _score_mapping(case: EpisodeMappingEvalCase, result: EpisodeMappingResult) -> list[str]:
    failures: list[str] = []
    by_video = {item.video_id: item for item in result.items}

    for video_id, expected in case.required_mappings.items():
        item = by_video.get(video_id)
        if item is None:
            failures.append(f"video {video_id} missing from mapping")
            continue
        actual = (item.season, item.episode_start, item.episode_end)
        if actual != expected:
            failures.append(
                f"video {video_id} expected S{expected[0]:02d}E{expected[1]:02d}-E{expected[2]:02d}, "
                f"got S{actual[0]:02d}E{actual[1]:02d}-E{actual[2]:02d}"
            )

    for video_id in sorted(case.forbidden_video_ids):
        if video_id in by_video:
            item = by_video[video_id]
            failures.append(
                f"video {video_id} should be omitted but mapped to "
                f"S{item.season:02d}E{item.episode_start:02d}-E{item.episode_end:02d}"
            )

    expected_or_forbidden = set(case.required_mappings) | set(case.forbidden_video_ids)
    for video_id in sorted(set(by_video) - expected_or_forbidden):
        failures.append(f"video {video_id} was unexpectedly mapped")

    for video_id, expected_subtitles in case.expected_subtitles.items():
        item = by_video.get(video_id)
        if item is None:
            continue
        actual_subtitles = set(item.subtitle_ids)
        if actual_subtitles != expected_subtitles:
            failures.append(
                f"video {video_id} expected subtitles {sorted(expected_subtitles)}, "
                f"got {sorted(actual_subtitles)}"
            )

    used_subtitles = {
        subtitle_id for item in result.items for subtitle_id in item.subtitle_ids
    }
    forbidden_subtitles = used_subtitles & case.forbidden_subtitle_ids
    if forbidden_subtitles:
        failures.append(
            f"forbidden subtitle ids were mapped: {sorted(forbidden_subtitles)}"
        )

    return failures


def _basic_sequence_case() -> EpisodeMappingEvalCase:
    return EpisodeMappingEvalCase(
        name="basic_sequence_with_op_omitted",
        series_dir="测试动画",
        videos=(
            _video(1, "测试动画 - 01.mkv", size=120_000_000),
            _video(2, "测试动画 - 02.mkv", size=121_000_000),
            _video(3, "测试动画 NCOP.mkv", size=12_000_000),
        ),
        subtitles=(
            _subtitle(101, "测试动画 - 01.chs.ass"),
            _subtitle(102, "测试动画 - 02.chs.ass"),
            _subtitle(103, "测试动画 NCOP.ass", size=50_000),
        ),
        season_episode_counts={1: 12},
        regular_seasons_zh={1: _regular_season(1, 12, lang="zh")},
        regular_seasons_en={1: _regular_season(1, 12, lang="en")},
        required_mappings={1: (1, 1, 1), 2: (1, 2, 2)},
        expected_subtitles={1: {101}, 2: {102}},
        forbidden_video_ids={3},
        forbidden_subtitle_ids={103},
    )


def _season_two_case() -> EpisodeMappingEvalCase:
    return EpisodeMappingEvalCase(
        name="season_two_marker",
        series_dir="测试动画 S2",
        videos=(
            _video(1, "测试动画 - 01.mkv"),
            _video(2, "测试动画 - 02.mkv"),
            _video(3, "测试动画 ED.mkv", size=11_000_000),
        ),
        subtitles=(
            _subtitle(101, "测试动画 - 01.ass"),
            _subtitle(102, "测试动画 - 02.ass"),
            _subtitle(103, "测试动画 ED.ass", size=45_000),
        ),
        season_episode_counts={1: 12, 2: 12},
        regular_seasons_zh={
            1: _regular_season(1, 12, lang="zh"),
            2: _regular_season(2, 12, lang="zh"),
        },
        regular_seasons_en={
            1: _regular_season(1, 12, lang="en"),
            2: _regular_season(2, 12, lang="en"),
        },
        required_mappings={1: (2, 1, 1), 2: (2, 2, 2)},
        expected_subtitles={1: {101}, 2: {102}},
        forbidden_video_ids={3},
        forbidden_subtitle_ids={103},
    )


def _ova_case() -> EpisodeMappingEvalCase:
    specials_zh = SeasonDetails(
        id=100,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="OVA", overview="OVA 特别篇"),
            Episode(episode_number=2, name="OAD", overview="OAD 附赠动画"),
        ],
    )
    specials_en = SeasonDetails(
        id=100,
        season_number=0,
        episodes=[
            Episode(episode_number=1, name="OVA", overview="OVA special"),
            Episode(episode_number=2, name="OAD", overview="OAD special"),
        ],
    )
    return EpisodeMappingEvalCase(
        name="ova_oad_to_s00_with_extras_omitted",
        series_dir="测试动画 Specials",
        videos=(
            _video(1, "测试动画 OVA.mkv", size=150_000_000),
            _video(2, "测试动画 OAD 2.mkv", size=151_000_000),
            _video(3, "测试动画 OP.mkv", size=10_000_000),
            _video(4, "测试动画 PV1.mkv", size=9_000_000),
        ),
        subtitles=(
            _subtitle(101, "测试动画 OVA.ass"),
            _subtitle(102, "测试动画 OAD 2.ass"),
            _subtitle(103, "测试动画 OP.ass", size=40_000),
        ),
        season_episode_counts={0: 2, 1: 12},
        regular_seasons_zh={1: _regular_season(1, 12, lang="zh")},
        regular_seasons_en={1: _regular_season(1, 12, lang="en")},
        specials_zh=specials_zh,
        specials_en=specials_en,
        required_mappings={1: (0, 1, 1), 2: (0, 2, 2)},
        expected_subtitles={1: {101}, 2: {102}},
        forbidden_video_ids={3, 4},
        forbidden_subtitle_ids={103},
    )


def _multi_subtitle_case() -> EpisodeMappingEvalCase:
    return EpisodeMappingEvalCase(
        name="multiple_subtitles_same_episode",
        series_dir="测试动画",
        videos=(_video(1, "测试动画 S01E01.mkv"),),
        subtitles=(
            _subtitle(101, "测试动画 S01E01.chs.ass"),
            _subtitle(102, "测试动画 S01E01.cht.srt"),
            _subtitle(103, "测试动画 S01E02.ass"),
        ),
        season_episode_counts={1: 12},
        regular_seasons_zh={1: _regular_season(1, 12, lang="zh")},
        regular_seasons_en={1: _regular_season(1, 12, lang="en")},
        required_mappings={1: (1, 1, 1)},
        expected_subtitles={1: {101, 102}},
        forbidden_video_ids=set(),
        forbidden_subtitle_ids={103},
    )


def _existing_inventory_case() -> EpisodeMappingEvalCase:
    return EpisodeMappingEvalCase(
        name="existing_inventory_offset",
        series_dir="测试动画",
        videos=(
            _video(1, "测试动画 - 03.mkv"),
            _video(2, "测试动画 - 04.mkv"),
        ),
        subtitles=(
            _subtitle(101, "测试动画 - 03.ass"),
            _subtitle(102, "测试动画 - 04.ass"),
        ),
        season_episode_counts={1: 12},
        regular_seasons_zh={1: _regular_season(1, 12, lang="zh")},
        regular_seasons_en={1: _regular_season(1, 12, lang="en")},
        existing_episode_numbers_by_season={1: (1, 2)},
        required_mappings={1: (1, 3, 3), 2: (1, 4, 4)},
        expected_subtitles={1: {101}, 2: {102}},
        forbidden_video_ids=set(),
    )


def _semantic_title_existing_inventory_case() -> EpisodeMappingEvalCase:
    return EpisodeMappingEvalCase(
        name="semantic_title_existing_inventory",
        series_dir="クール de M",
        videos=(
            _video(
                1,
                "クール de M ～崩れないオンナ～ [中文字幕] [404988].mp4",
                size=100_000_000,
            ),
        ),
        subtitles=(),
        season_episode_counts={1: 2},
        tmdb_id=301686,
        series_name_zh_cn="高冷的M女",
        year=2025,
        regular_seasons_zh={
            1: SeasonDetails(
                id=30168601,
                season_number=1,
                episodes=[
                    Episode(episode_number=1, name="出会い", overview=""),
                    Episode(episode_number=2, name="崩れないオンナ", overview=""),
                ],
            )
        },
        regular_seasons_en={
            1: SeasonDetails(
                id=30168602,
                season_number=1,
                episodes=[
                    Episode(episode_number=1, name="Encounter", overview=""),
                    Episode(episode_number=2, name="The Unbreakable Woman", overview=""),
                ],
            )
        },
        existing_episode_numbers_by_season={1: (1,)},
        required_mappings={1: (1, 2, 2)},
        expected_subtitles={1: set()},
        forbidden_video_ids=set(),
    )


EPISODE_MAPPING_EVAL_CASES = (
    pytest.param(
        _basic_sequence_case(),
        id="basic-sequence",
        marks=(pytest.mark.llm_smoke, pytest.mark.llm_eval),
    ),
    pytest.param(_season_two_case(), id="season-two", marks=pytest.mark.llm_eval),
    pytest.param(_ova_case(), id="ova-oad", marks=pytest.mark.llm_eval),
    pytest.param(
        _multi_subtitle_case(), id="multi-subtitle", marks=pytest.mark.llm_eval
    ),
    pytest.param(
        _existing_inventory_case(), id="existing-inventory", marks=pytest.mark.llm_eval
    ),
    pytest.param(
        _semantic_title_existing_inventory_case(),
        id="semantic-title-existing-inventory",
        marks=pytest.mark.llm_eval,
    ),
)


@pytest.mark.integration
@pytest.mark.parametrize("case", EPISODE_MAPPING_EVAL_CASES)
def test_episode_mapping_real_llm_api_output(
    case: EpisodeMappingEvalCase,
    tmp_path: Path,
    integration_openai_settings: OpenAISettings,
) -> None:
    llm = openai_llm_from_settings(integration_openai_settings)
    scan = ScanResult(
        series_dir=tmp_path / case.series_dir,
        videos=list(case.videos),
        subtitles=list(case.subtitles),
    )
    messages = build_episode_mapping_messages(
        tmdb_id=case.tmdb_id,
        series_name_zh_cn=case.series_name_zh_cn,
        year=case.year,
        series_dir=case.series_dir,
        season_episode_counts=case.season_episode_counts,
        regular_seasons_zh=case.regular_seasons_zh,
        regular_seasons_en=case.regular_seasons_en,
        specials_zh=case.specials_zh,
        specials_en=case.specials_en,
        videos=scan.videos,
        subtitles=scan.subtitles,
        existing_episode_numbers_by_season=case.existing_episode_numbers_by_season,
    )

    raw_output = llm.chat(messages, temperature=0.0, max_output_tokens=4096)
    raw_output_path = tmp_path / f"{case.name}.llm-output.json"

    try:
        result = parse_episode_mapping_output(
            raw_output,
            expected_tmdb_id=case.tmdb_id,
            video_ids={video.id for video in scan.videos},
            subtitle_ids={subtitle.id for subtitle in scan.subtitles},
            season_episode_counts=case.season_episode_counts,
            existing_episode_numbers_by_season=case.existing_episode_numbers_by_season,
        )
    except LLMOutputError as exc:
        _fail_eval(
            case,
            integration_openai_settings,
            raw_output_path=raw_output_path,
            raw_output=raw_output,
            failures=[f"LLM output failed parser validation: {exc}"],
        )

    failures = _score_mapping(case, result)
    if failures:
        _fail_eval(
            case,
            integration_openai_settings,
            raw_output_path=raw_output_path,
            raw_output=raw_output,
            failures=failures,
            parsed=result,
        )

    _write_raw_output(
        raw_output_path,
        case=case,
        settings=integration_openai_settings,
        raw_output=raw_output,
        parsed=result,
    )
