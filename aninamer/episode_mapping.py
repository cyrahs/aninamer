from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Mapping, Sequence

from aninamer.errors import LLMOutputError, OpenAIError
from aninamer.json_utils import extract_first_json_object
from aninamer.llm_client import LLMClient
from aninamer.prompts import build_episode_mapping_messages
from aninamer.scanner import ScanResult
from aninamer.tmdb_client import SeasonDetails

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeMapItem:
    video_id: int
    season: int
    episode_start: int
    episode_end: int
    subtitle_ids: tuple[int, ...]


@dataclass(frozen=True)
class EpisodeMappingResult:
    tmdb_id: int
    items: tuple[EpisodeMapItem, ...]


def parse_episode_mapping_output(
    text: str,
    *,
    expected_tmdb_id: int,
    video_ids: set[int],
    subtitle_ids: set[int],
    season_episode_counts: Mapping[int, int],
) -> EpisodeMappingResult:
    try:
        json_text = extract_first_json_object(text)
    except ValueError as exc:
        raise LLMOutputError(f"invalid json: {exc}") from exc

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid json: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise LLMOutputError("expected JSON object")

    if set(data.keys()) != {"tmdb", "eps"}:
        raise LLMOutputError("expected object with only 'tmdb' and 'eps' keys")

    tmdb_id = data.get("tmdb")
    if not isinstance(tmdb_id, int):
        raise LLMOutputError("tmdb must be int")

    if tmdb_id != expected_tmdb_id:
        raise LLMOutputError(
            f"tmdb id {tmdb_id} does not match expected {expected_tmdb_id}"
        )

    eps = data.get("eps")
    if not isinstance(eps, list):
        raise LLMOutputError("eps must be list")

    items: list[EpisodeMapItem] = []
    used_videos: set[int] = set()
    used_subtitles: set[int] = set()
    season_claims: dict[int, set[int]] = {}

    for idx, entry in enumerate(eps, start=1):
        if not isinstance(entry, dict):
            raise LLMOutputError(f"eps[{idx}] must be object")
        if set(entry.keys()) != {"v", "s", "e1", "e2", "u"}:
            raise LLMOutputError(
                f"eps[{idx}] must have only keys 'v', 's', 'e1', 'e2', 'u'"
            )

        v = entry.get("v")
        if not isinstance(v, int):
            raise LLMOutputError(f"eps[{idx}].v must be int")
        if v not in video_ids:
            raise LLMOutputError(f"eps[{idx}].v {v} not in video ids")
        if v in used_videos:
            raise LLMOutputError(f"video id {v} appears more than once")
        used_videos.add(v)

        s = entry.get("s")
        if not isinstance(s, int):
            raise LLMOutputError(f"eps[{idx}].s must be int")
        if s not in season_episode_counts:
            if s == 0:
                raise LLMOutputError("season 0 not in season_episode_counts")
            raise LLMOutputError(f"season {s} not in season_episode_counts")

        e1 = entry.get("e1")
        e2 = entry.get("e2")
        if not isinstance(e1, int):
            raise LLMOutputError(f"eps[{idx}].e1 must be int")
        if not isinstance(e2, int):
            raise LLMOutputError(f"eps[{idx}].e2 must be int")
        if e1 < 1 or e2 < 1:
            raise LLMOutputError(f"eps[{idx}] episodes must be >= 1")
        if e1 > e2:
            raise LLMOutputError(f"eps[{idx}] episode range {e1}-{e2} is invalid")

        max_count = season_episode_counts[s]
        if e2 > max_count:
            raise LLMOutputError(
                f"eps[{idx}] episode range {e1}-{e2} exceeds season {s} count {max_count}"
            )

        u = entry.get("u")
        if not isinstance(u, list):
            raise LLMOutputError(f"eps[{idx}].u must be list")

        seen_in_item: set[int] = set()
        subtitle_list: list[int] = []
        for sub_id in u:
            if not isinstance(sub_id, int):
                raise LLMOutputError(f"eps[{idx}].u must contain only ints")
            if sub_id not in subtitle_ids:
                raise LLMOutputError(
                    f"eps[{idx}].u subtitle id {sub_id} not in subtitle ids"
                )
            if sub_id in seen_in_item:
                raise LLMOutputError(
                    f"eps[{idx}].u has duplicate subtitle id {sub_id}"
                )
            if sub_id in used_subtitles:
                raise LLMOutputError(
                    f"subtitle id {sub_id} used by multiple items"
                )
            seen_in_item.add(sub_id)
            used_subtitles.add(sub_id)
            subtitle_list.append(sub_id)

        claimed = season_claims.setdefault(s, set())
        for episode_number in range(e1, e2 + 1):
            if episode_number in claimed:
                raise LLMOutputError(
                    f"episode overlap in season {s} at episode {episode_number}"
                )
            claimed.add(episode_number)

        items.append(
            EpisodeMapItem(
                video_id=v,
                season=s,
                episode_start=e1,
                episode_end=e2,
                subtitle_ids=tuple(subtitle_list),
            )
        )

    return EpisodeMappingResult(tmdb_id=tmdb_id, items=tuple(items))


def map_episodes_with_llm(
    *,
    tmdb_id: int,
    series_name_zh_cn: str,
    year: int | None,
    season_episode_counts: Mapping[int, int],
    specials_zh: SeasonDetails | None,
    specials_en: SeasonDetails | None,
    scan: ScanResult,
    existing_s00_files: Sequence[str] | None = None,
    llm: LLMClient,
    max_output_tokens: int = 2048,
    max_attempts: int = 3,
) -> EpisodeMappingResult:
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    logger.info(
        "episode_map: start tmdb_id=%s video_count=%s subtitle_count=%s max_output_tokens=%s",
        tmdb_id,
        len(scan.videos),
        len(scan.subtitles),
        max_output_tokens,
    )
    messages = build_episode_mapping_messages(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        series_dir=scan.series_dir.name,
        season_episode_counts=season_episode_counts,
        specials_zh=specials_zh,
        specials_en=specials_en,
        videos=scan.videos,
        subtitles=scan.subtitles,
        existing_s00_files=existing_s00_files,
    )
    logger.debug(
        "episode_map: llm_prompt=%s",
        [{"role": msg.role, "content": msg.content} for msg in messages],
    )
    logger.info("episode_map: llm_call message_count=%s", len(messages))
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.chat(
                messages,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            )
            logger.debug("episode_map: raw_llm_output=%s", response)
            result = parse_episode_mapping_output(
                response,
                expected_tmdb_id=tmdb_id,
                video_ids={video.id for video in scan.videos},
                subtitle_ids={subtitle.id for subtitle in scan.subtitles},
                season_episode_counts=season_episode_counts,
            )
            mapped_subtitles_count = sum(
                len(item.subtitle_ids) for item in result.items
            )
            logger.info(
                "episode_map: parsed mapped_items_count=%s mapped_subtitles_count=%s",
                len(result.items),
                mapped_subtitles_count,
            )
            return result
        except (OpenAIError, LLMOutputError) as exc:
            if attempt >= max_attempts:
                raise
            logger.warning(
                "episode_map: llm_retry attempt=%s/%s error=%s",
                attempt,
                max_attempts,
                exc,
            )

    raise RuntimeError("episode_map: unreachable")
