from __future__ import annotations

from typing import Mapping, Sequence

from aninamer.llm_client import ChatMessage
from aninamer.scanner import FileCandidate
from aninamer.tmdb_client import SeasonDetails
from aninamer.tmdb_client import TvSearchResult


def _format_field(value: object) -> str:
    if value is None:
        return '""'
    if isinstance(value, str):
        cleaned = value.replace("\n", " ").replace("\r", " ")
    else:
        cleaned = str(value)
    if cleaned == "":
        return '""'
    return cleaned


def _single_line(value: str) -> str:
    return value.replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _clean_cell(value: str | None, *, max_chars: int | None = None) -> str:
    if value is None:
        return ""
    cleaned = _single_line(value).replace("|", " ")
    if max_chars is not None and max_chars >= 0 and len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned


def build_tmdb_tv_id_select_messages(
    dirname: str,
    candidates: Sequence[TvSearchResult],
    *,
    max_candidates: int = 5,
) -> list[ChatMessage]:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    trimmed = list(candidates)[:max_candidates]
    allowed_ids = [candidate.id for candidate in trimmed]

    system_content = (
        "Select the correct TMDB TV id from the candidate list. "
        "Respond with ONLY valid JSON in the form {\"tmdb\": <int>} with no other keys, "
        "no markdown, and no commentary. The id must be selected from the candidates."
    )

    lines = [
        f"dirname: {dirname}",
        "candidates:",
        "id|name|first_air_date|original_name|popularity|vote_count",
    ]
    for candidate in trimmed:
        parts = [
            str(candidate.id),
            _format_field(candidate.name),
            _format_field(candidate.first_air_date),
            _format_field(candidate.original_name),
            _format_field(candidate.popularity),
            _format_field(candidate.vote_count),
        ]
        lines.append("|".join(parts))

    lines.append(f"allowed ids: [{', '.join(str(value) for value in allowed_ids)}]")
    lines.append('required output schema: {"tmdb": <one of allowed ids>}')

    user_content = "\n".join(lines)

    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]


def build_tmdb_title_clean_messages(dirname: str) -> list[ChatMessage]:
    system_content = (
        "Extract the canonical TV series title from a noisy folder name for TMDB search. "
        "Respond with ONLY valid JSON in the form {\"title\": \"...\"} with no other keys, "
        "no markdown, and no commentary. The title should exclude release groups, "
        "quality tags, season/episode markers, and other non-title metadata."
    )

    lines = [
        f"dirname: {dirname}",
        'required output schema: {"title": "<string>"}',
    ]
    user_content = "\n".join(lines)

    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]


def build_episode_mapping_messages(
    *,
    tmdb_id: int,
    series_name_zh_cn: str,
    year: int | None,
    season_episode_counts: Mapping[int, int],
    specials_zh: SeasonDetails | None,
    specials_en: SeasonDetails | None,
    videos: Sequence[FileCandidate],
    subtitles: Sequence[FileCandidate],
    existing_s00_files: Sequence[str] | None = None,
    max_special_overview_chars: int = 160,
) -> list[ChatMessage]:
    system_content = (
        "Map episode video files to TMDB season/episode numbers. "
        "Output ONLY valid JSON with no markdown or extra text. "
        'Use the exact schema {"tmdb": <int>, "eps": [{"v": <int>, "s": <int>, '
        '"e1": <int>, "e2": <int>, "u": [<int>...]}]}. '
        "Include ONLY regular episodes (seasons >=1) and OVA/OAD specials in season 0. "
        "Omit OP/ED/PV/trailer/promo/NCOP/NCED/recap/credits/shorts/extras and any uncertain items. "
        "Never map two videos to the same episode range. "
        "If duplicate releases exist, choose only one (prefer larger size)."
    )

    lines: list[str] = []
    lines.append("schema (no extra keys):")
    lines.append(
        '{"tmdb": <int>, "eps": [{"v": <int>, "s": <int>, "e1": <int>, "e2": <int>, "u": [<int>...]}]}'
    )
    lines.append("rules:")
    lines.append("only output items to rename")
    lines.append("s must be 0 or in season_episode_counts keys")
    lines.append("for s==0: 1..season_episode_counts[0]")
    lines.append("for s>=1: 1..season_episode_counts[s]")
    lines.append("sort eps by v ascending")
    lines.append("u must contain only subtitle ids for that episode video; otherwise leave u empty")
    lines.append("put OVA/OAD in S00")
    lines.append("prefer matching OVA/OAD using TMDB specials name/overview that mention OVA/OAD")
    lines.append("if no explicit OVA/OAD info, assume local OVA/OAD order matches TMDB specials order")

    lines.append("TMDB:")
    lines.append(f"tmdb_id: {tmdb_id}")
    lines.append(f"series_name_zh_cn: {_clean_cell(series_name_zh_cn)}")
    lines.append(f"year: {_format_field(year)}")
    lines.append("season_episode_counts:")
    for season_number in sorted(season_episode_counts.keys()):
        count = season_episode_counts[season_number]
        lines.append(f"S{season_number:02d}={count}")

    if 0 in season_episode_counts and (specials_zh or specials_en):
        lines.append("specials (season 0):")
        lines.append("ep|name_zh|overview_zh_snippet|name_en|overview_en_snippet")
        zh_lookup = {
            episode.episode_number: episode for episode in specials_zh.episodes
        } if specials_zh else {}
        en_lookup = {
            episode.episode_number: episode for episode in specials_en.episodes
        } if specials_en else {}
        episode_numbers = sorted(set(zh_lookup.keys()) | set(en_lookup.keys()))
        for ep_number in episode_numbers:
            zh_episode = zh_lookup.get(ep_number)
            en_episode = en_lookup.get(ep_number)
            name_zh = _clean_cell(zh_episode.name if zh_episode else None)
            overview_zh = _clean_cell(
                zh_episode.overview if zh_episode else None,
                max_chars=max_special_overview_chars,
            )
            name_en = _clean_cell(en_episode.name if en_episode else None)
            overview_en = _clean_cell(
                en_episode.overview if en_episode else None,
                max_chars=max_special_overview_chars,
            )
            lines.append(
                f"{ep_number}|{name_zh}|{overview_zh}|{name_en}|{overview_en}"
            )

    if existing_s00_files:
        lines.append("existing destination S00 files:")
        lines.append("name")
        for name in existing_s00_files:
            cleaned = _clean_cell(name)
            if cleaned:
                lines.append(cleaned)

    lines.append("FILES:")
    lines.append("videos:")
    lines.append("id|rel_path|size_bytes")
    for video in videos:
        rel_path = _clean_cell(video.rel_path)
        lines.append(f"{video.id}|{rel_path}|{video.size_bytes}")

    lines.append("subtitles:")
    lines.append("id|rel_path|size_bytes")
    for subtitle in subtitles:
        rel_path = _clean_cell(subtitle.rel_path)
        lines.append(f"{subtitle.id}|{rel_path}|{subtitle.size_bytes}")

    user_content = "\n".join(lines)

    return [
        ChatMessage(role="system", content=system_content),
        ChatMessage(role="user", content=user_content),
    ]
