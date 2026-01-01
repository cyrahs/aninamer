from __future__ import annotations

from typing import Sequence

from aninamer.llm_client import ChatMessage
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
