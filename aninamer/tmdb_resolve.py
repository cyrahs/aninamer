from __future__ import annotations

import json
from typing import Sequence

from aninamer.errors import LLMOutputError
from aninamer.llm_client import LLMClient
from aninamer.prompts import build_tmdb_tv_id_select_messages
from aninamer.tmdb_client import TvSearchResult


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()

    if stripped.endswith("```") and stripped != "```":
        inner = stripped[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        return inner

    return text


def parse_selected_tmdb_tv_id(text: str, *, allowed_ids: set[int]) -> int:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"invalid json: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise LLMOutputError("expected JSON object")

    if set(data.keys()) != {"tmdb"}:
        raise LLMOutputError("expected object with only 'tmdb' key")

    tmdb_id = data.get("tmdb")
    if not isinstance(tmdb_id, int):
        raise LLMOutputError("tmdb must be int")

    if tmdb_id not in allowed_ids:
        raise LLMOutputError(f"tmdb id {tmdb_id} not in allowed ids")

    return tmdb_id


def resolve_tmdb_tv_id_with_llm(
    dirname: str,
    candidates: Sequence[TvSearchResult],
    llm: LLMClient,
    *,
    max_candidates: int = 5,
) -> int:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must be non-empty")

    if len(candidate_list) == 1:
        return candidate_list[0].id

    messages = build_tmdb_tv_id_select_messages(
        dirname,
        candidate_list,
        max_candidates=max_candidates,
    )
    response = llm.chat(messages, temperature=0.0, max_output_tokens=64)

    allowed_ids = {candidate.id for candidate in candidate_list[:max_candidates]}

    return parse_selected_tmdb_tv_id(response, allowed_ids=allowed_ids)
