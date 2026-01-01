from __future__ import annotations

import json
import logging
from typing import Sequence

from aninamer.errors import LLMOutputError
from aninamer.json_utils import extract_first_json_object
from aninamer.llm_client import LLMClient
from aninamer.prompts import build_tmdb_tv_id_select_messages
from aninamer.tmdb_client import TvSearchResult

logger = logging.getLogger(__name__)


def parse_selected_tmdb_tv_id(text: str, *, allowed_ids: set[int]) -> int:
    try:
        cleaned = extract_first_json_object(text)
    except ValueError:
        cleaned = text.strip()
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

    logger.info(
        "tmdb_resolve: start dirname=%s candidate_count=%s max_candidates=%s",
        dirname,
        len(candidate_list),
        max_candidates,
    )
    if len(candidate_list) == 1:
        logger.info(
            "tmdb_resolve: single_candidate id=%s",
            candidate_list[0].id,
        )
        return candidate_list[0].id

    messages = build_tmdb_tv_id_select_messages(
        dirname,
        candidate_list,
        max_candidates=max_candidates,
    )

    allowed_ids = {candidate.id for candidate in candidate_list[:max_candidates]}
    logger.info(
        "tmdb_resolve: llm_call allowed_ids=%s",
        sorted(allowed_ids),
    )
    response = llm.chat(messages, temperature=0.0, max_output_tokens=64)
    logger.debug("tmdb_resolve: raw_llm_output=%s", response)

    selected_id = parse_selected_tmdb_tv_id(response, allowed_ids=allowed_ids)
    logger.info("tmdb_resolve: selected id=%s", selected_id)
    return selected_id
