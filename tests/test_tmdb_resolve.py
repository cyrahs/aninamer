from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from aninamer.errors import LLMOutputError
from aninamer.llm_client import ChatMessage
from aninamer.prompts import build_tmdb_tv_id_select_messages
from aninamer.tmdb_client import TvSearchResult
from aninamer.tmdb_resolve import parse_selected_tmdb_tv_id, resolve_tmdb_tv_id_with_llm


@dataclass
class FakeLLM:
    reply: str
    calls: int = 0
    last_messages: list[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        self.last_messages = list(messages)
        return self.reply


def _cand(i: int, name: str, date: str | None = None) -> TvSearchResult:
    return TvSearchResult(
        id=i,
        name=name,
        first_air_date=date,
        original_name=None,
        popularity=1.0,
        vote_count=10,
    )


def test_build_tmdb_tv_id_select_messages_contains_allowed_ids_and_candidate_lines() -> None:
    cands = [_cand(1, "A", "2020-01-01"), _cand(2, "B", None)]
    msgs = build_tmdb_tv_id_select_messages("MyDir", cands, max_candidates=5)

    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"

    user = msgs[1].content
    assert "MyDir" in user
    assert "allowed" in user.lower()
    assert "1|A|2020-01-01" in user
    assert "2|B|" in user  # first_air_date missing -> empty field


def test_parse_selected_tmdb_tv_id_accepts_valid() -> None:
    assert parse_selected_tmdb_tv_id('{"tmdb": 123}', allowed_ids={123, 456}) == 123


def test_parse_selected_tmdb_tv_id_rejects_non_json() -> None:
    with pytest.raises(LLMOutputError):
        parse_selected_tmdb_tv_id("not json", allowed_ids={1})


def test_parse_selected_tmdb_tv_id_rejects_extra_keys() -> None:
    with pytest.raises(LLMOutputError):
        parse_selected_tmdb_tv_id('{"tmdb": 1, "x": 2}', allowed_ids={1})


def test_parse_selected_tmdb_tv_id_accepts_string_value() -> None:
    assert parse_selected_tmdb_tv_id('{"tmdb": "1"}', allowed_ids={1}) == 1


def test_parse_selected_tmdb_tv_id_rejects_non_numeric_string() -> None:
    with pytest.raises(LLMOutputError):
        parse_selected_tmdb_tv_id('{"tmdb": "abc"}', allowed_ids={1})


def test_parse_selected_tmdb_tv_id_rejects_id_not_allowed() -> None:
    with pytest.raises(LLMOutputError):
        parse_selected_tmdb_tv_id('{"tmdb": 999}', allowed_ids={1, 2})


def test_resolve_tmdb_tv_id_with_llm_short_circuits_single_candidate() -> None:
    llm = FakeLLM(reply='{"tmdb": 1}')
    cands = [_cand(1, "Only")]
    tv_id = resolve_tmdb_tv_id_with_llm("Dir", cands, llm)

    assert tv_id == 1
    assert llm.calls == 0


def test_resolve_tmdb_tv_id_with_llm_calls_llm_and_parses() -> None:
    llm = FakeLLM(reply='{"tmdb": 2}')
    cands = [_cand(1, "A"), _cand(2, "B")]
    tv_id = resolve_tmdb_tv_id_with_llm("Dir", cands, llm)

    assert tv_id == 2
    assert llm.calls == 1
    assert llm.last_messages is not None
    assert llm.last_messages[0].role == "system"
    assert llm.last_messages[1].role == "user"


def test_resolve_tmdb_tv_id_with_llm_respects_max_candidates() -> None:
    llm = FakeLLM(reply='{"tmdb": 3}')
    cands = [_cand(1, "A"), _cand(2, "B"), _cand(3, "C"), _cand(999, "Z")]

    # max_candidates=3 means 999 is not allowed
    tv_id = resolve_tmdb_tv_id_with_llm("Dir", cands, llm, max_candidates=3)
    assert tv_id == 3

    # If LLM tries to pick 999, it should error
    llm2 = FakeLLM(reply='{"tmdb": 999}')
    with pytest.raises(LLMOutputError):
        resolve_tmdb_tv_id_with_llm("Dir", cands, llm2, max_candidates=3)
