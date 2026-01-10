from __future__ import annotations

from dataclasses import dataclass

import pytest

from aninamer.errors import LLMOutputError
from aninamer.llm_client import ChatMessage
from aninamer.prompts import build_tmdb_tv_id_select_messages
from aninamer.tmdb_client import TvSearchResult
from aninamer.tmdb_resolve import (
    parse_selected_tmdb_tv_id,
    resolve_tmdb_tv_id_with_llm,
)


def _candidate(
    tmdb_id: int,
    name: str,
    *,
    first_air_date: str | None = None,
    original_name: str | None = None,
    popularity: float | None = None,
    vote_count: int | None = None,
    genre_ids: tuple[int, ...] | None = None,
    origin_country: tuple[str, ...] | None = None,
) -> TvSearchResult:
    return TvSearchResult(
        id=tmdb_id,
        name=name,
        first_air_date=first_air_date,
        original_name=original_name,
        popularity=popularity,
        vote_count=vote_count,
        genre_ids=genre_ids,
        origin_country=origin_country,
    )


@dataclass
class FakeLLM:
    response: str
    calls: list[tuple[list[ChatMessage], float, int]]

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls.append((list(messages), temperature, max_output_tokens))
        return self.response


def test_build_tmdb_tv_id_select_messages_formats_content() -> None:
    candidates = [
        _candidate(
            1,
            "Show\nOne",
            popularity=1.2,
            vote_count=10,
            genre_ids=(16,),
            origin_country=("JP",),
        ),
        _candidate(
            2,
            "Show Two",
            first_air_date="2021-01-01",
            original_name="Orig\rName",
        ),
    ]

    messages = build_tmdb_tv_id_select_messages(
        "Series Dir",
        candidates,
        max_candidates=1,
    )

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"

    user_content = messages[1].content
    assert "dirname: Series Dir" in user_content
    assert (
        "id|name|first_air_date|original_name|origin_country|genre_ids" in user_content
    )
    assert "Show\nOne" not in user_content
    assert '1|Show One|""|""|JP|16' in user_content
    assert "Show Two" not in user_content
    assert "allowed ids: [1]" in user_content
    assert 'required output schema: {"tmdb": <one of allowed ids>}' in user_content


def test_build_tmdb_tv_id_select_messages_requires_candidates() -> None:
    with pytest.raises(ValueError):
        build_tmdb_tv_id_select_messages("Series Dir", [])


def test_build_tmdb_tv_id_select_messages_rejects_bad_max_candidates() -> None:
    candidates = [_candidate(1, "Show")]
    with pytest.raises(ValueError):
        build_tmdb_tv_id_select_messages("Series Dir", candidates, max_candidates=0)


def test_parse_selected_tmdb_tv_id_accepts_valid() -> None:
    assert parse_selected_tmdb_tv_id('{"tmdb": 10}', allowed_ids={10}) == 10


def test_parse_selected_tmdb_tv_id_accepts_code_fence() -> None:
    text = '```json\n{"tmdb": 111}\n```'
    assert parse_selected_tmdb_tv_id(text, allowed_ids={111}) == 111


def test_parse_selected_tmdb_tv_id_accepts_string_value() -> None:
    assert parse_selected_tmdb_tv_id('{"tmdb": "10"}', allowed_ids={10}) == 10


@pytest.mark.parametrize(
    "text, match",
    [
        ("not json", "invalid json"),
        ("[]", "expected JSON object"),
        ('{"tmdb": 1, "extra": 2}', "only 'tmdb'"),
        ('{"tmdb": "abc"}', "tmdb must be int"),
        ('{"tmdb": 2}', "not in allowed"),
    ],
)
def test_parse_selected_tmdb_tv_id_rejects_invalid(text: str, match: str) -> None:
    with pytest.raises(LLMOutputError, match=match):
        parse_selected_tmdb_tv_id(text, allowed_ids={1})


def test_resolve_tmdb_tv_id_with_llm_short_circuits() -> None:
    candidates = [_candidate(7, "Only Show")]
    llm = FakeLLM('{"tmdb": 7}', calls=[])

    result = resolve_tmdb_tv_id_with_llm("Series", candidates, llm)

    assert result == 7
    assert llm.calls == []


def test_resolve_tmdb_tv_id_with_llm_calls_and_truncates() -> None:
    candidates = [
        _candidate(1, "Show A"),
        _candidate(2, "Show B"),
        _candidate(3, "Show C"),
    ]
    llm = FakeLLM('{"tmdb": 2}', calls=[])

    result = resolve_tmdb_tv_id_with_llm(
        "Series",
        candidates,
        llm,
        max_candidates=2,
    )

    assert result == 2
    assert len(llm.calls) == 1
    _messages, temperature, max_output_tokens = llm.calls[0]
    assert temperature == 0.0
    assert max_output_tokens == 1024


def test_resolve_tmdb_tv_id_with_llm_rejects_unlisted_id() -> None:
    candidates = [
        _candidate(1, "Show A"),
        _candidate(2, "Show B"),
        _candidate(3, "Show C"),
    ]
    llm = FakeLLM('{"tmdb": 3}', calls=[])

    with pytest.raises(LLMOutputError):
        resolve_tmdb_tv_id_with_llm(
            "Series",
            candidates,
            llm,
            max_candidates=2,
        )
