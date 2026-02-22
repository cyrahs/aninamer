from __future__ import annotations

import pytest

from aninamer.errors import LLMOutputError
from aninamer.json_utils import extract_first_json_object
from aninamer.tmdb_resolve import parse_selected_tmdb_tv_id


def test_extract_first_json_object_plain() -> None:
    assert extract_first_json_object('{"a":1}') == '{"a":1}'


def test_extract_first_json_object_with_fence() -> None:
    text = "```json\n{\"tmdb\": 123}\n```"
    assert extract_first_json_object(text) == '{"tmdb": 123}'


def test_extract_first_json_object_with_prefix_suffix() -> None:
    text = "Here you go:\n{\"tmdb\": 5}\nThanks"
    assert extract_first_json_object(text) == '{"tmdb": 5}'


def test_extract_first_json_object_raises_if_none() -> None:
    with pytest.raises(ValueError):
        extract_first_json_object("no json here")


def test_parse_selected_tmdb_tv_id_accepts_fenced_json() -> None:
    out = parse_selected_tmdb_tv_id("```json\n{\"tmdb\": 1}\n```", allowed_ids={1, 2})
    assert out == 1


def test_parse_selected_tmdb_tv_id_still_strict_schema() -> None:
    with pytest.raises(LLMOutputError):
        parse_selected_tmdb_tv_id("```json\n{\"tmdb\": 1, \"x\": 2}\n```", allowed_ids={1})
