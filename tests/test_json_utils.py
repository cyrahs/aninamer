from __future__ import annotations

import pytest

from aninamer.json_utils import extract_first_json_object


def test_extract_first_json_object_from_fenced_block() -> None:
    text = "prefix\n```json\n{\"tmdb\": 1}\n```\nsuffix"
    assert extract_first_json_object(text) == '{"tmdb": 1}'


def test_extract_first_json_object_with_wrapped_text() -> None:
    text = "hello {\"a\": 1} trailing"
    assert extract_first_json_object(text) == '{"a": 1}'


def test_extract_first_json_object_raises_when_missing() -> None:
    with pytest.raises(ValueError):
        extract_first_json_object("no json here")
