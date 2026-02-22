from __future__ import annotations

import pytest

from aninamer.name_clean import (
    build_tmdb_query_variants,
    clean_tmdb_query,
    extract_tmdb_id_tag,
)


def test_clean_tmdb_query_strips_release_tags_and_season_marker() -> None:
    name = "[DMG&SumiSora&VCB-Studio] Mahouka_Koukou.no.Rettousei S3 [Ma10p_1080p]"
    assert clean_tmdb_query(name) == "Mahouka Koukou no Rettousei"


def test_clean_tmdb_query_removes_chinese_season_marker() -> None:
    name = "Test \u7b2c3\u5b63"
    assert clean_tmdb_query(name) == "Test"


def test_build_tmdb_query_variants_shortens_cleaned_name() -> None:
    name = "My Long Anime Title Part Two [1080p]"
    variants = build_tmdb_query_variants(name, max_variants=6)
    assert variants == [
        "My Long Anime Title Part Two [1080p]",
        "My Long Anime Title Part Two",
        "My Long Anime Title",
        "My Long",
    ]


def test_build_tmdb_query_variants_requires_positive_max() -> None:
    with pytest.raises(ValueError):
        build_tmdb_query_variants("Name", max_variants=0)


def test_extract_tmdb_id_tag_parses_single_tag() -> None:
    assert extract_tmdb_id_tag("Series {tmdb-123}") == 123
    assert extract_tmdb_id_tag("Series {TMDB-456}") == 456


def test_extract_tmdb_id_tag_rejects_invalid_or_multiple() -> None:
    with pytest.raises(ValueError):
        extract_tmdb_id_tag("Series {tmdb-abc}")
    with pytest.raises(ValueError):
        extract_tmdb_id_tag("Series {tmdb-1} {tmdb-2}")
