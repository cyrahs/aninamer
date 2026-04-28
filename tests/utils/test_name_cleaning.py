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


def test_clean_tmdb_query_keeps_trailing_the_animation_suffix() -> None:
    assert (
        clean_tmdb_query("ながちち永井さん THE ANIMATION")
        == "ながちち永井さん THE ANIMATION"
    )
    assert (
        clean_tmdb_query("Nagachichi Nagai-san The Animation")
        == "Nagachichi Nagai-san The Animation"
    )


def test_build_tmdb_query_variants_shortens_cleaned_name() -> None:
    name = "My Long Anime Title Part Two [1080p]"
    variants = build_tmdb_query_variants(name, max_variants=6)
    assert variants == [
        "My Long Anime Title Part Two [1080p]",
        "My Long Anime Title Part Two",
        "My Long Anime Title",
        "My Long",
    ]


def test_build_tmdb_query_variants_includes_title_without_the_animation_suffix() -> None:
    variants = build_tmdb_query_variants("ながちち永井さん THE ANIMATION")

    assert variants[:2] == ["ながちち永井さん THE ANIMATION", "ながちち永井さん"]


def test_build_tmdb_query_variants_strips_the_animation_after_cleaned_query() -> None:
    variants = build_tmdb_query_variants("[Group] Foo THE ANIMATION [1080p]")

    assert variants[:3] == [
        "[Group] Foo THE ANIMATION [1080p]",
        "Foo THE ANIMATION",
        "Foo",
    ]


def test_build_tmdb_query_variants_strips_the_animation_after_unbracketed_release_tag() -> None:
    variants = build_tmdb_query_variants("Foo THE ANIMATION 1080p")

    assert variants[:3] == [
        "Foo THE ANIMATION 1080p",
        "Foo THE ANIMATION",
        "Foo",
    ]


def test_build_tmdb_query_variants_includes_traditional_chinese_fallback() -> None:
    variants = build_tmdb_query_variants("向日葵在夜晚绽放")

    assert variants == ["向日葵在夜晚绽放", "向日葵在夜晚綻放"]


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
