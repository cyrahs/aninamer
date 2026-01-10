"""Tests for TMDB title language fallback logic using translations API."""

from __future__ import annotations

import json

import pytest

from aninamer.tmdb_client import (
    CHINESE_COUNTRY_FALLBACK_ORDER,
    TMDBClient,
    TvDetails,
    TvTranslation,
)


class TestChineseCountryFallbackOrder:
    """Tests for the country code fallback order constant."""

    def test_fallback_order_is_correct(self) -> None:
        assert CHINESE_COUNTRY_FALLBACK_ORDER == (
            "CN",
            "SG",
            "HK",
            "TW",
        )

    def test_cn_is_first(self) -> None:
        assert CHINESE_COUNTRY_FALLBACK_ORDER[0] == "CN"


class FakeHttpResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body
        self.headers: dict[str, str] = {}


class FakeTransport:
    """A fake transport for testing TMDBClient."""

    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        """
        Args:
            responses: A dict mapping URL substrings to response data.
        """
        self.responses = responses
        self.calls: list[str] = []

    def __call__(
        self, url: str, headers: dict[str, str], timeout: float
    ) -> FakeHttpResponse:
        self.calls.append(url)
        for key, data in self.responses.items():
            if key in url:
                return FakeHttpResponse(200, json.dumps(data).encode())
        return FakeHttpResponse(404, b'{"status_message": "not found"}')


def _make_tv_details_response(
    tv_id: int, name: str, original_name: str | None = None
) -> dict[str, object]:
    return {
        "id": tv_id,
        "name": name,
        "original_name": original_name,
        "first_air_date": "2020-01-01",
        "seasons": [{"season_number": 1, "episode_count": 12}],
    }


def _make_translations_response(
    translations: list[dict[str, object]],
) -> dict[str, object]:
    return {"translations": translations}


def _make_translation(
    country: str, lang: str, name: str | None, overview: str | None = None
) -> dict[str, object]:
    return {
        "iso_3166_1": country,
        "iso_639_1": lang,
        "data": {"name": name, "overview": overview},
    }


class TestGetTvTranslations:
    """Tests for TMDBClient.get_tv_translations method."""

    def test_parses_translations(self) -> None:
        transport = FakeTransport(
            {
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("CN", "zh", "进击的巨人"),
                        _make_translation("TW", "zh", "進擊的巨人"),
                        _make_translation("US", "en", "Attack on Titan"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        translations = client.get_tv_translations(123)

        assert len(translations) == 3
        assert translations[0] == TvTranslation(
            iso_3166_1="CN", iso_639_1="zh", name="进击的巨人", overview=None
        )
        assert translations[1] == TvTranslation(
            iso_3166_1="TW", iso_639_1="zh", name="進擊的巨人", overview=None
        )

    def test_handles_empty_translations(self) -> None:
        transport = FakeTransport(
            {
                "/tv/123/translations": {"translations": []},
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        translations = client.get_tv_translations(123)

        assert translations == []

    def test_handles_missing_translations_key(self) -> None:
        transport = FakeTransport(
            {
                "/tv/123/translations": {"id": 123},
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        translations = client.get_tv_translations(123)

        assert translations == []


class TestResolveSeriesTitle:
    """Tests for TMDBClient.resolve_series_title method."""

    def test_returns_cn_when_available(self) -> None:
        """Should return CN translation when available."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("CN", "zh", "进击的巨人"),
                        _make_translation("TW", "zh", "進擊的巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "进击的巨人"
        assert details.id == 123

    def test_fallback_to_sg(self) -> None:
        """Should fall back to SG when CN not available."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("SG", "zh", "进击的巨人"),
                        _make_translation("TW", "zh", "進擊的巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "进击的巨人"

    def test_fallback_to_hk(self) -> None:
        """Should fall back to HK when CN and SG not available."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("HK", "zh", "進擊的巨人"),
                        _make_translation("JP", "ja", "進撃の巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "進擊的巨人"

    def test_fallback_to_tw(self) -> None:
        """Should fall back to TW when earlier options not available."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("TW", "zh", "進擊的巨人"),
                        _make_translation("US", "en", "Attack on Titan"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "進擊的巨人"

    def test_fallback_to_original_name(self) -> None:
        """Should fall back to original_name when no Chinese translations."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("US", "en", "Attack on Titan"),
                        _make_translation("JP", "ja", "進撃の巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "進撃の巨人"  # Falls back to original_name

    def test_fallback_to_details_name_when_no_original(self) -> None:
        """Should fall back to details name when no original_name."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", None
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("US", "en", "Attack on Titan"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "Attack on Titan"

    def test_returns_unknown_when_all_empty(self) -> None:
        """Should return 'Unknown' when all titles are empty."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(123, "", None),
                "/tv/123/translations": _make_translations_response([]),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        assert title == "Unknown"

    def test_skips_empty_translation_names(self) -> None:
        """Should skip translations with empty names."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("CN", "zh", ""),  # Empty
                        _make_translation("CN", "zh", None),  # None
                        _make_translation("TW", "zh", "進擊的巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        # Should skip empty CN and use TW
        assert title == "進擊的巨人"

    def test_custom_country_order(self) -> None:
        """Should respect custom country code order."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("CN", "zh", "进击的巨人"),
                        _make_translation("TW", "zh", "進擊的巨人"),
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        # Ask for TW first
        title, details = client.resolve_series_title(123, country_codes=("TW", "CN"))

        assert title == "進擊的巨人"

    def test_uses_first_translation_for_country(self) -> None:
        """Should use the first translation found for a country code."""
        transport = FakeTransport(
            {
                "/tv/123?language=zh-CN": _make_tv_details_response(
                    123, "Attack on Titan", "進撃の巨人"
                ),
                "/tv/123/translations": _make_translations_response(
                    [
                        _make_translation("CN", "zh", "进击的巨人"),
                        _make_translation("CN", "zh", "巨人进击"),  # Second CN entry
                    ]
                ),
            }
        )
        client = TMDBClient(api_key="fake", transport=transport)

        title, details = client.resolve_series_title(123)

        # Should use first CN translation
        assert title == "进击的巨人"
