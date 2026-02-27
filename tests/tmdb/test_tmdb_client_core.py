from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse
from dataclasses import dataclass
from typing import Any

import pytest

from aninamer.tmdb_client import (
    Episode,
    HttpResponse,
    SeasonDetails,
    TMDBClient,
    TMDBError,
    TvDetails,
    TvSearchResult,
)

class CapturingTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
        self.calls.append((url, headers, timeout))
        if not self._responses:
            raise AssertionError("no response queued")
        return self._responses.pop(0)


def test_client_requires_api_key() -> None:
    with pytest.raises(ValueError):
        TMDBClient("")


def test_search_tv_parses_results_and_query_params() -> None:
    payload = {
        "results": [
            {
                "id": 1,
                "name": "Show A",
                "first_air_date": "2020-01-02",
                "original_name": "Orig A",
                "popularity": 12.5,
                "vote_count": 100,
            },
            {
                "id": 2,
                "name": "Show B",
            },
        ]
    }
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps(payload).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    results = client.search_tv("My Show")

    assert [r.id for r in results] == [1, 2]
    assert results[0].year == 2020
    assert results[1].first_air_date is None
    assert results[1].original_name is None
    assert results[1].popularity is None
    assert results[1].vote_count is None

    url, headers, timeout = transport.calls[0]
    parsed = urlparse(url)
    assert parsed.path.endswith("/search/tv")
    params = parse_qs(parsed.query)
    assert params["api_key"] == ["key"]
    assert params["query"] == ["My Show"]
    assert params["language"] == ["zh-CN"]
    assert params["page"] == ["1"]
    assert headers["Accept"] == "application/json"
    assert headers["User-Agent"] == "aninamer/0.1"
    assert timeout == 30.0


def test_get_tv_details_parses_and_sorts_seasons() -> None:
    payload = {
        "id": 99,
        "name": "Show",
        "original_name": "Original",
        "first_air_date": "2019-03-04",
        "poster_path": "/poster.jpg",
        "seasons": [
            {"season_number": 2, "episode_count": 12},
            {"season_number": 1, "episode_count": 10},
        ],
    }
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps(payload).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    details = client.get_tv_details(99)

    assert details.id == 99
    assert details.name == "Show"
    assert details.year == 2019
    assert details.poster_path == "/poster.jpg"
    assert [s.season_number for s in details.seasons] == [1, 2]
    assert [s.episode_count for s in details.seasons] == [10, 12]

    url, _headers, _timeout = transport.calls[0]
    parsed = urlparse(url)
    assert parsed.path.endswith("/tv/99")


def test_get_season_parses_episodes_and_falls_back_to_argument() -> None:
    payload = {
        "id": 12,
        "episodes": [
            {"episode_number": 2, "name": "B", "overview": "two"},
            {"episode_number": 1, "name": "A"},
        ],
    }
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps(payload).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    season = client.get_season(10, 0)

    assert season.id == 12
    assert season.season_number == 0
    assert season.episode_count == 2
    assert [e.episode_number for e in season.episodes] == [1, 2]
    assert season.episodes[0].name == "A"
    assert season.episodes[1].overview == "two"


def test_get_season_uses_response_season_number_when_present() -> None:
    payload = {
        "id": 22,
        "season_number": 3,
        "episodes": [{"episode_number": 1, "name": "E1"}],
    }
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps(payload).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    season = client.get_season(10, 9)

    assert season.season_number == 3


def test_non_2xx_response_raises_tmdb_error() -> None:
    transport = CapturingTransport([HttpResponse(404, b"{}", {})])
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    with pytest.raises(TMDBError):
        client.search_tv("Missing")


def test_invalid_json_raises_tmdb_error() -> None:
    transport = CapturingTransport([HttpResponse(200, b"{", {})])
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    with pytest.raises(TMDBError):
        client.search_tv("Bad JSON")


def test_missing_results_raises_tmdb_error() -> None:
    transport = CapturingTransport([HttpResponse(200, b"{}", {})])
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    with pytest.raises(TMDBError):
        client.search_tv("No Results")


def test_missing_required_fields_raise_tmdb_error() -> None:
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps({"id": 1, "name": "Show"}).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    with pytest.raises(TMDBError):
        client.get_tv_details(1)


def test_missing_episodes_raise_tmdb_error() -> None:
    transport = CapturingTransport(
        [HttpResponse(200, json.dumps({"id": 1}).encode("utf-8"), {})]
    )
    client = TMDBClient("key", base_url="https://example.test/3", transport=transport)

    with pytest.raises(TMDBError):
        client.get_season(1, 1)


@dataclass
class FakeTransport:
    """A simple transport stub that returns canned responses based on URL path."""
    routes: dict[str, tuple[int, dict[str, Any]]]
    calls: list[str]

    def __call__(self, url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
        self.calls.append(url)
        path = urlparse(url).path

        for suffix, (status, payload) in self.routes.items():
            if path.endswith(suffix):
                body = json.dumps(payload).encode("utf-8")
                return HttpResponse(status=status, body=body, headers={"Content-Type": "application/json"})

        return HttpResponse(status=404, body=b"{}", headers={"Content-Type": "application/json"})


def test_search_tv_builds_url_and_parses_results() -> None:
    transport = FakeTransport(
        routes={
            "/search/tv": (
                200,
                {
                    "page": 1,
                    "results": [
                        {
                            "id": 111,
                            "name": "测试动画",
                            "first_air_date": "2020-04-01",
                            "original_name": "Test Anime",
                            "popularity": 12.3,
                            "vote_count": 100,
                        }
                    ],
                },
            )
        },
        calls=[],
    )
    client = TMDBClient(api_key="KEY", transport=transport)

    results = client.search_tv("foo bar", language="zh-CN", page=2)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, TvSearchResult)
    assert r.id == 111
    assert r.name == "测试动画"
    assert r.year == 2020

    assert len(transport.calls) == 1
    url = transport.calls[0]
    parsed = urlparse(url)
    assert parsed.path.endswith("/search/tv")

    qs = parse_qs(parsed.query)
    assert qs["api_key"] == ["KEY"]
    assert qs["language"] == ["zh-CN"]
    assert qs["query"] == ["foo bar"]
    assert qs["page"] == ["2"]


def test_get_tv_details_parses_and_sorts_seasons_2() -> None:
    transport = FakeTransport(
        routes={
            "/tv/111": (
                200,
                {
                    "id": 111,
                    "name": "动画中文名",
                    "original_name": "Anime Original",
                    "first_air_date": "2020-04-01",
                    "seasons": [
                        {"season_number": 2, "episode_count": 12},
                        {"season_number": 0, "episode_count": 3},
                        {"season_number": 1, "episode_count": 24},
                    ],
                },
            )
        },
        calls=[],
    )
    client = TMDBClient(api_key="KEY", transport=transport)

    details = client.get_tv_details(111, language="zh-CN")
    assert isinstance(details, TvDetails)
    assert details.id == 111
    assert details.name == "动画中文名"
    assert details.year == 2020

    # sorted by season_number asc
    assert [s.season_number for s in details.seasons] == [0, 1, 2]
    assert [s.episode_count for s in details.seasons] == [3, 24, 12]


def test_get_season_parses_and_sorts_episodes() -> None:
    transport = FakeTransport(
        routes={
            "/tv/111/season/0": (
                200,
                {
                    "id": 999,
                    "season_number": 0,
                    "episodes": [
                        {"episode_number": 2, "name": "Special 2", "overview": "B"},
                        {"episode_number": 1, "name": "Special 1", "overview": "A"},
                    ],
                },
            )
        },
        calls=[],
    )
    client = TMDBClient(api_key="KEY", transport=transport)

    season = client.get_season(111, 0, language="en-US")
    assert isinstance(season, SeasonDetails)
    assert season.id == 999
    assert season.season_number == 0
    assert season.episode_count == 2
    assert [e.episode_number for e in season.episodes] == [1, 2]
    assert all(isinstance(e, Episode) for e in season.episodes)

    url = transport.calls[0]
    qs = parse_qs(urlparse(url).query)
    assert qs["language"] == ["en-US"]


def test_non_2xx_raises_tmdb_error() -> None:
    transport = FakeTransport(
        routes={
            "/tv/111": (
                401,
                {"status_code": 7, "status_message": "Invalid API key"},
            )
        },
        calls=[],
    )
    client = TMDBClient(api_key="KEY", transport=transport)

    with pytest.raises(TMDBError) as exc:
        client.get_tv_details(111)

    assert "401" in str(exc.value)


def test_invalid_json_raises_tmdb_error_2() -> None:
    class BadJsonTransport:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, url: str, headers: dict[str, str], timeout: float) -> HttpResponse:
            self.calls.append(url)
            return HttpResponse(status=200, body=b"not-json", headers={"Content-Type": "application/json"})

    transport = BadJsonTransport()
    client = TMDBClient(api_key="KEY", transport=transport)

    with pytest.raises(TMDBError):
        client.search_tv("x")
