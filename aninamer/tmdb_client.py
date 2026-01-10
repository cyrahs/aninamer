from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Callable
from urllib import error, parse, request

logger = logging.getLogger(__name__)


class TMDBError(Exception):
    pass


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]


Transport = Callable[[str, dict[str, str], float], HttpResponse]


def default_transport(
    url: str, headers: dict[str, str], timeout: float
) -> HttpResponse:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            body = resp.read()
            resp_headers = {key: value for key, value in resp.headers.items()}
            return HttpResponse(status=status, body=body, headers=resp_headers)
    except error.HTTPError as exc:
        body = exc.read()
        resp_headers = (
            {key: value for key, value in exc.headers.items()} if exc.headers else {}
        )
        return HttpResponse(status=exc.code, body=body, headers=resp_headers)
    except Exception as exc:
        raise TMDBError(f"network error for {url}: {exc}") from exc


@dataclass(frozen=True)
class TvSearchResult:
    id: int
    name: str
    first_air_date: str | None
    original_name: str | None
    popularity: float | None
    vote_count: int | None
    genre_ids: tuple[int, ...] | None = None
    origin_country: tuple[str, ...] | None = None

    @property
    def year(self) -> int | None:
        return _parse_year(self.first_air_date)


@dataclass(frozen=True)
class SeasonSummary:
    season_number: int
    episode_count: int


@dataclass(frozen=True)
class TvDetails:
    id: int
    name: str
    original_name: str | None
    first_air_date: str | None
    seasons: list[SeasonSummary]

    @property
    def year(self) -> int | None:
        return _parse_year(self.first_air_date)


@dataclass(frozen=True)
class Episode:
    episode_number: int
    name: str | None
    overview: str | None


@dataclass(frozen=True)
class SeasonDetails:
    id: int | None
    season_number: int
    episodes: list[Episode]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)


def _parse_year(value: str | None) -> int | None:
    if not value:
        return None
    year_text = value.split("-", 1)[0]
    if len(year_text) != 4:
        return None
    try:
        return int(year_text)
    except ValueError:
        return None


# Country code fallback order for Chinese titles (used with TMDB translations API)
# These are ISO 3166-1 country codes
CHINESE_COUNTRY_FALLBACK_ORDER: tuple[str, ...] = (
    "CN",  # Simplified Chinese - China
    "SG",  # Simplified Chinese - Singapore
    "HK",  # Traditional Chinese - Hong Kong
    "TW",  # Traditional Chinese - Taiwan
)


@dataclass(frozen=True)
class TvTranslation:
    """A single translation entry from TMDB."""

    iso_3166_1: str  # Country code (e.g., "CN", "TW")
    iso_639_1: str  # Language code (e.g., "zh")
    name: str | None  # Translated series name
    overview: str | None  # Translated overview


def _require_key(data: dict[str, object], key: str, url: str) -> object:
    if key not in data:
        raise TMDBError(f"missing '{key}' in response from {url}")
    return data[key]


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


class TMDBClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.themoviedb.org/3",
        timeout: float = 30.0,
        transport: Transport | None = None,
        user_agent: str = "aninamer/0.1",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport or default_transport
        self._headers = {"Accept": "application/json", "User-Agent": user_agent}
        self._last_url = ""

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        data = self._get_json(
            "/search/tv",
            {
                "query": query,
                "language": language,
                "page": page,
                "include_adult": "true",
            },
        )
        return self._parse_tv_search_results(data)

    def search_tv_anime(
        self,
        query: str,
        *,
        language: str = "zh-CN",
        max_pages: int = 5,
    ) -> list[TvSearchResult]:
        """
        Search for anime TV shows. Filters by Animation genre (16).
        Falls back to all results if no animation matches found.
        """
        all_results: list[TvSearchResult] = []
        for page in range(1, max_pages + 1):
            data = self._get_json(
                "/search/tv",
                {
                    "query": query,
                    "language": language,
                    "page": page,
                    "include_adult": "true",
                },
            )
            page_results = self._parse_tv_search_results(data)
            all_results.extend(page_results)
            if len(page_results) < 20:  # No more pages
                break

        # Filter for animation: Animation genre (16)
        anime_results = [r for r in all_results if r.genre_ids and 16 in r.genre_ids]

        if anime_results:
            logger.info(
                "tmdb: search_tv_anime query=%s anime_count=%s total_count=%s",
                query,
                len(anime_results),
                len(all_results),
            )
            return anime_results

        # Fallback: return all results if no animation found
        logger.info(
            "tmdb: search_tv_anime query=%s no_anime_found, returning_all=%s",
            query,
            len(all_results),
        )
        return all_results

    def _parse_tv_search_results(self, data: dict[str, object]) -> list[TvSearchResult]:
        if "results" not in data:
            raise TMDBError(f"missing 'results' in response from {self._last_url}")
        results_raw = data["results"]
        if not isinstance(results_raw, list):
            raise TMDBError(f"invalid 'results' in response from {self._last_url}")

        results: list[TvSearchResult] = []
        for item in results_raw:
            if not isinstance(item, dict):
                raise TMDBError(
                    f"invalid result entry in response from {self._last_url}"
                )
            raw_id = _require_key(item, "id", self._last_url)
            raw_name = _require_key(item, "name", self._last_url)
            if not isinstance(raw_id, int) or not isinstance(raw_name, str):
                raise TMDBError(
                    f"invalid result fields in response from {self._last_url}"
                )
            # Parse genre_ids as tuple of ints
            raw_genre_ids = item.get("genre_ids")
            genre_ids: tuple[int, ...] | None = None
            if isinstance(raw_genre_ids, list):
                genre_ids = tuple(g for g in raw_genre_ids if isinstance(g, int))

            # Parse origin_country as tuple of strings
            raw_origin_country = item.get("origin_country")
            origin_country: tuple[str, ...] | None = None
            if isinstance(raw_origin_country, list):
                origin_country = tuple(
                    c for c in raw_origin_country if isinstance(c, str)
                )

            results.append(
                TvSearchResult(
                    id=raw_id,
                    name=raw_name,
                    first_air_date=_optional_str(item.get("first_air_date")),
                    original_name=_optional_str(item.get("original_name")),
                    popularity=_optional_float(item.get("popularity")),
                    vote_count=_optional_int(item.get("vote_count")),
                    genre_ids=genre_ids,
                    origin_country=origin_country,
                )
            )
        logger.info(
            "tmdb: search_tv parsed results_count=%s",
            len(results),
        )
        return results

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        data = self._get_json(
            f"/tv/{tv_id}",
            {"language": language},
        )
        raw_id = _require_key(data, "id", self._last_url)
        raw_name = _require_key(data, "name", self._last_url)
        raw_seasons = _require_key(data, "seasons", self._last_url)
        if not isinstance(raw_id, int) or not isinstance(raw_name, str):
            raise TMDBError(f"invalid tv details in response from {self._last_url}")
        if not isinstance(raw_seasons, list):
            raise TMDBError(f"invalid 'seasons' in response from {self._last_url}")

        seasons: list[SeasonSummary] = []
        for item in raw_seasons:
            if not isinstance(item, dict):
                raise TMDBError(f"invalid season in response from {self._last_url}")
            season_number = _require_key(item, "season_number", self._last_url)
            episode_count = _require_key(item, "episode_count", self._last_url)
            if not isinstance(season_number, int) or not isinstance(episode_count, int):
                raise TMDBError(
                    f"invalid season fields in response from {self._last_url}"
                )
            seasons.append(
                SeasonSummary(season_number=season_number, episode_count=episode_count)
            )
        seasons.sort(key=lambda season: season.season_number)
        logger.info(
            "tmdb: get_tv_details done tv_id=%s language=%s seasons_count=%s",
            tv_id,
            language,
            len(seasons),
        )
        return TvDetails(
            id=raw_id,
            name=raw_name,
            original_name=_optional_str(data.get("original_name")),
            first_air_date=_optional_str(data.get("first_air_date")),
            seasons=seasons,
        )

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        data = self._get_json(
            f"/tv/{tv_id}/season/{season_number}",
            {"language": language},
        )
        episodes_raw = _require_key(data, "episodes", self._last_url)
        if not isinstance(episodes_raw, list):
            raise TMDBError(f"invalid 'episodes' in response from {self._last_url}")

        parsed_season_number = data.get("season_number")
        if not isinstance(parsed_season_number, int):
            parsed_season_number = season_number

        parsed_id = data.get("id") if isinstance(data.get("id"), int) else None

        episodes: list[Episode] = []
        for item in episodes_raw:
            if not isinstance(item, dict):
                raise TMDBError(f"invalid episode in response from {self._last_url}")
            episode_number = _require_key(item, "episode_number", self._last_url)
            if not isinstance(episode_number, int):
                raise TMDBError(
                    f"invalid episode fields in response from {self._last_url}"
                )
            episodes.append(
                Episode(
                    episode_number=episode_number,
                    name=_optional_str(item.get("name")),
                    overview=_optional_str(item.get("overview")),
                )
            )
        episodes.sort(key=lambda episode: episode.episode_number)
        logger.info(
            "tmdb: get_season done tv_id=%s season_number=%s language=%s episodes_count=%s",
            tv_id,
            season_number,
            language,
            len(episodes),
        )
        return SeasonDetails(
            id=parsed_id,
            season_number=parsed_season_number,
            episodes=episodes,
        )

    def get_tv_translations(self, tv_id: int) -> list[TvTranslation]:
        """
        Fetch all available translations for a TV show.

        Uses TMDB's /tv/{tv_id}/translations endpoint.

        Returns:
            List of TvTranslation objects containing translated names/overviews.
        """
        data = self._get_json(f"/tv/{tv_id}/translations", {})
        translations_raw = data.get("translations")
        if not isinstance(translations_raw, list):
            logger.warning(
                "tmdb: get_tv_translations tv_id=%s missing translations list",
                tv_id,
            )
            return []

        translations: list[TvTranslation] = []
        for item in translations_raw:
            if not isinstance(item, dict):
                continue
            iso_3166_1 = item.get("iso_3166_1")
            iso_639_1 = item.get("iso_639_1")
            if not isinstance(iso_3166_1, str) or not isinstance(iso_639_1, str):
                continue

            # Extract name from nested "data" object
            data_obj = item.get("data")
            name: str | None = None
            overview: str | None = None
            if isinstance(data_obj, dict):
                name = _optional_str(data_obj.get("name"))
                overview = _optional_str(data_obj.get("overview"))

            translations.append(
                TvTranslation(
                    iso_3166_1=iso_3166_1,
                    iso_639_1=iso_639_1,
                    name=name,
                    overview=overview,
                )
            )

        logger.info(
            "tmdb: get_tv_translations tv_id=%s translations_count=%s",
            tv_id,
            len(translations),
        )
        return translations

    def resolve_series_title(
        self,
        tv_id: int,
        *,
        country_codes: tuple[str, ...] = CHINESE_COUNTRY_FALLBACK_ORDER,
    ) -> tuple[str, TvDetails]:
        """
        Resolve the best available series title using TMDB translations API.

        Fetches translations and returns the first non-empty title matching
        one of the country codes in order. Falls back to original_name if
        no translation is found.

        Args:
            tv_id: The TMDB TV show ID.
            country_codes: Tuple of ISO 3166-1 country codes to try in order.
                          Defaults to ("CN", "SG", "HK", "TW").

        Returns:
            A tuple of (resolved_title, details).
            The details are fetched with language="zh-CN" for consistency.
        """
        # Fetch details in zh-CN for consistent metadata (seasons, year, etc.)
        details = self.get_tv_details(tv_id, language="zh-CN")

        # Fetch all translations
        translations = self.get_tv_translations(tv_id)

        # Build a map of country code -> translation name
        translation_by_country: dict[str, str] = {}
        for trans in translations:
            name = (trans.name or "").strip()
            if name and trans.iso_3166_1 not in translation_by_country:
                translation_by_country[trans.iso_3166_1] = name

        # Try each country code in order
        for country in country_codes:
            if country in translation_by_country:
                title = translation_by_country[country]
                logger.info(
                    "tmdb: resolve_series_title tv_id=%s found country=%s title=%s",
                    tv_id,
                    country,
                    title,
                )
                return title, details

        # Fall back to original_name
        original = (details.original_name or "").strip()
        if original:
            logger.info(
                "tmdb: resolve_series_title tv_id=%s fallback_to_original title=%s",
                tv_id,
                original,
            )
            return original, details

        # Last resort: use the details name
        fallback = (details.name or "").strip()
        if not fallback:
            fallback = "Unknown"
        logger.warning(
            "tmdb: resolve_series_title tv_id=%s no_translation_found fallback=%s",
            tv_id,
            fallback,
        )
        return fallback, details

    def _get_json(self, path: str, params: dict[str, object]) -> dict[str, object]:
        sanitized_params = {
            key: value for key, value in params.items() if key != "api_key"
        }
        logger.debug("tmdb: request path=%s params=%s", path, sanitized_params)
        url = self._build_url(path, {**params, "api_key": self._api_key})
        self._last_url = url
        response = self._transport(url, dict(self._headers), self._timeout)
        if response.status < 200 or response.status >= 300:
            logger.warning(
                "tmdb: request failed path=%s status=%s", path, response.status
            )
            raise TMDBError(f"tmdb request failed ({response.status}) for {url}")
        try:
            data = json.loads(response.body)
        except Exception as exc:
            logger.warning("tmdb: invalid json path=%s", path)
            raise TMDBError(f"invalid json from {url}") from exc
        if not isinstance(data, dict):
            raise TMDBError(f"unexpected json from {url}")
        return data

    def _build_url(self, path: str, params: dict[str, object]) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        base_url = f"{self._base_url}{normalized_path}"
        query = parse.urlencode(params)
        return f"{base_url}?{query}" if query else base_url
