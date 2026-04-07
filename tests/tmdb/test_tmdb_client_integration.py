from __future__ import annotations

import pytest

from aninamer.config import TmdbConfig
from aninamer.pipeline import tmdb_client_from_settings


@pytest.mark.integration
def test_tmdb_real_search_details_and_season_smoke(
    integration_tmdb_settings: TmdbConfig,
) -> None:
    client = tmdb_client_from_settings(integration_tmdb_settings)

    # Use a stable, widely known query to reduce flakiness.
    results = client.search_tv("Attack on Titan", language="en-US", page=1)
    assert results, "TMDB search_tv returned no results for a common query"

    tv_id = results[0].id

    # Fetch details in English
    details_en = client.get_tv_details(tv_id, language="en-US")
    assert details_en.id == tv_id
    assert isinstance(details_en.name, str) and details_en.name.strip() != ""
    assert details_en.seasons, "Expected at least one season in tv details"

    # Ensure seasons are sorted by season_number as our client promises
    season_numbers = [s.season_number for s in details_en.seasons]
    assert season_numbers == sorted(season_numbers)

    # Choose the first normal season (season_number > 0) that has episodes
    season_num = next((s.season_number for s in details_en.seasons if s.season_number > 0 and s.episode_count > 0), None)
    if season_num is None:
        pytest.skip("No normal seasons found for the chosen TV id (unexpected but possible)")

    season = client.get_season(tv_id, season_num, language="en-US")
    assert season.season_number == season_num
    assert season.episodes, "Expected season episodes list to be non-empty"

    ep_nums = [e.episode_number for e in season.episodes]
    assert ep_nums == sorted(ep_nums), "Expected episodes to be sorted by episode_number"

    # Also fetch zh-CN details to ensure localization path works (name may or may not be Chinese depending on TMDB data)
    details_zh = client.get_tv_details(tv_id, language="zh-CN")
    assert isinstance(details_zh.name, str) and details_zh.name.strip() != ""
