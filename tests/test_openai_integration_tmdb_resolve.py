from __future__ import annotations

import os

import pytest

from aninamer.openai_llm_client import openai_llm_from_env
from aninamer.tmdb_client import TvSearchResult
from aninamer.tmdb_resolve import resolve_tmdb_tv_id_with_llm


def _env_present(name: str) -> bool:
    v = os.getenv(name)
    return v is not None and v.strip() != ""


@pytest.mark.integration
def test_resolve_tmdb_id_with_real_llm_smoke() -> None:
    # Require env vars to run
    if not (_env_present("OPENAI_API_KEY") and _env_present("OPENAI_MODEL")):
        pytest.skip("OPENAI_API_KEY and OPENAI_MODEL not set")

    # Optional base URL / reasoning effort
    llm = openai_llm_from_env()

    # Make it deterministic: dirname exactly matches one candidate name.
    dirname = "完全匹配作品名"
    candidates = [
        TvSearchResult(
            id=111,
            name="完全匹配作品名",
            first_air_date="2020-01-01",
            original_name="Exact Match",
            popularity=1.0,
            vote_count=10,
        ),
        TvSearchResult(
            id=222,
            name="其他作品名",
            first_air_date="2020-01-01",
            original_name="Other",
            popularity=1000.0,
            vote_count=999999,
        ),
    ]

    chosen = resolve_tmdb_tv_id_with_llm(dirname, candidates, llm, max_candidates=2)
    assert chosen == 111
