from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aninamer.apply import apply_rename_plan
from aninamer.cli import main
from aninamer.llm_client import ChatMessage
from aninamer.plan import PlannedMove, RenamePlan
from aninamer.tmdb_client import SeasonSummary, TvDetails, TvSearchResult


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# -------------------------
# Apply: default two_stage=False
# -------------------------


def test_apply_default_single_stage_no_temp_dir(tmp_path: Path) -> None:
    series_dir = tmp_path / "series"
    out_root = tmp_path / "out"

    src = series_dir / "ep1.mkv"
    dst = out_root / "Show (2020) {tmdb-1}" / "S01" / "Show S01E01.mkv"
    _write(src, b"video")

    plan = RenamePlan(
        tmdb_id=1,
        series_name_zh_cn="Show",
        year=2020,
        series_dir=series_dir,
        output_root=out_root,
        moves=(PlannedMove(src=src, dst=dst, kind="video", src_id=1),),
    )

    res = apply_rename_plan(plan, dry_run=False)
    assert res.temp_dir is None
    assert dst.exists()
    assert not src.exists()

    # Ensure no staging dir created
    if out_root.exists():
        assert not any(p.name.startswith(".aninamer_tmp_") for p in out_root.iterdir())


# -------------------------
# Monitor CLI
# -------------------------


@dataclass
class FakeLLMMapping:
    calls: int = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        # Extract expected tmdb_id from prompt (user message contains `tmdb_id: <int>`)
        user = next((m.content for m in messages if m.role == "user"), "")
        m = re.search(r"tmdb_id:\s*(\d+)", user)
        tmdb_id = int(m.group(1)) if m else 0
        # Each series dir in this test has 1 video id=1 and 1 subtitle id=2
        return f'{{"tmdb":{tmdb_id},"eps":[{{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}}]}}'


@dataclass
class FakeLLMTmdbId:
    reply: str
    calls: int = 0

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        return self.reply


class FakeTMDBClientMonitor:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str]] = []
        self.details_calls: list[int] = []

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        self.search_calls.append((query, language))

        # Distinguish shows by query contents
        if "ShowA" in query:
            return [
                TvSearchResult(
                    id=100,
                    name="测试动画A",
                    first_air_date="2020-01-01",
                    original_name="A",
                    popularity=1.0,
                    vote_count=1,
                )
            ]
        if "ShowB" in query:
            return [
                TvSearchResult(
                    id=200,
                    name="测试动画B",
                    first_air_date="2021-01-01",
                    original_name="B",
                    popularity=1.0,
                    vote_count=1,
                )
            ]
        return []

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ) -> list[TvSearchResult]:
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        self.details_calls.append(tv_id)
        if tv_id == 100:
            return TvDetails(
                id=100,
                name="测试动画A",
                original_name="A",
                first_air_date="2020-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=1)],
            )
        if tv_id == 200:
            return TvDetails(
                id=200,
                name="测试动画B",
                original_name="B",
                first_air_date="2021-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=1)],
            )
        raise AssertionError("unexpected tv_id")

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN"):
        # no specials for these tests
        from aninamer.tmdb_client import SeasonDetails

        return SeasonDetails(id=None, season_number=season_number, episodes=[])

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = self.get_tv_details(tv_id)
        return details.name, details


def test_cli_monitor_once_apply_processes_each_subdir_and_writes_state(
    tmp_path: Path,
) -> None:
    in_root = tmp_path / "in"
    out_root = tmp_path / "out"
    state_file = tmp_path / "state.json"
    log_path = tmp_path / "logs"

    # Two incoming series dirs
    show_a = in_root / "ShowA [1080p]"
    show_b = in_root / "ShowB [1080p]"

    _write(show_a / "ep1.mkv", b"videoA")
    _write(show_a / "ep1.ass", "国国国".encode("utf-8"))
    _write(show_b / "ep1.mkv", b"videoB")
    _write(show_b / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDBClientMonitor()
    llm_map = FakeLLMMapping()

    # TMDB-id selection LLM should not be called because FakeTMDB returns a single candidate
    llm_id = FakeLLMTmdbId(reply='{"tmdb": 999}')

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(in_root),
            str(out_root),
            "--apply",
            "--once",
            "--state-file",
            str(state_file),
            "--settle-seconds",
            "0",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0

    # Files should be moved to two different destinations (different tmdb ids / names)
    dst_a = out_root / "测试动画A (2020) {tmdb-100}" / "S01" / "测试动画A S01E01.mkv"
    dst_b = out_root / "测试动画B (2021) {tmdb-200}" / "S01" / "测试动画B S01E01.mkv"
    assert dst_a.exists()
    assert dst_b.exists()

    # Sources removed
    assert not (show_a / "ep1.mkv").exists()
    assert not (show_b / "ep1.mkv").exists()
    assert not show_a.exists()
    assert not show_b.exists()

    # State file written and includes both series dirs
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert data["pending"] == []
    assert data["planned"] == []
    assert "failed" not in data

    # Mapping LLM called twice (once per show)
    assert llm_map.calls == 2
    # TMDB-id selection LLM not called (single candidate shortcut)
    assert llm_id.calls == 0

    # Second monitor run should have no additional work.
    rc2 = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(in_root),
            str(out_root),
            "--apply",
            "--once",
            "--state-file",
            str(state_file),
            "--settle-seconds",
            "0",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc2 == 0
    assert llm_map.calls == 2
