from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.cli import main
from aninamer.llm_client import ChatMessage
from aninamer.tmdb_client import Episode, SeasonDetails, SeasonSummary, TvDetails, TvSearchResult


@dataclass
class FakeLLM:
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


class FakeTMDBClient:
    def search_tv(self, query: str, *, language: str = "zh-CN", page: int = 1) -> list[TvSearchResult]:
        # Always return 1 candidate so TMDB-id LLM is skipped.
        return [
            TvSearchResult(
                id=100,
                name="测试动画",
                first_air_date="2020-01-01",
                original_name="Test Anime",
                popularity=1.0,
                vote_count=10,
            )
        ]

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return TvDetails(
            id=tv_id,
            name="测试动画",
            original_name="Test Anime",
            first_air_date="2020-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN") -> SeasonDetails:
        return SeasonDetails(id=None, season_number=season_number, episodes=[Episode(episode_number=1, name="OVA", overview="OVA")] if season_number == 0 else [])


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_plan_default_writes_plan_under_log_path(tmp_path: Path) -> None:
    series_dir = tmp_path / "InputSeries"
    out_root = tmp_path / "OutMount"
    log_path = tmp_path / "local_logs"

    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDBClient()

    # Mapping LLM: map video id 1 -> S01E01 and subtitle id 2
    llm_map = FakeLLM(reply='{"tmdb":100,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}')

    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--tmdb",
            "100",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0

    # Plan should be in log_path/plans/*.rename_plan.json
    plans_dir = log_path / "plans"
    plan_files = list(plans_dir.glob("*.rename_plan.json"))
    assert len(plan_files) == 1

    plan_data = json.loads(plan_files[0].read_text(encoding="utf-8"))
    assert plan_data["version"] == 1
    assert plan_data["tmdb_id"] == 100

    # Log file should be under log_path/aninamer.log
    assert (log_path / "aninamer.log").exists()


def test_apply_default_writes_rollback_under_log_path(tmp_path: Path) -> None:
    series_dir = tmp_path / "InputSeries"
    out_root = tmp_path / "OutMount"
    log_path = tmp_path / "local_logs"
    plan_file = tmp_path / "plan_in_tmp.rename_plan.json"  # could be anywhere (read-only)

    _write(series_dir / "ep1.mkv", b"video")

    # Write a minimal valid plan JSON by invoking 'plan' once (easier than handcrafting)
    tmdb = FakeTMDBClient()
    llm_map = FakeLLM(reply='{"tmdb":100,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}')
    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_file),
            "--tmdb",
            "100",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0
    assert plan_file.exists()

    # Apply using CLI; rollback default should go into log_path/plans/
    rc2 = main(
        [
            "--log-path",
            str(log_path),
            "apply",
            str(plan_file),
        ]
    )
    assert rc2 == 0

    rb_files = list((log_path / "plans").glob("*.rollback_plan.json"))
    assert len(rb_files) == 1
    rb = json.loads(rb_files[0].read_text(encoding="utf-8"))
    assert rb["version"] == 1
    assert rb["tmdb_id"] == 100


def test_monitor_default_state_file_under_log_path(tmp_path: Path) -> None:
    in_root = tmp_path / "in_mount"
    out_root = tmp_path / "out_mount"
    log_path = tmp_path / "local_logs"

    series_dir = in_root / "ShowA [1080p]"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDBClient()
    llm_map = FakeLLM(reply='{"tmdb":100,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}')

    # monitor --apply --once => should write state file under log_path by default
    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            str(in_root),
            "--out",
            str(out_root),
            "--apply",
            "--once",
            "--tmdb",
            "100",
            "--settle-seconds",
            "0",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0

    state_file = log_path / "monitor_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 3
    assert str(series_dir.resolve()) in set(data["processed"])
