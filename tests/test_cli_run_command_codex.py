from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.cli import _default_plan_paths, main
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
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.details_calls: list[tuple[int, str]] = []
        self.season_calls: list[tuple[int, int, str]] = []

    def search_tv(self, query: str, *, language: str = "zh-CN", page: int = 1) -> list[TvSearchResult]:
        self.search_calls.append((query, language, page))
        return [
            TvSearchResult(
                id=100,
                name="测试动画",
                first_air_date="2020-01-01",
                original_name="Test Anime",
                popularity=1.0,
                vote_count=10,
            ),
            TvSearchResult(
                id=200,
                name="其他动画",
                first_air_date="2019-01-01",
                original_name="Other",
                popularity=2.0,
                vote_count=20,
            ),
        ]

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        self.details_calls.append((tv_id, language))
        return TvDetails(
            id=tv_id,
            name="测试动画",
            original_name="Test Anime",
            first_air_date="2020-01-01",
            seasons=[
                SeasonSummary(season_number=0, episode_count=1),
                SeasonSummary(season_number=1, episode_count=1),
            ],
        )

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN") -> SeasonDetails:
        self.season_calls.append((tv_id, season_number, language))
        if season_number != 0:
            raise AssertionError("FakeTMDBClient only expects S00 calls in this test")
        return SeasonDetails(
            id=999,
            season_number=0,
            episodes=[Episode(episode_number=1, name="OVA", overview="OVA 特别篇")],
        )


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_cli_run_dry_run_default_writes_plan_but_does_not_move(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    series_dir = tmp_path / "InputSeries"
    out_root = tmp_path / "Out"
    plan_file = tmp_path / "rename_plan.json"
    rollback_file = tmp_path / "rollback_plan.json"
    log_path = tmp_path / "logs"

    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    fake_tmdb = FakeTMDBClient()
    llm_id = FakeLLM(reply='{"tmdb": 100}')
    llm_map = FakeLLM(reply='{"tmdb":100,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}')

    rc = main(
        [
            "--log-path",
            str(log_path),
            "run",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_file),
            "--max-candidates",
            "2",
            "--max-output-tokens",
            "512",
        ],
        tmdb_client_factory=lambda: fake_tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0
    assert plan_file.exists()
    assert not rollback_file.exists()

    # No file moves should have happened
    assert (series_dir / "ep1.mkv").exists()
    assert (series_dir / "ep1.ass").exists()

    series_folder = out_root / "测试动画 (2020) {tmdb-100}"
    dst_video = series_folder / "S01" / "测试动画 S01E01.mkv"
    dst_sub = series_folder / "S01" / "测试动画 S01E01.chs.ass"
    assert not dst_video.exists()
    assert not dst_sub.exists()

    # Plan JSON is valid
    data = json.loads(plan_file.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["tmdb_id"] == 100
    assert len(data["moves"]) == 2

    out = capsys.readouterr().out
    assert f"wrote plan to {plan_file}" in out
    assert "moves: videos=1 subtitles=1" in out


def test_cli_run_apply_applies_moves(tmp_path: Path) -> None:
    series_dir = tmp_path / "InputSeries"
    out_root = tmp_path / "Out"
    plan_file = tmp_path / "rename_plan.json"
    log_path = tmp_path / "logs"

    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    fake_tmdb = FakeTMDBClient()
    llm_id = FakeLLM(reply='{"tmdb": 100}')
    llm_map = FakeLLM(reply='{"tmdb":100,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}')

    rc = main(
        [
            "--log-path",
            str(log_path),
            "run",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_file),
            "--apply",
        ],
        tmdb_client_factory=lambda: fake_tmdb,
        llm_for_tmdb_id_factory=lambda: llm_id,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0
    assert plan_file.exists()

    series_folder = out_root / "测试动画 (2020) {tmdb-100}"
    dst_video = series_folder / "S01" / "测试动画 S01E01.mkv"
    dst_sub = series_folder / "S01" / "测试动画 S01E01.chs.ass"
    assert dst_video.exists()
    assert dst_sub.exists()
    assert not (series_dir / "ep1.mkv").exists()
    assert not (series_dir / "ep1.ass").exists()

    _, rollback_file = _default_plan_paths(log_path, series_dir)
    assert rollback_file.exists()
