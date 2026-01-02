from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aninamer.cli import _default_plan_paths, _is_settled, main
from aninamer.tmdb_client import SeasonSummary, TvDetails, TvSearchResult


@dataclass
class FakeLLM:
    reply: str
    calls: int = 0

    def chat(
        self,
        messages: Sequence[object],
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> str:
        self.calls += 1
        return self.reply


@dataclass
class FakeTMDB:
    details: TvDetails

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        return []

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return self.details

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> object:
        raise AssertionError("unexpected specials lookup")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_cli_monitor_once_plans_without_apply(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_a = input_root / "SeriesA"
    series_b = input_root / "SeriesB"
    _write(series_a / "ep1.mkv", b"video")
    _write(series_a / "ep1.ass", "国国国".encode("utf-8"))
    _write(series_b / "ep1.mkv", b"video")
    _write(series_b / "ep1.ass", "国国国".encode("utf-8"))

    out_root = tmp_path / "Out"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            str(input_root),
            "--out",
            str(out_root),
            "--once",
            "--tmdb",
            "123",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0

    plan_a, rollback_a = _default_plan_paths(log_path, series_a)
    plan_b, rollback_b = _default_plan_paths(log_path, series_b)
    assert not plan_a.exists()
    assert not plan_b.exists()
    assert not rollback_a.exists()
    assert not rollback_b.exists()

    state_file = log_path / "monitor_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 3
    baseline = set(data["baseline"])
    assert str(series_a.resolve(strict=False)) in baseline
    assert str(series_b.resolve(strict=False)) in baseline
    assert data["pending"] == []
    assert data["planned"] == []
    assert data["processed"] == []
    assert data["failed"] == []
    assert mapping_llm.calls == 0

    assert (series_a / "ep1.mkv").exists()
    assert (series_a / "ep1.ass").exists()
    assert (series_b / "ep1.mkv").exists()
    assert (series_b / "ep1.ass").exists()


def test_is_settled_ignores_skip_dirs(tmp_path: Path) -> None:
    series_dir = tmp_path / "SeriesA"
    sample_dir = series_dir / "sample"
    _write(series_dir / "ep1.mkv", b"video")
    _write(sample_dir / "clip.mkv", b"video")

    now = time.time()
    old_time = now - 120
    os.utime(series_dir / "ep1.mkv", (old_time, old_time))
    os.utime(sample_dir / "clip.mkv", (now, now))
    os.utime(sample_dir, (now, now))
    os.utime(series_dir, (old_time, old_time))

    assert _is_settled(series_dir, 30, now=now)


def test_cli_monitor_once_pending_when_not_settled(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesA"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    out_root = tmp_path / "Out"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            str(input_root),
            "--out",
            str(out_root),
            "--once",
            "--tmdb",
            "123",
            "--settle-seconds",
            "3600",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0

    plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
    assert not plan_path.exists()
    assert not rollback_path.exists()

    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 3
    assert str(series_dir.resolve(strict=False)) in data["pending"]
    assert data["planned"] == []
    assert data["processed"] == []
    assert data["failed"] == []
    assert mapping_llm.calls == 0

    assert (series_dir / "ep1.mkv").exists()
    assert (series_dir / "ep1.ass").exists()


def test_cli_monitor_apply_marks_state_and_moves(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesA"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    out_root = tmp_path / "Out"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            str(input_root),
            "--out",
            str(out_root),
            "--once",
            "--apply",
            "--tmdb",
            "123",
            "--settle-seconds",
            "0",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0

    plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
    assert plan_path.exists()
    assert rollback_path.exists()

    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 3
    assert str(series_dir.resolve(strict=False)) in data["processed"]
    assert data["pending"] == []
    assert data["planned"] == []
    assert data["failed"] == []

    series_folder = out_root / "Show (2020) {tmdb-123}"
    dst_video = series_folder / "S01" / "Show S01E01.mkv"
    dst_sub = series_folder / "S01" / "Show S01E01.chs.ass"
    assert dst_video.exists()
    assert dst_sub.exists()
    assert not (series_dir / "ep1.mkv").exists()
    assert not (series_dir / "ep1.ass").exists()


def test_cli_monitor_marks_failed_and_skips_future_runs(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesFail"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))

    out_root = tmp_path / "Out"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(details=details)
    mapping_llm = FakeLLM(reply="not json")

    args = [
        "--log-path",
        str(log_path),
        "monitor",
        str(input_root),
        "--out",
        str(out_root),
        "--once",
        "--tmdb",
        "123",
        "--settle-seconds",
        "0",
        "--include-existing",
    ]

    rc1 = main(
        args,
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    assert rc1 == 0

    plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
    assert not plan_path.exists()
    assert not rollback_path.exists()

    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 3
    resolved = str(series_dir.resolve(strict=False))
    assert resolved in data["failed"]
    assert resolved not in data["pending"]
    assert resolved not in data["planned"]
    assert resolved not in data["processed"]
    calls_after_first = mapping_llm.calls
    assert calls_after_first > 0

    rc2 = main(
        args,
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    assert rc2 == 0
    assert mapping_llm.calls == calls_after_first
