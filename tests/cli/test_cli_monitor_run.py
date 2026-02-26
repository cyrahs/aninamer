from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import aninamer.cli as cli_module
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

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ) -> list[TvSearchResult]:
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return self.details

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> object:
        raise AssertionError("unexpected specials lookup")

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        return self.details.name, self.details


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
            "--watch",
            str(input_root),
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
    assert data["version"] == 4
    pending = set(data["pending"])
    assert str(series_a.resolve(strict=False)) in pending
    assert str(series_b.resolve(strict=False)) in pending
    assert data["planned"] == []
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
            "--watch",
            str(input_root),
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
    assert data["version"] == 4
    assert str(series_dir.resolve(strict=False)) in data["pending"]
    assert data["planned"] == []
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
            "--watch",
            str(input_root),
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
    assert data["version"] == 4
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
    assert not series_dir.exists()


def test_cli_monitor_apply_deletes_recursive_empty_dirs(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesA"
    nested = series_dir / "season1" / "batch"
    _write(nested / "ep1.mkv", b"video")
    _write(nested / "ep1.ass", "国国国".encode("utf-8"))
    # Pre-existing empty subtree should also be removed by recursive prune.
    (series_dir / "empty_a" / "empty_b").mkdir(parents=True, exist_ok=True)

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
            "--watch",
            str(input_root),
            str(out_root),
            "--once",
            "--apply",
            "--tmdb",
            "123",
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    assert not series_dir.exists()
    assert not (input_root / "archive" / "SeriesA").exists()


def test_cli_monitor_apply_archives_non_empty_dir_and_ignores_archive(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesA"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    _write(series_dir / "note.txt", b"keep")

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

    args = [
        "--log-path",
        str(log_path),
        "monitor",
        "--watch",
        str(input_root),
        str(out_root),
        "--once",
        "--apply",
        "--tmdb",
        "123",
        "--settle-seconds",
        "0",
    ]

    rc1 = main(
        args,
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    assert rc1 == 0
    assert mapping_llm.calls == 1

    archived_dir = input_root / "archive" / "SeriesA"
    assert archived_dir.exists()
    assert (archived_dir / "note.txt").exists()
    assert not series_dir.exists()

    rc2 = main(
        args,
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    assert rc2 == 0
    assert mapping_llm.calls == 1


def test_cli_monitor_archive_name_collision_appends_suffix(tmp_path: Path) -> None:
    input_root = tmp_path / "Input"
    series_dir = input_root / "SeriesA"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", "国国国".encode("utf-8"))
    _write(series_dir / "note.txt", b"keep")
    _write(input_root / "archive" / "SeriesA" / "old.txt", b"old")

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
            "--watch",
            str(input_root),
            str(out_root),
            "--once",
            "--apply",
            "--tmdb",
            "123",
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    assert (input_root / "archive" / "SeriesA").exists()
    assert (input_root / "archive" / "SeriesA.1" / "note.txt").exists()
    assert not series_dir.exists()


def test_cli_monitor_skips_finalize_if_new_files_appear_after_apply(
    tmp_path: Path, monkeypatch
) -> None:
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

    original_apply = cli_module._do_apply_from_plan

    def _apply_and_add_new_files(
        plan,
        *,
        rollback_file: Path,
        dry_run: bool,
        two_stage: bool,
    ) -> int:
        applied_count = original_apply(
            plan,
            rollback_file=rollback_file,
            dry_run=dry_run,
            two_stage=two_stage,
        )
        _write(plan.series_dir / "late.mkv", b"late")
        _write(plan.series_dir / "late.ass", "国国国".encode("utf-8"))
        return applied_count

    monkeypatch.setattr(cli_module, "_do_apply_from_plan", _apply_and_add_new_files)

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(input_root),
            str(out_root),
            "--once",
            "--apply",
            "--tmdb",
            "123",
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    assert (series_dir / "late.mkv").exists()
    assert not (input_root / "archive" / "SeriesA").exists()

    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    resolved = str(series_dir.resolve(strict=False))
    assert resolved not in set(data["failed"])


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
        "--watch",
        str(input_root),
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
    assert data["version"] == 4
    resolved = str(series_dir.resolve(strict=False))
    assert resolved in data["failed"]
    assert resolved not in data["pending"]
    assert resolved not in data["planned"]
    calls_after_first = mapping_llm.calls
    assert calls_after_first > 0

    rc2 = main(
        args,
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )
    assert rc2 == 0
    assert mapping_llm.calls == calls_after_first
