from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from aninamer.cli import main
from aninamer.llm_client import ChatMessage
from aninamer.tmdb_client import SeasonSummary, TvDetails, TvSearchResult


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


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
        # Pull tmdb_id from user prompt line like: "tmdb_id: <id>"
        user = next((m.content for m in messages if m.role == "user"), "")
        tmdb_id = 0
        for line in user.splitlines():
            if line.strip().startswith("tmdb_id:"):
                tmdb_id = int(line.split(":", 1)[1].strip())
                break
        # In these tests scanner assigns: video id=1, subtitle id=2
        return f'{{"tmdb":{tmdb_id},"eps":[{{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}}]}}'


class FakeTMDBClientMonitor:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str]] = []
        self.details_calls: list[int] = []

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        self.search_calls.append((query, language))

        # Return different IDs based on directory name tokens to avoid output collisions.
        if "ShowNew" in query:
            return [
                TvSearchResult(
                    id=101,
                    name="测试动画New",
                    first_air_date="2020-01-01",
                    original_name="New",
                    popularity=1.0,
                    vote_count=1,
                )
            ]
        if "ShowPending" in query:
            return [
                TvSearchResult(
                    id=102,
                    name="测试动画Pending",
                    first_air_date="2021-01-01",
                    original_name="Pending",
                    popularity=1.0,
                    vote_count=1,
                )
            ]
        # default
        return [
            TvSearchResult(
                id=100,
                name="测试动画Old",
                first_air_date="2019-01-01",
                original_name="Old",
                popularity=1.0,
                vote_count=1,
            )
        ]

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ) -> list[TvSearchResult]:
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        self.details_calls.append(tv_id)
        if tv_id == 101:
            return TvDetails(
                id=101,
                name="测试动画New",
                original_name="New",
                first_air_date="2020-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=1)],
            )
        if tv_id == 102:
            return TvDetails(
                id=102,
                name="测试动画Pending",
                original_name="Pending",
                first_air_date="2021-01-01",
                seasons=[SeasonSummary(season_number=1, episode_count=1)],
            )
        return TvDetails(
            id=100,
            name="测试动画Old",
            original_name="Old",
            first_air_date="2019-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN"):
        # No specials in these tests
        from aninamer.tmdb_client import SeasonDetails

        return SeasonDetails(id=None, season_number=season_number, episodes=[])

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = self.get_tv_details(tv_id)
        return details.name, details


def test_monitor_first_run_processes_existing_by_default(tmp_path: Path) -> None:
    in_root = tmp_path / "in_mount"
    out_root = tmp_path / "out_mount"
    log_path = tmp_path / "logs"

    show_old = in_root / "ShowOld [1080p]"
    _write(show_old / "ep1.mkv", b"video")
    _write(show_old / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDBClientMonitor()
    llm_map = FakeLLMMapping()

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
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc == 0

    # Existing dir should be processed on first run.
    dst_video = (
        out_root / "测试动画Old (2019) {tmdb-100}" / "S01" / "测试动画Old S01E01.mkv"
    )
    assert dst_video.exists()
    assert not show_old.exists()

    # State file should exist and record processed dir.
    state_file = log_path / "monitor_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 4
    assert data["failed"] == []
    assert data["pending"] == []
    assert data["planned"] == []

    assert llm_map.calls == 1


def test_monitor_second_run_processes_another_dir(tmp_path: Path) -> None:
    in_root = tmp_path / "in_mount"
    out_root = tmp_path / "out_mount"
    log_path = tmp_path / "logs"

    # First run: process existing dir.
    show_old = in_root / "ShowOld [1080p]"
    _write(show_old / "ep1.mkv", b"video_old")
    _write(show_old / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDBClientMonitor()
    llm_map = FakeLLMMapping()

    rc1 = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(in_root),
            str(out_root),
            "--apply",
            "--once",
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc1 == 0
    assert llm_map.calls == 1

    # Add another dir for the next run.
    show_new = in_root / "ShowNew [1080p]"
    _write(show_new / "ep1.mkv", b"video_new")
    _write(show_new / "ep1.ass", "国国国".encode("utf-8"))

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
            "--settle-seconds",
            "0",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc2 == 0
    assert llm_map.calls == 2

    # New dir should have been processed and moved
    dst_video = (
        out_root / "测试动画New (2020) {tmdb-101}" / "S01" / "测试动画New S01E01.mkv"
    )
    assert dst_video.exists()
    assert not (show_new / "ep1.mkv").exists()

    # State should record processed
    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["pending"] == []
    assert data["planned"] == []
    assert data["failed"] == []


def test_monitor_settle_seconds_defers_processing_until_stable(tmp_path: Path) -> None:
    in_root = tmp_path / "in_mount"
    out_root = tmp_path / "out_mount"
    log_path = tmp_path / "logs"

    # Run once with empty root to create state file.
    tmdb = FakeTMDBClientMonitor()
    llm_map = FakeLLMMapping()

    rc0 = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(in_root),
            str(out_root),
            "--apply",
            "--once",
            "--settle-seconds",
            "15",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc0 == 0
    assert llm_map.calls == 0

    # Create a dir; it will be pending but NOT settled yet.
    show_p = in_root / "ShowPending [1080p]"
    _write(show_p / "ep1.mkv", b"video_pending")
    _write(show_p / "ep1.ass", "国国国".encode("utf-8"))

    rc1 = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(in_root),
            str(out_root),
            "--apply",
            "--once",
            "--settle-seconds",
            "15",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc1 == 0

    # Not processed yet due to settle window
    assert (show_p / "ep1.mkv").exists()
    assert not any(out_root.rglob("测试动画Pending*"))

    # Should be recorded as pending (so it can be processed on later runs)
    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert str(show_p.resolve()) in set(data.get("pending", []))

    # Now age the files/dir to be older than 15 seconds
    old_time = time.time() - 30
    os.utime(show_p / "ep1.mkv", (old_time, old_time))
    os.utime(show_p / "ep1.ass", (old_time, old_time))
    os.utime(show_p, (old_time, old_time))

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
            "--settle-seconds",
            "15",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )
    assert rc2 == 0

    # Now it should be processed
    dst_video = (
        out_root
        / "测试动画Pending (2021) {tmdb-102}"
        / "S01"
        / "测试动画Pending S01E01.mkv"
    )
    assert dst_video.exists()
    assert not (show_p / "ep1.mkv").exists()
