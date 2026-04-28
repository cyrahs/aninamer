from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from aninamer.config import (
    ApiConfig,
    AppConfig,
    DatabaseConfig,
    OpenAISettings,
    TmdbConfig,
    WatchRootConfig,
    WorkerConfig,
)
from aninamer.monitoring import SeriesDiscoveryResult
from aninamer.worker import AninamerWorker


@dataclass
class FakeStore:
    last_scan_at: str | None = None

    def snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(last_scan_at=self.last_scan_at)


def _config(tmp_path: Path, *, root_count: int = 1) -> AppConfig:
    return AppConfig(
        log_path=tmp_path / "logs",
        database=DatabaseConfig(postgres_dsn="postgresql://unused"),
        tmdb=TmdbConfig(api_key="tmdb-key"),
        openai=OpenAISettings(api_key="openai-key", model="gpt-test"),
        notifications=None,
        notifications_warning=None,
        api=ApiConfig(token="secret"),
        worker=WorkerConfig(
            settle_seconds=0,
            scan_interval_seconds=1,
            health_stale_after_seconds=300,
        ),
        watch_roots=tuple(
            WatchRootConfig(
                key=f"root-{index}",
                input_root=tmp_path / f"input-{index}",
                output_root=tmp_path / f"output-{index}",
            )
            for index in range(root_count)
        ),
    )


def test_worker_health_reports_unhealthy_when_all_watch_roots_are_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path, root_count=2)
    worker = AninamerWorker(  # type: ignore[arg-type]
        config,
        FakeStore(last_scan_at="2026-04-13T07:40:01+00:00"),
    )
    worker._mark_worker_started()

    def unavailable(_input_root: Path) -> SeriesDiscoveryResult:
        return SeriesDiscoveryResult(
            series_dirs=[],
            unavailable=True,
            error_message="OSError: [Errno 107] Transport endpoint is not connected",
        )

    monkeypatch.setattr("aninamer.worker.discover_series_dirs_status", unavailable)

    worker._discover_new_jobs()

    health = worker.health_status()
    assert health.healthy is False
    assert health.reason == "all watch roots are unavailable"
    assert health.unavailable_watch_root_keys == ("root-0", "root-1")


def test_worker_health_stays_healthy_when_at_least_one_watch_root_is_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config(tmp_path, root_count=2)
    worker = AninamerWorker(  # type: ignore[arg-type]
        config,
        FakeStore(last_scan_at="2026-04-13T07:40:01+00:00"),
    )
    worker._mark_worker_started()

    def mixed(input_root: Path) -> SeriesDiscoveryResult:
        if input_root.name == "input-0":
            return SeriesDiscoveryResult(series_dirs=[], unavailable=True)
        return SeriesDiscoveryResult(series_dirs=[])

    monkeypatch.setattr("aninamer.worker.discover_series_dirs_status", mixed)

    worker._discover_new_jobs()

    health = worker.health_status()
    assert health.healthy is True
    assert health.unavailable_watch_root_keys == ("root-0",)
