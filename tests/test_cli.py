from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.cli import _default_plan_paths, main
from aninamer.plan import PlannedMove, RenamePlan
from aninamer.plan_io import read_rename_plan_json, write_rename_plan_json
from aninamer.tmdb_client import SeasonDetails, SeasonSummary, TvDetails, TvSearchResult


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
    search_results: list[TvSearchResult]
    details: TvDetails
    specials_zh: SeasonDetails | None = None
    specials_en: SeasonDetails | None = None
    search_queries: list[str] = field(default_factory=list)

    def search_tv(
        self, query: str, *, language: str = "zh-CN", page: int = 1
    ) -> list[TvSearchResult]:
        self.search_queries.append(query)
        return list(self.search_results)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return self.details

    def get_season(
        self, tv_id: int, season_number: int, *, language: str = "zh-CN"
    ) -> SeasonDetails:
        if language.startswith("zh"):
            if self.specials_zh is None:
                raise AssertionError("unexpected zh specials call")
            return self.specials_zh
        if self.specials_en is None:
            raise AssertionError("unexpected en specials call")
        return self.specials_en


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_cli_plan_writes_plan_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    series_dir = tmp_path / "Series"
    _write(series_dir / "ep1.mkv", b"video")
    _write(series_dir / "ep1.ass", b"sub")
    out_root = tmp_path / "out"
    plan_path = tmp_path / "plans" / "rename_plan.json"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(search_results=[], details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--tmdb",
            "123",
            "--plan-file",
            str(plan_path),
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    plan = read_rename_plan_json(plan_path)
    assert plan.tmdb_id == 123
    assert len(plan.moves) == 2

    out = capsys.readouterr().out
    assert str(plan_path) in out
    assert "Moves:" in out


def test_cli_plan_uses_llm_for_tmdb_id(tmp_path: Path) -> None:
    series_dir = tmp_path / "Series"
    _write(series_dir / "ep1.mkv", b"video")
    out_root = tmp_path / "out"
    plan_path = tmp_path / "rename_plan.json"
    log_path = tmp_path / "logs"

    search_results = [
        TvSearchResult(
            id=111,
            name="Alpha",
            first_air_date="2019-01-01",
            original_name=None,
            popularity=None,
            vote_count=None,
        ),
        TvSearchResult(
            id=456,
            name="Beta",
            first_air_date="2020-01-01",
            original_name=None,
            popularity=None,
            vote_count=None,
        ),
    ]
    details = TvDetails(
        id=456,
        name="Beta",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(search_results=search_results, details=details)
    llm_for_id = FakeLLM(reply='{"tmdb":456}')
    mapping_llm = FakeLLM(
        reply='{"tmdb":456,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--plan-file",
            str(plan_path),
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_tmdb_id_factory=lambda: llm_for_id,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    assert llm_for_id.calls == 1
    assert tmdb.search_queries == [series_dir.name] * 3
    plan = read_rename_plan_json(plan_path)
    assert plan.tmdb_id == 456


def test_cli_apply_writes_rollback_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    plan_path = tmp_path / "rename_plan.json"
    rollback_path = tmp_path / "rollback_plan.json"
    log_path = tmp_path / "logs"
    write_rename_plan_json(plan_path, plan)

    rc = main(
        [
            "--log-path",
            str(log_path),
            "apply",
            str(plan_path),
            "--dry-run",
            "--rollback-file",
            str(rollback_path),
        ]
    )

    assert rc == 0
    rollback = read_rename_plan_json(rollback_path)
    assert rollback.tmdb_id == plan.tmdb_id
    assert rollback.moves[0].src == dst
    assert rollback.moves[0].dst == src

    out = capsys.readouterr().out
    assert str(rollback_path) in out


def test_cli_plan_defaults_to_log_path(tmp_path: Path) -> None:
    series_dir = tmp_path / "Series"
    _write(series_dir / "ep1.mkv", b"video")
    out_root = tmp_path / "out"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(search_results=[], details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--tmdb",
            "123",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc == 0
    plan_path, rollback_path = _default_plan_paths(log_path, series_dir)
    assert plan_path.exists()
    assert not rollback_path.exists()


def test_cli_plan_rejects_plan_file_under_series_dir(tmp_path: Path) -> None:
    series_dir = tmp_path / "Series"
    _write(series_dir / "ep1.mkv", b"video")
    out_root = tmp_path / "out"
    plan_path = series_dir / "rename_plan.json"
    log_path = tmp_path / "logs"

    details = TvDetails(
        id=123,
        name="Show",
        original_name=None,
        first_air_date="2020-01-01",
        seasons=[SeasonSummary(season_number=1, episode_count=1)],
    )
    tmdb = FakeTMDB(search_results=[], details=details)
    mapping_llm = FakeLLM(
        reply='{"tmdb":123,"eps":[{"v":1,"s":1,"e1":1,"e2":1,"u":[]}]}'
    )

    rc = main(
        [
            "--log-path",
            str(log_path),
            "plan",
            str(series_dir),
            "--out",
            str(out_root),
            "--tmdb",
            "123",
            "--plan-file",
            str(plan_path),
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: mapping_llm,
    )

    assert rc != 0
    assert not plan_path.exists()
