from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from aninamer.cli import main
from aninamer.llm_client import ChatMessage
from aninamer.tmdb_client import SeasonSummary, TvDetails


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
        user = next((m.content for m in messages if m.role == "user"), "")
        tmdb_id = 0
        for line in user.splitlines():
            if line.strip().startswith("tmdb_id:"):
                tmdb_id = int(line.split(":", 1)[1].strip())
                break
        return f'{{"tmdb":{tmdb_id},"eps":[{{"v":1,"s":1,"e1":1,"e2":1,"u":[2]}}]}}'


class FakeTMDB:
    def search_tv(self, query: str, *, language: str = "zh-CN", page: int = 1):
        raise AssertionError("tmdb search should not be used when --tmdb is set")

    def search_tv_anime(
        self, query: str, *, language: str = "zh-CN", max_pages: int = 1
    ):
        return self.search_tv(query, language=language, page=1)

    def get_tv_details(self, tv_id: int, *, language: str = "zh-CN") -> TvDetails:
        return TvDetails(
            id=tv_id,
            name="测试动画",
            original_name=None,
            first_air_date="2022-01-01",
            seasons=[SeasonSummary(season_number=1, episode_count=1)],
        )

    def get_season(self, tv_id: int, season_number: int, *, language: str = "zh-CN"):
        raise AssertionError("unexpected specials lookup")

    def resolve_series_title(
        self, tv_id: int, *, country_codes: tuple[str, ...] = ()
    ) -> tuple[str, TvDetails]:
        details = self.get_tv_details(tv_id)
        return details.name, details


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_cli_monitor_multi_watch_pairs_apply_and_archive(tmp_path: Path) -> None:
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    log_path = tmp_path / "logs"

    series_a = src_a / "ShowA"
    series_b = src_b / "ShowB"
    _write(series_a / "ep1.mkv", b"video_a")
    _write(series_a / "ep1.ass", "国国国".encode("utf-8"))
    _write(series_b / "ep1.mkv", b"video_b")
    _write(series_b / "ep1.ass", "国国国".encode("utf-8"))

    tmdb = FakeTMDB()
    llm_map = FakeLLMMapping()

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(src_a),
            str(out_a),
            "--watch",
            str(src_b),
            str(out_b),
            "--apply",
            "--once",
            "--tmdb",
            "123",
            "--settle-seconds",
            "0",
            "--include-existing",
        ],
        tmdb_client_factory=lambda: tmdb,
        llm_for_mapping_factory=lambda: llm_map,
    )

    assert rc == 0
    assert llm_map.calls == 2

    dst_rel_video = Path("测试动画 (2022) {tmdb-123}") / "S01" / "测试动画 S01E01.mkv"
    dst_rel_sub = Path("测试动画 (2022) {tmdb-123}") / "S01" / "测试动画 S01E01.chs.ass"
    assert (out_a / dst_rel_video).exists()
    assert (out_a / dst_rel_sub).exists()
    assert (out_b / dst_rel_video).exists()
    assert (out_b / dst_rel_sub).exists()

    assert not (series_a / "ep1.mkv").exists()
    assert not (series_b / "ep1.mkv").exists()

    plans_dir = log_path / "plans"
    assert len(list(plans_dir.glob("*.rename_plan.json"))) == 2
    assert len(list(plans_dir.glob("*.rollback_plan.json"))) == 2

    state_file = log_path / "monitor_state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert data["pending"] == []
    assert data["planned"] == []
    assert "failed" not in data


def test_cli_monitor_rejects_same_input_root_with_different_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    log_path = tmp_path / "logs"

    rc = main(
        [
            "--log-path",
            str(log_path),
            "monitor",
            "--watch",
            str(src),
            str(out_a),
            "--watch",
            str(src),
            str(out_b),
            "--once",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "configured with multiple output roots" in err
