from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import pytest

from aninamer.cli import _search_tmdb_candidates
from aninamer.episode_mapping import (
    EpisodeMapItem,
    EpisodeMappingResult,
    map_episodes_with_llm,
)
from aninamer.openai_llm_client import (
    openai_llm_for_tmdb_id_from_env,
    openai_llm_from_env,
)
from aninamer.plan import RenamePlan, build_rename_plan
from aninamer.plan_io import read_rename_plan_json
from aninamer.scanner import SUBTITLE_EXTS, VIDEO_EXTS, scan_series_dir
from aninamer.tmdb_client import TMDBClient, TMDBError
from aninamer.tmdb_resolve import resolve_tmdb_tv_id_with_llm


DATA_DIR = Path(__file__).parent / "data" / "real_cases"

SIMPLIFIED_SAMPLE = "\u4e3a"

_EP_PATTERN = re.compile(r"S(\d{2})E(\d{2})(?:-E(\d{2}))?")


@dataclass(frozen=True)
class TreeEntry:
    rel_path: Path
    is_dir: bool


@dataclass(frozen=True)
class RealCase:
    name: str
    tree_path: Path
    plan_path: Path


def _load_cases() -> list[RealCase]:
    tree_files = sorted(DATA_DIR.glob("*.tree"))
    json_files = sorted(DATA_DIR.glob("*.json"))

    tree_by_stem = {path.stem: path for path in tree_files}
    json_by_stem = {path.stem: path for path in json_files}

    missing_json = sorted(set(tree_by_stem) - set(json_by_stem))
    missing_tree = sorted(set(json_by_stem) - set(tree_by_stem))
    if missing_json or missing_tree:
        raise AssertionError(
            "real_cases must have paired .tree/.json files; "
            f"missing_json={missing_json or None} missing_tree={missing_tree or None}"
        )

    cases: list[RealCase] = []
    for stem in sorted(tree_by_stem):
        cases.append(
            RealCase(
                name=stem,
                tree_path=tree_by_stem[stem],
                plan_path=json_by_stem[stem],
            )
        )
    return cases


CASES = _load_cases()


def _normalize_tree_line(line: str) -> str:
    return line.replace("\u00a0", " ")


def _env_present(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip() != ""


def _parse_tree_entries(tree_path: Path) -> tuple[Path, list[TreeEntry]]:
    lines = tree_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"tree file is empty: {tree_path}")

    root = Path(lines[0].rstrip("/"))
    items: list[tuple[int, str]] = []
    for raw in lines[1:]:
        line = _normalize_tree_line(raw)
        if "── " not in line:
            continue
        prefix, name = line.split("── ", 1)
        depth = len(prefix) // 4
        items.append((depth, name))

    entries: list[TreeEntry] = []
    stack: list[str] = []
    for idx, (depth, name) in enumerate(items):
        stack = stack[:depth]
        stack.append(name)
        next_depth = items[idx + 1][0] if idx + 1 < len(items) else -1
        is_dir = next_depth > depth
        entries.append(TreeEntry(rel_path=Path(*stack), is_dir=is_dir))

    return root, entries


def _materialize_tree(root: Path, entries: list[TreeEntry]) -> None:
    for entry in entries:
        path = root / entry.rel_path
        if entry.is_dir:
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in SUBTITLE_EXTS:
            path.write_text(SIMPLIFIED_SAMPLE, encoding="utf-8")
        else:
            path.write_bytes(b"")


def _parse_episode_range(path: Path) -> tuple[int, int, int]:
    match = _EP_PATTERN.search(path.name)
    if not match:
        raise AssertionError(f"missing episode pattern in {path}")
    season = int(match.group(1))
    e1 = int(match.group(2))
    e2 = int(match.group(3)) if match.group(3) else e1
    return season, e1, e2


def _mapping_from_plan(plan: RenamePlan) -> EpisodeMappingResult:
    plan_moves = plan.moves
    subtitles_by_key: dict[tuple[int, int, int], list[int]] = {}
    videos: list[tuple[int, tuple[int, int, int]]] = []

    for move in plan_moves:
        key = _parse_episode_range(move.dst)
        if move.kind == "video":
            videos.append((move.src_id, key))
        else:
            subtitles_by_key.setdefault(key, []).append(move.src_id)

    items: list[EpisodeMapItem] = []
    used_subs: set[int] = set()
    for video_id, key in sorted(videos, key=lambda item: item[0]):
        subs = tuple(sorted(subtitles_by_key.get(key, [])))
        used_subs.update(subs)
        season, e1, e2 = key
        items.append(
            EpisodeMapItem(
                video_id=video_id,
                season=season,
                episode_start=e1,
                episode_end=e2,
                subtitle_ids=subs,
            )
        )

    all_subs = {move.src_id for move in plan_moves if move.kind == "subtitle"}
    assert used_subs == all_subs

    return EpisodeMappingResult(tmdb_id=plan.tmdb_id, items=tuple(items))


def _relative_moves(
    plan_moves: tuple, *, series_dir: Path, output_root: Path
) -> dict[int, tuple[str, str, str]]:
    moves: dict[int, tuple[str, str, str]] = {}
    for move in plan_moves:
        src_rel = Path(move.src).relative_to(series_dir).as_posix()
        dst_rel = Path(move.dst).relative_to(output_root).as_posix()
        moves[move.src_id] = (move.kind, src_rel, dst_rel)
    return moves


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_real_case_plan_matches(case: RealCase, tmp_path: Path) -> None:
    plan = read_rename_plan_json(case.plan_path)
    tree_root, entries = _parse_tree_entries(case.tree_path)

    assert tree_root.name == Path(plan.series_dir).name

    series_dir = tmp_path / tree_root.name
    _materialize_tree(series_dir, entries)

    scan = scan_series_dir(series_dir)

    plan_videos = [move for move in plan.moves if move.kind == "video"]
    plan_subtitles = [move for move in plan.moves if move.kind == "subtitle"]

    assert len(scan.videos) == len(plan_videos)
    assert len(scan.subtitles) == len(plan_subtitles)

    for candidate in scan.videos + scan.subtitles:
        assert not candidate.rel_path.startswith(("Bonus/", "CDs/", "Scans/"))

    expected_src_by_id = {
        move.src_id: Path(move.src).relative_to(plan.series_dir).as_posix()
        for move in plan.moves
    }
    actual_src_by_id = {
        candidate.id: candidate.rel_path
        for candidate in scan.videos + scan.subtitles
    }
    assert expected_src_by_id == actual_src_by_id

    for move in plan.moves:
        ext = Path(move.src).suffix.lower()
        if move.kind == "video":
            assert ext in VIDEO_EXTS
        else:
            assert ext in SUBTITLE_EXTS

    mapping = _mapping_from_plan(plan)
    output_root = tmp_path / "out"
    built_plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn=plan.series_name_zh_cn,
        year=plan.year,
        tmdb_id=plan.tmdb_id,
        output_root=output_root,
    )

    expected_moves = _relative_moves(
        plan.moves,
        series_dir=Path(plan.series_dir),
        output_root=Path(plan.output_root),
    )
    actual_moves = _relative_moves(
        built_plan.moves,
        series_dir=series_dir,
        output_root=output_root,
    )

    assert expected_moves == actual_moves


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_real_case_end_to_end_with_real_llm(
    case: RealCase, tmp_path: Path
) -> None:
    if not (
        _env_present("OPENAI_API_KEY")
        and _env_present("OPENAI_MODEL")
        and _env_present("TMDB_API_KEY")
    ):
        pytest.skip("OPENAI_API_KEY, OPENAI_MODEL, or TMDB_API_KEY not set")

    plan = read_rename_plan_json(case.plan_path)
    tree_root, entries = _parse_tree_entries(case.tree_path)

    series_dir = tmp_path / tree_root.name
    _materialize_tree(series_dir, entries)

    scan = scan_series_dir(series_dir)

    tmdb = TMDBClient(api_key=os.environ["TMDB_API_KEY"].strip(), timeout=30.0)
    candidates = _search_tmdb_candidates(
        tmdb,
        series_dir.name,
        llm_title_factory=openai_llm_for_tmdb_id_from_env,
    )

    if len(candidates) == 1:
        tmdb_id = candidates[0].id
    else:
        tmdb_id = resolve_tmdb_tv_id_with_llm(
            series_dir.name,
            candidates,
            openai_llm_for_tmdb_id_from_env(),
            max_candidates=min(10, len(candidates)),
        )

    assert tmdb_id == plan.tmdb_id

    series_name_zh_cn, details = tmdb.resolve_series_title(tmdb_id)
    assert series_name_zh_cn == plan.series_name_zh_cn
    assert details.year == plan.year

    season_episode_counts = {
        season.season_number: season.episode_count for season in details.seasons
    }

    specials_zh = None
    specials_en = None
    if 0 in season_episode_counts:
        try:
            specials_zh = tmdb.get_season(tmdb_id, 0, language="zh-CN")
        except TMDBError:
            specials_zh = None
        try:
            specials_en = tmdb.get_season(tmdb_id, 0, language="en-US")
        except TMDBError:
            specials_en = None

    llm = openai_llm_from_env()
    mapping = map_episodes_with_llm(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=details.year,
        season_episode_counts=season_episode_counts,
        specials_zh=specials_zh,
        specials_en=specials_en,
        scan=scan,
        llm=llm,
        max_output_tokens=8196,
    )

    output_root = tmp_path / "out"
    built_plan = build_rename_plan(
        scan=scan,
        mapping=mapping,
        series_name_zh_cn=series_name_zh_cn,
        year=details.year,
        tmdb_id=tmdb_id,
        output_root=output_root,
    )

    expected_moves = _relative_moves(
        plan.moves,
        series_dir=Path(plan.series_dir),
        output_root=Path(plan.output_root),
    )
    actual_moves = _relative_moves(
        built_plan.moves,
        series_dir=series_dir,
        output_root=output_root,
    )

    assert expected_moves == actual_moves
