from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

from aninamer.episode_mapping import EpisodeMapItem, EpisodeMappingResult
from aninamer.errors import PlanValidationError
from aninamer.scanner import FileCandidate, ScanResult
from aninamer.subtitles import detect_chinese_sub_variant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannedMove:
    src: Path
    dst: Path
    kind: Literal["video", "subtitle"]
    src_id: int


@dataclass(frozen=True)
class RenamePlan:
    tmdb_id: int
    series_name_zh_cn: str
    year: int | None
    series_dir: Path
    output_root: Path
    moves: tuple[PlannedMove, ...]


_ILLEGAL_PATH_CHARS = set('<>:"/\\|?*')


def sanitize_path_component(name: str) -> str:
    cleaned = "".join(" " if ch in _ILLEGAL_PATH_CHARS else ch for ch in name)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.rstrip(" .")
    if not cleaned:
        return "Unknown"
    if all(ch == "." for ch in cleaned):
        return "Unknown"
    return cleaned


def format_series_root_folder(
    series_name_zh_cn: str, year: int | None, tmdb_id: int
) -> str:
    series_part = sanitize_path_component(series_name_zh_cn)
    tag = f"{{tmdb-{tmdb_id}}}"
    if year is None:
        return f"{series_part} {tag}"
    return f"{series_part} ({year}) {tag}"


def format_season_folder(season: int) -> str:
    return f"S{season:02d}"


def format_episode_base(
    series_name_zh_cn: str, season: int, e1: int, e2: int
) -> str:
    series_part = sanitize_path_component(series_name_zh_cn)
    if e1 == e2:
        return f"{series_part} S{season:02d}E{e1:02d}"
    return f"{series_part} S{season:02d}E{e1:02d}-E{e2:02d}"


def _resolve_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _ensure_within_output_root(dst: Path, output_root: Path) -> Path:
    dst_resolved = _resolve_path(dst)
    output_root_resolved = _resolve_path(output_root)
    if not dst_resolved.is_relative_to(output_root_resolved):
        raise PlanValidationError(
            f"destination {dst_resolved} is outside output_root {output_root_resolved}"
        )
    return dst_resolved


def _disambiguate_subtitle_destination(
    dst: Path, used_dests: set[Path]
) -> Path:
    if dst not in used_dests:
        return dst
    suffix = dst.suffix
    stem = dst.name[:-len(suffix)] if suffix else dst.name
    counter = 1
    while True:
        if suffix:
            candidate = dst.with_name(f"{stem}.{counter}{suffix}")
        else:
            candidate = dst.with_name(f"{stem}.{counter}")
        if candidate not in used_dests:
            return candidate
        counter += 1


def _candidate_by_id(
    candidates: list[FileCandidate],
) -> dict[int, FileCandidate]:
    return {candidate.id: candidate for candidate in candidates}


def build_rename_plan(
    *,
    scan: ScanResult,
    mapping: EpisodeMappingResult,
    series_name_zh_cn: str,
    year: int | None,
    tmdb_id: int,
    output_root: Path,
    allow_existing_dest: bool = False,
) -> RenamePlan:
    if not scan.series_dir.exists() or not scan.series_dir.is_dir():
        raise ValueError("scan.series_dir must be an existing directory")
    if not isinstance(output_root, Path):
        raise ValueError("output_root must be a Path")
    if tmdb_id != mapping.tmdb_id:
        raise PlanValidationError(
            f"tmdb_id {tmdb_id} does not match mapping tmdb_id {mapping.tmdb_id}"
        )

    logger.info(
        "plan: start tmdb_id=%s output_root=%s item_count=%s",
        tmdb_id,
        output_root,
        len(mapping.items),
    )
    series_dir = _resolve_path(scan.series_dir)
    output_root_resolved = _resolve_path(output_root)
    series_folder = format_series_root_folder(series_name_zh_cn, year, tmdb_id)
    video_by_id = _candidate_by_id(scan.videos)
    sub_by_id = _candidate_by_id(scan.subtitles)
    used_dests: set[Path] = set()
    moves: list[PlannedMove] = []
    items: tuple[EpisodeMapItem, ...] = mapping.items

    for item in items:
        video = video_by_id.get(item.video_id)
        if video is None:
            raise PlanValidationError(f"video id {item.video_id} not found in scan")
        video_src = _resolve_path(series_dir / video.rel_path)
        if not video_src.is_file():
            raise PlanValidationError(
                f"video id {video.id} source {video_src} does not exist"
            )

        season_folder = format_season_folder(item.season)
        episode_base = format_episode_base(
            series_name_zh_cn,
            item.season,
            item.episode_start,
            item.episode_end,
        )
        video_name = f"{episode_base}{video.ext}"
        video_dst = output_root_resolved / series_folder / season_folder / video_name
        video_dst = _ensure_within_output_root(video_dst, output_root_resolved)
        if video_dst in used_dests:
            raise PlanValidationError(f"destination collision at {video_dst}")
        if not allow_existing_dest and video_dst.exists():
            raise PlanValidationError(f"destination already exists: {video_dst}")

        moves.append(
            PlannedMove(
                src=video_src,
                dst=video_dst,
                kind="video",
                src_id=video.id,
            )
        )
        used_dests.add(video_dst)

        for sub_id in item.subtitle_ids:
            subtitle = sub_by_id.get(sub_id)
            if subtitle is None:
                raise PlanValidationError(
                    f"subtitle id {sub_id} not found in scan"
                )
            subtitle_src = _resolve_path(series_dir / subtitle.rel_path)
            if not subtitle_src.is_file():
                raise PlanValidationError(
                    f"subtitle id {subtitle.id} source {subtitle_src} does not exist"
                )

            variant_suffix = detect_chinese_sub_variant(subtitle_src).dot_suffix
            subtitle_name = f"{episode_base}{variant_suffix}{subtitle.ext}"
            subtitle_dst = (
                output_root_resolved / series_folder / season_folder / subtitle_name
            )
            subtitle_dst = _disambiguate_subtitle_destination(
                subtitle_dst, used_dests
            )
            subtitle_dst = _ensure_within_output_root(
                subtitle_dst, output_root_resolved
            )
            if not allow_existing_dest and subtitle_dst.exists():
                raise PlanValidationError(
                    f"destination already exists: {subtitle_dst}"
                )

            moves.append(
                PlannedMove(
                    src=subtitle_src,
                    dst=subtitle_dst,
                    kind="subtitle",
                    src_id=subtitle.id,
                )
            )
            used_dests.add(subtitle_dst)

    sorted_moves = tuple(
        sorted(moves, key=lambda move: (move.kind, move.dst.as_posix()))
    )

    video_moves = sum(1 for move in sorted_moves if move.kind == "video")
    subtitle_moves = sum(1 for move in sorted_moves if move.kind == "subtitle")
    logger.info(
        "plan: built total_moves=%s video_moves=%s subtitle_moves=%s series_folder_name=%s",
        len(sorted_moves),
        video_moves,
        subtitle_moves,
        series_folder,
    )
    logger.debug(
        "plan: sample_destinations=%s",
        [move.dst.as_posix() for move in sorted_moves[:10]],
    )

    return RenamePlan(
        tmdb_id=tmdb_id,
        series_name_zh_cn=series_name_zh_cn,
        year=year,
        series_dir=series_dir,
        output_root=output_root_resolved,
        moves=sorted_moves,
    )
