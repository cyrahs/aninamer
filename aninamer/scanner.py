from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
import os
from pathlib import Path

from aninamer.diagnostics import logger as diagnostics_logger

logger = logging.getLogger(__name__)
SCAN_SAMPLE_LIMIT = 20

VIDEO_EXTS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".m4v",
    ".mov",
    ".ts",
    ".m2ts",
    ".webm",
}

SUBTITLE_EXTS = {
    ".ass",
    ".ssa",
    ".srt",
    ".sub",
    ".vtt",
    ".idx",
    ".sup",
}

SKIP_DIR_NAMES = {
    "sample",
    "samples",
    "trailer",
    "trailers",
    "bonus",
    "extra",
    "extras",
    "sp",
    "sps",
    "cd",
    "cds",
    "music",
    "musics",
    "scans",
    "scan",
    "menu",
    "menus",
    "preview",
    "previews",
    "映像特典",
}


@dataclass(frozen=True)
class FileCandidate:
    id: int
    rel_path: str
    ext: str
    size_bytes: int


@dataclass(frozen=True)
class ScanResult:
    series_dir: Path
    videos: list[FileCandidate]
    subtitles: list[FileCandidate]


def scan_series_dir(series_dir: Path) -> ScanResult:
    if not series_dir.exists() or not series_dir.is_dir():
        raise ValueError("series_dir must be an existing directory")

    logger.info("scan: start series_dir=%s", series_dir)

    video_items: list[tuple[str, str, int]] = []
    subtitle_items: list[tuple[str, str, int]] = []
    all_file_items: list[tuple[str, str, int, float]] = []
    extension_counts: Counter[str] = Counter()

    for root, dirs, files in os.walk(
        series_dir,
        followlinks=False,
        onerror=_raise_walk_error,
    ):
        dirs[:] = [name for name in dirs if name.casefold() not in SKIP_DIR_NAMES]
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            ext = file_path.suffix.lower()
            stat_result = file_path.stat()
            rel_path = file_path.relative_to(series_dir).as_posix()
            normalized_ext = ext or "<none>"
            extension_counts[normalized_ext] += 1
            all_file_items.append(
                (rel_path, normalized_ext, stat_result.st_size, stat_result.st_mtime)
            )
            if not ext:
                continue
            if ext in VIDEO_EXTS:
                size_bytes = stat_result.st_size
                video_items.append((rel_path, ext, size_bytes))
            elif ext in SUBTITLE_EXTS:
                size_bytes = stat_result.st_size
                subtitle_items.append((rel_path, ext, size_bytes))

    video_items.sort(key=lambda item: item[0])
    subtitle_items.sort(key=lambda item: item[0])
    all_file_items.sort(key=lambda item: item[0])

    videos: list[FileCandidate] = []
    for idx, (rel_path, ext, size_bytes) in enumerate(video_items, start=1):
        videos.append(
            FileCandidate(
                id=idx,
                rel_path=rel_path,
                ext=ext,
                size_bytes=size_bytes,
            )
        )

    subtitles: list[FileCandidate] = []
    start_id = len(videos) + 1
    for offset, (rel_path, ext, size_bytes) in enumerate(subtitle_items):
        subtitles.append(
            FileCandidate(
                id=start_id + offset,
                rel_path=rel_path,
                ext=ext,
                size_bytes=size_bytes,
            )
        )

    logger.info(
        "scan: done videos=%s subtitles=%s",
        len(videos),
        len(subtitles),
    )
    diagnostics_logger.info(
        "scan_snapshot: series_dir=%s total_files=%s extension_counts=%s sample_files=%s",
        series_dir,
        len(all_file_items),
        dict(sorted(extension_counts.items())),
        [
            {
                "rel_path": rel_path,
                "ext": ext,
                "size_bytes": size_bytes,
                "mtime": mtime,
            }
            for rel_path, ext, size_bytes, mtime in all_file_items[:SCAN_SAMPLE_LIMIT]
        ],
    )
    logger.debug(
        "scan: sample_videos=%s",
        [candidate.rel_path for candidate in videos[:10]],
    )
    logger.debug(
        "scan: sample_subtitles=%s",
        [candidate.rel_path for candidate in subtitles[:10]],
    )

    return ScanResult(series_dir=series_dir, videos=videos, subtitles=subtitles)


def _raise_walk_error(exc: OSError) -> None:
    raise exc
