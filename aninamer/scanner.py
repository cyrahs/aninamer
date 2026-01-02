from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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

    for root, dirs, files in os.walk(series_dir, followlinks=False):
        dirs[:] = [name for name in dirs if name.casefold() not in SKIP_DIR_NAMES]
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            ext = file_path.suffix.lower()
            if not ext:
                continue
            if ext in VIDEO_EXTS:
                rel_path = file_path.relative_to(series_dir).as_posix()
                size_bytes = file_path.stat().st_size
                video_items.append((rel_path, ext, size_bytes))
            elif ext in SUBTITLE_EXTS:
                rel_path = file_path.relative_to(series_dir).as_posix()
                size_bytes = file_path.stat().st_size
                subtitle_items.append((rel_path, ext, size_bytes))

    video_items.sort(key=lambda item: item[0])
    subtitle_items.sort(key=lambda item: item[0])

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
    logger.debug(
        "scan: sample_videos=%s",
        [candidate.rel_path for candidate in videos[:10]],
    )
    logger.debug(
        "scan: sample_subtitles=%s",
        [candidate.rel_path for candidate in subtitles[:10]],
    )

    return ScanResult(series_dir=series_dir, videos=videos, subtitles=subtitles)
