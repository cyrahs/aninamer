from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.scanner import scan_series_dir


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_scan_series_dir_requires_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        scan_series_dir(missing)

    file_path = tmp_path / "not_a_dir.mkv"
    file_path.write_bytes(b"0")
    with pytest.raises(ValueError):
        scan_series_dir(file_path)


def test_scan_series_dir_collects_and_sorts_candidates(tmp_path: Path) -> None:
    series_dir = tmp_path / "Series"
    series_dir.mkdir()

    _write_file(series_dir / "b.MKV", b"bb")
    _write_file(series_dir / "a.mp4", b"aaaa")
    _write_file(series_dir / "subdir" / "c.AVI", b"c")

    _write_file(series_dir / "subs" / "zz.SRT", b"zzz")
    _write_file(series_dir / "subs" / "aa.ass", b"aaaaa")

    _write_file(series_dir / "cover.jpg", b"jpg")
    _write_file(series_dir / "notes.nfo", b"nfo")

    result = scan_series_dir(series_dir)

    assert result.series_dir == series_dir
    assert [v.rel_path for v in result.videos] == [
        "a.mp4",
        "b.MKV",
        "subdir/c.AVI",
    ]
    assert [v.ext for v in result.videos] == [".mp4", ".mkv", ".avi"]
    assert [v.id for v in result.videos] == [1, 2, 3]

    assert [s.rel_path for s in result.subtitles] == [
        "subs/aa.ass",
        "subs/zz.SRT",
    ]
    assert [s.ext for s in result.subtitles] == [".ass", ".srt"]
    assert [s.id for s in result.subtitles] == [4, 5]

    sizes = {c.rel_path: c.size_bytes for c in result.videos + result.subtitles}
    assert sizes["a.mp4"] == 4
    assert sizes["b.MKV"] == 2
    assert sizes["subdir/c.AVI"] == 1
    assert sizes["subs/aa.ass"] == 5
    assert sizes["subs/zz.SRT"] == 3



def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_scan_series_dir_filters_and_sorts_and_assigns_ids(tmp_path: Path) -> None:
    root = tmp_path / "Show"

    # Videos (mixed case extensions, nested)
    _write(root / "ep2.MKV", b"2222")                 # size 4
    _write(root / "ep1.mkv", b"111")                  # size 3
    _write(root / "nested" / "ep3.mp4", b"333333")    # size 6

    # Subtitles
    _write(root / "Subs" / "ep1.ass", b"a")           # size 1
    _write(root / "Subs" / "ep2.srt", b"bb")          # size 2
    _write(root / "nested" / "ep3.vtt", b"ccc")       # size 3

    # Useless resources (must be ignored)
    _write(root / "fonts" / "a.ttf", b"x")
    _write(root / "cover.jpg", b"y")
    _write(root / "info.nfo", b"z")
    _write(root / "archive.zip", b"z")

    result = scan_series_dir(root)

    # Basic structure
    assert result.series_dir == root
    assert len(result.videos) == 3
    assert len(result.subtitles) == 3

    # Videos are sorted by rel_path and have stable IDs starting at 1
    video_rel_paths = [v.rel_path for v in result.videos]
    assert video_rel_paths == sorted(video_rel_paths)
    assert [v.id for v in result.videos] == [1, 2, 3]

    # Extensions are normalized to lowercase
    assert [v.ext for v in result.videos] == [".mkv", ".mkv", ".mp4"]

    # Sizes are correct
    size_by_rel = {v.rel_path: v.size_bytes for v in result.videos}
    assert size_by_rel["ep1.mkv"] == 3
    assert size_by_rel["ep2.MKV"] == 4
    assert size_by_rel["nested/ep3.mp4"] == 6

    # Subtitles are sorted and IDs start after videos (unique across both lists)
    sub_rel_paths = [s.rel_path for s in result.subtitles]
    assert sub_rel_paths == sorted(sub_rel_paths)
    assert [s.id for s in result.subtitles] == [4, 5, 6]
    assert {s.ext for s in result.subtitles} == {".ass", ".srt", ".vtt"}

    # Ensure ignored files didn't sneak in
    all_rel = set(video_rel_paths) | set(sub_rel_paths)
    assert "fonts/a.ttf" not in all_rel
    assert "cover.jpg" not in all_rel
    assert "info.nfo" not in all_rel
    assert "archive.zip" not in all_rel


def test_scan_series_dir_skips_named_directories(tmp_path: Path) -> None:
    root = tmp_path / "Show"

    _write(root / "SPs" / "sp1.mkv", b"x")
    _write(root / "Bonus" / "bonus1.ass", b"y")
    _write(root / "映像特典" / "extra1.mp4", b"z")
    _write(root / "Previews" / "preview1.srt", b"w")
    _write(root / "main" / "ep1.mkv", b"ok")
    _write(root / "main" / "ep1.srt", b"sub")

    result = scan_series_dir(root)

    assert [v.rel_path for v in result.videos] == ["main/ep1.mkv"]
    assert [s.rel_path for s in result.subtitles] == ["main/ep1.srt"]


def test_scan_series_dir_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError):
        scan_series_dir(file_path)


def test_scan_series_dir_rejects_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(ValueError):
        scan_series_dir(missing)
