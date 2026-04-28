from __future__ import annotations

import errno
from pathlib import Path

from aninamer.monitoring import discover_series_dirs, discover_series_dirs_status


def test_discover_series_dirs_returns_empty_when_mount_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def broken_iterdir(self: Path):  # noqa: ANN202
        if self == tmp_path:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return original_iterdir(self)

    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", broken_iterdir)

    assert discover_series_dirs(tmp_path) == []
    result = discover_series_dirs_status(tmp_path)
    assert result.series_dirs == []
    assert result.unavailable is True
    assert result.error_message is not None
    assert "Transport endpoint is not connected" in result.error_message


def test_discover_series_dirs_marks_root_unavailable_when_entries_cannot_stat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    child = tmp_path / "ShowA"
    child.mkdir()

    original_is_dir = Path.is_dir

    def broken_is_dir(self: Path) -> bool:
        if self == child:
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", broken_is_dir)

    assert discover_series_dirs(tmp_path) == []
    result = discover_series_dirs_status(tmp_path)
    assert result.series_dirs == []
    assert result.unavailable is True
    assert result.error_message is not None
    assert "Transport endpoint is not connected" in result.error_message
