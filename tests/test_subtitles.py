from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.subtitles import (
    ChineseSubtitleVariant,
    detect_chinese_sub_suffix,
    detect_chinese_sub_variant,
    detect_variant_from_filename,
    detect_variant_from_text,
)


@pytest.mark.parametrize(
    "name",
    [
        "Show.[CHS].ass",
        "Show.chs.ass",
        "Show_chs_.ass",
        "Show-zh-hans.srt",
        "Show.zh_cn.srt",
        "Show.zh-cn.srt",
        "Show.gb.ass",
        "Show.hans.ass",
    ],
)
def test_detect_variant_from_filename_simplified_ascii(name: str) -> None:
    assert detect_variant_from_filename(name) == ChineseSubtitleVariant.CHS


@pytest.mark.parametrize(
    "name",
    [
        "Show.[CHT].ass",
        "Show.cht.ass",
        "Show-zh-hant.srt",
        "Show.zh_tw.srt",
        "Show.zh-tw.srt",
        "Show.big5.ass",
        "Show.hant.ass",
    ],
)
def test_detect_variant_from_filename_traditional_ascii(name: str) -> None:
    assert detect_variant_from_filename(name) == ChineseSubtitleVariant.CHT


def test_detect_variant_from_filename_chinese_tokens() -> None:
    assert (
        detect_variant_from_filename("动画_简体.ass")
        == ChineseSubtitleVariant.CHS
    )
    assert (
        detect_variant_from_filename("动画-繁体.srt")
        == ChineseSubtitleVariant.CHT
    )


def test_detect_variant_from_filename_no_hint() -> None:
    assert detect_variant_from_filename("video.chsfoo.srt") is None
    assert detect_variant_from_filename("video.zh-cnish.srt") is None


def test_detect_variant_from_text_simplified() -> None:
    assert detect_variant_from_text("为国车门") == ChineseSubtitleVariant.CHS


def test_detect_variant_from_text_traditional() -> None:
    assert detect_variant_from_text("為國車門") == ChineseSubtitleVariant.CHT


def test_detect_variant_from_text_equal() -> None:
    assert detect_variant_from_text("為为") is None


def test_detect_chinese_sub_variant_uses_filename_hint(tmp_path: Path) -> None:
    path = tmp_path / "episode.chs.srt"
    path.write_text("為國", encoding="utf-8")
    assert detect_chinese_sub_variant(path) == ChineseSubtitleVariant.CHS


def test_detect_chinese_sub_variant_binary_sup(tmp_path: Path) -> None:
    path = tmp_path / "episode.sup"
    path.write_bytes(b"\x00\x01\x02")
    assert detect_chinese_sub_variant(path) == ChineseSubtitleVariant.CHI


def test_detect_chinese_sub_variant_from_text(tmp_path: Path) -> None:
    path = tmp_path / "episode.srt"
    path.write_text("为国", encoding="utf-8")
    assert detect_chinese_sub_variant(path) == ChineseSubtitleVariant.CHS


def test_detect_chinese_sub_suffix(tmp_path: Path) -> None:
    path = tmp_path / "episode.chs.ass"
    path.write_text("为国", encoding="utf-8")
    assert detect_chinese_sub_suffix(path) == ".chs"
