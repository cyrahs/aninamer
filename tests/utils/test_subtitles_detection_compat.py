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


def test_detect_variant_from_filename_chs() -> None:
    assert detect_variant_from_filename("[SubsPlease] Show - 01 [CHS].ass") == ChineseSubtitleVariant.CHS
    assert detect_variant_from_filename("Show.01.zh-Hans.srt") == ChineseSubtitleVariant.CHS
    assert detect_variant_from_filename("Show.01.简体.ass") == ChineseSubtitleVariant.CHS
    assert detect_variant_from_filename("Show.01.SC.ass") == ChineseSubtitleVariant.CHS
    assert detect_variant_from_filename("Show.01.JPSC.ass") == ChineseSubtitleVariant.CHS


def test_detect_variant_from_filename_cht() -> None:
    assert detect_variant_from_filename("Show.01.[CHT].ass") == ChineseSubtitleVariant.CHT
    assert detect_variant_from_filename("Show.01.zh-Hant.srt") == ChineseSubtitleVariant.CHT
    assert detect_variant_from_filename("Show.01.繁体.ass") == ChineseSubtitleVariant.CHT
    assert detect_variant_from_filename("Show.01.BIG5.ass") == ChineseSubtitleVariant.CHT
    assert detect_variant_from_filename("Show.01.TC.ass") == ChineseSubtitleVariant.CHT
    assert detect_variant_from_filename("Show.01.JPTC.ass") == ChineseSubtitleVariant.CHT


def test_detect_variant_from_filename_none() -> None:
    assert detect_variant_from_filename("Show.01.ass") is None
    assert detect_variant_from_filename("Show.01.chinese.ass") is None  # not a strong variant hint
    assert detect_variant_from_filename("Show.01.scfoo.ass") is None
    assert detect_variant_from_filename("Show.01.tcbar.ass") is None


def test_detect_variant_from_text_chs_vs_cht() -> None:
    # Strong simplified signal
    text_chs = "国国国 后后后 门门门 见见见 发发发 这这这"
    assert detect_variant_from_text(text_chs) == ChineseSubtitleVariant.CHS

    # Strong traditional signal
    text_cht = "國國國 後後後 門門門 見見見 發發發 這這這"
    assert detect_variant_from_text(text_cht) == ChineseSubtitleVariant.CHT

    # No distinguishing characters -> None
    assert detect_variant_from_text("你好世界 こんにちは") is None


def test_detect_chinese_sub_variant_prefers_filename_hint_over_content(tmp_path: Path) -> None:
    p = tmp_path / "ep01.CHT.ass"
    # Content is simplified-ish, but filename says CHT
    p.write_text("国国国 后后后", encoding="utf-8")
    assert detect_chinese_sub_variant(p) == ChineseSubtitleVariant.CHT


def test_detect_chinese_sub_variant_uses_content_when_no_filename_hint(tmp_path: Path) -> None:
    p = tmp_path / "ep01.ass"
    p.write_text("國國國 後後後 門門門 見見見", encoding="utf-8")
    assert detect_chinese_sub_variant(p) == ChineseSubtitleVariant.CHT


def test_detect_chinese_sub_variant_returns_chi_when_unknown(tmp_path: Path) -> None:
    p = tmp_path / "ep01.ass"
    p.write_text("你好世界", encoding="utf-8")
    assert detect_chinese_sub_variant(p) == ChineseSubtitleVariant.CHI


def test_detect_chinese_sub_variant_binary_sup_skips_content(tmp_path: Path) -> None:
    p = tmp_path / "ep01.sup"
    p.write_bytes(b"\x00\x01\x02\x03\x04\xff")
    # no filename hint, binary -> unknown
    assert detect_chinese_sub_variant(p) == ChineseSubtitleVariant.CHI


def test_detect_chinese_sub_suffix(tmp_path: Path) -> None:
    p = tmp_path / "ep01.ass"
    p.write_text("国国国 后后后", encoding="utf-8")
    assert detect_chinese_sub_suffix(p) == ".chs"
