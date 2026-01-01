from __future__ import annotations

from enum import Enum
from pathlib import Path
import re


class ChineseSubtitleVariant(Enum):
    CHS = "chs"
    CHT = "cht"
    CHI = "chi"

    @property
    def dot_suffix(self) -> str:
        return f".{self.value}"


_SIMPLIFIED_ASCII_TOKENS = ("chs", "hans", "zh-hans", "zh_cn", "zh-cn", "gb")
_TRADITIONAL_ASCII_TOKENS = ("cht", "hant", "zh-hant", "zh_tw", "zh-tw", "big5")
_SIMPLIFIED_WORDS = ("简体", "简中")
_TRADITIONAL_WORDS = ("繁体", "繁中")

_SIMPLIFIED_CHARS = set("为国云马门见车长乐书这爱气网与万广后台里发复钟东")
_TRADITIONAL_CHARS = set("為國雲馬門見車長樂書這愛氣網與萬廣後臺裡發復鐘東")

_DECODE_ENCODINGS = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030")


def _compile_ascii_token_patterns(tokens: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for token in tokens
    )


_SIMPLIFIED_PATTERNS = _compile_ascii_token_patterns(_SIMPLIFIED_ASCII_TOKENS)
_TRADITIONAL_PATTERNS = _compile_ascii_token_patterns(_TRADITIONAL_ASCII_TOKENS)


def detect_variant_from_filename(filename: str) -> ChineseSubtitleVariant | None:
    for pattern in _SIMPLIFIED_PATTERNS:
        if pattern.search(filename):
            return ChineseSubtitleVariant.CHS
    for word in _SIMPLIFIED_WORDS:
        if word in filename:
            return ChineseSubtitleVariant.CHS
    for pattern in _TRADITIONAL_PATTERNS:
        if pattern.search(filename):
            return ChineseSubtitleVariant.CHT
    for word in _TRADITIONAL_WORDS:
        if word in filename:
            return ChineseSubtitleVariant.CHT
    return None


def detect_variant_from_text(text: str) -> ChineseSubtitleVariant | None:
    simplified_count = 0
    traditional_count = 0
    for ch in text:
        if ch in _SIMPLIFIED_CHARS:
            simplified_count += 1
        if ch in _TRADITIONAL_CHARS:
            traditional_count += 1
    if simplified_count > traditional_count:
        return ChineseSubtitleVariant.CHS
    if traditional_count > simplified_count:
        return ChineseSubtitleVariant.CHT
    return None


def _decode_subtitle_bytes(data: bytes) -> str:
    for encoding in _DECODE_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def detect_chinese_sub_variant(
    path: Path, *, max_bytes: int = 65536
) -> ChineseSubtitleVariant:
    hint = detect_variant_from_filename(path.name)
    if hint is not None:
        return hint
    if path.suffix.lower() == ".sup":
        return ChineseSubtitleVariant.CHI
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
    text = _decode_subtitle_bytes(data)
    from_text = detect_variant_from_text(text)
    if from_text is not None:
        return from_text
    return ChineseSubtitleVariant.CHI


def detect_chinese_sub_suffix(path: Path) -> str:
    return detect_chinese_sub_variant(path).dot_suffix
