from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
import re
from typing import Sequence

DEFAULT_THREADS = 8


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

# High-frequency characters that differ between simplified and traditional Chinese.
# Selected for dialogue-heavy content like anime subtitles.
_SIMPLIFIED_CHARS = set(
    "为国云马门见车长乐书这爱气网与万广后台里发复钟东"  # original set
    "说时来会过对话听开头觉点样经认关现离让给请学问"  # high-frequency dialogue
    "还没虽该谁写买卖读语词饭馆银钱"  # additional common words
)
_TRADITIONAL_CHARS = set(
    "為國雲馬門見車長樂書這愛氣網與萬廣後臺裡發復鐘東"  # original set
    "說時來會過對話聽開頭覺點樣經認關現離讓給請學問"  # high-frequency dialogue
    "還沒雖該誰寫買賣讀語詞飯館銀錢"  # additional common words
)

_DECODE_ENCODINGS = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030")


def _compile_ascii_token_patterns(
    tokens: tuple[str, ...],
) -> tuple[re.Pattern[str], ...]:
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


def detect_chinese_sub_variants_batch(
    paths: Sequence[Path],
    *,
    max_bytes: int = 65536,
    max_workers: int = DEFAULT_THREADS,
) -> dict[Path, ChineseSubtitleVariant]:
    """Detect Chinese subtitle variants for multiple files in parallel.

    Useful when working with network-mounted filesystems (e.g., rclone mount)
    where parallel I/O significantly improves throughput.

    Args:
        paths: Sequence of subtitle file paths to analyze.
        max_bytes: Maximum bytes to read from each file for content analysis.
        max_workers: Number of threads to use (default: 8).

    Returns:
        Dictionary mapping each path to its detected variant.
    """
    if not paths:
        return {}

    def _detect(p: Path) -> tuple[Path, ChineseSubtitleVariant]:
        return p, detect_chinese_sub_variant(p, max_bytes=max_bytes)

    results: dict[Path, ChineseSubtitleVariant] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for path, variant in executor.map(_detect, paths):
            results[path] = variant
    return results


def detect_chinese_sub_suffixes_batch(
    paths: Sequence[Path],
    *,
    max_bytes: int = 65536,
    max_workers: int = DEFAULT_THREADS,
) -> dict[Path, str]:
    """Detect Chinese subtitle suffixes for multiple files in parallel.

    Args:
        paths: Sequence of subtitle file paths to analyze.
        max_bytes: Maximum bytes to read from each file for content analysis.
        max_workers: Number of threads to use (default: 8).

    Returns:
        Dictionary mapping each path to its detected suffix (e.g., ".chs").
    """
    variants = detect_chinese_sub_variants_batch(
        paths, max_bytes=max_bytes, max_workers=max_workers
    )
    return {p: v.dot_suffix for p, v in variants.items()}
