from __future__ import annotations

import re

_BRACKET_PATTERNS = (
    re.compile(r"\[[^\[\]]*\]"),
    re.compile(r"\([^()]*\)"),
    re.compile(r"\{[^{}]*\}"),
)

_RELEASE_TOKENS = (
    "2160p",
    "1080p",
    "720p",
    "480p",
    "4k",
    "x264",
    "x265",
    "h264",
    "h265",
    "hevc",
    "avc",
    "aac",
    "flac",
    "10bit",
    "8bit",
    "hi10p",
    "ma10p",
    "bdrip",
    "bluray",
    "bd",
    "web",
    "webrip",
    "web-dl",
    "hdr",
    "dv",
    "remux",
    "vcb",
    "vcb-studio",
    "batch",
)

_TOKEN_PATTERN = re.compile(
    rf"(?i)(?<!\w)(?:{'|'.join(re.escape(token) for token in _RELEASE_TOKENS)})(?!\w)"
)

_CHINESE_NUMERALS = "\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341"
_SEASON_PATTERNS = (
    re.compile(r"(?i)(?<!\w)s\d{1,2}(?!\w)"),
    re.compile(r"(?i)(?<!\w)season\s*\d{1,2}(?!\w)"),
    re.compile(r"(?i)(?<!\w)\d{1,2}(?:st|nd|rd|th)\s*season(?!\w)"),
    re.compile(rf"\u7b2c\s*(?:\d+|[{_CHINESE_NUMERALS}]+)\s*\u5b63"),
)
_TMDB_TAG_PATTERN = re.compile(r"\{\s*tmdb-([^{}]+)\s*\}", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _strip_bracketed_segments(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        for pattern in _BRACKET_PATTERNS:
            text = pattern.sub(" ", text)
    return text


def _strip_unbalanced_brackets(text: str) -> str:
    return re.sub(r"[\[\]\(\)\{\}]", " ", text)


def _strip_season_markers(text: str) -> str:
    for pattern in _SEASON_PATTERNS:
        text = pattern.sub(" ", text)
    return text


def clean_tmdb_query(name: str) -> str:
    working = name.replace("_", " ").replace(".", " ")
    working = _strip_bracketed_segments(working)
    working = _strip_unbalanced_brackets(working)
    working = _strip_season_markers(working)
    working = _TOKEN_PATTERN.sub(" ", working)
    working = _normalize_whitespace(working).strip()
    if working:
        return working

    fallback = name
    fallback = _strip_bracketed_segments(fallback)
    fallback = _strip_unbalanced_brackets(fallback)
    fallback = fallback.replace("_", " ").replace(".", " ")
    return _normalize_whitespace(fallback).strip()


def build_tmdb_query_variants(name: str, *, max_variants: int = 6) -> list[str]:
    if max_variants < 1:
        raise ValueError("max_variants must be >= 1")

    variants: list[str] = []
    base = _normalize_whitespace(name).strip()
    if base:
        variants.append(base)

    cleaned = clean_tmdb_query(name)
    if cleaned:
        variants.append(cleaned)

    words = cleaned.split()
    for count in (8, 6, 4, 2):
        if len(words) > count:
            variants.append(" ".join(words[:count]))

    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        if len(variant) < 2:
            continue
        if variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)

    return deduped[:max_variants]


def extract_tmdb_id_tag(name: str) -> int | None:
    matches = _TMDB_TAG_PATTERN.findall(name)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("multiple tmdb tags found in name")
    raw = matches[0].strip()
    if not raw.isdigit():
        raise ValueError("tmdb tag must be numeric")
    tmdb_id = int(raw)
    if tmdb_id < 1:
        raise ValueError("tmdb id must be >= 1")
    return tmdb_id
