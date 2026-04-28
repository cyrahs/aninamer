from __future__ import annotations

from functools import lru_cache

from opencc import OpenCC


@lru_cache(maxsize=1)
def _s2t_converter() -> OpenCC:
    return OpenCC("s2t")


def to_traditional_chinese(text: str) -> str:
    return _s2t_converter().convert(text)
