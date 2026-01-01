from __future__ import annotations

import json


def extract_first_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            _obj, end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        return text[idx : idx + end]
    raise ValueError("no JSON object found")
