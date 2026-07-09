from __future__ import annotations

import struct
from pathlib import Path

from fastapi.testclient import TestClient

from aninamer.api import create_app


STATIC_DIR = Path("aninamer/api/static")


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _ico_sizes(path: Path) -> list[tuple[int, int]]:
    data = path.read_bytes()
    reserved, image_type, count = struct.unpack("<HHH", data[:6])
    assert (reserved, image_type) == (0, 1)
    sizes: list[tuple[int, int]] = []
    for index in range(count):
        offset = 6 + (index * 16)
        width = data[offset] or 256
        height = data[offset + 1] or 256
        sizes.append((width, height))
    return sizes


def test_favicon_bundle_dimensions_and_manifest() -> None:
    expected_png_sizes = {
        "favicon-16x16.png": (16, 16),
        "favicon-32x32.png": (32, 32),
        "favicon-48x48.png": (48, 48),
        "apple-touch-icon.png": (180, 180),
        "android-chrome-192x192.png": (192, 192),
        "android-chrome-512x512.png": (512, 512),
    }

    assert {
        name: _png_size(STATIC_DIR / name) for name in expected_png_sizes
    } == expected_png_sizes
    assert _ico_sizes(STATIC_DIR / "favicon.ico") == [
        (16, 16),
        (32, 32),
        (48, 48),
    ]

    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for asset_name in (
        "favicon.ico",
        "favicon-16x16.png",
        "favicon-32x32.png",
        "apple-touch-icon.png",
        "site.webmanifest",
    ):
        assert f'/ui/{asset_name}' in index_html


def test_favicon_bundle_is_served_with_cache_validators() -> None:
    app = create_app(  # type: ignore[arg-type]
        object(),
        store=object(),
        worker=object(),
    )

    with TestClient(app) as client:
        root_favicon_response = client.get("/favicon.ico")
        favicon_response = client.get("/ui/favicon.ico")
        png_response = client.get("/ui/favicon-32x32.png")
        manifest_response = client.get("/ui/site.webmanifest")
        cached_response = client.get(
            "/ui/favicon-32x32.png",
            headers={"If-None-Match": png_response.headers["etag"]},
        )

    assert root_favicon_response.status_code == 200
    assert root_favicon_response.content == favicon_response.content
    assert favicon_response.status_code == 200
    assert favicon_response.headers["content-type"] == "image/vnd.microsoft.icon"
    assert png_response.status_code == 200
    assert png_response.headers["content-type"] == "image/png"
    assert manifest_response.status_code == 200
    assert manifest_response.headers["content-type"].startswith(
        "application/manifest+json"
    )
    manifest = manifest_response.json()
    assert manifest["start_url"] == "/ui/"
    assert manifest["scope"] == "/ui/"
    assert manifest["icons"] == [
        {
            "src": "/ui/android-chrome-192x192.png",
            "sizes": "192x192",
            "type": "image/png",
        },
        {
            "src": "/ui/android-chrome-512x512.png",
            "sizes": "512x512",
            "type": "image/png",
        },
    ]
    assert cached_response.status_code == 304
