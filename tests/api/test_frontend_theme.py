from __future__ import annotations

import re
from pathlib import Path

from aninamer.api.app import _ui_cache_control


STATIC_DIR = Path("aninamer/api/static")


def _dark_theme_variables(styles: str) -> dict[str, str]:
    match = re.search(
        r':root\[data-resolved-theme="dark"\]\s*\{(?P<body>.*?)\n\}',
        styles,
        re.DOTALL,
    )
    assert match is not None
    return {
        name: value.strip()
        for name, value in re.findall(
            r"--([a-z0-9-]+):\s*([^;]+);",
            match.group("body"),
        )
    }


def _relative_luminance(hex_color: str) -> float:
    channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    brighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (brighter + 0.05) / (darker + 0.05)


def test_theme_control_bootstraps_before_stylesheet() -> None:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert 'data-theme="system" data-resolved-theme="light"' in html
    assert html.index("aninamer.theme") < html.index('rel="stylesheet"')
    assert 'new Set(["system", "light", "dark"])' in html
    assert "allowed.has(stored) ? stored : \"system\"" in html
    assert 'typeof window.matchMedia === "function"' in html
    assert 'id="themeColor" name="theme-color"' in html
    assert 'id="themeControl" class="theme-control" role="group"' in html
    assert 'aria-label="配色"' in html
    assert 'data-theme-choice="system"' in html
    assert 'data-theme-choice="light"' in html
    assert 'data-theme-choice="dark"' in html
    assert 'aria-label="跟随系统"' in html
    assert 'aria-label="浅色模式"' in html
    assert 'aria-label="深色模式"' in html
    assert html.count('aria-hidden="true" focusable="false"') == 3
    assert "themeSelect" not in html

    asset_versions = re.findall(
        r'/ui/(?:styles\.css|app\.js)\?v=([^"\s]+)',
        html,
    )
    assert len(asset_versions) == 2
    assert len(set(asset_versions)) == 1


def test_theme_runtime_persists_preference_and_tracks_system_changes() -> None:
    source = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'const THEME_STORAGE_KEY = "aninamer.theme"' in source
    assert 'window.matchMedia("(prefers-color-scheme: dark)")' in source
    assert "function normalizeThemePreference(value)" in source
    assert '["system", "light", "dark"].includes(value)' in source
    assert 'document.documentElement.dataset.theme = preference' in source
    assert 'document.documentElement.dataset.resolvedTheme = effectiveTheme' in source
    assert 'document.documentElement.dataset.theme === "system"' in source
    assert 'systemThemeQuery.addEventListener("change", handleSystemThemeChange)' in source
    assert 'window.addEventListener("storage"' in source
    assert "meta.content = THEME_COLORS[effectiveTheme]" in source
    assert "storageSet(THEME_STORAGE_KEY, preference)" in source
    assert 'document.querySelector("#themeControl")' in source
    assert 'document.querySelectorAll("[data-theme-choice]")' in source
    assert 'els.themeControl.addEventListener("click"' in source
    assert "option.dataset.themeChoice" in source
    assert 'option.setAttribute(\n      "aria-pressed"' in source
    assert "themeSelect" not in source
    assert "function storageGet(key)" in source
    assert "function storageSet(key, value)" in source
    assert "function storageRemove(key)" in source


def test_dark_palette_uses_semantic_surfaces_and_accessible_contrast() -> None:
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    variables = _dark_theme_variables(styles)

    assert "color-scheme: dark" in styles
    assert variables["bg"] == "#0b1217"
    assert variables["panel"] == "#121c23"
    assert variables["field"] == "#17242c"
    assert variables["control-border"] == "#5b7380"
    assert variables["focus-ring"] == "rgb(94 234 212 / 55%)"
    assert variables["accent-soft"] == "#12352f"
    assert variables["danger-soft"] == "#3b1b1b"
    assert variables["warn-soft"] == "#382b15"
    assert variables["success-soft"] == "#12342b"
    assert "background: var(--surface-subtle)" in styles
    assert "border-bottom: 1px solid var(--line)" in styles
    assert "background: var(--mobile-sidebar)" in styles

    contrast_pairs = [
        (variables["text"], variables["bg"]),
        (variables["muted"], variables["panel"]),
        (variables["accent-text"], variables["panel"]),
        (variables["danger"], variables["danger-soft"]),
        (variables["warn"], variables["warn-soft"]),
        (variables["success"], variables["success-soft"]),
        (variables["control-border"], variables["field"]),
        ("#ffffff", variables["warn-fill"]),
    ]
    text_pairs = contrast_pairs[:6]
    non_text_pairs = contrast_pairs[6:7]
    assert all(
        _contrast_ratio(foreground, background) >= 4.5
        for foreground, background in text_pairs
    )
    assert all(
        _contrast_ratio(foreground, background) >= 3
        for foreground, background in non_text_pairs
    )
    assert _contrast_ratio(*contrast_pairs[-1]) >= 4.5


def test_theme_control_stays_usable_on_small_screens() -> None:
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")
    mobile = styles[styles.index("@media (max-width: 640px)") :]

    assert ".topbar-actions {" in mobile
    assert "grid-template-columns: auto minmax(0, 1fr)" in mobile
    assert ".theme-option {" in mobile
    assert "width: 44px" in mobile
    assert "min-height: 44px" in mobile


def test_theme_icon_group_has_distinct_pressed_and_hover_states() -> None:
    styles = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert ".theme-control {" in styles
    assert "border-radius: 999px" in styles
    assert ".theme-option svg {" in styles
    assert 'stroke: currentColor' in styles
    assert '.theme-option[aria-pressed="true"] {' in styles
    assert "background: var(--accent-soft)" in styles


def test_ui_cache_policy_prevents_mixed_static_asset_versions() -> None:
    assert _ui_cache_control("/ui") == "no-store"
    assert _ui_cache_control("/ui/") == "no-store"
    assert _ui_cache_control("/ui/app.js") == (
        "no-cache, max-age=0, must-revalidate"
    )
    assert _ui_cache_control("/ui/styles.css") == (
        "no-cache, max-age=0, must-revalidate"
    )
    assert _ui_cache_control("/ui/favicon.ico") is None
    assert _ui_cache_control("/healthz") is None
