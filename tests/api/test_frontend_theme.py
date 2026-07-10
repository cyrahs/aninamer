from __future__ import annotations

import re
from pathlib import Path


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
    assert 'id="themeColor" name="theme-color"' in html
    assert '<select id="themeSelect" name="themeSelect">' in html
    assert '<option value="system">跟随系统</option>' in html
    assert '<option value="light">浅色</option>' in html
    assert '<option value="dark">深色</option>' in html


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
    assert "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)" in mobile
    assert ".theme-control," in mobile
    assert ".theme-control select" in mobile
    assert "min-width: 0" in mobile
