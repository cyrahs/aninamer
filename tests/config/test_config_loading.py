from __future__ import annotations

from pathlib import Path

import pytest

from aninamer.config import (
    load_config,
    load_openai_settings_from_env_or_config,
    load_tmdb_settings_from_env_or_config,
)


def _write_full_config(path: Path) -> None:
    path.write_text(
        """
log_path = "./logs"

[database]
postgres_dsn = "postgresql://aninamer:aninamer@127.0.0.1:5432/aninamer"

[tmdb]
api_key = "tmdb-key"
timeout = 45

[openai]
api_key = "openai-key"
model = "gpt-5.2"
base_url = "https://api.openai.com"
timeout = 75
reasoning_effort_chore = "low"
reasoning_effort_mapping = "medium"

[notifications]
base_url = "https://notify.example.test"
bearer_token = "notify-token"
timeout_seconds = 12

[api]
token = "secret"

[worker]
settle_seconds = 10
scan_interval_seconds = 5
health_stale_after_seconds = 15
auto_apply = true

[[watch_roots]]
key = "downloads"
input_root = "/input"
output_root = "/output"
""".strip(),
        encoding="utf-8",
    )


def test_load_config_parses_worker_and_watch_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path)

    config = load_config(config_path)

    assert config.log_path == Path("./logs")
    assert config.database.postgres_dsn == (
        "postgresql://aninamer:aninamer@127.0.0.1:5432/aninamer"
    )
    assert config.tmdb.api_key == "tmdb-key"
    assert config.tmdb.timeout == 45.0
    assert config.openai.api_key == "openai-key"
    assert config.openai.model == "gpt-5.2"
    assert config.openai.base_url == "https://api.openai.com"
    assert config.openai.timeout == 75.0
    assert config.openai.reasoning_effort_chore == "low"
    assert config.openai.reasoning_effort_mapping == "medium"
    assert config.notifications is not None
    assert config.notifications.base_url == "https://notify.example.test"
    assert config.notifications.bearer_token == "notify-token"
    assert config.notifications.timeout_seconds == 12.0
    assert config.notifications_warning is None
    assert config.api.token == "secret"
    assert config.api.bind == "127.0.0.1"
    assert config.api.port == 8091
    assert config.worker.settle_seconds == 10
    assert config.worker.scan_interval_seconds == 5
    assert config.worker.health_stale_after_seconds == 15
    assert config.worker.auto_apply is True
    assert len(config.watch_roots) == 1
    assert config.watch_roots[0].key == "downloads"
    assert config.watch_roots[0].input_root == Path("/input")
    assert config.watch_roots[0].output_root == Path("/output")


def test_load_config_disables_notifications_when_incomplete(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
log_path = "./logs"

[database]
postgres_dsn = "postgresql://aninamer:aninamer@127.0.0.1:5432/aninamer"

[tmdb]
api_key = "tmdb-key"

[openai]
api_key = "openai-key"
model = "gpt-5.2"

[notifications]
base_url = "https://notify.example.test"

[api]
token = "secret"

[[watch_roots]]
key = "downloads"
input_root = "/input"
output_root = "/output"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.notifications is None
    assert config.notifications_warning is not None


def test_load_openai_settings_from_env_or_config_falls_back_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path)

    monkeypatch.setenv("ANINAMER_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_CHORE", raising=False)
    monkeypatch.delenv("OPENAI_REASONING_EFFORT_MAPPING", raising=False)

    settings = load_openai_settings_from_env_or_config()

    assert settings.api_key == "openai-key"
    assert settings.model == "gpt-5.2"
    assert settings.base_url == "https://api.openai.com"
    assert settings.timeout == 75.0
    assert settings.reasoning_effort_chore == "low"
    assert settings.reasoning_effort_mapping == "medium"


def test_load_tmdb_settings_from_env_or_config_falls_back_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_full_config(config_path)

    monkeypatch.setenv("ANINAMER_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("TMDB_TIMEOUT", raising=False)

    settings = load_tmdb_settings_from_env_or_config()

    assert settings.api_key == "tmdb-key"
    assert settings.timeout == 45.0


def test_load_openai_settings_from_env_or_config_prefers_env_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_config = tmp_path / "missing.toml"
    monkeypatch.setenv("ANINAMER_CONFIG_PATH", str(missing_config))
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test")
    monkeypatch.setenv("OPENAI_TIMEOUT", "90")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_CHORE", "high")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT_MAPPING", "medium")

    settings = load_openai_settings_from_env_or_config()

    assert settings.api_key == "env-openai-key"
    assert settings.model == "env-model"
    assert settings.base_url == "https://openai.example.test"
    assert settings.timeout == 90.0
    assert settings.reasoning_effort_chore == "high"
    assert settings.reasoning_effort_mapping == "medium"


def test_load_tmdb_settings_from_env_or_config_prefers_env_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_config = tmp_path / "missing.toml"
    monkeypatch.setenv("ANINAMER_CONFIG_PATH", str(missing_config))
    monkeypatch.setenv("TMDB_API_KEY", "env-tmdb-key")
    monkeypatch.setenv("TMDB_TIMEOUT", "55")

    settings = load_tmdb_settings_from_env_or_config()

    assert settings.api_key == "env-tmdb-key"
    assert settings.timeout == 55.0
