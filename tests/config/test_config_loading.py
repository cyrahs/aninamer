from __future__ import annotations

from pathlib import Path

from aninamer.config import load_config


def test_load_config_parses_worker_and_watch_roots(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
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
auto_apply = true

[[watch_roots]]
key = "downloads"
input_root = "/input"
output_root = "/output"
""".strip(),
        encoding="utf-8",
    )

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
