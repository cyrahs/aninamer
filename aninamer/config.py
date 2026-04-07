from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ApiConfig:
    token: str
    bind: str = "127.0.0.1"
    port: int = 8091


@dataclass(frozen=True)
class DatabaseConfig:
    postgres_dsn: str


@dataclass(frozen=True)
class TmdbConfig:
    api_key: str
    timeout: float = 30.0


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com"
    timeout: float = 60.0
    reasoning_effort_chore: str = "low"
    reasoning_effort_mapping: str | None = None


@dataclass(frozen=True)
class NotificationConfig:
    base_url: str
    bearer_token: str
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class WorkerConfig:
    settle_seconds: int = 30
    scan_interval_seconds: int = 60
    auto_apply: bool = False
    two_stage: bool = False
    max_candidates: int = 5
    max_output_tokens: int = 2048
    allow_existing_dest: bool = False


@dataclass(frozen=True)
class WatchRootConfig:
    key: str
    input_root: Path
    output_root: Path


@dataclass(frozen=True)
class AppConfig:
    log_path: Path
    database: DatabaseConfig
    tmdb: TmdbConfig
    openai: OpenAISettings
    notifications: NotificationConfig | None
    notifications_warning: str | None
    api: ApiConfig
    worker: WorkerConfig
    watch_roots: tuple[WatchRootConfig, ...]


def default_config_path() -> Path:
    raw = os.getenv("ANINAMER_CONFIG_PATH", "").strip()
    if raw:
        return Path(raw)
    return Path("config.toml")


def load_tmdb_settings_from_env_or_config(path: Path | None = None) -> TmdbConfig:
    api_key = _read_optional_env("TMDB_API_KEY")
    timeout = _read_optional_float_env("TMDB_TIMEOUT")

    if api_key is not None:
        return TmdbConfig(api_key=api_key, timeout=timeout or 30.0)

    try:
        config = load_config(path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "TMDB settings require TMDB_API_KEY or config.toml [tmdb]"
        ) from exc

    return TmdbConfig(api_key=config.tmdb.api_key, timeout=timeout or config.tmdb.timeout)


def load_openai_settings_from_env_or_config(
    path: Path | None = None,
) -> OpenAISettings:
    api_key = _read_optional_env("OPENAI_API_KEY")
    model = _read_optional_env("OPENAI_MODEL")
    base_url = _read_optional_env("OPENAI_BASE_URL")
    timeout = _read_optional_float_env("OPENAI_TIMEOUT")
    reasoning_effort_chore = _read_optional_env("OPENAI_REASONING_EFFORT_CHORE")
    reasoning_effort_mapping = _read_optional_env("OPENAI_REASONING_EFFORT_MAPPING")

    if api_key is not None and model is not None:
        return OpenAISettings(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.openai.com",
            timeout=timeout or 60.0,
            reasoning_effort_chore=reasoning_effort_chore or "low",
            reasoning_effort_mapping=reasoning_effort_mapping,
        )

    try:
        config = load_config(path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(
            "OpenAI settings require OPENAI_API_KEY and OPENAI_MODEL or config.toml [openai]"
        ) from exc

    return OpenAISettings(
        api_key=api_key or config.openai.api_key,
        model=model or config.openai.model,
        base_url=base_url or config.openai.base_url,
        timeout=timeout or config.openai.timeout,
        reasoning_effort_chore=(
            reasoning_effort_chore or config.openai.reasoning_effort_chore or "low"
        ),
        reasoning_effort_mapping=(
            reasoning_effort_mapping or config.openai.reasoning_effort_mapping
        ),
    )


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or default_config_path()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise ValueError("config must be a TOML object")

    log_path_raw = data.get("log_path", "./logs")
    if not isinstance(log_path_raw, str) or not log_path_raw.strip():
        raise ValueError("log_path must be a non-empty string")
    log_path = Path(log_path_raw)

    database_data = data.get("database")
    if not isinstance(database_data, dict):
        raise ValueError("[database] is required")
    postgres_dsn = database_data.get("postgres_dsn")
    if not isinstance(postgres_dsn, str) or not postgres_dsn.strip():
        raise ValueError("database.postgres_dsn is required")

    tmdb_data = data.get("tmdb")
    if not isinstance(tmdb_data, dict):
        raise ValueError("[tmdb] is required")
    tmdb_api_key = _require_non_empty_string(tmdb_data.get("api_key"), "tmdb.api_key")
    tmdb_timeout = _require_positive_float(tmdb_data.get("timeout", 30.0), "tmdb.timeout")

    openai_data = data.get("openai")
    if not isinstance(openai_data, dict):
        raise ValueError("[openai] is required")
    openai_api_key = _require_non_empty_string(
        openai_data.get("api_key"),
        "openai.api_key",
    )
    openai_model = _require_non_empty_string(openai_data.get("model"), "openai.model")
    openai_base_url = _require_non_empty_string(
        openai_data.get("base_url", "https://api.openai.com"),
        "openai.base_url",
    )
    openai_timeout = _require_positive_float(
        openai_data.get("timeout", 60.0),
        "openai.timeout",
    )
    reasoning_effort_chore = _require_optional_non_empty_string(
        openai_data.get("reasoning_effort_chore", "low"),
        "openai.reasoning_effort_chore",
    )
    reasoning_effort_mapping = _require_optional_non_empty_string(
        openai_data.get("reasoning_effort_mapping"),
        "openai.reasoning_effort_mapping",
    )

    notifications: NotificationConfig | None = None
    notifications_warning: str | None = None
    notifications_data = data.get("notifications")
    if notifications_data is None:
        notifications_warning = "notifications disabled: [notifications] is missing"
    elif not isinstance(notifications_data, dict):
        raise ValueError("[notifications] must be an object")
    else:
        raw_base_url = notifications_data.get("base_url")
        raw_bearer_token = notifications_data.get("bearer_token")
        if (
            isinstance(raw_base_url, str)
            and raw_base_url.strip()
            and isinstance(raw_bearer_token, str)
            and raw_bearer_token.strip()
        ):
            notifications = NotificationConfig(
                base_url=raw_base_url.strip(),
                bearer_token=raw_bearer_token.strip(),
                timeout_seconds=_require_positive_float(
                    notifications_data.get("timeout_seconds", 10.0),
                    "notifications.timeout_seconds",
                ),
            )
        else:
            notifications_warning = (
                "notifications disabled: notifications.base_url and "
                "notifications.bearer_token are required"
            )

    api_data = data.get("api")
    if not isinstance(api_data, dict):
        raise ValueError("[api] is required")
    token = api_data.get("token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("api.token is required")
    bind = api_data.get("bind", "127.0.0.1")
    if not isinstance(bind, str) or not bind.strip():
        raise ValueError("api.bind must be a non-empty string")
    port = api_data.get("port", 8091)
    if isinstance(port, bool) or not isinstance(port, int) or port <= 0:
        raise ValueError("api.port must be a positive integer")

    worker_data = data.get("worker", {})
    if not isinstance(worker_data, dict):
        raise ValueError("[worker] must be an object")
    worker = WorkerConfig(
        settle_seconds=_require_non_negative_int(
            worker_data.get("settle_seconds", 30), "worker.settle_seconds"
        ),
        scan_interval_seconds=_require_non_negative_int(
            worker_data.get("scan_interval_seconds", 60),
            "worker.scan_interval_seconds",
        ),
        auto_apply=_require_bool(worker_data.get("auto_apply", False), "worker.auto_apply"),
        two_stage=_require_bool(worker_data.get("two_stage", False), "worker.two_stage"),
        max_candidates=_require_positive_int(
            worker_data.get("max_candidates", 5), "worker.max_candidates"
        ),
        max_output_tokens=_require_positive_int(
            worker_data.get("max_output_tokens", 2048),
            "worker.max_output_tokens",
        ),
        allow_existing_dest=_require_bool(
            worker_data.get("allow_existing_dest", False),
            "worker.allow_existing_dest",
        ),
    )

    watch_root_data = data.get("watch_roots")
    if not isinstance(watch_root_data, list) or not watch_root_data:
        raise ValueError("watch_roots must be a non-empty array of tables")
    watch_roots: list[WatchRootConfig] = []
    seen_keys: set[str] = set()
    for index, entry in enumerate(watch_root_data):
        if not isinstance(entry, dict):
            raise ValueError(f"watch_roots[{index}] must be an object")
        key = entry.get("key")
        input_root = entry.get("input_root")
        output_root = entry.get("output_root")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"watch_roots[{index}].key must be a non-empty string")
        if key in seen_keys:
            raise ValueError(f"watch root key {key} is duplicated")
        if not isinstance(input_root, str) or not input_root.strip():
            raise ValueError(
                f"watch_roots[{index}].input_root must be a non-empty string"
            )
        if not isinstance(output_root, str) or not output_root.strip():
            raise ValueError(
                f"watch_roots[{index}].output_root must be a non-empty string"
            )
        seen_keys.add(key)
        watch_roots.append(
            WatchRootConfig(
                key=key,
                input_root=Path(input_root),
                output_root=Path(output_root),
            )
        )

    return AppConfig(
        log_path=log_path,
        database=DatabaseConfig(postgres_dsn=postgres_dsn.strip()),
        tmdb=TmdbConfig(api_key=tmdb_api_key, timeout=tmdb_timeout),
        openai=OpenAISettings(
            api_key=openai_api_key,
            model=openai_model,
            base_url=openai_base_url,
            timeout=openai_timeout,
            reasoning_effort_chore=reasoning_effort_chore or "low",
            reasoning_effort_mapping=reasoning_effort_mapping,
        ),
        notifications=notifications,
        notifications_warning=notifications_warning,
        api=ApiConfig(token=token.strip(), bind=bind.strip(), port=port),
        worker=worker,
        watch_roots=tuple(watch_roots),
    )


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be bool")
    return value


def _require_non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _require_optional_non_empty_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string or null")
    return value.strip()


def _require_positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{label} must be a positive number")
    return float(value)


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _read_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return cleaned


def _read_optional_float_env(name: str) -> float | None:
    value = _read_optional_env(name)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed
