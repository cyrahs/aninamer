from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

import psycopg
import pytest

from aninamer.config import (
    ApiConfig,
    AppConfig,
    DatabaseConfig,
    NotificationConfig,
    OpenAISettings,
    TmdbConfig,
    WatchRootConfig,
    WorkerConfig,
    load_openai_settings_from_env_or_config,
    load_tmdb_settings_from_env_or_config,
)
from aninamer.errors import OpenAIError
from aninamer.llm_client import ChatMessage
from aninamer.openai_llm_client import openai_llm_for_tmdb_id_from_settings
from aninamer.store import RuntimeStore


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _wait_for_postgres(dsn: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return
        except psycopg.Error:
            time.sleep(0.5)
    raise RuntimeError("postgres test container did not become ready in time")


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    configured = os.getenv("ANINAMER_TEST_POSTGRES_DSN", "").strip()
    if configured:
        _wait_for_postgres(configured)
        return configured

    if not _docker_available():
        pytest.skip("postgres tests require ANINAMER_TEST_POSTGRES_DSN or docker")

    port = _free_port()
    container_name = f"aninamer-test-postgres-{uuid.uuid4().hex[:8]}"
    env = [
        "-e",
        "POSTGRES_USER=aninamer",
        "-e",
        "POSTGRES_PASSWORD=aninamer",
        "-e",
        "POSTGRES_DB=aninamer_test",
    ]
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{port}:5432",
            *env,
            "postgres:16-alpine",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"failed to start postgres docker container: {run.stderr.strip()}")

    dsn = f"postgresql://aninamer:aninamer@127.0.0.1:{port}/aninamer_test"
    try:
        _wait_for_postgres(dsn)
        yield dsn
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            capture_output=True,
            text=True,
        )


@pytest.fixture
def runtime_store(postgres_dsn: str) -> RuntimeStore:
    store = RuntimeStore(postgres_dsn)
    with psycopg.connect(postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "TRUNCATE TABLE notifications, job_artifacts, job_requests, jobs "
                "RESTART IDENTITY CASCADE"
            )
            cur.execute("UPDATE runtime_state SET last_scan_at = NULL WHERE id = 1")
    return RuntimeStore(postgres_dsn)


@pytest.fixture(scope="session")
def integration_openai_settings() -> OpenAISettings:
    try:
        settings = load_openai_settings_from_env_or_config()
    except ValueError as exc:
        pytest.skip(str(exc))
    try:
        llm = openai_llm_for_tmdb_id_from_settings(settings)
        llm.chat(
            [ChatMessage(role="user", content='Return {"tmdb": 1} only.')],
            max_output_tokens=4096,
        )
    except OpenAIError as exc:
        pytest.skip(f"OpenAI integration unavailable: {exc}")
    return settings


@pytest.fixture(scope="session")
def integration_tmdb_settings() -> TmdbConfig:
    try:
        return load_tmdb_settings_from_env_or_config()
    except ValueError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def app_config(tmp_path: Path, postgres_dsn: str) -> AppConfig:
    return AppConfig(
        log_path=tmp_path / "logs",
        database=DatabaseConfig(postgres_dsn=postgres_dsn),
        tmdb=TmdbConfig(api_key="tmdb-key"),
        openai=OpenAISettings(
            api_key="openai-key",
            model="gpt-test",
        ),
        notifications=None,
        notifications_warning="notifications disabled: [notifications] is missing",
        api=ApiConfig(token="secret"),
        worker=WorkerConfig(
            settle_seconds=0,
            scan_interval_seconds=1,
            auto_apply=False,
        ),
        watch_roots=(
            WatchRootConfig(
                key="downloads",
                input_root=tmp_path / "input",
                output_root=tmp_path / "output",
            ),
        ),
    )


@pytest.fixture
def app_config_with_notifications(app_config: AppConfig) -> AppConfig:
    return replace(
        app_config,
        notifications=NotificationConfig(
            base_url="https://notify.example.test",
            bearer_token="notify-token",
            timeout_seconds=10.0,
        ),
        notifications_warning=None,
    )
