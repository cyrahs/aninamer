from __future__ import annotations

import signal
import threading

import uvicorn

from aninamer.api import create_app
from aninamer.config import AppConfig, load_config
from aninamer.logging_utils import configure_logging
from aninamer.store import RuntimeStore
from aninamer.worker import AninamerWorker


def run_service(config: AppConfig | None = None) -> None:
    resolved = config or load_config()
    configure_logging()
    store = RuntimeStore(resolved.database.postgres_dsn)
    worker = AninamerWorker(resolved, store)
    app = create_app(resolved, store=store, worker=worker)

    shutdown_requested = threading.Event()

    def _handle_signal(signum: int, _frame: object | None) -> None:
        shutdown_requested.set()

    original_sigterm = signal.signal(signal.SIGTERM, _handle_signal)
    original_sigint = signal.signal(signal.SIGINT, _handle_signal)
    worker_thread = threading.Thread(
        target=worker.run_forever,
        args=(shutdown_requested.is_set,),
        name="aninamer-worker",
        daemon=True,
    )
    worker_thread.start()
    try:
        uvicorn.run(app, host=resolved.api.bind, port=resolved.api.port)
    finally:
        shutdown_requested.set()
        worker_thread.join(timeout=5.0)
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)
