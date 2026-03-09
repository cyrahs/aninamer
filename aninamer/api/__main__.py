from __future__ import annotations

import uvicorn

from aninamer.config import load_config
from aninamer.store import RuntimeStore
from aninamer.worker import AninamerWorker

from .app import create_app


def main() -> None:
    config = load_config()
    store = RuntimeStore(config.database.postgres_dsn)
    worker = AninamerWorker(config, store)
    app = create_app(config, store=store, worker=worker)
    uvicorn.run(app, host=config.api.bind, port=config.api.port)


if __name__ == "__main__":
    main()
