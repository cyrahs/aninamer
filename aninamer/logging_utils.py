from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: str = "INFO", log_path: Path | str | None = None) -> None:
    level_name = level.upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    log_dir = Path(log_path) if log_path is not None else Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_aninamer", False):
            root.removeHandler(handler)
            handler.close()

    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler._aninamer = True  # type: ignore[attr-defined]
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / "aninamer.log", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler._aninamer = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
