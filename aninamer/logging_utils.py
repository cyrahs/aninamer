from __future__ import annotations

import logging

DIAGNOSTIC_LOGGER_NAME = "aninamer.diagnostics"


def configure_logging(level: str = "INFO") -> None:
    level_name = level.upper()
    numeric_level = getattr(logging, level_name, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

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
