from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

DEFAULT_LOG_RETENTION_DAYS = 7
DIAGNOSTIC_LOGGER_NAME = "aninamer.diagnostics"


class _OnlyLoggerFilter(logging.Filter):
    def __init__(self, logger_name: str) -> None:
        super().__init__()
        self._logger_name = logger_name
        self._prefix = f"{logger_name}."

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self._logger_name or record.name.startswith(self._prefix)


class _ExcludeLoggerFilter(logging.Filter):
    def __init__(self, logger_name: str) -> None:
        super().__init__()
        self._logger_name = logger_name
        self._prefix = f"{logger_name}."

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != self._logger_name and not record.name.startswith(
            self._prefix
        )


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
    console_handler.addFilter(_ExcludeLoggerFilter(DIAGNOSTIC_LOGGER_NAME))
    console_handler._aninamer = True  # type: ignore[attr-defined]
    root.addHandler(console_handler)

    file_handler = TimedRotatingFileHandler(
        log_dir / "aninamer.log",
        when="midnight",
        backupCount=DEFAULT_LOG_RETENTION_DAYS,
        encoding="utf-8",
        utc=True,
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_ExcludeLoggerFilter(DIAGNOSTIC_LOGGER_NAME))
    file_handler._aninamer = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)

    diagnostics_handler = TimedRotatingFileHandler(
        log_dir / "diagnostics.log",
        when="midnight",
        backupCount=DEFAULT_LOG_RETENTION_DAYS,
        encoding="utf-8",
        utc=True,
    )
    diagnostics_handler.setLevel(numeric_level)
    diagnostics_handler.setFormatter(formatter)
    diagnostics_handler.addFilter(_OnlyLoggerFilter(DIAGNOSTIC_LOGGER_NAME))
    diagnostics_handler._aninamer = True  # type: ignore[attr-defined]
    root.addHandler(diagnostics_handler)
