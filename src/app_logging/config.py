"""Utilities to configure application logging outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

DEFAULT_LOG_FILENAME = "app.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(
    *,
    log_dir: Path,
    level: int = logging.INFO,
    filename: str = DEFAULT_LOG_FILENAME,
    propagate_existing: bool = True,
) -> Path:
    """Configure root logging to mirror messages into console and file handlers."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT)

    _ensure_console_handler(root_logger, level=level, formatter=formatter)
    _ensure_file_handler(root_logger, log_path=log_path, level=level, formatter=formatter)

    root_logger.propagate = propagate_existing
    return log_path


def _ensure_console_handler(logger: logging.Logger, *, level: int, formatter: logging.Formatter) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def _ensure_file_handler(
    logger: logging.Logger,
    *,
    log_path: Path,
    level: int,
    formatter: logging.Formatter,
) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path:
            handler.setLevel(level)
            handler.setFormatter(formatter)
            return

    file_handler: Optional[logging.FileHandler]
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
