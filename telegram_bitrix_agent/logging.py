"""Настройка структурированного логирования."""
from __future__ import annotations

import logging
from typing import Any, Dict

try:  # pragma: no cover - structlog может отсутствовать в окружении тестов
    import structlog
except ImportError:  # pragma: no cover
    structlog = None  # type: ignore


class PlainLogger:
    """Простая обёртка, имитирующая интерфейс structlog.BoundLogger."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def bind(self, **event: Any) -> "PlainLogger":
        clone = PlainLogger(self._logger.name)
        clone._context = {**self._context, **event}
        return clone

    def _log(self, level: int, event: str, **payload: Any) -> None:
        merged = {**self._context, **payload}
        message = f"{event} | {merged}" if merged else event
        self._logger.log(level, message)

    def info(self, event: str, **payload: Any) -> None:
        self._log(logging.INFO, event, **payload)

    def warning(self, event: str, **payload: Any) -> None:
        self._log(logging.WARNING, event, **payload)

    def error(self, event: str, **payload: Any) -> None:
        self._log(logging.ERROR, event, **payload)


def get_logger(name: str) -> Any:
    """Возвращает логгер с интерфейсом BoundLogger."""

    if structlog is None:  # pragma: no cover
        return PlainLogger(name)
    return structlog.get_logger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Инициализирует structlog и стандартный логгер."""

    if structlog is None:  # pragma: no cover
        logging.basicConfig(level=level)
        return
    timestamper = structlog.processors.TimeStamper(fmt="ISO")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            timestamper,
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("message"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(ensure_ascii=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=level)


def bind_event(logger: Any, **event: Any) -> Any:
    """Возвращает логгер с добавленным контекстом."""

    return logger.bind(**event)
