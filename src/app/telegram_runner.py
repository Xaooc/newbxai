"""Утилиты для подготовки и запуска Telegram-бота."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from src.adapters.telegram_bot import TelegramBotAdapter, TelegramBotConfig

from src.config import load_env_file
from src.logging.logger import build_interaction_logger
from src.orchestrator.agent import Orchestrator, OrchestratorSettings
from src.orchestrator.model_client import ModelClientError, build_default_model_client
from src.state.manager import AgentStateManager

logger = logging.getLogger(__name__)


class TelegramTokenMissingError(RuntimeError):
    """Выбрасывается, если токен Telegram-бота не найден."""


def build_orchestrator(mode: str, storage_dir: Path, log_dir: Path) -> Orchestrator:
    """Создаёт экземпляр оркестратора с инициализацией хранилищ."""

    storage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_manager = AgentStateManager(storage_dir=storage_dir)
    interaction_logger = build_interaction_logger(log_dir=log_dir)
    settings = OrchestratorSettings(mode=mode)

    model_client = None
    try:
        model_client = build_default_model_client(settings.model_name)
    except ModelClientError as exc:  # pragma: no cover - ветка зависит от внешнего API
        logger.warning(
            "ChatGPT недоступен: %s. Оркестратор запустится в режиме заглушки.",
            exc,
        )

    return Orchestrator(
        state_manager=state_manager,
        interaction_logger=interaction_logger,
        settings=settings,
        model_client=model_client,
    )


def create_telegram_adapter(env_path: str | Path | None = ".env") -> "TelegramBotAdapter":
    """Создаёт Telegram-адаптер, подготавливая окружение при необходимости."""

    orchestrator, config = _prepare_bot_dependencies(env_path)
    from src.adapters.telegram_bot import TelegramBotAdapter  # локальный импорт

    return TelegramBotAdapter(orchestrator=orchestrator, config=config)


def launch_telegram_bot(env_path: str | Path | None = ".env") -> int:
    """Запускает polling-бота и возвращает код завершения процесса."""

    try:
        adapter = create_telegram_adapter(env_path)
    except TelegramTokenMissingError:
        logger.error(
            "Переменная TELEGRAM_BOT_TOKEN не задана. Укажите токен бота в .env или окружении."
        )
        return 1

    adapter.run()
    return 0


def _prepare_bot_dependencies(env_path: str | Path | None) -> Tuple[Orchestrator, "TelegramBotConfig"]:
    """Загружает окружение и возвращает оркестратор вместе с конфигурацией бота."""

    if env_path:
        load_env_file(env_path)

    config = _build_config_from_env()
    orchestrator = build_orchestrator(config.mode, config.state_dir, config.log_dir)
    return orchestrator, config


def _build_config_from_env() -> "TelegramBotConfig":
    """Формирует `TelegramBotConfig` на основе переменных окружения."""

    from src.adapters.telegram_bot import TelegramBotConfig, parse_allowed_chats  # ленивый импорт

    mode = os.getenv("AGENT_MODE", "shadow")
    state_dir = Path(os.getenv("STATE_DIR", "./data/state"))
    log_dir = Path(os.getenv("LOG_DIR", "./data/logs"))
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise TelegramTokenMissingError()

    allowed_chats = parse_allowed_chats(os.getenv("TELEGRAM_ALLOWED_CHATS"))
    poll_interval = _read_poll_interval(os.getenv("TELEGRAM_POLL_INTERVAL"))

    error_chat_id = _read_error_chat_id(os.getenv("TELEGRAM_ERROR_CHAT_ID"))

    worker_threads = _read_worker_threads(os.getenv("TELEGRAM_WORKER_THREADS"))

    return TelegramBotConfig(
        token=token,
        mode=mode,
        state_dir=state_dir,
        log_dir=log_dir,
        allowed_chats=allowed_chats,
        poll_interval=poll_interval,
        error_chat_id=error_chat_id,
        worker_threads=worker_threads,
    )


def _read_poll_interval(raw_value: Optional[str]) -> float:
    """Безопасно читает интервал опроса Telegram."""

    if not raw_value:
        return 0.0
    candidate = raw_value.strip()
    if not candidate:
        return 0.0
    try:
        value = float(candidate)
    except ValueError:
        logger.warning(
            "Некорректное значение TELEGRAM_POLL_INTERVAL=%s. Используем значение по умолчанию (0.0).",
            raw_value,
        )
        return 0.0
    if value < 0:
        logger.warning(
            "Отрицательное значение TELEGRAM_POLL_INTERVAL=%s не поддерживается. Используем 0.0.",
            raw_value,
        )
        return 0.0
    return value


def _read_error_chat_id(raw_value: Optional[str]) -> Optional[int]:
    """Преобразует значение переменной окружения в идентификатор чата."""

    if not raw_value:
        return None
    candidate = raw_value.strip()
    if not candidate:
        return None
    try:
        return int(candidate)
    except ValueError:
        logger.warning(
            "Некорректное значение TELEGRAM_ERROR_CHAT_ID=%s. Уведомления об ошибках отключены.",
            raw_value,
        )
        return None


def _read_worker_threads(raw_value: Optional[str]) -> int:
    """Читает размер пула потоков для обработки сообщений Telegram."""

    default_value = 8
    if not raw_value:
        return default_value
    candidate = raw_value.strip()
    if not candidate:
        return default_value
    try:
        value = int(candidate)
    except ValueError:
        logger.warning(
            "Некорректное значение TELEGRAM_WORKER_THREADS=%s. Используем значение по умолчанию (%s).",
            raw_value,
            default_value,
        )
        return default_value
    if value < 1:
        logger.warning(
            "Значение TELEGRAM_WORKER_THREADS=%s должно быть положительным. Используем %s.",
            raw_value,
            default_value,
        )
        return default_value
    return value


__all__ = [
    "TelegramTokenMissingError",
    "build_orchestrator",
    "create_telegram_adapter",
    "launch_telegram_bot",
]
