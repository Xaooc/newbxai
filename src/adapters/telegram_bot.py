"""Telegram-адаптер для взаимодействия с оркестратором."""

from __future__ import annotations

import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from src.orchestrator.agent import Orchestrator
from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096


@dataclass
class TelegramBotConfig:
    """Настройки Telegram-бота."""

    token: str
    mode: str = "shadow"
    state_dir: Path = field(default_factory=lambda: Path("./data/state"))
    log_dir: Path = field(default_factory=lambda: Path("./data/logs"))
    allowed_chats: Optional[Set[int]] = None
    poll_interval: float = 0.0
    error_chat_id: Optional[int] = None
    worker_threads: int = 8

    def is_chat_allowed(self, chat_id: int) -> bool:
        """Возвращает признак, что чат разрешён."""

        if not self.allowed_chats:
            return True
        return chat_id in self.allowed_chats


class TelegramBotAdapter:
    """Интеграция Telegram с оркестратором."""

    def __init__(self, orchestrator: Orchestrator, config: TelegramBotConfig) -> None:
        self._orchestrator = orchestrator
        self._config = config
        workers = max(1, config.worker_threads)
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="telegram-worker",
        )

    def run(self) -> None:
        """Запускает polling-бота."""

        application = ApplicationBuilder().token(self._config.token).build()

        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(MessageHandler(filters.COMMAND, self._handle_unknown_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
        application.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, self._handle_non_text_message))
        application.add_error_handler(self._handle_error)

        logger.info(
            "Старт Telegram-бота в режиме %s (state_dir=%s, log_dir=%s)",
            self._config.mode,
            self._config.state_dir,
            self._config.log_dir,
        )
        try:
            application.run_polling(poll_interval=self._config.poll_interval)
        finally:
            self._executor.shutdown(wait=True)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Отправляет приветствие при команде /start."""

        if not update.effective_chat or not update.effective_user or not update.message:
            return
        if not self._check_access(update):
            await self._notify_restricted(update)
            return
        message = (
            "Здравствуйте! Это AI-менеджер Bitrix24. "
            "Отправьте текстовую задачу, и я сформирую план действий. "
            f"Текущий режим безопасности: {self._config.mode}."
        )
        telegram_message = update.message
        await self._reply_with_text(telegram_message, message)

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает текстовые сообщения."""

        if not update.effective_chat or not update.effective_user or not update.message:
            return
        if not self._check_access(update):
            await self._notify_restricted(update)
            return
        telegram_message = update.message
        text = telegram_message.text or ""
        if not text.strip():
            await self._reply_with_text(
                telegram_message,
                "Пока поддерживаются только непустые текстовые сообщения.",
            )
            return
        user_id = str(update.effective_user.id)
        typing_done = asyncio.Event()
        typing_task = asyncio.create_task(
            self._typing_indicator_loop(context, update.effective_chat.id, typing_done)
        )
        try:
            response = await self._run_in_executor(
                self._orchestrator.process_message,
                user_id,
                text,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Ошибка обработки сообщения от пользователя %s", user_id)
            await self._reply_with_text(
                telegram_message,
                "Произошла ошибка при обработке запроса. Попробуйте повторить позже или обратитесь к администратору.",
            )
            await self._send_error_notification(
                context,
                f"⚠️ Ошибка при обработке сообщения пользователя {user_id}: {exc}",
            )
            return
        finally:
            typing_done.set()
            try:
                await typing_task
            except Exception:  # noqa: BLE001 - ошибки индикации не должны прерывать логику
                logger.debug("Ошибка при отправке статуса набора текста", exc_info=True)

        await self._reply_with_text(telegram_message, response)

    async def _handle_non_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Сообщает пользователю о поддержке только текстового формата."""

        if not update.message or not update.effective_chat:
            return
        if not self._check_access(update):
            await self._notify_restricted(update)
            return
        await self._reply_with_text(
            update.message,
            "Сейчас могу работать только с текстом. Пожалуйста, опишите задачу словами и отправьте сообщение повторно.",
        )

    async def _handle_unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Сообщает об отсутствии поддержки произвольных команд."""

        if not update.message or not update.effective_chat:
            return
        if not self._check_access(update):
            await self._notify_restricted(update)
            return
        await self._reply_with_text(
            update.message,
            "Команда не поддерживается. Отправьте текстовое сообщение с задачей.",
        )

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Логирует глобальные ошибки Telegram-бота."""

        logger.exception("Исключение в Telegram-боте", exc_info=context.error)
        await self._send_error_notification(
            context,
            "⚠️ Telegram-бот столкнулся с ошибкой. Проверьте логи для деталей.",
        )

    def _check_access(self, update: Update) -> bool:
        """Проверяет, разрешён ли чат для работы с ботом."""

        if not update.effective_chat:
            return False
        chat_id = update.effective_chat.id
        if self._config.is_chat_allowed(chat_id):
            return True
        logger.warning("Сообщение из запрещённого чата %s отклонено", chat_id)
        return False

    async def _notify_restricted(self, update: Update) -> None:
        """Отправляет сообщение о запрете доступа."""

        if update.message:
            await self._reply_with_text(
                update.message,
                "Доступ к этому боту ограничен. Обратитесь к администратору.",
            )

    async def _reply_with_text(self, telegram_message: Message, text: str) -> None:
        """Отправляет текст, разбивая его на допустимые Telegram фрагменты."""

        for chunk in split_message_for_telegram(text, TELEGRAM_MESSAGE_LIMIT):
            await telegram_message.reply_text(chunk)

    async def _send_error_notification(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        message: str,
    ) -> None:
        """Отправляет уведомление об ошибке в административный чат, если он настроен."""

        chat_id = self._config.error_chat_id
        if not chat_id:
            return
        application = getattr(context, "application", None)
        if not application or not getattr(application, "bot", None):
            return
        try:
            await application.bot.send_message(chat_id=chat_id, text=message)
        except Exception:  # noqa: BLE001 - уведомление не должно прерывать работу бота
            logger.warning("Не удалось отправить уведомление об ошибке в чат %s", chat_id)

    async def _typing_indicator_loop(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        done: asyncio.Event,
    ) -> None:
        """Периодически отправляет статус "typing" пока запрос обрабатывается."""

        bot = getattr(context, "bot", None)
        if bot is None:
            application = getattr(context, "application", None)
            bot = getattr(application, "bot", None) if application else None
        if bot is None:
            return

        try:
            while True:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                try:
                    await asyncio.wait_for(done.wait(), timeout=4)
                except asyncio.TimeoutError:
                    continue
                break
        except Exception:  # noqa: BLE001 - сбой индикации не критичен
            logger.debug("Не удалось отправить индикацию набора текста", exc_info=True)


    async def _run_in_executor(self, func, *args):  # type: ignore[no-untyped-def]
        """Выполняет синхронную функцию в пуле потоков."""

        loop = asyncio.get_running_loop()
        callback = functools.partial(func, *args)
        return await loop.run_in_executor(self._executor, callback)


def parse_allowed_chats(values: Optional[str]) -> Optional[Set[int]]:
    """Преобразует строку идентификаторов чатов в множество."""

    if not values:
        return None
    allowed: Set[int] = set()
    for raw in values.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        try:
            allowed.add(int(candidate))
        except ValueError:
            logger.warning("Игнорируем некорректный идентификатор чата: %s", candidate)
    return allowed or None


def split_message_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    """Делит текст на последовательность сообщений, укладывающихся в лимит Telegram."""

    if limit <= 0:
        return [text]

    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in text.splitlines(keepends=True):
        remaining_line = line
        while len(remaining_line) > limit:
            if current:
                chunks.append("".join(current))
                current = []
                current_len = 0
            chunks.append(remaining_line[:limit])
            remaining_line = remaining_line[limit:]
        if not remaining_line:
            continue

        if current_len + len(remaining_line) > limit:
            if current:
                chunks.append("".join(current))
            current = []
            current_len = 0

        current.append(remaining_line)
        current_len += len(remaining_line)

    if current:
        chunks.append("".join(current))

    # При совпадении длины строки с лимитом `splitlines` не добавит пустой фрагмент, поэтому
    # результат всегда непустой. Однако для надёжности возвращаем хотя бы одно пустое сообщение.
    return chunks or [""]
