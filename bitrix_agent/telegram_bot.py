"""Telegram-бот для автономного агента Bitrix24."""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from .agent import BitrixAutonomousAgent
from .config import BitrixConfig
from .environment import load_env_file


logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов для запуска бота."""

    parser = argparse.ArgumentParser(description="Telegram-бот автономного агента Bitrix24")
    parser.add_argument("--token", help="Токен Telegram-бота. По умолчанию используется TELEGRAM_BOT_TOKEN")
    parser.add_argument("--env-file", default=".env", help="Файл окружения с настройками Bitrix24")
    parser.add_argument(
        "--allowed-chat",
        action="append",
        type=int,
        help="ID чатов, которым разрешён доступ. Можно указать несколько флагов",
    )
    parser.add_argument("--verbose", action="store_true", help="Включить подробное логирование")
    return parser


def configure_logging(verbose: bool) -> None:
    """Настроить вывод логов."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


def chunk_message(message: str, limit: int = 4096) -> List[str]:
    """Разбить длинное сообщение на части, учитывая ограничения Telegram."""

    if limit <= 0:
        raise ValueError("Лимит должен быть положительным")
    if not message:
        return [""]
    return [message[i : i + limit] for i in range(0, len(message), limit)]


class TelegramIO:
    """Обработчик ввода-вывода, общающийся с пользователем через Telegram."""

    def __init__(self, sender: Callable[[str], None]) -> None:
        self._sender = sender
        self._answers: "queue.Queue[str]" = queue.Queue()
        self._waiting = threading.Event()

    def ask(self, question: str) -> str:
        """Отправить вопрос пользователю и дождаться ответа."""

        self._waiting.set()
        self._send(question)
        try:
            answer = self._answers.get()
        finally:
            self._waiting.clear()
        return answer

    def notify(self, message: str) -> None:
        """Отправить уведомление пользователю."""

        self._send(message)

    def supply_answer(self, answer: str) -> bool:
        """Передать ответ пользователя во внутреннюю очередь."""

        if not self._waiting.is_set():
            return False
        self._answers.put(answer)
        return True

    def is_waiting(self) -> bool:
        """Проверить, ожидается ли ответ от пользователя."""

        return self._waiting.is_set()

    def _send(self, message: str) -> None:
        for part in chunk_message(message):
            self._sender(part)


@dataclass
class TelegramAgentSession:
    """Состояние активной сессии агента в конкретном чате."""

    io: TelegramIO
    task: asyncio.Task[None]
    goal: str


class TelegramMessageDispatcher:
    """Передача сообщений в чат с учётом ограничений и потоков."""

    def __init__(self, application: Application, chat_id: int, loop: asyncio.AbstractEventLoop) -> None:
        self._application = application
        self._chat_id = chat_id
        self._loop = loop

    def send_text(self, message: str, *, parse_mode: ParseMode | None = None) -> None:
        """Отправить сообщение в чат, безопасно для сторонних потоков."""

        asyncio.run_coroutine_threadsafe(
            self._application.bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode=parse_mode,
            ),
            self._loop,
        )


class TelegramAgentBot:
    """Координатор работы агента через Telegram."""

    def __init__(
        self,
        *,
        token: str,
        config: BitrixConfig,
        allowed_chats: Optional[Iterable[int]] = None,
    ) -> None:
        self._token = token
        self._config = config
        self._allowed_chats = set(allowed_chats or [])
        self._sessions: Dict[int, TelegramAgentSession] = {}

    def run(self) -> None:
        """Запустить бота и начать обработку обновлений."""

        application = self._build_application()
        logger.info("Запуск Telegram-бота")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    def _build_application(self) -> Application:
        """Построить приложение Telegram с зарегистрированными хендлерами."""

        application = ApplicationBuilder().token(self._token).build()
        application.add_handler(CommandHandler("start", self._handle_start))
        application.add_handler(CommandHandler("help", self._handle_help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        return application

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Приветствие и краткая инструкция по использованию."""

        if update.effective_chat is None or update.message is None:
            return
        if not self._is_chat_allowed(update.effective_chat.id):
            await self._deny_access(update)
            return
        await update.message.reply_text(
            "Привет! Отправьте цель в одном сообщении, и агент выполнит задачу в Bitrix24."
        )

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Показать справку пользователю."""

        if update.effective_chat is None or update.message is None:
            return
        if not self._is_chat_allowed(update.effective_chat.id):
            await self._deny_access(update)
            return
        await update.message.reply_text(
            "Просто опишите цель. При необходимости бот задаст уточняющие вопросы."
        )

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработать произвольное текстовое сообщение пользователя."""

        if update.effective_chat is None or update.message is None or update.message.text is None:
            return
        chat_id = update.effective_chat.id
        if not self._is_chat_allowed(chat_id):
            await self._deny_access(update)
            return

        text = update.message.text.strip()
        if not text:
            await update.message.reply_text("Сообщение пустое. Сформулируйте цель полностью.")
            return

        session = self._sessions.get(chat_id)
        if session:
            if session.io.supply_answer(text):
                await update.message.reply_text("Спасибо, продолжаю работу.")
                return
            if not session.task.done():
                await update.message.reply_text(
                    "Агент ещё работает над предыдущей целью. Дождитесь завершения или ответьте на вопрос."
                )
                return

        await self._start_new_session(chat_id, text, context)

    async def _start_new_session(
        self,
        chat_id: int,
        text: str,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Создать новую сессию агента и запустить её в фоне."""

        application = context.application
        loop = asyncio.get_running_loop()
        dispatcher = TelegramMessageDispatcher(application, chat_id, loop)

        def sender(message: str) -> None:
            dispatcher.send_text(message)

        io = TelegramIO(sender)

        async def worker() -> None:
            try:
                dispatcher.send_text("Цель принята, начинаю работу.")
                agent = BitrixAutonomousAgent(config=self._config, io=io)
                result = await asyncio.to_thread(agent.run, text)
                await self._announce_results(result, dispatcher)
            except Exception as exc:  # noqa: BLE001 - хотим показать любую ошибку пользователю
                logger.exception("Ошибка при выполнении агента")
                await application.bot.send_message(chat_id=chat_id, text=f"Ошибка: {exc}")
            finally:
                self._sessions.pop(chat_id, None)

        task = asyncio.create_task(worker())
        self._sessions[chat_id] = TelegramAgentSession(io=io, task=task, goal=text)

    async def _announce_results(self, result: Dict[str, object], dispatcher: TelegramMessageDispatcher) -> None:
        """Отправить пользователю финальный отчёт и подробности."""

        report = result.get("report") if isinstance(result, dict) else None
        if isinstance(report, str) and report.strip():
            dispatcher.send_text(f"Отчёт агента:\n{report}")
        else:
            dispatcher.send_text("Работа завершена, отчёт не сформирован.")

        if isinstance(result, dict):
            details = result.get("results")
        else:
            details = None

        if details:
            try:
                summary = json.dumps(details, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                summary = str(details)
            escaped = html.escape(summary)
            dispatcher.send_text(
                f"Подробности выполнения:\n<pre>{escaped}</pre>",
                parse_mode=ParseMode.HTML,
            )

    def _is_chat_allowed(self, chat_id: int) -> bool:
        """Проверить, разрешён ли доступ для заданного чата."""

        return not self._allowed_chats or chat_id in self._allowed_chats

    async def _deny_access(self, update: Update) -> None:
        """Ответить пользователю, если чат не входит в белый список."""

        if update.message is not None:
            await update.message.reply_text("Этот бот доступен только для разрешённых чатов.")


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI-точка входа для запуска Telegram-бота."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.env_file:
        try:
            load_env_file(args.env_file)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    configure_logging(args.verbose)

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Не найден токен Telegram-бота. Укажите --token или переменную окружения TELEGRAM_BOT_TOKEN."
        )

    try:
        config = BitrixConfig.from_env()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    bot = TelegramAgentBot(token=token, config=config, allowed_chats=args.allowed_chat)
    bot.run()


if __name__ == "__main__":  # pragma: no cover - запуск модуля напрямую
    main()
