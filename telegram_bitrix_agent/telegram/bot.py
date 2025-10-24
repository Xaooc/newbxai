"""Модуль запуска Telegram-бота на aiogram с вебхуком."""
from __future__ import annotations

import asyncio

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import CommandStart
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from ..assistant import AssistantOrchestrator
from ..config import Settings
from ..logging import configure_logging


class TelegramAssistantBot:
    """Настраивает aiogram-бота и связывает его с оркестратором."""

    def __init__(self, settings: Settings, orchestrator: AssistantOrchestrator) -> None:
        self._settings = settings
        self._orchestrator = orchestrator
        self._bot = Bot(token=settings.bot_token)
        self._dispatcher = Dispatcher()
        self._router = Router()
        self._setup_routes()
        self._dispatcher.include_router(self._router)

    @property
    def router(self) -> Router:
        """Возвращает основной роутер aiogram (используется в тестах)."""

        return self._router

    @property
    def dispatcher(self) -> Dispatcher:
        """Предоставляет доступ к диспетчеру."""

        return self._dispatcher

    def _setup_routes(self) -> None:
        @self._router.message(CommandStart())
        async def handle_start(message: types.Message) -> None:
            """Приветствие для новой сессии."""

            await message.answer("Привет! Я помогу с задачами Битрикс24.")

        @self._router.message()
        async def handle_message(message: types.Message) -> None:
            """Передаёт текст оркестратору."""

            if not message.text:
                await message.answer("Нужен текст запроса.")
                return
            chat_id = str(message.chat.id)
            response = await self._orchestrator.handle(chat_id, message.text)
            await message.answer(response)

        @self._router.callback_query()
        async def handle_callback(callback: types.CallbackQuery) -> None:
            """Обрабатывает нажатия на inline-кнопки."""

            data = callback.data or ""
            if not data:
                await callback.answer("Нет данных для обработки", show_alert=False)
                return
            if callback.message and callback.message.chat:
                chat_id = str(callback.message.chat.id)
            elif callback.from_user:
                chat_id = str(callback.from_user.id)
            else:
                chat_id = "unknown"
            response = await self._orchestrator.handle(chat_id, data)
            if callback.message:
                await callback.message.answer(response)
            else:
                await self._bot.send_message(chat_id, response)
            await callback.answer()

    async def on_startup(self, _: Dispatcher) -> None:
        """Устанавливает вебхук при запуске."""

        if self._settings.webhook_url:
            await self._bot.set_webhook(self._settings.webhook_url)

    async def on_shutdown(self, _: Dispatcher) -> None:
        """Удаляет вебхук при остановке."""

        if self._settings.webhook_url:
            await self._bot.delete_webhook()

    def create_web_app(self) -> web.Application:
        """Создаёт aiohttp-приложение для вебхука."""

        app = web.Application()
        handler = SimpleRequestHandler(dispatcher=self._dispatcher, bot=self._bot)
        handler.register(app, path="/")
        setup_application(app, self._dispatcher, on_startup=[self.on_startup], on_shutdown=[self.on_shutdown])
        return app


async def run_webhook(settings: Settings, orchestrator: AssistantOrchestrator) -> None:
    """Точка входа для запуска aiohttp-сервера."""

    configure_logging()
    bot = TelegramAssistantBot(settings, orchestrator)
    app = bot.create_web_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=8000)
    await site.start()
    while True:
        await asyncio.sleep(3600)
