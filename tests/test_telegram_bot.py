from __future__ import annotations

from types import SimpleNamespace

import asyncio
import pytest

pytest.importorskip("aiogram")

from telegram_bitrix_agent.config import Settings
from telegram_bitrix_agent.telegram.bot import TelegramAssistantBot


class DummyOrchestrator:
    """Фиктивный оркестратор для проверки обработчиков Telegram."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def handle(self, chat_id: str, message: str) -> str:  # noqa: D401
        self.calls.append((chat_id, message))
        return "ok"


class DummyMessage:
    """Минимальный объект сообщения для тестов callback_query."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(id=123)
        self.responses: list[str] = []

    async def answer(self, text: str) -> None:  # noqa: D401
        self.responses.append(text)


class DummyCallback:
    """Заглушка callback_query без Telegram API."""

    def __init__(self, data: str, message: DummyMessage | None = None) -> None:
        self.data = data
        self.message = message
        self.from_user = SimpleNamespace(id=321)
        self._answered = False

    async def answer(self, text: str | None = None, show_alert: bool = False) -> None:  # noqa: D401
        self._answered = True
        self.alert_text = text
        self.show_alert = show_alert


def test_callback_query_triggers_orchestrator() -> None:
    async def scenario() -> None:
        settings = Settings(
            bot_token="token",
            openai_api_key="key",
            bitrix_webhook="https://example.com",
        )
        orchestrator = DummyOrchestrator()
        bot = TelegramAssistantBot(settings, orchestrator)
        assert bot.router.callback_query.handlers, "Должен быть зарегистрирован обработчик callback_query"

        handler = bot.router.callback_query.handlers[0].callback
        message = DummyMessage()
        callback = DummyCallback("test", message=message)
        await handler(callback)

        assert orchestrator.calls == [("123", "test")]
        assert message.responses == ["ok"]
        assert callback._answered is True

    asyncio.run(scenario())
