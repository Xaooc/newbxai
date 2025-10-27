from __future__ import annotations

from types import SimpleNamespace

import asyncio
import pytest

pytest.importorskip("aiogram")

from telegram_bitrix_agent.config import Settings
from telegram_bitrix_agent.telegram.bot import TelegramAssistantBot


class DummyOrchestrator:
    """Фиктивный оркестратор для проверки обработчиков Telegram."""

    def __init__(self, response: str = "ok") -> None:
        self.calls: list[tuple[str, str]] = []
        self.response = response

    async def handle(self, chat_id: str, message: str) -> str:  # noqa: D401
        self.calls.append((chat_id, message))
        return self.response


class DummyMessage:
    """Минимальный объект сообщения для тестов callback_query."""

    def __init__(self, text: str | None = None) -> None:
        self.text = text
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
            bot_token="123456:TEST",
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


def test_long_message_split_into_chunks() -> None:
    async def scenario() -> None:
        long_response = "A" * 5000
        settings = Settings(
            bot_token="123456:TEST",
            openai_api_key="key",
            bitrix_webhook="https://example.com",
        )
        orchestrator = DummyOrchestrator(response=long_response)
        bot = TelegramAssistantBot(settings, orchestrator)
        handlers = [h.callback for h in bot.router.message.handlers]
        handler = next(h for h in handlers if h.__name__ == "handle_message")

        message = DummyMessage(text="привет")
        await handler(message)

        assert orchestrator.calls == [("123", "привет")]
        assert "".join(message.responses) == long_response
        assert all(len(chunk) <= 4096 for chunk in message.responses)
        assert len(message.responses) == 2

    asyncio.run(scenario())
