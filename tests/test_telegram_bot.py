"""Тесты утилит Telegram-адаптера."""

from types import SimpleNamespace
from typing import Callable, List, Tuple

import pytest

pytest.importorskip("telegram")

import asyncio

from src.adapters.telegram_bot import (  # noqa: E402
    TELEGRAM_MESSAGE_LIMIT,
    TelegramBotAdapter,
    TelegramBotConfig,
    split_message_for_telegram,
)


def test_split_message_preserves_short_text() -> None:
    """Сообщение короче лимита не должно дробиться."""

    text = "Короткий ответ"
    chunks = split_message_for_telegram(text)

    assert chunks == [text]


def test_split_message_splits_by_lines() -> None:
    """Длинный текст разбивается по границам строк."""

    limit = 20
    text = "Первая строка\nВторая строка, чуть длиннее\nТретья"

    chunks = split_message_for_telegram(text, limit=limit)

    for chunk in chunks:
        assert len(chunk) <= limit

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_split_message_handles_overflow_line() -> None:
    """Строки длиннее лимита дробятся на несколько частей."""

    long_line = "А" * (TELEGRAM_MESSAGE_LIMIT + 10)
    chunks = split_message_for_telegram(long_line)

    assert len(chunks) == 2
    assert all(len(chunk) <= TELEGRAM_MESSAGE_LIMIT for chunk in chunks)
    assert "".join(chunks) == long_line


class FakeMessage:
    """Заглушка для telegram.Message."""

    def __init__(self, text: str | None, chat_id: int) -> None:
        self.text = text
        self.chat_id = chat_id
        self.replies: List[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeBot:
    """Заглушка Telegram-бота для уведомлений."""

    def __init__(self) -> None:
        self.sent: List[Tuple[int, str]] = []
        self.actions: List[Tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))


class FakeOrchestrator:
    """Заглушка оркестратора, имитирующая обработку сообщения."""

    def __init__(self, *, should_fail: bool = False, alert_message: str | None = None) -> None:
        self.should_fail = should_fail
        self.alert_message = alert_message
        self.calls: List[Tuple[str, str]] = []
        self._alert_handler: Callable[[str], None] | None = None

    def process_message(self, user_id: str, message: str) -> str:
        if self.should_fail:
            raise RuntimeError("boom")
        self.calls.append((user_id, message))
        if self.alert_message and self._alert_handler:
            self._alert_handler(self.alert_message)
        return "Готово"

    def register_alert_handler(self, handler: Callable[[str], None]) -> None:
        self._alert_handler = handler


def _make_update(chat_id: int, user_id: int, message: FakeMessage) -> SimpleNamespace:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=user_id),
        message=message,
    )


def _make_context(bot: FakeBot | None = None) -> SimpleNamespace:
    application = SimpleNamespace(bot=bot or FakeBot())
    return SimpleNamespace(application=application)


def test_handle_text_message_success(monkeypatch):
    """Текстовое сообщение обрабатывается и отправляет ответ пользователя."""

    async def fake_executor(self, func, *args):  # type: ignore[no-untyped-def]
        return func(*args)

    monkeypatch.setattr(
        TelegramBotAdapter,
        "_run_in_executor",
        fake_executor,
    )

    orchestrator = FakeOrchestrator()
    config = TelegramBotConfig(token="x")
    adapter = TelegramBotAdapter(orchestrator, config)

    message = FakeMessage("Привет", chat_id=1)
    update = _make_update(chat_id=1, user_id=42, message=message)
    context = _make_context()

    async def scenario() -> None:
        await adapter._handle_text_message(update, context)

    asyncio.run(scenario())
    adapter._executor.shutdown(wait=True)

    assert orchestrator.calls == [("42", "Привет")]
    assert message.replies == ["Готово"]
    assert context.application.bot.actions


def test_handle_text_message_error_notifies_admin(monkeypatch):
    """При ошибке оркестратора пользователь и администратор получают уведомления."""

    async def fake_executor(self, func, *args):  # type: ignore[no-untyped-def]
        return func(*args)

    monkeypatch.setattr(
        TelegramBotAdapter,
        "_run_in_executor",
        fake_executor,
    )

    bot = FakeBot()
    orchestrator = FakeOrchestrator(should_fail=True)
    config = TelegramBotConfig(token="x", error_chat_id=999)
    adapter = TelegramBotAdapter(orchestrator, config)

    message = FakeMessage("Привет", chat_id=1)
    update = _make_update(chat_id=1, user_id=7, message=message)
    context = _make_context(bot)

    async def scenario() -> None:
        await adapter._handle_text_message(update, context)

    asyncio.run(scenario())
    adapter._executor.shutdown(wait=True)

    assert message.replies
    assert bot.sent and bot.sent[-1][0] == 999


def test_service_alert_forwarded_to_admin():
    """Сервисные алерты оркестратора пересылаются в административный чат."""

    orchestrator = FakeOrchestrator(alert_message="Повторяющиеся ошибки Bitrix24")
    config = TelegramBotConfig(token="x", error_chat_id=321)
    adapter = TelegramBotAdapter(orchestrator, config)

    message = FakeMessage("Запрос", chat_id=1)
    update = _make_update(chat_id=1, user_id=99, message=message)
    bot = FakeBot()
    context = _make_context(bot)

    async def scenario() -> None:
        await adapter._handle_text_message(update, context)

    asyncio.run(scenario())
    adapter._executor.shutdown(wait=True)

    assert bot.sent
    assert any("Повторяющиеся ошибки Bitrix24" in text for _, text in bot.sent)


def test_non_text_message_returns_hint():
    """Нетекстовое сообщение сопровождается подсказкой."""

    orchestrator = FakeOrchestrator()
    config = TelegramBotConfig(token="x")
    adapter = TelegramBotAdapter(orchestrator, config)

    message = FakeMessage(None, chat_id=1)
    update = _make_update(chat_id=1, user_id=5, message=message)
    context = _make_context()

    async def scenario() -> None:
        await adapter._handle_non_text_message(update, context)

    asyncio.run(scenario())
    adapter._executor.shutdown(wait=True)

    assert message.replies[-1].startswith("Сейчас могу работать только с текстом")


def test_access_denied_notifies_user():
    """Сообщение из запрещённого чата отклоняется."""

    orchestrator = FakeOrchestrator()
    config = TelegramBotConfig(token="x", allowed_chats={1})
    adapter = TelegramBotAdapter(orchestrator, config)

    message = FakeMessage("Запрос", chat_id=2)
    update = _make_update(chat_id=2, user_id=5, message=message)
    context = _make_context()

    async def scenario() -> None:
        await adapter._handle_text_message(update, context)

    asyncio.run(scenario())
    adapter._executor.shutdown(wait=True)

    assert message.replies[-1] == "Доступ к этому боту ограничен. Обратитесь к администратору."
    assert not orchestrator.calls
