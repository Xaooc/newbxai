"""Тесты для TelegramIO."""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("telegram")

from bitrix_agent.telegram_bot import TelegramIO, chunk_message


def test_telegram_io_waits_for_answer() -> None:
    """Проверить, что ask блокируется до получения supply_answer."""

    sent_messages: list[str] = []
    message_event = threading.Event()

    def sender(message: str) -> None:
        sent_messages.append(message)
        message_event.set()

    io = TelegramIO(sender)

    result_container: list[str] = []

    def worker() -> None:
        result_container.append(io.ask("Какова цель?"))

    thread = threading.Thread(target=worker)
    thread.start()

    assert message_event.wait(timeout=1), "Вопрос не был отправлен"
    assert sent_messages == ["Какова цель?"], "Сообщение отправлено некорректно"
    assert io.is_waiting(), "IO должен ожидать ответа"

    assert io.supply_answer("Создать сделку"), "Ответ не был принят"

    thread.join(timeout=1)
    assert result_container == ["Создать сделку"], "Ответ не был возвращён"
    assert not io.is_waiting(), "Ожидание должно завершиться"


def test_telegram_io_rejects_unsolicited_answer() -> None:
    """Проверить, что ответ без вопроса отклоняется."""

    io = TelegramIO(lambda _: None)

    assert not io.is_waiting(), "Не должно быть ожиданий"
    assert not io.supply_answer("что-то"), "Ответ не должен приниматься без вопроса"


def test_chunk_message_splits_long_text() -> None:
    """Проверить корректность разбиения длинного текста."""

    text = "abcdef"
    assert chunk_message(text, limit=2) == ["ab", "cd", "ef"]
    assert chunk_message("", limit=5) == [""]


def test_chunk_message_rejects_non_positive_limit() -> None:
    """Проверить, что некорректный лимит вызывает ошибку."""

    try:
        chunk_message("text", limit=0)
    except ValueError:
        pass
    else:  # pragma: no cover - защитная ветка
        raise AssertionError("Ожидалось исключение ValueError")
