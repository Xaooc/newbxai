"""Утилита для запуска Telegram-бота без Docker."""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from telegram_bitrix_agent.main import main as start_bot


def run() -> None:
    """Загружает переменные окружения и запускает асинхронный цикл бота."""
    load_dotenv()
    asyncio.run(start_bot())


if __name__ == "__main__":
    run()
