"""Конфигурация приложения Telegram Bitrix ассистента."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Настройки, считываемые из переменных окружения."""

    bot_token: str
    openai_api_key: str
    bitrix_webhook: str
    webhook_url: Optional[str] = None
    redis_dsn: Optional[str] = None

    @staticmethod
    def from_env() -> "Settings":
        """Собирает настройки из переменных окружения."""

        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        bitrix_webhook = os.getenv("BITRIX_WEBHOOK_URL")
        if not bot_token:
            raise RuntimeError("Не задана переменная TELEGRAM_BOT_TOKEN")
        if not openai_api_key:
            raise RuntimeError("Не задана переменная OPENAI_API_KEY")
        if not bitrix_webhook:
            raise RuntimeError("Не задана переменная BITRIX_WEBHOOK_URL")
        webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
        redis_dsn = os.getenv("REDIS_DSN")
        return Settings(
            bot_token=bot_token,
            openai_api_key=openai_api_key,
            bitrix_webhook=bitrix_webhook,
            webhook_url=webhook_url,
            redis_dsn=redis_dsn,
        )
