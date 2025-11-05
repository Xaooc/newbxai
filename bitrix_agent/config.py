"""Настройки доступа к Bitrix24."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class BitrixConfig:
    """Конфигурация вебхука Bitrix24."""

    base_url: str

    @classmethod
    def from_env(cls) -> "BitrixConfig":
        """Создать конфигурацию, используя переменные окружения."""

        base_url = os.getenv("BITRIX_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "Не задана переменная окружения BITRIX_BASE_URL. "
                "Укажите её в файле окружения или переменных системы."
            )
        if not base_url.endswith("/"):
            base_url += "/"
        return cls(base_url=base_url)
