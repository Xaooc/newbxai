"""Загрузчики текстовых промптов."""
from __future__ import annotations

from pathlib import Path


def load_system_prompt() -> str:
    """Возвращает системный промпт ассистента."""

    path = Path(__file__).with_name("assistant_system_ru.md")
    return path.read_text(encoding="utf-8")


def load_error_tips() -> dict[str, list[str]]:
    """Загружает рекомендации по кодам ошибок."""

    path = Path(__file__).with_name("error_tips.md")
    tips: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        code, payload = line.split("=", 1)
        tips[code.strip()] = [item.strip() for item in payload.split(";") if item.strip()]
    return tips
