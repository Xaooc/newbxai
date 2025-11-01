"""Telegram-точка входа для AI-менеджера Bitrix24."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# При запуске `python src/main_entrypoint.py` интерпретатор устанавливает sys.path
# в каталог `src`, поэтому пакет `src.*` становится недоступен. Добавляем корень
# проекта вручную, чтобы импорты работали в обоих сценариях запуска.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from src.app import launch_telegram_bot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main(argv: list[str] | None = None) -> int:
    """Запускает Telegram-бота для взаимодействия с пользователями."""

    parser = argparse.ArgumentParser(description="AI-менеджер Bitrix24 — Telegram-бот")
    parser.add_argument(
        "--env",
        default=".env",
        help="Путь к файлу окружения (по умолчанию .env в корне проекта)",
    )

    args = parser.parse_args(argv)
    return launch_telegram_bot(args.env)


if __name__ == "__main__":
    sys.exit(main())
