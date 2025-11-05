"""CLI-запуск автономного агента Bitrix24."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Iterable

from .agent import BitrixAutonomousAgent
from .config import BitrixConfig
from .environment import load_env_file


def build_parser() -> argparse.ArgumentParser:
    """Создать парсер аргументов командной строки."""

    parser = argparse.ArgumentParser(description="Автономный агент Bitrix24")
    parser.add_argument("--prompt", required=True, help="Цель, которую должен выполнить агент")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Путь к файлу окружения с настройками Bitrix24",
    )
    parser.add_argument("--verbose", action="store_true", help="Включить подробный лог")
    return parser


def configure_logging(verbose: bool) -> None:
    """Настроить уровень логирования."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level)


def run_agent(prompt: str) -> dict:
    """Запустить агента и вернуть результат выполнения."""

    agent = BitrixAutonomousAgent(config=BitrixConfig.from_env())
    return agent.run(prompt)


def main(argv: Iterable[str] | None = None) -> None:
    """Точка входа CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.env_file:
        try:
            load_env_file(args.env_file)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    configure_logging(args.verbose)
    try:
        result = run_agent(args.prompt)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - запуск как скрипт
    main()
