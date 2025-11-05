"""Примитивный ввод-вывод для агента."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class IOHandler(Protocol):
    """Протокол для общения агента с оператором."""

    def ask(self, question: str) -> str:
        ...

    def notify(self, message: str) -> None:
        ...


@dataclass
class ConsoleIO:
    """Ввод-вывод через стандартные потоки."""

    def ask(self, question: str) -> str:
        return input(question + " ")

    def notify(self, message: str) -> None:
        print(message)
