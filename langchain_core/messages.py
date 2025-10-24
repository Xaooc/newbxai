"""Минимальные определения сообщений для совместимости."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseMessage:
    """Базовый тип сообщения."""

    content: str
    type: str


@dataclass
class HumanMessage(BaseMessage):
    """Сообщение пользователя."""

    def __init__(self, content: str) -> None:
        super().__init__(content=content, type="user")


@dataclass
class AIMessage(BaseMessage):
    """Сообщение ассистента."""

    def __init__(self, content: str) -> None:
        super().__init__(content=content, type="assistant")
