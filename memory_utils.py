"""Утилиты для совместимой работы с памятью чата в разных версиях LangChain."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    # Новые версии LangChain (до рефакторинга) поставляют готовую память здесь.
    from langchain.memory import ConversationBufferMemory as _ConversationBufferMemory

    ConversationBufferMemory = _ConversationBufferMemory
except ImportError:  # pragma: no cover - поддержка версии 1.x со split-пакетами
    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import BaseMessage

    class ConversationBufferMemory:  # type: ignore[override] - сохраняем совместимый API
        """
        Упрощённая реализация ConversationBufferMemory для новых версий LangChain.

        Хранит сообщения в оперативной памяти через InMemoryChatMessageHistory и
        предоставляет тот же интерфейс, который ожидают агенты LangChain.
        """

        def __init__(
            self,
            *,
            memory_key: str = "chat_history",
            return_messages: bool = True,
        ) -> None:
            self.memory_key = memory_key
            self.return_messages = return_messages
            self.chat_memory = InMemoryChatMessageHistory()

        def load_memory_variables(self, _: Mapping[str, Any]) -> Dict[str, Any]:
            messages: List[BaseMessage] = list(self.chat_memory.messages)
            if self.return_messages:
                return {self.memory_key: messages}
            text = "\n".join(msg.content for msg in messages)
            return {self.memory_key: text}

        def save_context(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
            user_content = self._extract_text(inputs)
            ai_content = self._extract_text(outputs)
            if user_content:
                self.chat_memory.add_user_message(user_content)
            if ai_content:
                self.chat_memory.add_ai_message(ai_content)

        def clear(self) -> None:
            self.chat_memory.clear()

        @staticmethod
        def _extract_text(payload: Mapping[str, Any]) -> Optional[str]:
            for key in ("input", "content", "text"):
                if key in payload and payload[key]:
                    return str(payload[key])
            if payload:
                first = next(iter(payload.values()))
                if isinstance(first, str):
                    return first
                if isinstance(first, Mapping) and "content" in first:
                    return str(first["content"])
            return None

__all__ = ["ConversationBufferMemory"]
