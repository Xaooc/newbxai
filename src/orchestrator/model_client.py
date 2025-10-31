"""Клиент для обращения к ChatGPT (модели семейства GPT-4.1)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from src.orchestrator.context_builder import build_state_summary

logger = logging.getLogger(__name__)


class ModelClientError(Exception):
    """Исключение, возникающее при ошибке общения с моделью."""


@dataclass
class ModelClient:
    """Минимальный HTTP-клиент для вызова ChatGPT."""

    model_name: str = "gpt-4.1"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 60
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if self.api_key is None:
            raise ModelClientError("Не задан API-ключ ChatGPT (переменная OPENAI_API_KEY)")

    def generate(
        self,
        system_prompt: str,
        state_snapshot: Dict[str, Any],
        user_message: str,
    ) -> str:
        """Запрашивает у модели ответ с блоками THOUGHT/ACTION/ASSISTANT."""

        url = self._build_url("/chat/completions")
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": self._build_messages(system_prompt, state_snapshot, user_message),
        }

        logger.debug("Отправляем запрос в модель", extra={"url": url, "model": self.model_name})

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ModelClientError(f"Ошибка сети при обращении к модели: {exc}") from exc

        if response.status_code != 200:
            raise ModelClientError(
                f"Модель вернула статус {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                f"Ответ модели не является корректным JSON: {response.text}"
            ) from exc

        content = self._extract_content(data)
        if not content:
            raise ModelClientError("Пустой ответ от модели")
        return content

    def _build_messages(
        self,
        system_prompt: str,
        state_snapshot: Dict[str, Any],
        user_message: str,
    ) -> List[Dict[str, str]]:
        summary = build_state_summary(state_snapshot)
        state_json = json.dumps(state_snapshot, ensure_ascii=False, indent=2)
        state_block = (
            "Краткая сводка состояния:\n"
            f"{summary}\n\n"
            "Текущее состояние агента (JSON):\n"
            f"```json\n{state_json}\n```"
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{state_block}\n\n"
                    f"Запрос пользователя: {user_message}\n\n"
                    "Сформируй ответ строго с блоками THOUGHT:/ACTION:/ASSISTANT:."
                ),
            },
        ]

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> Optional[str]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        return None


def build_default_model_client(model_name: str = "gpt-4.1") -> ModelClient:
    """Создаёт клиент, используя переменные окружения."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    return ModelClient(model_name=model_name, api_key=api_key, base_url=base_url)
