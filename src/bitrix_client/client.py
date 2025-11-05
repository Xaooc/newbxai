"""Клиент для вызова REST-методов Bitrix24.

Модуль предоставляет функцию `call_bitrix`, которую использует оркестратор.
Вызовы выполняются через вебхук Bitrix24. URL и токен должны храниться
в переменной окружения `BITRIX_WEBHOOK_URL` либо в конфигурационном файле,
который не попадает в репозиторий.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import requests

logger = logging.getLogger(__name__)


DEFAULT_WEBHOOK_URL = "https://portal.magnitmedia.ru/rest/132/1s0mz4mw8d42bfvk/"
SENSITIVE_PARAM_KEYS = (
    "token",
    "secret",
    "password",
    "auth",
    "key",
    "nonce",
    "login",
    "email",
    "phone",
    "handler",
)
REDACTED_VALUE = "***redacted***"
MAX_STRING_LENGTH = 256


class BitrixClientError(Exception):
    """Исключение, выбрасываемое при ошибке взаимодействия с Bitrix24."""


@dataclass
class BitrixClient:
    """Простой клиент для вызова REST-методов Bitrix24.

    Attributes:
        webhook_url: Полный URL вебхука Bitrix24. В формате
            `https://<portal>.bitrix24.ru/rest/<user>/<token>/`.
        timeout: Таймаут HTTP-запроса в секундах.
    """

    webhook_url: str
    timeout: int = 30

    def call_method(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        http_method: str = "POST",
    ) -> Dict[str, Any]:
        """Вызывает метод Bitrix24 и возвращает разобранный JSON-ответ.

        Args:
            method: Имя REST-метода (например, `crm.deal.list`).
            params: Параметры запроса.

        Returns:
            Ответ Bitrix24, преобразованный в словарь.

        Raises:
            BitrixClientError: Если URL не настроен, HTTP-запрос завершился ошибкой
                или Bitrix24 вернул сообщение об ошибке.
        """

        if not self.webhook_url:
            raise BitrixClientError("Не задан URL вебхука Bitrix24")

        payload = params or {}
        sanitized_payload = sanitize_for_logging(payload)
        url = f"{self.webhook_url.rstrip('/')}/{method}.json"
        logger.debug(
            "Выполняем вызов Bitrix24",
            extra={"method": method, "params": sanitized_payload, "http_method": http_method},
        )
        logger.info(
            "Запрос к Bitrix24",
            extra={"method": method, "params": sanitized_payload, "http_method": http_method},
        )

        http_method = http_method.upper()
        try:
            if http_method == "GET":
                response = requests.get(url, params=payload, timeout=self.timeout)
            elif http_method == "POST":
                response = requests.post(url, json=payload, timeout=self.timeout)
            else:
                raise BitrixClientError(f"Неподдерживаемый HTTP-метод: {http_method}")
        except requests.RequestException as exc:
            raise BitrixClientError(f"Ошибка сети при обращении к Bitrix24: {exc}") from exc

        if response.status_code != 200:
            raise BitrixClientError(
                f"Bitrix24 вернул статус {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise BitrixClientError(
                f"Не удалось декодировать ответ Bitrix24 как JSON: {response.text}"
            ) from exc

        logger.info(
            "Ответ от Bitrix24",
            extra={
                "method": method,
                "status_code": response.status_code,
                "response": sanitize_for_logging(data),
            },
        )
        if "error" in data:
            raise BitrixClientError(
                f"Bitrix24 сообщил об ошибке: {data['error']} — {data.get('error_description', 'нет описания')}"
            )

        return data


def build_default_client() -> BitrixClient:
    """Создаёт клиента Bitrix24 с учётом переменных окружения."""

    webhook_url = os.getenv("BITRIX_WEBHOOK_URL", DEFAULT_WEBHOOK_URL)
    return BitrixClient(webhook_url=webhook_url)


def call_bitrix(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    client: Optional[BitrixClient] = None,
    http_method: str = "POST",
) -> Dict[str, Any]:
    """Обёртка над клиентом Bitrix24 для использования оркестратором.

    Args:
        method: Имя REST-метода Bitrix24.
        params: Параметры вызова.
        client: Необязательный экземпляр `BitrixClient`. Если не передан,
            используется клиент по умолчанию, созданный функцией `build_default_client`.

    Returns:
        Ответ Bitrix24 (словарь).

    Raises:
        BitrixClientError: При любой ошибке общения с Bitrix24.
    """

    active_client = client or build_default_client()
    return active_client.call_method(method, params, http_method=http_method)


def sanitize_for_logging(payload: Any, *, _depth: int = 0, _visited: Optional[Set[int]] = None) -> Any:
    """РЈРґР°Р»СЏРµС‚ С‡СѓРІСЃС‚РІРёС‚РµР»СЊРЅС‹Рµ РґР°РЅРЅС‹Рµ Рё РѕРіСЂР°РЅРёС‡РёРІР°РµС‚ РґР»РёРЅСѓ СЃС‚СЂРѕРє РґР»СЏ Р»РѕРіРѕРІ."""

    max_depth = 5
    if _depth > max_depth:
        return "<truncated>"

    if _visited is None:
        _visited = set()

    obj_id = id(payload)
    if obj_id in _visited:
        return "<recursion>"
    _visited.add(obj_id)

    if isinstance(payload, dict):
        sanitized: Dict[str, Any] = {}
        for key, value in payload.items():
            key_str = str(key)
            if any(token in key_str.lower() for token in SENSITIVE_PARAM_KEYS):
                sanitized[key_str] = REDACTED_VALUE
            else:
                sanitized[key_str] = sanitize_for_logging(value, _depth=_depth + 1, _visited=_visited)
        return sanitized

    if isinstance(payload, list):
        return [sanitize_for_logging(item, _depth=_depth + 1, _visited=_visited) for item in payload]

    if isinstance(payload, tuple):
        return tuple(sanitize_for_logging(item, _depth=_depth + 1, _visited=_visited) for item in payload)

    if isinstance(payload, set):
        return {sanitize_for_logging(item, _depth=_depth + 1, _visited=_visited) for item in payload}

    if isinstance(payload, bytes):
        return f"<bytes:{len(payload)}>"

    if isinstance(payload, str):
        stripped = payload.strip()
        if len(stripped) > MAX_STRING_LENGTH:
            return f"{stripped[:MAX_STRING_LENGTH]}…"
        return stripped

    return payload
