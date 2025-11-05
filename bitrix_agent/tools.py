"""Инструменты взаимодействия с Bitrix24."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Mapping
from urllib.parse import parse_qsl, quote, urlencode, urlsplit

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - запасной вариант для окружений без requests
    from . import _compat_requests as requests

from .config import BitrixConfig


logger = logging.getLogger(__name__)

_RETRY_STATUS = {429} | set(range(500, 600))
_MAX_ATTEMPTS = 3
_INITIAL_DELAY = 0.5


class BitrixError(RuntimeError):
    """Исключение при ошибке Bitrix24."""


def _prepare_url(config: BitrixConfig, method: str) -> str:
    if method.startswith("/"):
        method = method.lstrip("/")
    return f"{config.base_url}{method}.json"


def _validate_response(payload: Mapping[str, Any]) -> None:
    """Проверить ответ Bitrix24."""

    if "error" in payload:
        raise BitrixError(f"Ошибка Bitrix24: {payload.get('error_description', payload['error'])}")


def _sanitize_url(url: str) -> str:
    """Скрыть чувствительные сегменты URL вебхука."""

    parsed = urlsplit(url)
    path = parsed.path
    if "/rest/" in path:
        tail = path.split("/rest/", 1)[1]
        parts = [part for part in tail.split("/") if part]
        if parts:
            method = parts[-1]
        else:  # pragma: no cover - защитный случай
            method = "unknown"
        return f"{parsed.netloc}/rest/.../{method}"
    return f"{parsed.netloc}{parsed.path}"


def _perform_request(verb: str, url: str, *, payload: Dict[str, Any] | None,
                     query: Dict[str, Any] | None, timeout: float,
                     log_label: str | None = None) -> Dict[str, Any]:
    """Отправить запрос с обязательными ретраями."""

    delay = _INITIAL_DELAY
    last_error: Exception | None = None
    response: requests.Response | None = None
    target = log_label or _sanitize_url(url)

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = requests.request(
                verb,
                url,
                json=payload if verb == "POST" else None,
                params=query if verb != "POST" else None,
                timeout=timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - сетевые ошибки редки
            last_error = exc
            logger.warning("Сбой запроса %s (%s/%s): %s", target, attempt, _MAX_ATTEMPTS, exc)
        else:
            if response.status_code in _RETRY_STATUS:
                last_error = None
                logger.warning(
                    "Лимит или ошибка сервера %s (%s/%s). Повтор через %.1f с.",
                    response.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                    delay,
                )
            else:
                response.raise_for_status()
                data = response.json()
                logger.debug("Ответ %s: %s", target, json.dumps(data, ensure_ascii=False))
                return data
        if attempt == _MAX_ATTEMPTS:
            break
        time.sleep(delay)
        delay *= 2
    if last_error:
        raise last_error
    if response is None:
        raise BitrixError("Не удалось выполнить запрос к Bitrix24")
    response.raise_for_status()
    data = response.json()
    logger.debug("Ответ %s: %s", target, json.dumps(data, ensure_ascii=False))
    return data


def bitrix_call(method: str, params: Dict[str, Any] | None = None, *, config: BitrixConfig | None = None,
                http_method: str | None = None, timeout: float = 10.0) -> Dict[str, Any]:
    """Вызвать одиночный метод Bitrix24."""

    cfg = config or BitrixConfig.from_env()
    params = params or {}
    url = _prepare_url(cfg, method)

    if http_method:
        verb = http_method.upper()
    elif params:
        verb = "POST"
    else:
        verb = "GET"

    payload = params if verb == "POST" else None
    query = params if verb != "POST" else None
    data = _perform_request(
        verb,
        url,
        payload=payload,
        query=query,
        timeout=timeout,
        log_label=method,
    )
    _validate_response(data)
    return data


def _encode_batch_commands(cmd: Mapping[str, str]) -> Dict[str, str]:
    """Закодировать параметры batch-команд."""

    encoded: Dict[str, str] = {}
    for alias, command in cmd.items():
        if "?" not in command:
            encoded[alias] = command
            continue
        method, query = command.split("?", 1)
        pairs = parse_qsl(query, keep_blank_values=True)
        encoded_query = urlencode(pairs, doseq=True, quote_via=quote, safe="")
        encoded[alias] = f"{method}?{encoded_query}"
    return encoded


def bitrix_batch(cmd: Mapping[str, str], *, config: BitrixConfig | None = None, halt: bool = False,
                 timeout: float = 10.0) -> Dict[str, Any]:
    """Выполнить batch-запрос к Bitrix24."""

    if len(cmd) > 50:
        raise ValueError("Batch поддерживает не более 50 команд")

    cfg = config or BitrixConfig.from_env()
    url = _prepare_url(cfg, "batch")
    payload = {"cmd": _encode_batch_commands(dict(cmd)), "halt": halt}
    data = _perform_request(
        "POST",
        url,
        payload=payload,
        query=None,
        timeout=timeout,
        log_label=f"batch[{len(cmd)}]",
    )
    _validate_response(data)
    return data
