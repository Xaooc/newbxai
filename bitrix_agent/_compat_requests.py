"""Минимальная совместимость с библиотекой requests."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class RequestException(Exception):
    """Базовое исключение совместимости."""


class HTTPError(RequestException):
    """Ошибка HTTP с кодом состояния."""

    def __init__(self, message: str, response: "Response" | None = None) -> None:
        super().__init__(message)
        self.response = response


@dataclass
class Response:
    """Упрощённый ответ, совместимый с requests.Response."""

    status_code: int
    _content: bytes
    url: str

    def json(self) -> Any:
        try:
            return json.loads(self._content.decode("utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - соответствует requests
            raise ValueError("Некорректный JSON") from exc

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise HTTPError(f"HTTP {self.status_code} для {self.url}", response=self)

    @property
    def text(self) -> str:
        return self._content.decode("utf-8")


def _build_url(url: str, params: Optional[Dict[str, Any]]) -> str:
    if not params:
        return url
    query = urllib.parse.urlencode(params, doseq=True)
    separator = "&" if urllib.parse.urlparse(url).query else "?"
    return f"{url}{separator}{query}"


def request(method: str, url: str, *, json: Optional[Dict[str, Any]] = None,
            params: Optional[Dict[str, Any]] = None, timeout: float | None = None) -> Response:
    data: bytes | None = None
    headers: Dict[str, str] = {}
    if json is not None:
        data = json_encode(json)
        headers["Content-Type"] = "application/json"
    full_url = _build_url(url, params)
    req = urllib.request.Request(full_url, data=data, method=method.upper(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout or None) as resp:
            content = resp.read()
            status = resp.getcode() or 0
    except urllib.error.HTTPError as exc:
        content = exc.read() if exc.fp else b""
        status = exc.code
    except urllib.error.URLError as exc:  # pragma: no cover - сетевые сбои
        raise RequestException(str(exc)) from exc
    return Response(status_code=status, _content=content, url=full_url)


def json_encode(data: Dict[str, Any]) -> bytes:
    return json.dumps(data).encode("utf-8")

