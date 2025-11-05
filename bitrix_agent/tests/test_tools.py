"""Тесты для инструментов bitrix_call и bitrix_batch."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from bitrix_agent.config import BitrixConfig
from bitrix_agent.tools import bitrix_batch, bitrix_call


class FakeResponse:
    """Заглушка ответа requests."""

    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(response=self)  # type: ignore[no-untyped-call]

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_bitrix_call_retries_on_429(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Убедиться, что 429 приводит к повторным попыткам."""

    attempts: List[Dict[str, Any]] = []
    responses = [
        FakeResponse(429, {"result": {}}),
        FakeResponse(200, {"result": {"ID": 1}}),
    ]

    def fake_request(method: str, url: str, json: Dict[str, Any] | None = None,
                     params: Dict[str, Any] | None = None, timeout: float | None = None) -> FakeResponse:
        attempts.append({"method": method, "url": url, "json": json, "params": params, "timeout": timeout})
        return responses.pop(0)

    monkeypatch.setattr("bitrix_agent.tools.requests.request", fake_request)
    monkeypatch.setattr("bitrix_agent.tools.time.sleep", lambda _: None)

    cfg = BitrixConfig(base_url="https://portal.magnitmedia.ru/rest/132/1s0mz4mw8d42bfvk/")
    with caplog.at_level("DEBUG"):
        result = bitrix_call("user.current", config=cfg)

    assert result == {"result": {"ID": 1}}
    assert len(attempts) == 2
    assert attempts[0]["params"] == {}
    assert any("user.current" in record.getMessage() for record in caplog.records)
    assert all("1s0mz4mw8d42bfvk" not in record.getMessage() for record in caplog.records)


def test_bitrix_batch_encodes_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Проверить, что batch кодирует специальные символы."""

    captured: Dict[str, Any] = {}

    def fake_request(method: str, url: str, json: Dict[str, Any] | None = None,
                     params: Dict[str, Any] | None = None, timeout: float | None = None) -> FakeResponse:
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return FakeResponse(200, {"result": {}})

    monkeypatch.setattr("bitrix_agent.tools.requests.request", fake_request)

    cfg = BitrixConfig(base_url="https://example.com/rest/")
    result = bitrix_batch({"user": "user.get?select[]=ID&filter[ACTIVE]=Y"}, config=cfg)

    assert result == {"result": {}}
    assert captured["method"] == "POST"
    assert captured["json"]["cmd"]["user"] == "user.get?select%5B%5D=ID&filter%5BACTIVE%5D=Y"
