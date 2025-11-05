"""Тесты клиента BitrixWebhookClient."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from bitrix_agent.client import BitrixWebhookClient
from bitrix_agent.config import BitrixConfig


@pytest.fixture
def client() -> BitrixWebhookClient:
    """Создать клиент с подменой сетевых вызовов."""

    cfg = BitrixConfig(base_url="https://example.com/rest/")
    return BitrixWebhookClient(config=cfg)


def test_crm_deal_add_uses_required_fields(monkeypatch: pytest.MonkeyPatch, client: BitrixWebhookClient) -> None:
    """Проверить, что crm_deal_add отправляет правильные параметры."""

    captured: Dict[str, Any] = {}

    def fake_call(method: str, params: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        captured["method"] = method
        captured["params"] = params
        return {"result": 123}

    monkeypatch.setattr("bitrix_agent.client.bitrix_call", fake_call)

    deal_id = client.crm_deal_add({"TITLE": "Test"})

    assert deal_id == 123
    assert captured["method"] == "crm.deal.add"
    assert captured["params"] == {"fields": {"TITLE": "Test"}}


def test_tasks_task_add_returns_task(monkeypatch: pytest.MonkeyPatch, client: BitrixWebhookClient) -> None:
    """Проверить, что tasks_task_add вытаскивает объект задачи из ответа."""

    def fake_call(method: str, params: Dict[str, Any], **_: Any) -> Dict[str, Any]:
        assert method == "tasks.task.add"
        assert params == {"fields": {"TITLE": "Задача"}}
        return {"result": {"task": {"id": 42, "title": "Задача"}}}

    monkeypatch.setattr("bitrix_agent.client.bitrix_call", fake_call)

    task = client.tasks_task_add({"TITLE": "Задача"})
    assert task == {"id": 42, "title": "Задача"}


def test_batch_proxy(monkeypatch: pytest.MonkeyPatch, client: BitrixWebhookClient) -> None:
    """Проверить, что batch вызывает соответствующий инструмент."""

    captured: Dict[str, Any] = {}

    def fake_batch(cmd: Dict[str, str], **_: Any) -> Dict[str, Any]:
        captured["cmd"] = cmd
        return {"result": {}}

    monkeypatch.setattr("bitrix_agent.client.bitrix_batch", fake_batch)

    result = client.batch({"user": "user.current"})
    assert result == {"result": {}}
    assert captured["cmd"] == {"user": "user.current"}
