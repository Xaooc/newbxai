"""Интеграционные тесты агента с подменой Bitrix API."""

from __future__ import annotations

from typing import Any

import pytest

from bitrix_agent.agent import BitrixAutonomousAgent
from bitrix_agent.io import IOHandler


class DummyIO(IOHandler):
    """Упрощённый IO-обработчик для тестов."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.questions: list[str] = []

    def ask(self, question: str) -> str:
        self.questions.append(question)
        if not self.answers:
            raise RuntimeError("Нет подготовленных ответов")
        return self.answers.pop(0)

    def notify(self, message: str) -> None:  # pragma: no cover - уведомления в тестах не требуются
        self.questions.append(f"notify:{message}")


@pytest.fixture(autouse=True)
def patch_bitrix(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_call(method: str, params: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        calls.append({"method": method, "params": params or {}})
        if method == "crm.deal.list":
            return {"result": []}
        if method == "crm.deal.add":
            return {"result": 321}
        if method == "crm.status.list":
            return {"result": [
                {"STATUS_ID": "NEW", "NAME": "Новая"},
                {"STATUS_ID": "PREP", "NAME": "Подготовка"},
            ]}
        if method == "user.get":
            return {"result": [
                {"ID": 132, "NAME": "Антон", "LAST_NAME": "Титовец"},
            ]}
        if method == "tasks.task.list":
            return {"result": {"tasks": []}}
        if method == "tasks.task.add":
            return {"result": {"task": {"id": 555}}}
        return {"result": True}

    monkeypatch.setattr("bitrix_agent.client.bitrix_call", fake_call)
    monkeypatch.setattr("bitrix_agent.client.bitrix_batch", lambda *args, **kwargs: {"result": {}})
    yield calls


def test_agent_creates_deal_and_handles_unknown(patch_bitrix):
    io = DummyIO([])
    agent = BitrixAutonomousAgent(io=io)
    result = agent.run("Создай сделку на 1 000 000 ₽ и назначь ответственным Антон Титовец")
    assert result["memory"]["knowns"]["last_deal_id"] == 321
    assert any(call["method"] == "crm.deal.add" for call in patch_bitrix)
    assert any(call["method"] == "user.get" for call in patch_bitrix)
    assert not any("ID ответственного" in question for question in io.questions)
    assert result["memory"]["knowns"].get("deal_stage_map")
    assert "Итог:" in result["report"]
    assert "Сделка #321" in result["report"]


def test_agent_creates_task(patch_bitrix):
    io = DummyIO([])
    agent = BitrixAutonomousAgent(io=io)
    result = agent.run("Создай задачу завтра 12:00")
    assert any(call["method"] == "tasks.task.add" for call in patch_bitrix)
    assert result["results"][-1]["step"] == "Подготовить отчёт"
    assert "Задача #555" in result["report"]
