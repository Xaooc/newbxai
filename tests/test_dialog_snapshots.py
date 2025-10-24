"""Диалоговые снапшоты для проверки полного ответа ассистента."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from telegram_bitrix_agent.assistant.graph.llm import ActionPlan, PlanResponse
from telegram_bitrix_agent.assistant.graph.runner import AssistantOrchestrator
from telegram_bitrix_agent.assistant.tools.formatter import Formatter
from telegram_bitrix_agent.assistant.tools.memory import MemoryStore


@dataclass
class SnapshotScenario:
    """Описание одного сценария диалога."""

    name: str
    message: str
    plan: PlanResponse
    schema_map: Dict[str, Dict[str, Any]]
    call_map: Dict[str, Any]
    expectations: List[str]


class SnapshotPlanner:
    """Планировщик, который возвращает предопределённый ответ."""

    def __init__(self, plan: PlanResponse) -> None:
        self._plan = plan

    async def plan(self, **_: Any) -> PlanResponse:  # noqa: D401
        return self._plan


class SnapshotGateway:
    """Шлюз Bitrix, использующий заранее подготовленные данные."""

    def __init__(self, schema_map: Dict[str, Dict[str, Any]], call_map: Dict[str, Any]) -> None:
        self._schema_map = schema_map
        self._call_map = call_map

    def search_methods(self, _: str) -> Dict[str, Any]:  # noqa: D401
        return {"candidates": [{"method": name, "score": 1.0} for name in self._schema_map]}

    def schema(self, method: str) -> Dict[str, Any]:  # noqa: D401
        schema = self._schema_map[method]
        return {
            "input_schema": {
                "required": schema.get("required", []),
                "properties": schema.get("properties", {}),
            },
            "description": schema.get("description", ""),
        }

    def call(self, method: str, params: Dict[str, Any]):  # noqa: D401
        payload = self._call_map[method]
        if callable(payload):  # pragma: no cover - гибкость для будущих сценариев
            return payload(params)
        return payload


SNAPSHOTS: List[SnapshotScenario] = [
    SnapshotScenario(
        name="deal_creation",
        message="Создай сделку на 150к для ИП Иванов сегодня",
        plan=PlanResponse(
            plan=["Проверить входные данные", "Создать сделку"],
            method="crm.deal.add",
            entity="crm.deal",
            params={
                "fields": {
                    "TITLE": "Сделка с ИП Иванов",
                    "OPPORTUNITY": 150000,
                    "CURRENCY_ID": "RUB",
                }
            },
        ),
        schema_map={
            "crm.deal.add": {
                "required": ["fields"],
                "properties": {
                    "fields": {
                        "type": "object",
                        "required": ["TITLE"],
                        "properties": {
                            "TITLE": {"description": "Название сделки"},
                            "OPPORTUNITY": {},
                            "CURRENCY_ID": {},
                        },
                    }
                },
            }
        },
        call_map={
            "crm.deal.add": {
                "ok": True,
                "status": 200,
                "data": {
                    "ID": 18452,
                    "TITLE": "Сделка с ИП Иванов",
                    "OPPORTUNITY": 150000,
                    "CURRENCY_ID": "RUB",
                    "STAGE_ID": "NEW",
                },
            }
        },
        expectations=[
            "Готово: Сделка создана: ID 18452",
            "Детали: Название: Сделка с ИП Иванов; Сумма: 150000 RUB; Стадия: NEW",
            "План: Проверить входные данные; Создать сделку",
            "Следующий шаг: Назначить ответственного?",
        ],
    ),
    SnapshotScenario(
        name="tasks_for_tomorrow",
        message="Покажи задачи по мне на завтра",
        plan=PlanResponse(
            plan=["Уточнить пользователя", "Получить задачи"],
            method="user.current",
            entity="user",
            params={},
            actions=[
                ActionPlan(method="user.current", entity="user", params={}),
                ActionPlan(
                    method="task.item.list",
                    entity="task",
                    params={"filter": {"RESPONSIBLE_ID": 42, "DEADLINE": "2024-05-15"}},
                ),
            ],
        ),
        schema_map={
            "user.current": {"required": [], "properties": {}},
            "task.item.list": {
                "required": ["filter"],
                "properties": {
                    "filter": {
                        "type": "object",
                        "required": ["RESPONSIBLE_ID"],
                        "properties": {
                            "RESPONSIBLE_ID": {"description": "ID ответственного"},
                            "DEADLINE": {"description": "Дата дедлайна"},
                        },
                    }
                },
            },
        },
        call_map={
            "user.current": {"ok": True, "status": 200, "data": {"ID": 42}},
            "task.item.list": {
                "ok": True,
                "status": 200,
                "data": [
                    {"ID": 301, "TITLE": "Подготовить отчёт"},
                    {"ID": 302, "TITLE": "Созвон с клиентом"},
                ],
            },
        },
        expectations=[
            "Готово: Найдено элементов: 2",
            "Детали: Примеры: Подготовить отчёт, Созвон с клиентом",
            "План: Уточнить пользователя; Получить задачи",
        ],
    ),
]


@pytest.mark.parametrize("scenario", SNAPSHOTS, ids=lambda item: item.name)
def test_dialog_snapshots(scenario: SnapshotScenario) -> None:
    async def scenario_run() -> None:
        planner = SnapshotPlanner(scenario.plan)
        gateway = SnapshotGateway(scenario.schema_map, scenario.call_map)
        orchestrator = AssistantOrchestrator(
            planner=planner,
            gateway=gateway,
            formatter=Formatter(),
            memory=MemoryStore(),
        )
        answer = await orchestrator.handle("chat", scenario.message)
        for fragment in scenario.expectations:
            assert fragment in answer

    asyncio.run(scenario_run())
