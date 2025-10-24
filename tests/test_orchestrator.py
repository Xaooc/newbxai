"""Интеграционные тесты для LangGraph ассистента."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from telegram_bitrix_agent.assistant.graph.llm import ActionPlan, PlanResponse
from telegram_bitrix_agent.assistant.graph.runner import AssistantOrchestrator
from telegram_bitrix_agent.assistant.tools.formatter import Formatter
from telegram_bitrix_agent.assistant.tools.memory import MemoryStore


class FakePlanner:
    """Заглушка планировщика GPT-5 с поддержкой последовательных ответов."""

    def __init__(self, responses: Dict[str, List[PlanResponse] | PlanResponse]) -> None:
        self._responses: Dict[str, List[PlanResponse]] = {}
        self._counters: Dict[str, int] = {}
        for message, payload in responses.items():
            if isinstance(payload, list):
                self._responses[message] = payload
            else:
                self._responses[message] = [payload]
            self._counters[message] = 0

    async def plan(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history,
        candidates,
        tool_context,
    ):  # noqa: D401, ANN001
        queue = self._responses[user_message]
        index = self._counters[user_message]
        self._counters[user_message] = min(index + 1, len(queue) - 1)
        return queue[index]


class FakeGateway:
    """Заглушка BitrixGateway."""

    def __init__(self, schema_map: Dict[str, Dict[str, Any]], call_map: Dict[str, Any]) -> None:
        self._schema_map = schema_map
        self._call_map = call_map
        self.last_params: Dict[str, Any] | None = None
        self.call_counters: Dict[str, int] = {}

    def search_methods(self, query: str) -> Dict[str, Any]:
        return {
            "candidates": [
                {
                    "method": method,
                    "score": 1.0,
                    "required_fields": schema.get("required", []),
                    "optional_fields": list(schema.get("properties", {}).keys()),
                }
                for method, schema in self._schema_map.items()
            ]
        }

    def schema(self, method: str) -> Dict[str, Any]:
        return {
            "input_schema": {
                "required": self._schema_map[method].get("required", []),
                "properties": self._schema_map[method].get("properties", {}),
            },
            "description": self._schema_map[method].get("description", ""),
        }

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.last_params = params
        self.call_counters[method] = self.call_counters.get(method, 0) + 1
        response = self._call_map.get(method)
        if isinstance(response, list):
            index = min(self.call_counters[method] - 1, len(response) - 1)
            return response[index]
        if callable(response):  # pragma: no cover - гибкость для будущих тестов
            return response(self.call_counters[method])
        return response or {"ok": False, "error": {"code": "UNKNOWN", "message": "Не настроено"}}


def test_deal_creation_success() -> None:
    async def scenario() -> None:
        responses = {
            "Создай сделку на 150к": PlanResponse(
                plan=["Проверить входные данные", "Вызвать crm.deal.add"],
                method="crm.deal.add",
                entity="crm.deal",
                params={"fields": {"TITLE": "Сделка", "OPPORTUNITY": 150000, "CURRENCY_ID": "RUB"}},
                missing_fields=[],
                memory_set={"preferred_currency": "RUB"},
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "crm.deal.add": {
                "required": ["fields"],
                "properties": {"fields": {"type": "object"}},
            }
        }
        call_map = {
            "crm.deal.add": {"ok": True, "status": 200, "data": {"ID": 18452, "TITLE": "Сделка"}}
        }
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Создай сделку на 150к")
        assert answer.startswith("Готово:")
        assert "Сделка создана" in answer
        assert "Детали" in answer
        assert "План:" in answer
        assert "Следующий шаг" in answer
        stored = await memory.tool_get("chat", ["preferred_currency"])
        assert stored["preferred_currency"] == "RUB"

    asyncio.run(scenario())


def test_missing_fields_request() -> None:
    async def scenario() -> None:
        responses = {
            "Покажи задачи на завтра": PlanResponse(
                plan=["Выбрать метод task.item.list"],
                method="task.item.list",
                entity="task",
                params={},
                missing_fields=[],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "task.item.list": {
                "required": ["filter"],
                "properties": {"filter": {"type": "object"}},
            }
        }
        gateway = FakeGateway(schema_map, call_map={})
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Покажи задачи на завтра")
        assert "Не хватает" in answer
        assert "filter" in answer
        assert "Уточните" in answer

    asyncio.run(scenario())


def test_nested_required_fields() -> None:
    async def scenario() -> None:
        responses = {
            "Создай сделку без названия": PlanResponse(
                plan=["Подготовить данные", "Вызвать crm.deал.add"],
                method="crm.deal.add",
                entity="crm.deal",
                params={"fields": {}},
                missing_fields=[],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "crm.deal.add": {
                "required": ["fields"],
                "properties": {
                    "fields": {
                        "type": "object",
                        "required": ["TITLE"],
                        "properties": {
                            "TITLE": {"description": "Название сделки"},
                        },
                    }
                },
            }
        }
        gateway = FakeGateway(schema_map, call_map={})
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Создай сделку без названия")
        assert "fields.TITLE" in answer or "TITLE" in answer
        assert "Название сделки" in answer

    asyncio.run(scenario())


def test_access_denied_error() -> None:
    async def scenario() -> None:
        responses = {
            "Обнови сделку": PlanResponse(
                plan=["Подготовить поля", "Вызвать crm.deal.update"],
                method="crm.deal.update",
                entity="crm.deal",
                params={"id": 1, "fields": {"TITLE": "Новая"}},
                missing_fields=[],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "crm.deal.update": {
                "required": ["id", "fields"],
                "properties": {"id": {}, "fields": {}},
            }
        }
        call_map = {
            "crm.deal.update": {
                "ok": False,
                "status": 403,
                "error": {"code": "ACCESS_DENIED", "message": "Недостаточно прав"},
            }
        }
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Обнови сделку")
        assert answer.startswith("Ошибка:")
        assert "права" in answer or "Что сделать" in answer

    asyncio.run(scenario())


def test_rate_limit_retry_success() -> None:
    async def scenario() -> None:
        responses = {
            "Запусти пакет": PlanResponse(
                plan=["Собрать команды", "Отправить batch"],
                method="batch",
                entity="batch",
                params={"cmd": {"deal": "crm.deal.list"}},
                missing_fields=[],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {"batch": {"required": ["cmd"], "properties": {"cmd": {}}}}
        call_map = {
            "batch": [
                {
                    "ok": False,
                    "status": 429,
                    "error": {"code": "RATE_LIMIT", "message": "Слишком много запросов", "details": {"retry_after": 0}},
                },
                {"ok": True, "status": 200, "data": {"processed": True}},
            ]
        }
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Запусти пакет")
        assert answer.startswith("Готово")
        assert gateway.call_counters["batch"] >= 2

    asyncio.run(scenario())


def test_multi_step_actions() -> None:
    async def scenario() -> None:
        responses = {
            "Покажи мои задачи": PlanResponse(
                plan=["Определить пользователя", "Получить задачи"],
                method="user.current",
                entity="user",
                params={},
                actions=[
                    ActionPlan(method="user.current", entity="user", params={}),
                    ActionPlan(
                        method="task.item.list",
                        entity="task",
                        params={"filter": {"RESPONSIBLE_ID": 1}},
                    ),
                ],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "user.current": {"required": [], "properties": {}},
            "task.item.list": {"required": ["filter"], "properties": {"filter": {"type": "object"}}},
        }
        call_map = {
            "user.current": {"ok": True, "status": 200, "data": {"ID": 1}},
            "task.item.list": {
                "ok": True,
                "status": 200,
                "data": [{"ID": 10, "TITLE": "Задача"}],
            },
        }
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Покажи мои задачи")
        assert "Найдено элементов" in answer or "Задача" in answer
        assert gateway.call_counters["user.current"] == 1
        assert gateway.call_counters["task.item.list"] == 1

    asyncio.run(scenario())


def test_not_found_error() -> None:
    async def scenario() -> None:
        responses = {
            "Найди компанию 999": PlanResponse(
                plan=["Вызвать crm.company.get"],
                method="crm.company.get",
                entity="crm.company",
                params={"id": 999},
                missing_fields=[],
            )
        }
        planner = FakePlanner(responses)
        schema_map = {
            "crm.company.get": {
                "required": ["id"],
                "properties": {"id": {}},
            }
        }
        call_map = {
            "crm.company.get": {
                "ok": False,
                "status": 404,
                "error": {"code": "NOT_FOUND", "message": "Компания не найдена"},
            }
        }
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Найди компанию 999")
        assert "Ошибка" in answer
        assert "Компания не найдена" in answer

    asyncio.run(scenario())


def test_memory_tracks_dialogue_and_get() -> None:
    async def scenario() -> None:
        responses = {
            "Привет": [
                PlanResponse(
                    plan=["Прочитать память"],
                    method="user.current",
                    entity="user",
                    params={},
                    missing_fields=[],
                    memory_get=["responsible_id"],
                ),
                PlanResponse(
                    plan=["Ответить пользователю"],
                    method="user.current",
                    entity="user",
                    params={"fallback": True},
                    missing_fields=[],
                ),
            ]
        }
        planner = FakePlanner(responses)
        schema_map = {"user.current": {"required": [], "properties": {}}}
        call_map = {"user.current": {"ok": True, "status": 200, "data": {"ID": 1}}}
        gateway = FakeGateway(schema_map, call_map)
        memory = MemoryStore()
        await memory.tool_set("chat", {"responsible_id": 42})
        orchestrator = AssistantOrchestrator(planner=planner, gateway=gateway, formatter=Formatter(), memory=memory)
        answer = await orchestrator.handle("chat", "Привет")
        assert "Готово" in answer or "ID" in answer
        history = await memory.get_history("chat")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert gateway.call_counters["user.current"] >= 1

    asyncio.run(scenario())
