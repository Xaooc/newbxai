"""Нагрузочный тест для проверки пропускной способности оркестратора."""
from __future__ import annotations

import asyncio
import time

from telegram_bitrix_agent.assistant.graph.llm import PlanResponse
from telegram_bitrix_agent.assistant.graph.runner import AssistantOrchestrator
from telegram_bitrix_agent.assistant.tools.formatter import Formatter
from telegram_bitrix_agent.assistant.tools.memory import MemoryStore


class LoadPlanner:
    """Планировщик, который всегда возвращает один и тот же план."""

    def __init__(self) -> None:
        self.plan_response = PlanResponse(
            plan=["Вызвать ping"],
            method="ping",
            entity="crm.deal",
            params={"fields": {"TITLE": "Ping"}},
        )

    async def plan(self, **_: object) -> PlanResponse:  # noqa: D401
        return self.plan_response


class LoadGateway:
    """Шлюз Bitrix для нагрузочного теста."""

    def __init__(self) -> None:
        self.counter = 0

    def search_methods(self, query: str):  # noqa: D401
        return {"candidates": [{"method": "ping", "score": 1.0}]}

    def schema(self, method: str):  # noqa: D401
        return {
            "input_schema": {"required": ["fields"], "properties": {"fields": {"type": "object"}}},
            "description": "",
        }

    def call(self, method: str, params):  # noqa: D401
        self.counter += 1
        return {"ok": True, "status": 200, "data": {"ID": self.counter, "TITLE": params["fields"]["TITLE"]}}


async def _run_load_test(orchestrator: AssistantOrchestrator, requests: int) -> float:
    start = time.perf_counter()

    async def invoke(index: int) -> str:
        chat_id = f"chat-{index}"
        return await orchestrator.handle(chat_id, "ping")

    await asyncio.gather(*(invoke(i) for i in range(requests)))
    return time.perf_counter() - start


def test_orchestrator_handles_50_rps() -> None:
    async def scenario() -> None:
        planner = LoadPlanner()
        gateway = LoadGateway()
        orchestrator = AssistantOrchestrator(
            planner=planner,
            gateway=gateway,
            formatter=Formatter(),
            memory=MemoryStore(),
        )
        duration = await _run_load_test(orchestrator, 50)
        assert gateway.counter == 50
        assert duration < 3.0, "Ожидается, что 50 запросов выполняются быстрее 3 секунд"

    asyncio.run(scenario())
