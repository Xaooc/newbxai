from __future__ import annotations

import asyncio

from telegram_bitrix_agent.assistant.graph.llm import PLANNER_TOOLS, PlannerToolContext
from telegram_bitrix_agent.assistant.tools.formatter import Formatter
from telegram_bitrix_agent.assistant.tools.memory import MemoryStore


class DummyGateway:
    """Простейший шлюз Bitrix для тестов инструментов."""

    def __init__(self) -> None:
        self.calls: dict[str, tuple[str, dict[str, object]] | str] = {}

    def search_methods(self, query: str):  # noqa: D401
        self.calls["search"] = query
        return {"candidates": [{"method": "crm.deal.add"}]}

    def schema(self, method: str):  # noqa: D401
        self.calls["schema"] = method
        return {"input_schema": {"required": [], "properties": {}}, "description": ""}

    def call(self, method: str, params):  # noqa: D401
        self.calls["call"] = (method, dict(params))
        return {"ok": True, "status": 200, "data": {"ID": 1, "TITLE": "Сделка"}}


def test_planner_tool_context_executes_all_tools() -> None:
    async def scenario() -> None:
        gateway = DummyGateway()
        memory = MemoryStore()
        context = PlannerToolContext(
            gateway=gateway,
            memory=memory,
            formatter=Formatter(),
            chat_id="chat",
        )

        search_result = await context.execute("bitrix.search_methods", {"query": "сделка"})
        assert search_result["candidates"], "Ожидаем список кандидатов"
        assert gateway.calls["search"] == "сделка"

        schema = await context.execute("bitrix.schema", {"method": "crm.deal.add"})
        assert "input_schema" in schema
        assert gateway.calls["schema"] == "crm.deal.add"

        call_result = await context.execute(
            "bitrix.call", {"method": "crm.deal.add", "params": {"fields": {}}}
        )
        assert call_result["ok"] is True
        assert gateway.calls["call"][0] == "crm.deal.add"

        await context.execute("memory.set", {"pairs": {"currency": "RUB"}})
        memory_values = await context.execute("memory.get", {"keys": ["currency", "missing"]})
        assert memory_values["currency"] == "RUB"

        formatted = await context.execute(
            "formatter.humanize",
            {"entity": "crm.deal", "data": {"ID": 99, "TITLE": "Тест"}, "locale": "ru"},
        )
        assert formatted["result"].startswith("Сделка")

    asyncio.run(scenario())


def test_planner_tool_spec_contains_required_names() -> None:
    names = {tool["function"]["name"] for tool in PLANNER_TOOLS}
    expected = {
        "bitrix.search_methods",
        "bitrix.schema",
        "bitrix.call",
        "formatter.humanize",
        "memory.get",
        "memory.set",
    }
    assert expected.issubset(names)
