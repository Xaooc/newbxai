"""Тесты планировщика."""

from bitrix_agent.memory import AgentMemory
from bitrix_agent.planner import SimplePlanner


def test_deal_plan_contains_create():
    memory = AgentMemory()
    planner = SimplePlanner(memory)
    steps = planner.build_plan("Создай новую сделку")
    actions = [step.action for step in steps]
    assert "crm.deal.add" in actions
    assert memory.next == steps[0].description
