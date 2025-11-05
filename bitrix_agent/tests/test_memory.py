"""Тесты для памяти агента."""

from bitrix_agent.memory import AgentMemory


def test_memory_to_json_and_progress():
    memory = AgentMemory(goal="Создать сделку")
    memory.plan = ["Шаг 1", "Шаг 2"]
    memory.update_next()
    assert memory.next == "Шаг 1"
    memory.mark_step_done("Шаг 1")
    assert memory.progress == ["Шаг 1"]
    assert memory.next == "Шаг 2"
    data = memory.to_json()
    assert data["goal"] == "Создать сделку"
    assert data["progress"] == ["Шаг 1"]
