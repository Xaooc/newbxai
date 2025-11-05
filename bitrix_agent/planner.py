"""Простой планировщик шагов агента."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .memory import AgentMemory


@dataclass
class PlanStep:
    """Описание одного шага."""

    description: str
    action: str


class SimplePlanner:
    """Формирует последовательность шагов на основании цели."""

    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory

    def build_plan(self, goal: str) -> List[PlanStep]:
        """Сформировать план по тексту цели."""

        steps: List[PlanStep] = []
        lowered = goal.lower()
        if "задач" in lowered or "task" in lowered:
            steps.extend(self._tasks_plan(goal))
        elif "сделк" in lowered or "deal" in lowered:
            steps.extend(self._deals_plan(goal))
        else:
            steps.append(PlanStep("Собрать вводные данные", "collect_context"))
            steps.append(PlanStep("Выполнить действие по описанию цели", "generic_action"))
        self.memory.plan = [step.description for step in steps]
        self.memory.update_next()
        return steps

    def _tasks_plan(self, goal: str) -> List[PlanStep]:
        steps: List[PlanStep] = [
            PlanStep("Проверить существующие задачи", "tasks.task.list"),
            PlanStep("Подготовить поля задачи", "prepare_task_fields"),
        ]
        if re.search(r"созд", goal.lower()):
            steps.append(PlanStep("Создать задачу", "tasks.task.add"))
        else:
            steps.append(PlanStep("Обновить задачу", "tasks.task.update"))
        steps.append(PlanStep("Подготовить отчёт", "summarize"))
        return steps

    def _deals_plan(self, goal: str) -> List[PlanStep]:
        steps: List[PlanStep] = [
            PlanStep("Уточнить параметры сделки", "collect_deal_context"),
            PlanStep("Проверить существующую сделку", "crm.deal.list"),
            PlanStep("Получить доступные стадии сделки", "collect_deal_stages"),
        ]
        if re.search(r"созд", goal.lower()):
            steps.append(PlanStep("Создать сделку", "crm.deal.add"))
        else:
            steps.append(PlanStep("Обновить сделку", "crm.deal.update"))
        steps.append(PlanStep("Подготовить отчёт", "summarize"))
        return steps
