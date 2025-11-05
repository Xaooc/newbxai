"""Память задачи агента."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentMemory:
    """Хранит ключевые элементы состояния задачи."""

    goal: str = ""
    knowns: Dict[str, Any] = field(default_factory=dict)
    unknowns: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    progress: List[str] = field(default_factory=list)
    next: str | None = None
    risks: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        """Преобразовать память к JSON-структуре."""

        return {
            "goal": self.goal,
            "knowns": self.knowns,
            "unknowns": self.unknowns,
            "plan": self.plan,
            "progress": self.progress,
            "next": self.next,
            "risks": self.risks,
        }

    def update_next(self) -> None:
        """Обновить поле next на основании плана."""

        self.next = self.plan[0] if self.plan else None

    def mark_step_done(self, description: str) -> None:
        """Перенести выполненный шаг в progress."""

        if self.plan and self.plan[0] == description:
            self.plan.pop(0)
        self.progress.append(description)
        self.update_next()

    def add_risk(self, risk: str) -> None:
        """Добавить описание риска."""

        if risk not in self.risks:
            self.risks.append(risk)
