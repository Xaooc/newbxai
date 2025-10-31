"""Управление состоянием agent_state."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Представление состояния агента."""

    goals: List[str] = field(default_factory=list)
    done: List[Dict[str, Any]] = field(default_factory=list)
    in_progress: List[Dict[str, Any]] = field(default_factory=list)
    objects: Dict[str, Any] = field(
        default_factory=lambda: {
            "current_deal_id": None,
            "current_contact_id": None,
            "current_company_id": None,
            "current_task_id": None,
        }
    )
    next_planned_actions: List[Dict[str, Any]] = field(default_factory=list)
    confirmations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует состояние в словарь для сериализации."""

        return {
            "goals": self.goals,
            "done": self.done,
            "in_progress": self.in_progress,
            "objects": self.objects,
            "next_planned_actions": self.next_planned_actions,
            "confirmations": self.confirmations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """Создаёт состояние из словаря."""

        return cls(
            goals=list(data.get("goals", [])),
            done=list(data.get("done", [])),
            in_progress=list(data.get("in_progress", [])),
            objects=dict(
                {
                    "current_deal_id": None,
                    "current_contact_id": None,
                    "current_company_id": None,
                    "current_task_id": None,
                },
                **data.get("objects", {}),
            ),
            next_planned_actions=list(data.get("next_planned_actions", [])),
            confirmations=dict(data.get("confirmations", {})),
        )


class AgentStateManager:
    """Класс для чтения и записи состояния на диск."""

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _state_file(self, user_id: str) -> Path:
        """Возвращает путь к файлу состояния для конкретного пользователя."""

        sanitized = user_id.replace("/", "_")
        return self.storage_dir / f"{sanitized}.json"

    def load_state(self, user_id: str) -> AgentState:
        """Загружает состояние из файла. Если файла нет — возвращает пустое состояние."""

        path = self._state_file(user_id)
        if not path.exists():
            logger.debug("Файл состояния не найден, создаём пустой", extra={"user_id": user_id})
            return AgentState()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Ошибка чтения состояния, возвращаем пустое", extra={"user_id": user_id, "error": str(exc)})
            return AgentState()

        return AgentState.from_dict(data)

    def save_state(self, user_id: str, state: AgentState) -> None:
        """Сохраняет состояние в файл."""

        path = self._state_file(user_id)
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Состояние сохранено", extra={"user_id": user_id})
