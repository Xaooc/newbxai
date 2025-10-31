"""Простейшая подсистема логирования взаимодействий."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Логгер, сохраняющий шаги работы агента в JSONL-файл."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_file(self, user_id: str) -> Path:
        sanitized = user_id.replace("/", "_")
        return self.log_dir / f"{sanitized}.jsonl"

    def log_model_response(self, user_id: str, user_message: str, model_response: Dict[str, Any]) -> None:
        """Сохраняет блоки THOUGHT/ACTION/ASSISTANT от модели."""

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "model_response",
            "user_message": user_message,
            "response": model_response,
        }
        self._append_record(user_id, record)

    def log_iteration(
        self,
        user_id: str,
        user_message: str,
        model_response: Dict[str, Any],
        state: Any,
        executed_actions: List[Dict[str, Any]],
        errors: List[str],
    ) -> None:
        """Логирует итог итерации обработки сообщения."""

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "iteration",
            "user_message": user_message,
            "model_response": model_response,
            "executed_actions": executed_actions,
            "errors": errors,
            "state": state.to_dict() if hasattr(state, "to_dict") else state,
        }
        self._append_record(user_id, record)

    def _append_record(self, user_id: str, record: Dict[str, Any]) -> None:
        path = self._log_file(user_id)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("Запись добавлена в лог", extra={"user_id": user_id, "type": record.get("type")})
