"""Подсистема логирования взаимодействий с поддержкой ротации."""

from __future__ import annotations

import json
import logging
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

import gzip

logger = logging.getLogger(__name__)


class InteractionLogger:
    """Логгер, сохраняющий шаги работы агента в JSONL-файл."""

    def __init__(
        self,
        log_dir: Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        max_archives: int = 10,
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_archives = max_archives
        self._lock = Lock()

    def _log_file(self, user_id: str) -> Path:
        sanitized = user_id.replace("/", "_")
        return self.log_dir / f"{sanitized}.jsonl"

    def log_model_response(self, user_id: str, user_message: str, model_response: Dict[str, Any]) -> None:
        """Сохраняет блоки THOUGHT/ACTION/ASSISTANT от модели."""

        record = {
            "timestamp": _utc_now_iso(),
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
            "timestamp": _utc_now_iso(),
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
        serialized = json.dumps(record, ensure_ascii=False) + "\n"

        with self._lock:
            try:
                self._rotate_if_needed(path)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(serialized)
                logger.debug(
                    "Запись добавлена в лог",
                    extra={"user_id": user_id, "type": record.get("type")},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Не удалось записать лог итерации: %s",
                    exc,
                    extra={"user_id": user_id, "type": record.get("type")},
                )

    def _rotate_if_needed(self, path: Path) -> None:
        if not path.exists() or path.stat().st_size < self.max_bytes:
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        archive_name = f"{path.name}.{timestamp}.gz"
        archive_path = path.parent / archive_name

        with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        path.unlink(missing_ok=True)
        self._prune_archives(path)

    def _prune_archives(self, path: Path) -> None:
        pattern = f"{path.name}.*.gz"
        archives = sorted(
            path.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for extra in archives[self.max_archives :]:
            with suppress(FileNotFoundError):
                extra.unlink()


def _utc_now_iso() -> str:
    """Возвращает ISO-временную метку в UTC с суффиксом Z."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
