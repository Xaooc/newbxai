"""Подсистема логирования взаимодействий с поддержкой ротации."""

from __future__ import annotations

import gzip
import json
import logging
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ArchiveUploadError(RuntimeError):
    """Исключение, возникающее при неудачной загрузке архива."""


class ArchiveUploader(Protocol):
    """Протокол для внешних хранилищ архивов логов."""

    def upload(
        self,
        archive_path: Path,
        *,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """Передаёт архив во внешнее хранилище."""


@dataclass
class ArchiveUploadMonitor:
    """Простейший мониторинг выгрузки архивов."""

    success: int = 0
    failures: int = 0
    retries: int = 0

    def report_success(self) -> None:
        self.success += 1

    def report_failure(self) -> None:
        self.failures += 1

    def report_retry(self) -> None:
        self.retries += 1

    def snapshot(self) -> Dict[str, int]:
        """Возвращает текущие счётчики."""

        return {"success": self.success, "failures": self.failures, "retries": self.retries}


class InteractionLogger:
    """Логгер, сохраняющий шаги работы агента в JSONL-файл."""

    def __init__(
        self,
        log_dir: Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        max_archives: int = 10,
        archive_uploader: Optional[ArchiveUploader] = None,
        monitor: Optional[ArchiveUploadMonitor] = None,
        upload_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.max_archives = max_archives
        self._lock = Lock()
        self._archive_uploader = archive_uploader
        self._monitor = monitor or ArchiveUploadMonitor()
        self._upload_attempts = max(1, upload_attempts)
        self._sleep = sleep_fn

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
                self._rotate_if_needed(user_id, path)
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

    def _rotate_if_needed(self, user_id: str, path: Path) -> None:
        if not path.exists() or path.stat().st_size < self.max_bytes:
            return

        timestamp_raw = datetime.now(UTC)
        timestamp_iso = timestamp_raw.isoformat().replace("+00:00", "Z")
        archive_suffix = timestamp_raw.strftime("%Y%m%dT%H%M%S")
        archive_name = f"{path.name}.{archive_suffix}.gz"
        archive_path = path.parent / archive_name

        with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        path.unlink(missing_ok=True)
        self._upload_archive(archive_path, user_id=user_id, timestamp=timestamp_iso)
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

    def _upload_archive(
        self,
        archive_path: Path,
        *,
        user_id: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        if not self._archive_uploader:
            return

        self._attempt_upload(archive_path, user_id=user_id, timestamp=timestamp)

    def sync_pending_archives(self, user_id: Optional[str] = None) -> None:
        """Отправляет существующие архивы в хранилище, если оно настроено."""

        if not self._archive_uploader:
            return

        archives = self._collect_archives(user_id)
        for archive_path in archives:
            archive_user, archive_timestamp = self._extract_archive_metadata(archive_path)
            try:
                self._attempt_upload(
                    archive_path,
                    user_id=archive_user,
                    timestamp=archive_timestamp,
                )
            except ArchiveUploadError:
                continue

    def _collect_archives(self, user_id: Optional[str]) -> List[Path]:
        if user_id:
            base = self._log_file(user_id).name
            pattern = f"{base}.*.gz"
            return sorted(self.log_dir.glob(pattern))
        return sorted(self.log_dir.glob("*.jsonl.*.gz"))

    def _extract_archive_metadata(self, archive_path: Path) -> tuple[Optional[str], Optional[str]]:
        stem = archive_path.with_suffix("").name  # удаляем .gz
        base, _, suffix = stem.rpartition(".jsonl.")
        user = base or None
        timestamp: Optional[str] = None
        if suffix:
            try:
                parsed = datetime.strptime(suffix, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
                timestamp = parsed.isoformat().replace("+00:00", "Z")
            except ValueError:
                timestamp = None
        return user, timestamp

    @property
    def archive_monitor(self) -> ArchiveUploadMonitor:
        """Возвращает мониторинг выгрузки архивов."""

        return self._monitor

    def _attempt_upload(
        self,
        archive_path: Path,
        *,
        user_id: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        if not self._archive_uploader:
            return

        attempts = 0
        delay = 0.5

        while attempts < self._upload_attempts:
            try:
                self._archive_uploader.upload(
                    archive_path,
                    user_id=user_id,
                    timestamp=timestamp,
                )
                self._monitor.report_success()
                logger.info(
                    "Архив логов отправлен во внешнее хранилище",
                    extra={"user_id": user_id, "archive": archive_path.name},
                )
                self._log_monitor_snapshot()
                return
            except ArchiveUploadError as exc:  # noqa: PERF203
                attempts += 1
                if attempts >= self._upload_attempts:
                    self._monitor.report_failure()
                    logger.warning(
                        "Не удалось выгрузить архив логов: %s",
                        exc,
                        extra={"user_id": user_id, "archive": archive_path.name},
                    )
                    self._log_monitor_snapshot(level=logging.WARNING)
                    return
                self._monitor.report_retry()
                logger.info(
                    "Повторяем попытку выгрузки архива (попытка %s из %s)",
                    attempts + 1,
                    self._upload_attempts,
                    extra={"user_id": user_id, "archive": archive_path.name},
                )
                self._log_monitor_snapshot()
                self._sleep(delay)
                delay = min(delay * 2, 4.0)

    def _log_monitor_snapshot(self, level: int = logging.INFO) -> None:
        logger.log(level, "Статистика отправки архивов: %s", self._monitor.snapshot())


def build_interaction_logger(
    log_dir: Path,
    *,
    max_bytes: int = 5 * 1024 * 1024,
    max_archives: int = 10,
    archive_uploader: Optional[ArchiveUploader] = None,
    monitor: Optional[ArchiveUploadMonitor] = None,
) -> InteractionLogger:
    """Создаёт `InteractionLogger`, автоматически подключая загрузчик из окружения."""

    if archive_uploader is None:
        from .archive_uploader import build_archive_uploader_from_env

        archive_uploader = build_archive_uploader_from_env()

    logger_instance = InteractionLogger(
        log_dir=log_dir,
        max_bytes=max_bytes,
        max_archives=max_archives,
        archive_uploader=archive_uploader,
        monitor=monitor,
    )
    logger_instance.sync_pending_archives()
    return logger_instance


def _utc_now_iso() -> str:
    """Возвращает ISO-временную метку в UTC с суффиксом Z."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
