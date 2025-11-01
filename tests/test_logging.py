"""Тесты для подсистемы логирования."""

from __future__ import annotations

from pathlib import Path

from src.app_logging.logger import (
    ArchiveUploadError,
    ArchiveUploadMonitor,
    InteractionLogger,
)


class DummyUploader:
    """Загрузчик-заглушка, фиксирующий вызовы."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str | None, str | None]] = []

    def upload(self, archive_path: Path, *, user_id: str | None = None, timestamp: str | None = None) -> None:
        self.calls.append((archive_path, user_id, timestamp))


def test_rotation_triggers_external_upload(tmp_path):
    """При ротации должен вызываться загрузчик архивов."""

    uploader = DummyUploader()
    logger = InteractionLogger(log_dir=tmp_path, max_bytes=200, archive_uploader=uploader)

    # Первая запись создаёт файл, вторая приводит к ротации.
    logger.log_iteration(
        "user-1",
        "первое сообщение" + " x" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )

    logger.log_iteration(
        "user-1",
        "второе сообщение" + " y" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )

    assert uploader.calls, "Загрузчик должен быть вызван при ротации"
    archive_path, user_id, timestamp = uploader.calls[-1]
    assert archive_path.name.endswith(".gz")
    assert user_id == "user-1"
    assert timestamp is not None


def test_sync_pending_archives_uploads_existing_files(tmp_path):
    """Метод синхронизации должен выгружать ранее созданные архивы."""

    uploader = DummyUploader()
    logger = InteractionLogger(log_dir=tmp_path, archive_uploader=uploader)

    archive_path = tmp_path / "user-2.jsonl.20240101T120000.gz"
    archive_path.write_bytes(b"test")

    logger.sync_pending_archives()

    assert uploader.calls, "Должен быть выгружен найденный архив"
    paths = {call[0] for call in uploader.calls}
    assert archive_path in paths


class FlakyUploader:
    """Загрузчик, который может временно падать."""

    def __init__(self, fail_before_success: int) -> None:
        self.fail_before_success = fail_before_success
        self.calls: int = 0

    def upload(self, archive_path: Path, *, user_id: str | None = None, timestamp: str | None = None) -> None:
        self.calls += 1
        if self.calls <= self.fail_before_success:
            raise ArchiveUploadError("temporary error")


def test_archive_upload_retries_recorded(tmp_path):
    """Ретраи при временной ошибке увеличивают счётчики мониторинга."""

    uploader = FlakyUploader(fail_before_success=2)
    monitor = ArchiveUploadMonitor()
    logger = InteractionLogger(
        log_dir=tmp_path,
        max_bytes=200,
        archive_uploader=uploader,
        monitor=monitor,
        upload_attempts=3,
        sleep_fn=lambda _: None,
    )

    logger.log_iteration(
        "user-3",
        "сообщение" + " x" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )
    logger.log_iteration(
        "user-3",
        "второе" + " y" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )

    assert uploader.calls == 3
    snapshot = monitor.snapshot()
    assert snapshot["success"] == 1
    assert snapshot["retries"] == 2
    assert snapshot["failures"] == 0


def test_archive_upload_final_failure_is_tracked(tmp_path):
    """Если все попытки исчерпаны, фиксируется неудача, но исключение не выбрасывается."""

    uploader = FlakyUploader(fail_before_success=5)
    monitor = ArchiveUploadMonitor()
    logger = InteractionLogger(
        log_dir=tmp_path,
        max_bytes=200,
        archive_uploader=uploader,
        monitor=monitor,
        upload_attempts=2,
        sleep_fn=lambda _: None,
    )

    logger.log_iteration(
        "user-4",
        "сообщение" + " x" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )
    logger.log_iteration(
        "user-4",
        "второе" + " y" * 80,
        {"THOUGHT": "", "ACTION": [], "ASSISTANT": ""},
        state={"objects": {}},
        executed_actions=[],
        errors=[],
    )

    snapshot = monitor.snapshot()
    assert snapshot["success"] == 0
    assert snapshot["failures"] == 1
    assert snapshot["retries"] >= 1
