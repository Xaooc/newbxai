"""Тесты для подсистемы логирования."""

from __future__ import annotations

from pathlib import Path

from src.logging.logger import InteractionLogger


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
