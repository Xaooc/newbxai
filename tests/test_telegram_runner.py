"""Тесты модуля запуска Telegram-бота."""

from pathlib import Path

import pytest
from src.app.telegram_runner import (
    TelegramTokenMissingError,
    _build_config_from_env,
    _prepare_bot_dependencies,
    _read_poll_interval,
    _read_worker_threads,
    launch_telegram_bot,
)


def test_build_config_from_env_requires_token(monkeypatch):
    """При отсутствии токена должно выбрасываться исключение."""

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(TelegramTokenMissingError):
        _build_config_from_env()


def test_build_config_from_env_reads_optional_fields(monkeypatch):
    """Дополнительные настройки считываются из окружения."""

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_ERROR_CHAT_ID", "321")
    monkeypatch.setenv("STATE_DIR", "./tmp/state")
    monkeypatch.setenv("LOG_DIR", "./tmp/logs")
    monkeypatch.setenv("TELEGRAM_WORKER_THREADS", "12")
    config = _build_config_from_env()

    assert config.error_chat_id == 321
    assert config.state_dir == Path("./tmp/state")
    assert config.log_dir == Path("./tmp/logs")
    assert config.worker_threads == 12


def test_prepare_bot_dependencies_loads_env(monkeypatch):
    """Подготовка зависимостей должна загружать окружение и инициализировать оркестратор."""

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    calls: dict[str, object] = {}

    def fake_load_env(path: str | Path | None) -> None:  # noqa: D401
        calls["env"] = path

    monkeypatch.setattr("src.app.telegram_runner.load_env_file", fake_load_env)

    fake_orchestrator = object()

    def fake_build(mode: str, storage_dir: Path, log_dir: Path):  # noqa: D401
        calls["mode"] = mode
        calls["storage"] = storage_dir
        calls["logs"] = log_dir
        return fake_orchestrator

    monkeypatch.setattr("src.app.telegram_runner.build_orchestrator", fake_build)

    orchestrator, config = _prepare_bot_dependencies(".env.test")

    assert calls["env"] == ".env.test"
    assert orchestrator is fake_orchestrator
    assert config.token == "token"


def test_read_poll_interval_handles_invalid_values() -> None:
    """Некорректные значения интервала должны приводить к 0.0."""

    assert _read_poll_interval(None) == 0.0
    assert _read_poll_interval("") == 0.0
    assert _read_poll_interval("abc") == 0.0
    assert _read_poll_interval("-5") == 0.0
    assert _read_poll_interval("1.5") == 1.5


def test_read_worker_threads_validates_values() -> None:
    """Размер пула потоков должен оставаться положительным целым."""

    assert _read_worker_threads(None) == 8
    assert _read_worker_threads("") == 8
    assert _read_worker_threads("abc") == 8
    assert _read_worker_threads("0") == 8
    assert _read_worker_threads("-3") == 8
    assert _read_worker_threads("5") == 5


def test_launch_returns_error_code_on_missing_token(monkeypatch):
    """Если адаптер создать не удалось, функция должна вернуть 1."""

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    result = launch_telegram_bot(None)
    assert result == 1


def test_launch_runs_adapter(monkeypatch):
    """Успешный запуск должен вызвать метод run адаптера."""

    class DummyAdapter:
        def __init__(self) -> None:
            self.started = False

        def run(self) -> None:
            self.started = True

    dummy = DummyAdapter()

    def fake_create(env_path: str | Path | None = ".env") -> DummyAdapter:  # noqa: D401
        return dummy

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setattr("src.app.telegram_runner.create_telegram_adapter", fake_create)

    result = launch_telegram_bot(None)

    assert result == 0
    assert dummy.started is True
