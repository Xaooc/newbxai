"""Общие настройки тестов."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Задать переменные окружения для тестов."""

    monkeypatch.setenv("BITRIX_BASE_URL", "https://portal.magnitmedia.ru/rest/132/1s0mz4mw8d42bfvk/")
    monkeypatch.setenv("BITRIX_DEFAULT_TZ", "Europe/Amsterdam")
