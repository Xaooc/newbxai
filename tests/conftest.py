"""Общие настройки тестов: путь импорта и заглушка `requests`."""

from __future__ import annotations

import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _DummyResponse:  # pragma: no cover - запасной вариант
        status_code = 200
        text = ""

        def json(self) -> dict:
            return {}

    def _not_implemented(*args, **kwargs):  # pragma: no cover - не должен вызываться
        raise RuntimeError("HTTP-вызовы в тестах не должны выполняться")

    requests_stub.post = _not_implemented
    requests_stub.get = _not_implemented
    requests_stub.RequestException = Exception
    sys.modules["requests"] = requests_stub


if "telegram" not in sys.modules:
    telegram_module = types.ModuleType("telegram")
    telegram_module.Message = type("Message", (), {})
    telegram_module.Update = type("Update", (), {})
    sys.modules["telegram"] = telegram_module

    constants_module = types.ModuleType("telegram.constants")
    constants_module.ChatAction = types.SimpleNamespace(TYPING="typing")
    sys.modules["telegram.constants"] = constants_module

    class _DummyFilter:
        def __and__(self, other):  # pragma: no cover - простая заглушка
            return self

        def __rand__(self, other):  # pragma: no cover
            return self

        def __invert__(self):  # pragma: no cover
            return self

    filters_module = types.ModuleType("telegram.ext.filters")
    filters_module.COMMAND = _DummyFilter()
    filters_module.TEXT = _DummyFilter()

    class _DummyApplication:
        def __init__(self) -> None:  # pragma: no cover - инфраструктурный код
            self.bot = types.SimpleNamespace(
                send_message=lambda *args, **kwargs: None,
                send_chat_action=lambda *args, **kwargs: None,
            )

        def add_handler(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def add_error_handler(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

        def run_polling(self, *args, **kwargs) -> None:  # pragma: no cover
            return None

    class _ApplicationBuilder:
        def __init__(self) -> None:
            self._token = ""

        def token(self, token: str) -> "_ApplicationBuilder":  # pragma: no cover
            self._token = token
            return self

        def build(self) -> _DummyApplication:  # pragma: no cover
            return _DummyApplication()

    class _CommandHandler:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            pass

    class _MessageHandler:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            pass

    context_types = types.SimpleNamespace(DEFAULT_TYPE=object())

    ext_module = types.ModuleType("telegram.ext")
    ext_module.ApplicationBuilder = _ApplicationBuilder
    ext_module.CommandHandler = _CommandHandler
    ext_module.ContextTypes = types.SimpleNamespace(DEFAULT_TYPE=context_types.DEFAULT_TYPE)
    ext_module.MessageHandler = _MessageHandler
    ext_module.filters = filters_module
    sys.modules["telegram.ext"] = ext_module

    # Совместимость с доступом вида `from telegram.ext import filters`
    sys.modules["telegram.ext.filters"] = filters_module
