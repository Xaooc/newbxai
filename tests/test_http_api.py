"""Тесты для HTTP API-адаптера."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.adapters.http_api import (  # noqa: E402
    AuthAttemptTracker,
    HttpAdapterConfig,
    _normalize_tokens,
    is_token_allowed,
    run_http_server,
)


class FakeClock:
    """Простые часы для детерминированных тестов блокировок."""

    def __init__(self) -> None:
        self._now = None

    def set(self, dt):
        self._now = dt

    def advance(self, seconds: int) -> None:
        from datetime import timedelta

        if self._now is None:
            raise RuntimeError("Время не инициализировано")
        self._now += timedelta(seconds=seconds)

    def __call__(self):
        if self._now is None:
            raise RuntimeError("Время не установлено")
        return self._now


def test_normalize_tokens_removes_empty_and_duplicates() -> None:
    tokens = (" dev ", "", "dev", "test", "test ")
    normalized = _normalize_tokens(tokens)
    assert normalized == ("dev", "test")


def test_is_token_allowed_strips_value() -> None:
    allowed = ("alpha", "beta")
    assert is_token_allowed(" beta\n", allowed) is True
    assert is_token_allowed("", allowed) is False
    assert is_token_allowed("gamma", allowed) is False


def test_run_http_server_requires_tokens() -> None:
    config = HttpAdapterConfig(allowed_tokens=())
    with pytest.raises(RuntimeError):
        run_http_server(config)


def test_auth_attempt_tracker_blocks_and_resets() -> None:
    from datetime import datetime, timezone

    clock = FakeClock()
    clock.set(datetime(2025, 1, 1, tzinfo=timezone.utc))
    tracker = AuthAttemptTracker(
        max_attempts=3,
        window_seconds=60,
        block_seconds=120,
        now_provider=clock,
    )

    key = "192.0.2.10"
    for attempt in range(2):
        attempts, blocked_until = tracker.register_failure(key)
        assert attempts == attempt + 1
        assert blocked_until is None

    attempts, blocked_until = tracker.register_failure(key)
    assert attempts == 3
    assert blocked_until is not None
    assert tracker.is_blocked(key) == blocked_until

    tracker.register_success(key)
    assert tracker.is_blocked(key) is None

    tracker.register_failure(key)
    clock.advance(61)
    assert tracker.is_blocked(key) is None
