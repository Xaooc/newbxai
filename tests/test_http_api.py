"""Тесты для HTTP API-адаптера."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.adapters.http_api import (  # noqa: E402
    HttpAdapterConfig,
    is_token_allowed,
    run_http_server,
    _normalize_tokens,
)


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
