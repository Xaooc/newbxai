"""Тесты для загрузки .env-файлов."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bitrix_agent.environment import load_env_file


def test_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / "test.env"
    env_file.write_text("FOO=bar\n# comment\nEMPTY=\nQUOTED='value'\n", encoding="utf-8")

    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    load_env_file(env_file)

    assert os.getenv("FOO") == "bar"
    assert os.getenv("QUOTED") == "value"
    assert "EMPTY" not in os.environ


def test_missing_env_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_env_file(tmp_path / "absent.env")
