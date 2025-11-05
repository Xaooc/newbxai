"""Загрузка настроек окружения из файла."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_env_file(path: str | os.PathLike[str]) -> None:
    """Загрузить переменные окружения из файла формата KEY=VALUE."""

    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Файл окружения {env_path} не найден")

    for line in _read_lines(env_path):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _read_lines(path: Path) -> Iterable[str]:
    """Прочитать файл построчно без символов перевода строки."""

    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            yield raw_line.strip()
