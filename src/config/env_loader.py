"""Загрузка переменных окружения из `.env`-файла."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_env_file(path: str | Path) -> Dict[str, str]:
    """Читает `.env` и подставляет значения в `os.environ`.

    Возвращает словарь загруженных значений (включая пропущенные из-за
    уже установленного окружения). Невалидные строки игнорируются с
    предупреждением.
    """

    file_path = Path(path).expanduser().resolve()
    loaded: Dict[str, str] = {}

    if not file_path.exists():
        logger.info("Файл окружения %s не найден, пропускаем загрузку", file_path)
        return loaded

    logger.info("Загружаем переменные окружения из %s", file_path)

    for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            logger.warning(
                "Строка %s:%d пропущена: отсутствует разделитель '='", file_path, line_number
            )
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            logger.warning("Строка %s:%d пропущена: пустой ключ", file_path, line_number)
            continue
        loaded[key] = value
        if key in os.environ:
            logger.debug(
                "Переменная %s уже определена во внешнем окружении, оставляем текущее значение", key
            )
            continue
        os.environ[key] = value
    return loaded
