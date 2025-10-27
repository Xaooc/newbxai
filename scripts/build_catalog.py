"""Строит индекс методов Bitrix24 на основе схем с подробным логированием."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


LOGGER = logging.getLogger("build_catalog")


@dataclass
class MethodIndex:
    """Описывает метод в индексе."""

    method: str
    required: List[str]
    optional: List[str]


def configure_logging(verbose: bool) -> None:
    """Настраивает вывод логов в консоль."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def iter_schema_files(schema_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """Возвращает отсортированный список файлов схем."""

    return sorted(schema_dir.glob("*.json"))


def build_index(schema_dir: pathlib.Path) -> Dict[str, MethodIndex]:
    """Сканирует директорию схем и собирает индекс с логированием шагов."""

    index: Dict[str, MethodIndex] = {}
    LOGGER.info("Начинаем построение индекса по каталогу %s", schema_dir)
    for path in iter_schema_files(schema_dir):
        LOGGER.debug("Обработка файла схемы %s", path.name)
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data.get("method")
        if method is None:
            LOGGER.warning("Файл %s пропущен: отсутствует ключ 'method'", path.name)
            continue
        required = data.get("required", [])
        properties = data.get("properties", {})
        optional = [name for name in properties.keys() if name not in required]
        index[method] = MethodIndex(method=method, required=required, optional=optional)
        LOGGER.debug(
            "Добавлена запись метода %s (обязательных: %d, необязательных: %d)",
            method,
            len(required),
            len(optional),
        )
    LOGGER.info("Индекс построен, всего методов: %d", len(index))
    return index


def dump_index(index: Dict[str, MethodIndex], destination: pathlib.Path) -> None:
    """Сохраняет индекс в файл и логирует результат."""

    payload = {
        method: {
            "required": item.required,
            "optional": item.optional,
        }
        for method, item in index.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Файл индекса сохранён по пути %s", destination)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "schemas",
        help="Путь к директории со схемами (по умолчанию ./schemas)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Полный путь к файлу индекса (по умолчанию <schema-dir>/index.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Включить подробные (DEBUG) логи",
    )
    return parser.parse_args(argv)


def resolve_output_path(schema_dir: pathlib.Path, output: Optional[pathlib.Path]) -> pathlib.Path:
    """Определяет путь для сохранения индекса."""

    if output is None:
        return schema_dir / "index.json"
    if output.is_dir():
        return output / "index.json"
    return output


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Точка входа для генерации файла index.json с логированием."""

    args = parse_args(argv)
    configure_logging(args.verbose)
    schema_dir = args.schema_dir.expanduser()
    if not schema_dir.exists():
        LOGGER.error("Директория со схемами не найдена: %s", schema_dir)
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")
    if not schema_dir.is_dir():
        LOGGER.error("Указанный путь не является директорией: %s", schema_dir)
        raise NotADirectoryError(f"Schema path is not a directory: {schema_dir}")
    schema_dir = schema_dir.resolve()

    output_arg = args.output.expanduser() if args.output else None
    output_path = resolve_output_path(schema_dir, output_arg)
    output_path = output_path.resolve()

    LOGGER.info(
        "Запуск генерации индекса (каталог схем: %s, файл назначения: %s, подробные логи: %s)",
        schema_dir,
        output_path,
        "да" if args.verbose else "нет",
    )
    index = build_index(schema_dir)
    dump_index(index, output_path)
    LOGGER.info("Генерация завершена успешно")


if __name__ == "__main__":
    main()
