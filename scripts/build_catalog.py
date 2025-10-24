"""Строит индекс методов Bitrix24 на основе схем."""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MethodIndex:
    """Описывает метод в индексе."""

    method: str
    required: List[str]
    optional: List[str]


def build_index(schema_dir: pathlib.Path) -> Dict[str, MethodIndex]:
    """Сканирует директорию схем и собирает индекс."""

    index: Dict[str, MethodIndex] = {}
    for path in schema_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        method = data["method"]
        required = data.get("required", [])
        properties = data.get("properties", {})
        optional = [name for name in properties.keys() if name not in required]
        index[method] = MethodIndex(method=method, required=required, optional=optional)
    return index


def main() -> None:
    """Точка входа для генерации файла index.json."""

    root = pathlib.Path(__file__).resolve().parent.parent / "schemas"
    index = build_index(root)
    payload = {
        method: {
            "required": item.required,
            "optional": item.optional,
        }
        for method, item in index.items()
    }
    (root / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
