"""Инструменты Bitrix24 для LangGraph."""
from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping

# orjson быстрее, но в тестовой среде можно использовать стандартный json.
try:
    import orjson
except ImportError:  # pragma: no cover
    import json as orjson  # type: ignore

from bitrix24_client import Bitrix24Client, Bitrix24Error


@dataclass
class MethodCandidate:
    """Кандидат метода, найденный поиском."""

    method: str
    score: float
    required_fields: List[str]
    optional_fields: List[str]


@dataclass
class SchemaEntry:
    """Схема метода из каталога."""

    method: str
    required: List[str]
    properties: Dict[str, Any]
    description: str | None = None


class MethodCatalog:
    """Загружает JSON-схемы методов Bitrix24."""

    def __init__(self, directory: pathlib.Path) -> None:
        self._directory = directory
        self._cache: Dict[str, SchemaEntry] = {}
        self._index: Dict[str, List[str]] = {}
        self._load()

    def _load(self) -> None:
        for path in self._directory.glob("*.json"):
            data = orjson.loads(path.read_bytes())
            method_name = data.get("method")
            if not method_name:
                # Индекс и вспомогательные файлы не содержат ключа method — пропускаем их.
                continue
            entry = SchemaEntry(
                method=method_name,
                required=list(data.get("required", [])),
                properties=dict(data.get("properties", {})),
                description=data.get("description"),
            )
            self._cache[entry.method] = entry
            keywords = self._extract_keywords(entry)
            self._index[entry.method] = keywords

    @staticmethod
    def _extract_keywords(entry: SchemaEntry) -> List[str]:
        tokens: List[str] = []
        tokens.extend(entry.method.replace(".", " ").split())
        tokens.extend(entry.required)
        for key, value in entry.properties.items():
            tokens.append(key)
            if isinstance(value, Mapping):
                if "description" in value and isinstance(value["description"], str):
                    tokens.extend(value["description"].split())
                if "title" in value and isinstance(value["title"], str):
                    tokens.extend(value["title"].split())
        return [token.lower() for token in tokens if isinstance(token, str)]

    def search(self, query: str, top_k: int = 5) -> List[MethodCandidate]:
        terms = [item.lower() for item in query.split() if item]
        scores: List[MethodCandidate] = []
        for method, keywords in self._index.items():
            overlap = sum(1 for term in terms if term in keywords)
            if overlap == 0:
                continue
            norm = math.sqrt(len(keywords))
            score = overlap / (norm or 1.0)
            entry = self._cache[method]
            scores.append(
                MethodCandidate(
                    method=method,
                    score=score,
                    required_fields=list(entry.required),
                    optional_fields=[name for name in entry.properties.keys() if name not in entry.required],
                )
            )
        scores.sort(key=lambda candidate: candidate.score, reverse=True)
        return scores[:top_k]

    def get(self, method: str) -> SchemaEntry:
        if method not in self._cache:
            raise KeyError(method)
        return self._cache[method]


class BitrixGateway:
    """Инкапсулирует доступ к Bitrix24 и схеме."""

    def __init__(self, client: Bitrix24Client, schema_dir: str | pathlib.Path) -> None:
        self._client = client
        self._catalog = MethodCatalog(pathlib.Path(schema_dir))

    def search_methods(self, query: str) -> Dict[str, Any]:
        candidates = self._catalog.search(query)
        return {
            "candidates": [
                {
                    "method": candidate.method,
                    "score": candidate.score,
                    "required_fields": candidate.required_fields,
                    "optional_fields": candidate.optional_fields,
                }
                for candidate in candidates
            ]
        }

    def schema(self, method: str) -> Dict[str, Any]:
        entry = self._catalog.get(method)
        return {
            "input_schema": {
                "required": entry.required,
                "properties": entry.properties,
            },
            "description": entry.description or "",
        }

    def call(self, method: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        payload: MutableMapping[str, Any] = dict(params)
        try:
            result = self._client._request(method, "POST", json_body=payload)  # noqa: SLF001
        except Bitrix24Error as exc:
            return {
                "ok": False,
                "status": exc.http_status or 500,
                "error": {
                    "code": exc.b24_code or "UNKNOWN",
                    "message": str(exc),
                },
            }
        return {
            "ok": True,
            "status": 200,
            "data": result,
        }


__all__ = ["BitrixGateway", "MethodCatalog", "MethodCandidate"]
