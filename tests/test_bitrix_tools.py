"""Тесты для инструментов Bitrix24."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from bitrix24_client import Bitrix24Error

from telegram_bitrix_agent.assistant.tools.bitrix import BitrixGateway, MethodCatalog


class DummyClient:
    """Минимальная заглушка клиента Bitrix24."""

    def __init__(self, responses: Dict[str, Any]) -> None:  # noqa: D401
        self._responses = responses

    def _request(self, method_name: str, http_method: str = "GET", **kwargs: Any) -> Any:  # noqa: ANN001, ANN003
        payload = self._responses.get(method_name)
        if isinstance(payload, Exception):
            raise payload
        return payload


def _write_schema(path: pathlib.Path, name: str, required: list[str], description: str) -> None:
    data = {
        "method": name,
        "required": required,
        "properties": {
            field: {"description": description}
            for field in required
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_method_catalog_skips_index(tmp_path: pathlib.Path) -> None:
    """Каталог должен пропускать вспомогательные файлы без ключа method."""

    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()

    (schema_dir / "index.json").write_text("{}", encoding="utf-8")
    _write_schema(schema_dir / "crm.deal.add.json", "crm.deal.add", ["fields"], "Название сделки")

    catalog = MethodCatalog(schema_dir)
    entry = catalog.get("crm.deal.add")
    assert entry.method == "crm.deal.add"
    results = catalog.search("Название сделки")
    assert results and results[0].method == "crm.deal.add"


def test_bitrix_gateway_call_handles_errors(tmp_path: pathlib.Path) -> None:
    """BitrixGateway.call должен нормализовывать исключения Bitrix24."""

    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    _write_schema(schema_dir / "user.current.json", "user.current", [], "")

    client = DummyClient({
        "user.current": Bitrix24Error("Недостаточно прав", http_status=403, b24_code="ACCESS_DENIED"),
    })
    gateway = BitrixGateway(client, schema_dir)
    response = gateway.call("user.current", {})
    assert response["ok"] is False
    assert response["status"] == 403
    assert response["error"]["code"] == "ACCESS_DENIED"

    client_success = DummyClient({"user.current": {"ID": 1}})
    gateway_success = BitrixGateway(client_success, schema_dir)
    result = gateway_success.call("user.current", {})
    assert result["ok"] is True
    assert result["data"] == {"ID": 1}
