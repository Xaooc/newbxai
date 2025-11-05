import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.app_logging.logger import InteractionLogger
from src.bitrix_client.client import BitrixClientError
from src.orchestrator.agent import Orchestrator, OrchestratorSettings
from src.state.manager import AgentStateManager


class FakeModelClient:
    """Простая заглушка GPT-клиента, возвращающая фиксированный ответ."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.captured_calls: List[Dict[str, Any]] = []

    def generate(
        self,
        system_prompt: str,
        state_snapshot: Dict[str, Any],
        user_message: str,
    ) -> str:  # noqa: D401
        self.captured_calls.append(
            {
                "system_prompt": system_prompt,
                "state": state_snapshot,
                "message": user_message,
            }
        )
        return self._response_text


def build_model_text(actions: List[Dict[str, Any]], assistant: str = "Готово") -> str:
    """Формирует текст ответа модели в нужном формате."""

    action_json = json.dumps(actions, ensure_ascii=False)
    return (
        "THOUGHT:\n"
        "- План действий\n"
        "ACTION:\n"
        f"{action_json}\n"
        "ASSISTANT:\n"
        f"{assistant}\n"
    )


@pytest.fixture()
def orchestrator_factory(tmp_path: Path):
    """Фабрика оркестратора с временными каталогами."""

    def _factory(mode: str, model_text: str, **settings_kwargs) -> Orchestrator:
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "logs"
        state_manager = AgentStateManager(storage_dir=state_dir)
        interaction_logger = InteractionLogger(log_dir=log_dir, max_bytes=1024)
        settings = OrchestratorSettings(mode=mode, **settings_kwargs)
        model_client = FakeModelClient(model_text)
        return Orchestrator(
            state_manager=state_manager,
            interaction_logger=interaction_logger,
            settings=settings,
            model_client=model_client,
        )

    return _factory


def test_shadow_mode_saves_plan_without_execution(orchestrator_factory, monkeypatch):
    """В режиме shadow шаги не выполняются, но план сохраняется в состоянии."""

    called = False

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        nonlocal called
        called = True
        return {"result": 1}

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    actions = [
        {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Тест"}}, "comment": "создаём"}
    ]
    orchestrator = orchestrator_factory("shadow", build_model_text(actions, assistant="План готов"))

    reply = orchestrator.process_message("user-shadow", "Создай сделку")

    assert "План готов" in reply
    assert called is False

    state = orchestrator.state_manager.load_state("user-shadow")
    assert state.last_plan["actions"] == actions
    assert state.done == []


def test_full_mode_executes_plan_immediately(orchestrator_factory, monkeypatch):
    """В рабочем режиме шаги выполняются сразу после получения плана."""

    recorded: List[Dict[str, Any]] = []

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        recorded.append({"method": method, "params": params, "http_method": http_method})
        return {"result": 321}

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    actions = [
        {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Новая"}}, "comment": "создаём"}
    ]
    orchestrator = orchestrator_factory("full", build_model_text(actions, assistant="План выполнен"))

    reply = orchestrator.process_message("user-full", "Создай сделку")

    assert recorded == [
        {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Новая"}}, "http_method": "POST"}
    ]
    assert "Что сделано" in reply
    state = orchestrator.state_manager.load_state("user-full")
    assert state.objects["current_deal_id"] == 321
    assert state.done, "История должна пополниться выполненным действием"


def test_risk_warning_added_for_sensitive_fields(orchestrator_factory, monkeypatch):
    """Агент предупреждает о изменении критичных полей без запроса подтверждения."""

    monkeypatch.setattr(
        "src.orchestrator.agent.call_bitrix",
        lambda method, params, http_method="POST": {"result": True},
    )

    actions = [
        {
            "method": "crm.deal.update",
            "params": {"id": 100, "fields": {"OPPORTUNITY": 90000}},
            "comment": "обновляем сумму",
        }
    ]
    orchestrator = orchestrator_factory(
        "full", build_model_text(actions, assistant="Сумма скорректирована")
    )

    reply = orchestrator.process_message("user-risk", "Обнови сумму сделки")

    assert "изменяет сумму" in reply
    assert "⚠️" in reply


def test_bitrix_error_is_reported_to_user(orchestrator_factory, monkeypatch):
    """Ошибка Bitrix24 преобразуется в понятное сообщение пользователю."""

    def failing_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        raise BitrixClientError("Bitrix вернул ошибку authorisation failed")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", failing_call)

    actions = [
        {"method": "crm.deal.get", "params": {"id": 555}, "comment": "читаем сделку"}
    ]
    orchestrator = orchestrator_factory("full", build_model_text(actions, assistant="Пробую"))

    reply = orchestrator.process_message("user-error", "Покажи сделку 555")

    assert "Не удалось выполнить действие" in reply
    assert "Bitrix24" in reply


def test_missing_fields_metrics_updated(orchestrator_factory):
    """Отсутствующие обязательные поля попадают в метрики состояния."""

    actions = [
        {
            "method": "crm.activity.add",
            "params": {"fields": {"OWNER_TYPE_ID": 1, "OWNER_ID": 1, "TYPE_ID": 1}},
            "comment": "создаём дело",
        }
    ]
    orchestrator = orchestrator_factory("full", build_model_text(actions))

    orchestrator.process_message("user-metrics", "Создай дело")
    state = orchestrator.state_manager.load_state("user-metrics")

    missing = state.metrics.get("missing_fields", {})
    assert missing
    assert any("SUBJECT" in key for key in missing.keys())


def test_risk_warning_metrics_updated(orchestrator_factory, monkeypatch):
    """Предупреждения о рискованных полях учитываются в метриках."""

    monkeypatch.setattr(
        "src.orchestrator.agent.call_bitrix",
        lambda method, params, http_method="POST": {"result": True},
    )

    actions = [
        {
            "method": "crm.deal.update",
            "params": {"id": 10, "fields": {"OPPORTUNITY": 50000}},
            "comment": "обновляем сумму",
        }
    ]
    orchestrator = orchestrator_factory("full", build_model_text(actions))

    orchestrator.process_message("user-risk", "Обнови сумму")
    state = orchestrator.state_manager.load_state("user-risk")

    risk_stats = state.metrics.get("risk_warnings", {})
    assert risk_stats.get("OPPORTUNITY") == 1


def test_retry_attempts_reported(orchestrator_factory, monkeypatch):
    """Число попыток вызывается в ответе при ошибке Bitrix24."""

    attempts = {"count": 0}

    def failing_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        attempts["count"] += 1
        raise BitrixClientError("temporary failure")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", failing_call)

    actions = [
        {"method": "crm.deal.get", "params": {"id": 1}, "comment": "читаем сделку"}
    ]
    orchestrator = orchestrator_factory(
        "full",
        build_model_text(actions),
        bitrix_max_retries=2,
        bitrix_retry_base_delay=0.0,
    )

    reply = orchestrator.process_message("user-retry", "Покажи сделку")

    assert "Попыток: 3" in reply
    assert attempts["count"] == 3
