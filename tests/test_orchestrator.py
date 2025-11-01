import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Tuple

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging.logger import InteractionLogger
from src.orchestrator.agent import Orchestrator, OrchestratorSettings
from src.state.manager import AgentState, AgentStateManager


class FakeModelClient:
    """Заглушка GPT-клиента, возвращающая заранее заданный ответ."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.captured_calls: List[Dict[str, Any]] = []

    def generate(self, system_prompt: str, state_snapshot: Dict[str, Any], user_message: str) -> str:  # noqa: D401
        self.captured_calls.append(
            {
                "system_prompt": system_prompt,
                "state": state_snapshot,
                "message": user_message,
            }
        )
        return self._response_text


@pytest.fixture()
def orchestrator_factory(tmp_path: Path):
    """Фабрика оркестратора для тестов."""

    def _factory(mode: str, model_text: str) -> Orchestrator:
        state_dir = tmp_path / "state"
        log_dir = tmp_path / "logs"
        state_manager = AgentStateManager(storage_dir=state_dir)
        interaction_logger = InteractionLogger(log_dir=log_dir, max_bytes=1024)
        settings = OrchestratorSettings(mode=mode)
        model_client = FakeModelClient(model_text)
        return Orchestrator(
            state_manager=state_manager,
            interaction_logger=interaction_logger,
            settings=settings,
            model_client=model_client,
        )

    return _factory


def build_model_text(actions: List[Dict[str, Any]], assistant: str = "Готово") -> str:
    """Формирует текст ответа модели с нужными блоками."""

    action_json = json.dumps(actions, ensure_ascii=False)
    return (
        "THOUGHT:\n"
        "- План действий\n"
        "ACTION:\n"
        f"{action_json}\n"
        "ASSISTANT:\n"
        f"{assistant}\n"
    )


def test_shadow_mode_stores_plan_without_calling_bitrix(orchestrator_factory, monkeypatch):
    """В режиме shadow действия не должны выполняться, только сохраняться в план."""

    called = False

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        nonlocal called
        called = True
        return {"result": 1}

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "shadow",
        build_model_text([
            {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Тестовая сделка"}}}
        ]),
    )

    reply = orchestrator.process_message("user-1", "Создай сделку")

    assert "Готово" in reply
    assert called is False

    state = orchestrator.state_manager.load_state("user-1")
    assert state.next_planned_actions == [
        {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Тестовая сделка"}}}
    ]
    assert not state.done


def test_user_summary_without_technical_terms(orchestrator_factory):
    """Резюме действий не должно содержать служебных названий методов."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant=""))

    summary = orchestrator._build_user_summary(
        [
            {
                "method": "crm.deal.add",
                "params": {"fields": {"TITLE": "Сделка"}},
                "result": {"result": {"ID": 10}},
            },
            {
                "method": "crm.timeline.comment.add",
                "params": {"fields": {"COMMENT": "Примечание"}},
                "result": {"result": True},
            },
        ]
    )

    assert "crm.deal.add" not in summary
    assert "crm.timeline.comment.add" not in summary
    assert summary.startswith("Что сделано:")


def test_full_mode_executes_safe_action(orchestrator_factory, monkeypatch):
    """В режиме full безопасный шаг должен выполниться и обновить состояние."""

    called: List[Dict[str, Any]] = []

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        called.append({"method": method, "params": params, "http_method": http_method})
        if method == "crm.deal.add":
            return {"result": 555}
        raise AssertionError("Неожиданный метод Bitrix")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "full",
        build_model_text([
            {"method": "crm.deal.add", "params": {"fields": {"TITLE": "Новая сделка"}}, "comment": "Создаём"}
        ],
        assistant="Сделка создана"),
    )

    reply = orchestrator.process_message("user-2", "Создай сделку")

    assert "Сделка создана" in reply
    assert "Что сделано:" in reply
    assert "Создана новая сделка" in reply
    assert "crm.deal.add" not in reply
    assert len(called) == 1
    assert called[0]["method"] == "crm.deal.add"
    assert called[0]["params"]["fields"]["TITLE"] == "Новая сделка"

    state = orchestrator.state_manager.load_state("user-2")
    assert state.objects["current_deal_id"] == 555
    assert state.done, "Запись об успешном действии должна быть добавлена"
    assert state.next_planned_actions == []


def test_validate_action_params_reports_missing_fields(orchestrator_factory):
    """Валидация сигнализирует о незаполненных обязательных полях."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant=""))

    errors = orchestrator._validate_action_params("crm.deal.add", {"fields": {"TITLE": ""}})
    combined = " ".join(errors)
    assert "не хватает обязательных данных" in combined
    assert "пустые поля" in combined
    assert "crm.deал.add" not in combined


def test_validate_action_params_structure_warning(orchestrator_factory):
    """Если структура параметров неверная, возвращается понятное сообщение."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant=""))

    errors = orchestrator._validate_action_params("crm.deal.add", {"fields": "plain"})
    assert any("структурировано" in msg for msg in errors)


def test_self_check_warnings_appended_to_reply(orchestrator_factory):
    """Self-check добавляет предупреждения в ответ пользователю."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant="Готово"))
    stale_at = (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    state = AgentState(
        goals=[],
        done=[{"description": "Создана сделка", "object_ids": {"deal_id": 5}}],
        in_progress=[{"description": ""}],
        confirmations={
            "deal": {
                "status": "requested",
                "requested_at": stale_at,
                "description": "Утвердить изменение сделки",
            }
        },
    )
    orchestrator.state_manager.save_state("user-self-check", state)

    reply = orchestrator.process_message("user-self-check", "Нужен отчёт")

    assert "Готово" in reply
    assert "⚠️" in reply
    assert "подтверждения" in reply or "активные объекты" in reply or "в работе" in reply


def test_confirmation_required_action_not_executed(orchestrator_factory, monkeypatch):
    """Рискованное действие должно запросить подтверждение и не вызываться в Bitrix."""

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        raise AssertionError("Не должно быть вызова Bitrix без подтверждения")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "full",
        build_model_text(
            [
                {
                    "method": "crm.deal.update",
                    "params": {"id": 321, "fields": {"OPPORTUNITY": 1000}},
                    "comment": "Меняем сумму",
                }
            ],
            assistant="Готово",
        ),
    )

    reply = orchestrator.process_message("user-3", "Измени сумму")

    assert "⚠️" in reply
    assert "Нужна ваша явная команда" in reply
    assert "crm.deal.update" not in reply

    state = orchestrator.state_manager.load_state("user-3")
    assert state.next_planned_actions, "Шаг должен остаться в плане"


def test_event_bind_requires_confirmation(orchestrator_factory, monkeypatch):
    """`event.bind` должен требовать подтверждения и не выполняться без него."""

    called = False

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        nonlocal called
        called = True
        return {"result": True}

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    action = {
        "method": "event.bind",
        "params": {"event": "OnCrmDealAdd", "handler": "https://example.com/hook"},
        "comment": "Настраиваем вебхук",
    }

    orchestrator = orchestrator_factory("full", build_model_text([action]))

    reply = orchestrator.process_message("user-4", "Подпиши событие")

    assert "⚠️" in reply
    assert "управления уведомлением" in reply.lower()
    assert "event.bind" not in reply
    assert called is False, "Без подтверждения вызова Bitrix быть не должно"

    state = orchestrator.state_manager.load_state("user-4")
    assert state.next_planned_actions == [action]
    confirmation = next(iter(state.confirmations.values()))
    assert confirmation["status"] == "requested"


def test_batch_updates_event_bindings_after_confirmation(orchestrator_factory, monkeypatch):
    """Пакетный вызов с изменением подписок должен обновлять `event_bindings`."""

    captured: List[Dict[str, Any]] = []

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        captured.append({"method": method, "params": params, "http_method": http_method})
        assert method == "batch"
        return {
            "result": {
                "result": {
                    "bind": True,
                    "list": {
                        "result": [
                            {
                                "event": "OnCrmDealAdd",
                                "handler": "https://example.com/hook",
                            }
                        ]
                    },
                }
            }
        }

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    action = {
        "method": "batch",
        "params": {
            "halt": 0,
            "cmd": {
                "bind": "event.bind?event=OnCrmDealAdd&handler=https://example.com/hook",
                "list": "event.get?",
            },
        },
        "requires_confirmation": True,
        "confirmed": True,
        "comment": "Обновляем подписки",
    }

    orchestrator = orchestrator_factory("full", build_model_text([action]))

    reply = orchestrator.process_message("user-5", "Обнови подписки")

    assert "Готово" in reply
    assert len(captured) == 1

    state = orchestrator.state_manager.load_state("user-5")
    assert state.event_bindings == [
        {"event": "OnCrmDealAdd", "handler": "https://example.com/hook"}
    ]
    assert state.next_planned_actions == []
    assert any(entry["description"].startswith("Подписка на событие обновлена") for entry in state.done)
    assert any(entry["description"].startswith("Получен список подписок") for entry in state.done)
    assert any(entry["description"].startswith("Выполнен пакетный вызов batch") for entry in state.done)
    assert state.confirmations, "Должна быть создана запись подтверждения"
    confirmation = next(iter(state.confirmations.values()))
    assert confirmation["status"] == "approved"




def test_batch_read_commands_execute_without_confirmation(orchestrator_factory, monkeypatch):
    """Пакет с чтением в canary должен выполниться без подтверждения."""

    captured: List[Dict[str, Any]] = []

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        captured.append({"method": method, "params": params})
        assert method == "batch"
        return {
            "result": {
                "result": {
                    "user": {"ID": "1"},
                    "deals": {"result": []},
                },
                "result_error": {},
            }
        }

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "canary",
        build_model_text(
            [
                {
                    "method": "batch",
                    "params": {
                        "cmd": {
                            "user": "user.current",
                            "deals": "crm.deal.list?filter[ASSIGNED_BY_ID]=1",
                        }
                    },
                }
            ]
        ),
    )

    reply = orchestrator.process_message("user-batch", "Сделай пакет чтения")

    assert "Готово" in reply
    assert len(captured) == 1

    state = orchestrator.state_manager.load_state("user-batch")
    assert not state.confirmations
    assert state.done[-1]["description"] == "Выполнен пакетный вызов batch"


def test_batch_with_update_requests_confirmation(orchestrator_factory, monkeypatch):
    """Пакет с изменением данных должен требовать подтверждение."""

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        raise AssertionError("Вызов Bitrix не должен выполняться без подтверждения")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "full",
        build_model_text(
            [
                {
                    "method": "batch",
                    "params": {
                        "cmd": {
                            "update": "crm.deal.update?id=42&fields[OPPORTUNITY]=1000",
                        }
                    },
                }
            ]
        ),
    )

    reply = orchestrator.process_message("user-batch-confirm", "Обнови сделку через batch")

    assert "⚠️" in reply

    state = orchestrator.state_manager.load_state("user-batch-confirm")
    assert state.confirmations
    assert state.next_planned_actions


def test_event_get_updates_state(orchestrator_factory, monkeypatch):
    """Вызов event.get должен обновить список подписок в состоянии."""

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "POST") -> Dict[str, Any]:
        assert method == "event.get"
        return {
            "result": [
                {"event": "onCrmDealAdd", "handler": "https://example.test/hook"},
                {"event": "OnTaskCommentAdd", "handler": "https://example.test/task"},
            ]
        }

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "canary",
        build_model_text(
            [
                {
                    "method": "event.get",
                    "params": {},
                }
            ],
            assistant="Готово",
        ),
    )

    reply = orchestrator.process_message("user-event-list", "Покажи подписки")

    assert "Готово" in reply
    state = orchestrator.state_manager.load_state("user-event-list")
    assert state.event_bindings == [
        {"event": "onCrmDealAdd", "handler": "https://example.test/hook"},
        {"event": "OnTaskCommentAdd", "handler": "https://example.test/task"},
    ]
    assert state.done[-1]["description"] == "Получен список подписок"


def test_crm_status_list_updates_done_history(orchestrator_factory, monkeypatch):
    """Чтение справочника через crm.status.list должно записывать агрегированную статистику."""

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "GET") -> Dict[str, Any]:
        assert method == "crm.status.list"
        assert params["filter"]["ENTITY_ID"] == "DEAL_STAGE"
        return {"result": [{"STATUS_ID": "NEW"}, {"STATUS_ID": "WON"}]}

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "canary",
        build_model_text(
            [
                {
                    "method": "crm.status.list",
                    "params": {"filter": {"ENTITY_ID": "DEAL_STAGE"}},
                    "comment": "Получаем статусы сделок",
                }
            ],
            assistant="Получены статусы",
        ),
    )

    reply = orchestrator.process_message("user-status", "Покажи статусы сделок")

    assert "Получены статусы" in reply

    state = orchestrator.state_manager.load_state("user-status")
    assert len(state.done) == 1
    done_entry = state.done[0]
    assert done_entry["description"] == "Получен справочник CRM"
    assert done_entry["object_ids"] == {"count": 2, "entity_id": "DEAL_STAGE"}


def test_crm_deal_category_reads_append_done(orchestrator_factory, monkeypatch):
    """Запросы crm.deal.category.* должны добавлять записи об объёме данных."""

    responses = iter(
        [
            {"result": [{"ID": 1, "NAME": "Основная"}, {"ID": 2, "NAME": "Эксперимент"}]},
            {"result": [{"STATUS_ID": "NEW"}, {"STATUS_ID": "WON"}, {"STATUS_ID": "LOSE"}]},
        ]
    )

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "GET") -> Dict[str, Any]:
        if method == "crm.deal.category.list":
            return next(responses)
        if method == "crm.deal.category.stage.list":
            assert params["id"] == 2
            return next(responses)
        raise AssertionError(f"Неожиданный метод {method}")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "canary",
        build_model_text(
            [
                {"method": "crm.deal.category.list", "params": {}},
                {"method": "crm.deal.category.stage.list", "params": {"id": 2}},
            ],
            assistant="Данные по направлениям и стадиям получены",
        ),
    )

    reply = orchestrator.process_message("user-categories", "Какие направления и стадии есть?")

    assert "Данные по направлениям и стадиям получены" in reply

    state = orchestrator.state_manager.load_state("user-categories")
    assert len(state.done) == 2
    categories_entry, stages_entry = state.done
    assert categories_entry["description"] == "Получен список направлений продаж"
    assert categories_entry["object_ids"] == {"count": 2}
    assert stages_entry["description"] == "Получен список стадий сделки"
    assert stages_entry["object_ids"] == {"count": 3, "category_id": 2}


def test_sonet_group_reads_update_done(orchestrator_factory, monkeypatch):
    """Чтение рабочих групп и их участников должно фиксироваться в истории."""

    responses = iter(
        [
            {"result": [{"ID": 10, "NAME": "Проект А"}]},
            {"result": [{"USER_ID": 42, "ROLE": "A"}, {"USER_ID": 43, "ROLE": "M"}]},
        ]
    )

    def fake_call(method: str, params: Dict[str, Any], http_method: str = "GET") -> Dict[str, Any]:
        if method == "sonet.group.get":
            return next(responses)
        if method == "sonet.group.user.get":
            assert params["GROUP_ID"] == 10
            return next(responses)
        raise AssertionError(f"Неожиданный метод {method}")

    monkeypatch.setattr("src.orchestrator.agent.call_bitrix", fake_call)

    orchestrator = orchestrator_factory(
        "canary",
        build_model_text(
            [
                {"method": "sonet.group.get", "params": {}},
                {"method": "sonet.group.user.get", "params": {"GROUP_ID": 10}},
            ],
            assistant="Информация о группах и участниках подготовлена",
        ),
    )

    reply = orchestrator.process_message("user-groups", "Покажи рабочие группы и участников")

    assert "Информация о группах и участниках подготовлена" in reply

    state = orchestrator.state_manager.load_state("user-groups")
    assert len(state.done) == 2
    groups_entry, members_entry = state.done
    assert groups_entry["description"] == "Получен список рабочих групп"
    assert groups_entry["object_ids"] == {"count": 1}
    assert members_entry["description"] == "Получен список участников группы"
    assert members_entry["object_ids"] == {"count": 2, "group_id": 10}

def test_interaction_logger_rotation(tmp_path: Path):
    """Проверяем, что логгер создаёт архивы и ограничивает их количество."""

    log_dir = tmp_path / "logs"
    logger_instance = InteractionLogger(log_dir=log_dir, max_bytes=200, max_archives=2)

    payload = {"THOUGHT": "t", "ACTION": [], "ASSISTANT": "a"}

    for idx in range(10):
        message = f"msg-{idx}" + "x" * 120  # увеличиваем размер записи
        logger_instance.log_model_response("user-logs", message, payload)
        logger_instance.log_iteration("user-logs", message, payload, state={}, executed_actions=[], errors=[])

    main_log = log_dir / "user-logs.jsonl"
    assert main_log.exists()

    archives = sorted(log_dir.glob("user-logs.jsonl.*.gz"))
    assert archives, "Архивы должны быть созданы"
    assert len(archives) <= 2, "Количество архивов ограничено max_archives"


def test_process_message_serializes_same_user(orchestrator_factory, monkeypatch):
    """Повторные запросы одного пользователя выполняются последовательно."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant="Готово"))
    first_enter = Event()
    second_enter = Event()
    release = Event()
    order: List[str] = []

    def fake_call(self, message: str, state: AgentState) -> Dict[str, Any]:  # type: ignore[override]
        thread_name = threading.current_thread().name
        order.append(thread_name)
        if len(order) == 1:
            first_enter.set()
            assert release.wait(timeout=1), "Основной поток обработки должен завершиться"
        else:
            second_enter.set()
        return {"THOUGHT": "План", "ACTION": [], "ASSISTANT": "Готово"}

    monkeypatch.setattr(Orchestrator, "_call_model", fake_call)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_first = pool.submit(orchestrator.process_message, "user-lock", "первый запрос")
        assert first_enter.wait(timeout=1)
        future_second = pool.submit(orchestrator.process_message, "user-lock", "второй запрос")
        assert not second_enter.wait(timeout=0.2), "Второй запрос не должен стартовать до завершения первого"
        release.set()
        assert second_enter.wait(timeout=1)
        reply_first = future_first.result(timeout=1)
        reply_second = future_second.result(timeout=1)

    assert reply_first
    assert reply_second
    assert len(order) == 2
    assert order[0] != order[1], "Обработку должны выполнять разные рабочие потоки"


def test_process_message_allows_parallel_users(orchestrator_factory, monkeypatch):
    """Разные пользователи обслуживаются параллельно и не блокируют друг друга."""

    orchestrator = orchestrator_factory("full", build_model_text([], assistant="Готово"))
    slow_enter = Event()
    fast_enter = Event()
    release = Event()
    order: List[Tuple[str, str]] = []

    def fake_call(self, message: str, state: AgentState) -> Dict[str, Any]:  # type: ignore[override]
        thread_name = threading.current_thread().name
        order.append((message, thread_name))
        if message == "медленный":
            slow_enter.set()
            assert release.wait(timeout=1), "Блокировка первого пользователя должна быть снята вручную"
        else:
            fast_enter.set()
        return {"THOUGHT": "План", "ACTION": [], "ASSISTANT": "Готово"}

    monkeypatch.setattr(Orchestrator, "_call_model", fake_call)

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_slow = pool.submit(orchestrator.process_message, "user-a", "медленный")
        assert slow_enter.wait(timeout=1)
        future_fast = pool.submit(orchestrator.process_message, "user-b", "быстрый")
        assert fast_enter.wait(timeout=0.5), "Другой пользователь должен выполняться параллельно"
        release.set()
        reply_fast = future_fast.result(timeout=1)
        reply_slow = future_slow.result(timeout=1)

    assert reply_slow
    assert reply_fast
    assert order[0][0] == "медленный"
    assert order[1][0] == "быстрый"
