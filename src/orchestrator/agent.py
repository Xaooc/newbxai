"""Оркестратор, управляющий коммуникацией с моделью и Bitrix24."""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs

from src.bitrix_client.client import BitrixClientError, call_bitrix, sanitize_for_logging
from src.app_logging.logger import InteractionLogger
from src.orchestrator.model_client import (
    ModelClient,
    ModelClientError,
    resolve_model_name,
)
from src.state.manager import AgentState, AgentStateManager

logger = logging.getLogger(__name__)

ErrorEntry = Union[str, Dict[str, Any]]



@dataclass
class BatchCommandInfo:
    """Описание подзапроса в batch."""

    key: str
    method: str
    query: str
    arguments: Dict[str, str]

    def payload_fields(self) -> Set[str]:
        """Возвращает множество полей, затронутых в параметрах подзапроса."""

        fields: Set[str] = set()
        for raw_key in self.arguments.keys():
            upper_key = raw_key.upper()
            fields.add(upper_key)
            bracket_field = self._extract_bracket_field(raw_key)
            if bracket_field:
                fields.add(bracket_field.upper())
        return fields

    def get_argument(self, name: str) -> Optional[str]:
        """Возвращает значение параметра по имени (если присутствует)."""

        return self.arguments.get(name)

    @staticmethod
    def _extract_bracket_field(raw_key: str) -> Optional[str]:
        """Извлекает имя поля из последней пары квадратных скобок."""

        if "[" not in raw_key or not raw_key.endswith("]"):
            return None
        return raw_key[raw_key.rfind("[") + 1 : -1] or None


READ_METHODS = {
    "user.current",
    "user.get",
    "crm.contact.list",
    "crm.contact.get",
    "crm.company.list",
    "crm.company.get",
    "crm.deal.list",
    "crm.deal.get",
    "crm.deal.category.list",
    "crm.deal.category.stage.list",
    "crm.status.list",
    "crm.activity.list",
    "tasks.task.list",
    "sonet.group.get",
    "sonet.group.user.get",
    "event.get",
}

SAFE_CREATE_METHODS = {
    "crm.deal.add",
    "crm.activity.add",
    "crm.timeline.comment.add",
    "tasks.task.add",
    "task.commentitem.add",
    "task.checklistitem.add",
}

UPDATE_METHODS = {
    "crm.deal.update",
    "tasks.task.update",
    "event.bind",
    "event.unbind",
}

EVENT_METHODS = {"event.bind", "event.get", "event.unbind"}
BATCH_METHODS = {"batch"}
ALL_ALLOWED_METHODS = READ_METHODS | SAFE_CREATE_METHODS | UPDATE_METHODS | BATCH_METHODS

ALLOWED_METHODS_BY_MODE = {
    "shadow": set(),
    "canary": READ_METHODS | SAFE_CREATE_METHODS | BATCH_METHODS,
    "full": ALL_ALLOWED_METHODS,
}

FRIENDLY_METHOD_NAMES: Dict[str, str] = {
    "user.current": "просмотр сведений о текущем сотруднике",
    "user.get": "поиск сотрудника",
    "crm.contact.list": "поиск контактов",
    "crm.contact.get": "просмотр контакта",
    "crm.company.list": "поиск компаний",
    "crm.company.get": "просмотр компании",
    "crm.deal.list": "поиск сделок",
    "crm.deal.get": "просмотр сделки",
    "crm.deal.add": "создание сделки",
    "crm.deal.update": "обновление сделки",
    "crm.deal.category.list": "просмотр направлений продаж",
    "crm.deal.category.stage.list": "просмотр этапов направления",
    "crm.status.list": "просмотр справочника статусов",
    "crm.activity.list": "поиск дел",
    "crm.activity.add": "создание дела",
    "crm.timeline.comment.add": "добавление комментария",
    "tasks.task.add": "создание задачи",
    "tasks.task.update": "обновление задачи",
    "tasks.task.list": "поиск задач",
    "task.commentitem.add": "добавление комментария к задаче",
    "task.checklistitem.add": "добавление пункта чек-листа",
    "sonet.group.get": "просмотр рабочей группы",
    "sonet.group.user.get": "просмотр состава рабочей группы",
    "event.bind": "настройка уведомления",
    "event.get": "просмотр активных уведомлений",
    "event.unbind": "отключение уведомления",
    "batch": "пакетное действие",
}

FRIENDLY_FIELD_NAMES: Dict[str, str] = {
    "OPPORTUNITY": "сумму",
    "ASSIGNED_BY_ID": "ответственного",
    "STAGE_ID": "этап",
    "CATEGORY_ID": "воронку",
    "RESPONSIBLE_ID": "ответственного",
    "DEADLINE": "крайний срок",
    "TITLE": "название",
    "SUBJECT": "тему",
    "COMMENT": "комментарий",
    "DESCRIPTION": "описание",
}


def _get_fields(params: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(params, dict):
        fields = params.get("fields")
        if isinstance(fields, dict):
            return fields
    return {}


def _extract_text(value: Any, limit: int = 120) -> Optional[str]:
    if not isinstance(value, str):
        return None
    clean = value.strip()
    if not clean:
        return None
    if len(clean) > limit:
        return clean[: limit - 1].rstrip() + "…"
    return clean


def _extract_count(result: Dict[str, Any]) -> Optional[int]:
    payload = result.get("result") if isinstance(result, dict) else None
    if isinstance(payload, dict):
        if isinstance(payload.get("total"), int):
            return payload["total"]
        items = payload.get("items")
        if isinstance(items, list):
            return len(items)
    if isinstance(payload, list):
        return len(payload)
    return None


ActionDescriptor = Callable[[Dict[str, Any], Dict[str, Any]], str]


def _describe_deal_add(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    title = _extract_text(_get_fields(params).get("TITLE") or _get_fields(params).get("title"))
    if title:
        return f"Создана новая сделка «{title}»."
    return "Создана новая сделка, она уже доступна в CRM."


def _describe_deal_update(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    fields = _get_fields(params)
    if fields:
        friendly_fields = sorted({FRIENDLY_FIELD_NAMES.get(key.upper(), key.lower()) for key in fields.keys()})
        if friendly_fields:
            readable = ", ".join(friendly_fields)
            return f"Обновлена сделка: уточнены {readable}."
    return "Обновлена информация по сделке."


def _describe_list_action(singular: str, plural: str, result: Dict[str, Any]) -> str:
    count = _extract_count(result)
    if count is None:
        return f"Получена информация по {plural}."
    if count == 0:
        return f"По вашему запросу {plural} не нашлось."
    if count == 1:
        return f"Найдена 1 {singular}."
    return f"Найдено {count} {plural}."


def _describe_activity_add(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    subject = _extract_text(_get_fields(params).get("SUBJECT"))
    if subject:
        return f"Запланировано новое дело «{subject}»."
    return "Запланировано новое дело."


def _describe_timeline_comment(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    comment = _extract_text(_get_fields(params).get("COMMENT"), limit=80)
    if comment:
        return f"Добавлен комментарий: «{comment}»."
    return "Добавлен комментарий в карточку."


def _describe_task_add(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    title = _extract_text(_get_fields(params).get("TITLE") or _get_fields(params).get("title"))
    if title:
        return f"Создана новая задача «{title}»."
    return "Создана новая задача."


def _describe_task_update(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    fields = _get_fields(params)
    if fields:
        friendly_fields = sorted({FRIENDLY_FIELD_NAMES.get(key.upper(), key.lower()) for key in fields.keys()})
        if friendly_fields:
            readable = ", ".join(friendly_fields)
            return f"Обновлена задача: скорректированы {readable}."
    return "Обновлена информация по задаче."


def _describe_task_comment(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    comment = _extract_text(_get_fields(params).get("POST_MESSAGE"), limit=80)
    if comment:
        return f"Оставлен комментарий в задаче: «{comment}»."
    return "Оставлен комментарий в задаче."


def _describe_checklist_item(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    title = _extract_text(_get_fields(params).get("TITLE"), limit=60)
    if title:
        return f"Добавлен пункт чек-листа «{title}»."
    return "Добавлен новый пункт чек-листа."


def _describe_event_bind(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    event_name = params.get("event") or params.get("EVENT")
    if isinstance(event_name, str) and event_name.strip():
        return f"Включено автоматическое уведомление «{event_name.strip()}»."
    return "Включено новое автоматическое уведомление."


def _describe_event_unbind(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    event_name = params.get("event") or params.get("EVENT")
    if isinstance(event_name, str) and event_name.strip():
        return f"Отключено уведомление «{event_name.strip()}»."
    return "Отключено одно из автоматических уведомлений."


def _describe_batch(params: Dict[str, Any], result: Dict[str, Any]) -> str:
    return "Выполнено несколько действий подряд, результаты проверены."


ACTION_DESCRIPTORS: Dict[str, ActionDescriptor] = {
    "crm.deal.add": _describe_deal_add,
    "crm.deal.update": _describe_deal_update,
    "crm.deal.list": lambda params, result: _describe_list_action("сделка", "сделок", result),
    "crm.deal.get": lambda params, result: "Получены актуальные данные сделки.",
    "crm.deal.category.list": lambda params, result: _describe_list_action("направление", "направлений", result),
    "crm.deal.category.stage.list": lambda params, result: _describe_list_action("этап", "этапов", result),
    "crm.status.list": lambda params, result: _describe_list_action("статус", "статусов", result),
    "crm.contact.list": lambda params, result: _describe_list_action("контакт", "контактов", result),
    "crm.contact.get": lambda params, result: "Получены актуальные данные контакта.",
    "crm.company.list": lambda params, result: _describe_list_action("компания", "компаний", result),
    "crm.company.get": lambda params, result: "Получены актуальные данные компании.",
    "crm.activity.list": lambda params, result: _describe_list_action("дело", "дел", result),
    "crm.activity.add": _describe_activity_add,
    "crm.timeline.comment.add": _describe_timeline_comment,
    "tasks.task.add": _describe_task_add,
    "tasks.task.update": _describe_task_update,
    "tasks.task.list": lambda params, result: _describe_list_action("задача", "задач", result),
    "task.commentitem.add": _describe_task_comment,
    "task.checklistitem.add": _describe_checklist_item,
    "user.current": lambda params, result: "Проверены данные вашей учётной записи.",
    "user.get": lambda params, result: _describe_list_action("сотрудник", "сотрудников", result),
    "sonet.group.get": lambda params, result: "Получена информация о рабочей группе.",
    "sonet.group.user.get": lambda params, result: _describe_list_action("участник", "участников группы", result),
    "event.bind": _describe_event_bind,
    "event.unbind": _describe_event_unbind,
    "event.get": lambda params, result: _describe_list_action("уведомление", "активных уведомлений", result),
    "batch": _describe_batch,
}

RISKY_FIELDS_BY_METHOD = {
    "crm.deal.add": {"fields": {"OPPORTUNITY", "ASSIGNED_BY_ID", "STAGE_ID", "CATEGORY_ID"}},
    "crm.deal.update": {"fields": {"OPPORTUNITY", "ASSIGNED_BY_ID", "STAGE_ID", "CATEGORY_ID"}},
    "crm.activity.add": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
    "tasks.task.add": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
    "tasks.task.update": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
}

ALWAYS_CONFIRM_METHODS = {"event.bind", "event.unbind"}

METHOD_PARAMETER_REQUIREMENTS = {
    "crm.deal.add": {
        "required": [("fields",), ("fields", ("TITLE", "title"))],
        "dict_fields": [("fields",)],
        "non_empty": [("fields", ("TITLE", "title"))],
    },
    "crm.deal.update": {
        "required": [(("id", "ID"),), ("fields",)],
        "dict_fields": [("fields",)],
        "non_empty": [("fields",)],
    },
    "crm.deal.get": {
        "required": [(("id", "ID"),)],
    },
    "crm.deal.category.stage.list": {
        "required": [(("id", "ID", "categoryId", "CATEGORY_ID"),)],
    },
    "crm.status.list": {
        "required": [("filter",), ("filter", ("ENTITY_ID", "entity_id"))],
        "dict_fields": [("filter",)],
    },
    "crm.activity.list": {
        "required": [("filter",), ("filter", ("OWNER_TYPE_ID", "owner_type_id"))],
        "dict_fields": [("filter",)],
    },
    "crm.activity.add": {
        "required": [
            ("fields",),
            ("fields", ("OWNER_TYPE_ID", "owner_type_id")),
            ("fields", ("OWNER_ID", "owner_id")),
            ("fields", ("TYPE_ID", "type_id")),
            ("fields", ("SUBJECT", "subject")),
        ],
        "dict_fields": [("fields",)],
        "non_empty": [("fields", ("SUBJECT", "subject"))],
    },
    "crm.timeline.comment.add": {
        "required": [
            ("fields",),
            ("fields", ("ENTITY_ID", "entity_id")),
            ("fields", ("ENTITY_TYPE", "entity_type")),
            ("fields", ("COMMENT", "comment")),
        ],
        "dict_fields": [("fields",)],
        "non_empty": [("fields", ("COMMENT", "comment"))],
    },
    "crm.contact.get": {
        "required": [(("id", "ID"),)],
    },
    "crm.company.get": {
        "required": [(("id", "ID"),)],
    },
    "tasks.task.add": {
        "required": [
            ("fields",),
            ("fields", ("TITLE", "title")),
            ("fields", ("DESCRIPTION", "description")),
            ("fields", ("RESPONSIBLE_ID", "responsible_id")),
        ],
        "dict_fields": [("fields",)],
        "non_empty": [
            ("fields", ("TITLE", "title")),
            ("fields", ("DESCRIPTION", "description")),
        ],
    },
    "tasks.task.update": {
        "required": [(("taskId", "TASK_ID", "id", "ID"),), ("fields",)],
        "dict_fields": [("fields",)],
        "non_empty": [("fields",)],
    },
    "task.commentitem.add": {
        "required": [(("taskId", "TASK_ID"),), ("fields",), ("fields", ("POST_MESSAGE", "post_message"))],
        "dict_fields": [("fields",)],
        "non_empty": [("fields", ("POST_MESSAGE", "post_message"))],
    },
    "task.checklistitem.add": {
        "required": [(("taskId", "TASK_ID"),), ("fields",), ("fields", ("TITLE", "title"))],
        "dict_fields": [("fields",)],
        "non_empty": [("fields", ("TITLE", "title"))],
    },
    "event.bind": {
        "required": [(("event", "EVENT"),), (("handler", "HANDLER"),)],
        "non_empty": [(("event", "EVENT"),), (("handler", "HANDLER"),)],
    },
    "event.unbind": {
        "required": [(("event", "EVENT"),), (("handler", "HANDLER"),)],
        "non_empty": [(("event", "EVENT"),), (("handler", "HANDLER"),)],
    },
}

DEFAULT_SYSTEM_PROMPT = (
    "Ты — AI-менеджер Bitrix24. Работай строго по инструкциям.\n"
    "Всегда отвечай в формате:\n"
    "THOUGHT:\n"
    "<твои размышления>\n"
    "ACTION:\n"
    "<JSON-массив шагов>\n"
    "ASSISTANT:\n"
    "<ответ пользователю на русском языке>.\n"
    "Разрешённые методы: user.current, user.get, crm.contact.list, crm.contact.get, crm.company.list,"
    " crm.company.get, crm.deal.list, crm.deal.get, crm.deal.add, crm.deal.update, crm.deal.category.list,"
    " crm.deal.category.stage.list, crm.status.list, crm.activity.list, crm.activity.add, crm.timeline.comment.add,"
    " tasks.task.add, tasks.task.update, tasks.task.list, task.commentitem.add, task.checklistitem.add,"
    " sonet.group.get, sonet.group.user.get, batch, event.bind, event.get, event.unbind. Запрещено использовать любые иные методы.\n"
    "Перед выполнением плана обязательно представь его пользователю, дождись явного подтверждения и только после этого повторяй шаги с confirmed: true. Пока подтверждения нет, каждый шаг в ACTION должен иметь requires_confirmation: true.\n"
    "Для поиска сотрудников используй только user.get. Методы crm.contact.* предназначены исключительно для клиентов и внешних контактов.\n"
    "Перед изменением сумм, стадий, ответственных, дедлайнов указывай requires_confirmation=true и жди подтверждения.\n"
    "В блоке ASSISTANT объясняй шаги простым языком: не упоминай внутренние идентификаторы или названия REST-методов,"
    " подчеркивай, чем итог полезен пользователю."
)


def _utc_iso_z() -> str:
    """Возвращает временную метку UTC в формате ISO 8601 с суффиксом Z."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _default_model_name() -> str:
    """Читает имя модели из окружения или возвращает значение по умолчанию."""

    return resolve_model_name(None)


@dataclass
class OrchestratorSettings:
    """Настройки оркестратора."""

    mode: str = "shadow"
    model_name: str = field(default_factory=_default_model_name)
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT


class Orchestrator:
    """Основной класс оркестрации диалога.

    В MVP фактический вызов GPT заменён на заглушку, которая возвращает
    фиксированный ответ. Реализация настоящего клиента описана в TODO.
    """

    def __init__(
        self,
        state_manager: AgentStateManager,
        interaction_logger: InteractionLogger,
        settings: OrchestratorSettings,
        model_client: ModelClient | None = None,
    ) -> None:
        self.state_manager = state_manager
        self.interaction_logger = interaction_logger
        self.settings = settings
        self.model_client = model_client
        self._locks_guard = Lock()
        self._user_locks: Dict[str, Lock] = {}

    def process_message(self, user_id: str, message: str) -> str:
        """Обрабатывает сообщение пользователя и возвращает ответ ассистента."""

        with self._acquire_user_lock(user_id):
            return self._process_message_locked(user_id, message)

    @contextmanager
    def _acquire_user_lock(self, user_id: str):
        """Гарантирует эксклюзивную обработку сообщений одного пользователя."""

        lock = self._get_user_lock(user_id)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _get_user_lock(self, user_id: str) -> Lock:
        """Возвращает (или создаёт) блокировку для конкретного пользователя."""

        sanitized = user_id or "_anonymous"
        with self._locks_guard:
            lock = self._user_locks.get(sanitized)
            if lock is None:
                lock = Lock()
                self._user_locks[sanitized] = lock
            return lock

    def _process_message_locked(self, user_id: str, message: str) -> str:
    def _process_message_locked(self, user_id: str, message: str) -> str:
        """Реализация обработки запроса под блокировкой конкретного пользователя."""

        state = self.state_manager.load_state(user_id)

        if message and (not state.goals or state.goals[0] != message):
            state.goals.insert(0, message)

        logger.debug("Загружено состояние", extra={"user_id": user_id, "state": state})

        self_check_warnings = self._run_context_self_check(state)
        if self_check_warnings:
            logger.info(
                "Self-check обнаружил потенциальные пробелы в контексте",
                extra={"user_id": user_id, "warnings": self_check_warnings},
            )

        model_response = self._call_model(message, state)
        self.interaction_logger.log_model_response(user_id, message, model_response)

        actions = model_response.get("ACTION", [])
        assistant_from_model = model_response.get("ASSISTANT", "")

        plan_summary = self._summarize_plan_actions(actions)
        self._store_last_plan(state, actions, plan_summary)

        executed_actions = []
        errors: List[ErrorEntry] = []

        if self.settings.mode == "shadow":
            logger.info("Режим shadow: действия не выполняются", extra={"user_id": user_id, "plan": actions})
        else:
            for action in actions:
                method = action.get("method")
                raw_params = action.get("params")
                if raw_params is None:
                    params = {}
                elif isinstance(raw_params, dict):
                    params = raw_params
                else:
                    errors.append(
                        self._format_action_error(
                            method,
                            "не получилось разобрать параметры. Уточните, что нужно сделать, и я перепланирую шаги.",
                        )
                    )
                    continue

                comment = action.get("comment", "")
                http_method = (action.get("http_method") or self._default_http_method(method or "")).upper()

                if not method:
                    errors.append(
                        "Одно из действий не распознано. Опишите задачу подробнее, и я предложу новую последовательность."
                    )
                    continue

                if not self._is_action_allowed(method, action):
                    errors.append(
                        self._format_action_error(
                            method,
                            f"это действие недоступно в режиме безопасности {self.settings.mode}.",
                        )
                    )
                    continue

                validation_errors, _missing_paths = self._validate_action_params(method, params)
                if validation_errors:
                    errors.extend(validation_errors)
                    continue

                try:
                    result = call_bitrix(method, params, http_method=http_method)
                    executed_actions.append(
                        {
                            "method": method,
                            "params": params,
                            "comment": comment,
                            "result": result,
                        }
                    )
                    self._update_state_from_action(state, action, result)
                except BitrixClientError as exc:
                    sanitized_params = sanitize_for_logging(params or {})
                    logger.error(
                        "Ошибка Bitrix24 при выполнении действия",
                        extra={"method": method, "error": str(exc), "params": sanitized_params},
                    )
                    user_message = self._format_action_error(
                        method,
                        "Bitrix24 отклонил запрос. Проверьте данные или повторите попытку позднее.",
                    )
                    errors.append(
                        {
                            "user_message": user_message,
                            "diagnostic": str(exc),
                            "method": method,
                            "params": sanitized_params,
                        }
                    )

        summary_text = self._build_user_summary(executed_actions)

        assistant_reply = self._compose_final_reply(
            assistant_from_model=assistant_from_model,
            plan_summary=plan_summary,
            execution_summary=summary_text,
            errors=errors,
            warnings=self_check_warnings,
        )

        self.state_manager.save_state(user_id, state)
        self.interaction_logger.log_iteration(user_id, message, model_response, state, executed_actions, errors)

        return assistant_reply or "Не удалось получить ответ от модели. Попробуйте повторить запрос позже."
    def _call_model(self, message: str, state: AgentState) -> Dict[str, Any]:
        """Обёртка вызова ChatGPT.

        В реальной реализации происходит обращение к API модели. При недоступности
        клиента возвращается предсказуемый ответ-заглушка.
        """

        system_prompt = self.settings.system_prompt_template or DEFAULT_SYSTEM_PROMPT
        if self.model_client is None:
            logger.warning("GPT-клиент недоступен: используется заглушка ответа.")
            return self._fallback_model_response("Клиент модели не настроен")

        try:
            raw_text = self.model_client.generate(system_prompt, state.to_dict(), message)
            parsed = self._parse_model_output(raw_text)
            return parsed
        except ModelClientError as exc:
            logger.warning("Ошибка GPT-клиента", extra={"error": str(exc)})
            return self._fallback_model_response(str(exc))

    def _parse_model_output(self, text: str) -> Dict[str, Any]:
        """Разбирает текст модели на блоки THOUGHT/ACTION/ASSISTANT."""

        sections: Dict[str, List[str]] = {"THOUGHT": [], "ACTION": [], "ASSISTANT": []}
        current: Optional[str] = None

        for line in text.splitlines():
            stripped = line.strip()
            header = stripped.rstrip(":").upper()
            if stripped.endswith(":") and header in sections:
                current = header
                sections[current] = []
                continue
            if current:
                sections[current].append(line)

        if any(not sections[key] and key != "ACTION" for key in sections):
            raise ModelClientError("Ответ модели не содержит обязательные блоки")

        thought = "\n".join(sections["THOUGHT"]).strip() or "План не предоставлен."
        assistant = "\n".join(sections["ASSISTANT"]).strip() or "Ответ не получен."
        action_block = "\n".join(sections["ACTION"]).strip()
        action_json = self._extract_action_json(action_block)

        try:
            actions = json.loads(action_json) if action_json else []
        except json.JSONDecodeError as exc:
            raise ModelClientError(f"Не удалось разобрать ACTION как JSON: {exc}") from exc

        if not isinstance(actions, list):
            raise ModelClientError("Блок ACTION должен быть массивом JSON")

        return {
            "THOUGHT": thought,
            "ACTION": actions,
            "ASSISTANT": assistant,
        }

    @staticmethod
    def _extract_action_json(block: str) -> str:
        """Извлекает JSON из блока ACTION, удаляя обёртку из кода."""

        cleaned = block.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return cleaned.strip()

    @staticmethod
    def _fallback_model_response(reason: str) -> Dict[str, Any]:
        """Возвращает предсказуемый ответ при недоступности модели."""

        return {
            "THOUGHT": "Модель недоступна, работаю в режиме заглушки.",
            "ACTION": [],
            "ASSISTANT": (
                "Автоматическое планирование временно недоступно: "
                f"{reason}. Я зафиксировал запрос и готов повторить попытку позже."
            ),
        }

    def _merge_reply_with_summary(self, base: str, summary: str) -> str:
        """Объединяет исходный ответ модели и краткое резюме."""

        base = (base or "").strip()
        if summary:
            if base:
                return f"{base}\n\n{summary}"
            return summary
        return base

    @staticmethod
    def _append_errors_to_reply(reply: str, errors: List[ErrorEntry]) -> str:
        """Добавляет блок предупреждений к ответу."""

        if not errors:
            return reply

        warning_lines = []
        for item in errors:
            if isinstance(item, dict):
                message = item.get("user_message") or item.get("message") or item.get("diagnostic") or str(item)
            else:
                message = item
            warning_lines.append(f"⚠️ {message}")

        appendix = "\n".join(warning_lines)
        if reply:
            return f"{reply}\n\n{appendix}"
        return appendix


    def _format_action_error(self, method: Optional[str], message: str) -> str:
        """Формирует человеко-понятное описание ошибки действия."""

        friendly = self._friendly_method_name(method)
        return f"Не удалось выполнить действие «{friendly}»: {message}".strip()

    def _run_context_self_check(self, state: AgentState) -> List[str]:
        """Проверяет полноту ключевых данных перед обращением к модели."""

        warnings: List[str] = []

        if not state.goals:
            warnings.append("Пока нет активной цели. Напишите, какую задачу решить.")

        objects = state.objects or {}
        tracked_ids = [
            objects.get("current_deal_id"),
            objects.get("current_contact_id"),
            objects.get("current_company_id"),
            objects.get("current_task_id"),
        ]
        if not any(tracked_ids):
            recent_done = [entry.get("object_ids") for entry in state.done[-5:]]
            if any(obj for obj in recent_done if obj):
                warnings.append(
                    "Не удалось определить активные объекты по последним действиям. "
                    "Напишите, с какой сделкой, задачей или клиентом работать дальше."
                )

        cutoff = datetime.now(UTC) - timedelta(hours=24)
        stale_confirmations: List[str] = []
        for key, record in state.confirmations.items():
            status = (record or {}).get("status")
            if status != "requested":
                continue
            requested_at = self._parse_iso_datetime((record or {}).get("requested_at"))
            if requested_at and requested_at < cutoff:
                description = (record or {}).get("description") or key
                stale_confirmations.append(description.strip() or key)
        if stale_confirmations:
            head = "; ".join(stale_confirmations[:3])
            tail = " и другие" if len(stale_confirmations) > 3 else ""
            warnings.append(
                f"Есть запросы, ожидающие подтверждения: {head}{tail}. "
                "Сообщите, можно ли продолжить."
            )

        if any(not (item.get("description") or "").strip() for item in state.in_progress):
            warnings.append(
                "Некоторые шаги помечены как «в работе», но без описания. "
                "Поясните, какой следующий шаг выполнить."
            )

        plan_record = state.plan_confirmation or {}
        if (plan_record.get("status") or "").lower() == "pending":
            requested_at = self._parse_iso_datetime(plan_record.get("requested_at"))
            if requested_at and requested_at < datetime.now(UTC) - timedelta(minutes=10):
                warnings.append(
                    "План всё ещё ожидает подтверждения. Напишите, можно ли выполнять шаги или что нужно поменять."
                )

        return warnings

    def _build_user_summary(self, executed_actions: List[Dict[str, Any]]) -> str:
        """Создаёт краткое резюме выполненных действий для пользователя."""

        descriptions: List[str] = []
        for entry in executed_actions:
            method = entry.get("method")
            params = entry.get("params") if isinstance(entry.get("params"), dict) else {}
            result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
            description = self._describe_action_for_user(method, params, result)
            if description:
                descriptions.append(description)

        if not descriptions:
            return ""

        lines = ["Что сделано:"]
        lines.extend(f"• {item}" for item in descriptions)
        return "\n".join(lines)

    def _describe_action_for_user(
        self,
        method: Optional[str],
        params: Dict[str, Any],
        result: Dict[str, Any],
    ) -> str:
        """Переводит выполненное действие в человеко-понятный текст."""

        if method and method in ACTION_DESCRIPTORS:
            descriptor = ACTION_DESCRIPTORS[method]
            try:
                return descriptor(params, result)
            except Exception as exc:  # pragma: no cover - защитный сценарий
                logger.debug(
                    "Не удалось построить описание действия",
                    extra={"method": method, "error": str(exc)},
                )
        friendly = self._friendly_method_name(method)
        return f"Выполнено действие «{friendly}». Результат уже доступен в Bitrix24."

    @staticmethod
    def _friendly_field_name(field: str) -> str:
        """Возвращает дружественное название поля Bitrix."""

        return FRIENDLY_FIELD_NAMES.get(field.upper(), field.lower())

    def _friendly_method_name(self, method: Optional[str]) -> str:
        """Возвращает дружественное название метода Bitrix."""

        if not method:
            return "запрошенное действие"
        return FRIENDLY_METHOD_NAMES.get(method, "запрошенное действие")

    @staticmethod
    def _parse_iso_datetime(raw_value: Optional[str]) -> Optional[datetime]:
        """Преобразует ISO-строку в datetime для self-check."""

        if not raw_value:
            return None
        value = raw_value.strip()
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            return None

    def _default_http_method(self, method: str) -> str:
        """Возвращает HTTP-метод по умолчанию для вызова Bitrix."""

        if method in READ_METHODS:
            return "GET"
        return "POST"

    def _validate_action_params(self, method: str, params: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Проверяет наличие обязательных параметров для метода."""

        spec = METHOD_PARAMETER_REQUIREMENTS.get(method)
        if not spec:
            return [], []

        errors: List[str] = []
        missing_paths: List[Tuple[Any, ...]] = []
        missing_required = False
        wrong_structure = False
        empty_values = False

        def get_value(path: Tuple[Any, ...]) -> Any:
            current: Any = params
            for segment in path:
                if isinstance(segment, tuple):
                    options = tuple(segment)
                else:
                    options = (str(segment),)
                if not isinstance(current, dict):
                    return None
                found = False
                for option in options:
                    if option in current:
                        current = current[option]
                        found = True
                        break
                if not found:
                    return None
            return current

        for path in spec.get("required", []):
            value = get_value(path)
            if not self._is_value_present(value):
                missing_required = True
                missing_paths.append(path)

        for path in spec.get("dict_fields", []):
            value = get_value(path)
            if value is None:
                continue
            if not isinstance(value, dict):
                wrong_structure = True

        for path in spec.get("non_empty", []):
            value = get_value(path)
            if value is None:
                continue
            if not self._is_value_present(value):
                empty_values = True

        friendly = self._friendly_method_name(method)
        if missing_required:
            errors.append(
                f"Для действия «{friendly}» не хватает обязательных данных. Пожалуйста, уточните детали, укажите их и повторите запрос."
            )
        if wrong_structure:
            errors.append(
                f"Для действия «{friendly}» данные нужно передать структурировано (таблица значений). Проверьте формат."
            )
        if empty_values:
            errors.append(
                f"Для действия «{friendly}» указаны пустые поля. Заполните их, чтобы я мог продолжить."
            )

        if missing_paths:
            friendly_names: List[str] = []
            for path in missing_paths:
                name = self._friendly_missing_field(path)
                if name.lower() in {"fields", "filter"}:
                    continue
                friendly_names.append(name)
            if friendly_names:
                deduped = list(dict.fromkeys(friendly_names))
                friendly_fields = ", ".join(f"«{name}»" for name in deduped)
                errors.append(f"Заполните поле {friendly_fields}.")

        return errors, [self._path_to_str(path) for path in missing_paths]

    def _friendly_missing_field(self, path: Tuple[Any, ...]) -> str:
        """Возвращает дружественное имя поля для сообщения об ошибке."""

        if not path:
            return "значение"
        last = path[-1]
        candidate: Optional[str] = None
        if isinstance(last, tuple):
            for option in last:
                if isinstance(option, str):
                    candidate = option
                    break
        elif isinstance(last, str):
            candidate = last

        if not candidate and len(path) >= 2 and isinstance(path[-2], str):
            candidate = path[-2]

        if not candidate:
            return "значение"
        return self._friendly_field_name(candidate)

    @staticmethod
    def _path_to_str(path: Tuple[Any, ...]) -> str:
        """Формирует строковое представление пути параметра."""

        parts: List[str] = []
        for segment in path:
            if isinstance(segment, tuple):
                parts.append("/".join(str(item) for item in segment))
            else:
                parts.append(str(segment))
        return ".".join(parts)

    @staticmethod
    def _is_value_present(value: Any) -> bool:
        """Проверяет, заполнено ли значение."""

        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, bool):
            return True
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, (list, dict, set, tuple)):
            return bool(value)
        return True

    def _is_action_allowed(self, method: str, action: Optional[Dict[str, Any]] = None) -> bool:
        """Проверяет, разрешён ли метод в текущем режиме безопасности."""

        allowed = ALLOWED_METHODS_BY_MODE.get(self.settings.mode, set())
        if method == "batch":
            if method not in allowed:
                return False
            try:
                commands = self._extract_batch_commands(action or {})
            except ValueError:
                return False
            for command in commands:
                if not command.method or command.method == "batch":
                    return False
                if command.method not in ALLOWED_METHODS_BY_MODE.get(self.settings.mode, set()):
                    return False
            return True
        return method in allowed

    def _check_confirmation_needed(self, state: AgentState, action: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Определяет, требуется ли подтверждение перед выполнением шага."""

        method = action.get("method", "")
        params = action.get("params") or {}
        explicit_request = action.get("requires_confirmation", False)
        confirmed_flag = action.get("confirmed", False)
        reason = action.get("confirmation_reason")

        if method in ALWAYS_CONFIRM_METHODS:
            explicit_request = True
            friendly = self._friendly_method_name(method)
            reason = reason or f"Для управления уведомлением («{friendly}») нужно ваше подтверждение."

        batch_commands: List[BatchCommandInfo] = []
        if method == "batch":
            try:
                batch_commands = self._extract_batch_commands(action)
            except ValueError as exc:
                return True, None, f"Некорректный пакетный вызов: {exc}"
            batch_requires_confirmation, batch_reason = self._evaluate_batch_confirmation(batch_commands)
            if batch_requires_confirmation:
                explicit_request = True
                reason = reason or batch_reason
            risky_fields: Set[str] = set()
            payload_fields: Set[str] = set()
        else:
            risky_fields = RISKY_FIELDS_BY_METHOD.get(method, {}).get("fields", set())
            fields_payload = params.get("fields") if isinstance(params.get("fields"), dict) else params
            payload_fields = set()
            if isinstance(fields_payload, dict):
                payload_fields = {key.upper() for key in fields_payload.keys()}

        auto_request = bool(risky_fields & payload_fields)
        needs_confirmation = explicit_request or auto_request

        if not needs_confirmation:
            return False, None, None

        key = self._build_confirmation_key(method, params)
        record = state.confirmations.get(key)

        confirmation_reason = reason
        if not confirmation_reason:
            confirmation_reason = self._build_confirmation_reason(method, params, risky_fields)

        if confirmed_flag or (record and record.get("status") == "approved"):
            updated_record = record or {}
            updated_record.update(
                {
                    "status": "approved",
                    "approved_at": _utc_iso_z(),
                    "action": action,
                    "reason": confirmation_reason,
                    "description": confirmation_reason,
                }
            )
            state.confirmations[key] = updated_record
            return False, key, updated_record["reason"]

        if record and record.get("status") == "denied":
            return True, key, record.get("reason", "операция ранее отклонена")

        state.confirmations[key] = {
            "status": "requested",
            "requested_at": _utc_iso_z(),
            "action": action,
            "reason": confirmation_reason,
            "description": confirmation_reason,
        }
        return True, key, confirmation_reason
    def _extract_batch_commands(self, action: Dict[str, Any]) -> List[BatchCommandInfo]:
        """Разбирает шаг batch в список подзапросов."""

        params = (action.get("params") or {}) if isinstance(action, dict) else {}
        cmd_payload = params.get("cmd")
        if cmd_payload is None:
            raise ValueError("Отсутствует параметр cmd для batch")

        if isinstance(cmd_payload, dict):
            items = list(cmd_payload.items())
        elif isinstance(cmd_payload, list):
            items = [(str(idx), value) for idx, value in enumerate(cmd_payload)]
        else:
            raise ValueError("Параметр cmd должен быть словарём или списком строк")

        commands: List[BatchCommandInfo] = []
        for key, raw in items:
            if not isinstance(raw, str):
                raise ValueError("Каждый подзапрос batch должен быть строкой")
            method_part, query = (raw.split("?", 1) + [""])[:2]
            method = method_part.strip()
            parsed_args = {
                arg_key: values[-1] if values else ""
                for arg_key, values in parse_qs(query, keep_blank_values=True).items()
            }
            commands.append(BatchCommandInfo(str(key), method, query, parsed_args))
        return commands

    def _evaluate_batch_confirmation(self, commands: List[BatchCommandInfo]) -> Tuple[bool, str]:
        """Определяет, нужен ли запрос подтверждения для пакетного вызова."""

        reasons: List[str] = []
        confirm = False
        for command in commands:
            sub_method = command.method
            if not sub_method:
                confirm = True
                reasons.append("не удалось определить одно из действий в пакете")
                continue
            if sub_method in ALWAYS_CONFIRM_METHODS:
                confirm = True
                reasons.append(
                    f"настройка уведомлений («{self._friendly_method_name(sub_method)}» внутри пакета)"
                )
                continue
            risky_fields = RISKY_FIELDS_BY_METHOD.get(sub_method, {}).get("fields", set())
            impacted_fields = command.payload_fields() & risky_fields
            if impacted_fields:
                confirm = True
                field_list = ", ".join(
                    sorted(self._friendly_field_name(field) for field in impacted_fields)
                )
                reasons.append(
                    f"изменение параметров ({field_list}) в рамках «{self._friendly_method_name(sub_method)}»"
                )
                continue
            if sub_method in UPDATE_METHODS and sub_method not in RISKY_FIELDS_BY_METHOD:
                confirm = True
                reasons.append(
                    f"в пакет включено действие «{self._friendly_method_name(sub_method)}», его нужно подтвердить отдельно"
                )
        if confirm:
            reason_text = "; ".join(dict.fromkeys(reasons)) if reasons else "Пакетный вызов содержит критичные изменения"
        else:
            reason_text = ""
        return confirm, reason_text

    def _build_confirmation_key(self, method: str, params: Dict[str, Any]) -> str:
        """Формирует ключ подтверждения на основе метода и параметров."""

        serialized = json.dumps({"method": method, "params": params}, sort_keys=True, ensure_ascii=False)
        return md5(serialized.encode("utf-8")).hexdigest()

    def _build_confirmation_reason(
        self,
        method: str,
        params: Dict[str, Any],
        risky_fields: set[str],
    ) -> str:
        """Генерирует описание рискованного действия."""

        friendly_method = self._friendly_method_name(method)
        if not risky_fields:
            return f"Нужно подтвердить действие «{friendly_method}» — оно влияет на важные данные."

        fields_payload = params.get("fields") if isinstance(params.get("fields"), dict) else params
        target_fields: List[str] = []
        if isinstance(fields_payload, dict):
            for field in risky_fields:
                if field in fields_payload:
                    target_fields.append(self._friendly_field_name(field))

        if target_fields:
            field_list = ", ".join(sorted(target_fields))
            return f"Планируется изменить {field_list}. Подтвердите, пожалуйста."
        return f"Действие «{friendly_method}» затрагивает критичные настройки. Нужна ваша проверка."

    @staticmethod
    def _normalize_user_text(message: str) -> str:
        """Приводит текст пользователя к нижнему регистру и убирает лишние пробелы."""

        return " ".join(message.strip().lower().split())

    def _maybe_handle_plan_confirmation_message(self, state: AgentState, message: Optional[str]) -> Optional[str]:
        """Определяет, подтвердил или отклонил пользователь текущий план."""

        if not message or not state.plan_confirmation:
            return None

        normalized = self._normalize_user_text(message)
        if not normalized:
            return None

        plan_status = (state.plan_confirmation or {}).get("status")
        if any(phrase in normalized for phrase in PLAN_DENY_PHRASES):
            if plan_status not in {None, "denied"}:
                state.plan_confirmation["status"] = "denied"
                state.plan_confirmation["denied_at"] = _utc_iso_z()
                state.next_planned_actions = []
                return "Остановил выполнение плана. Напишите, какие шаги скорректировать."
            return None

        if any(phrase in normalized for phrase in PLAN_CONFIRM_PHRASES):
            steps = state.plan_confirmation.get("steps", [])
            state.plan_confirmation.setdefault("summary", self._summarize_plan_actions(steps))
            state.plan_confirmation.setdefault("step_keys", self._extract_action_keys(steps))
            state.plan_confirmation["status"] = "approved"
            state.plan_confirmation["approved_at"] = _utc_iso_z()
            self._mark_plan_steps_confirmed(state, steps)
            return "План подтверждён, приступаю к выполнению согласованных шагов."

        return None

    def _hash_plan(self, actions: List[Dict[str, Any]]) -> str:
        """Возвращает md5-хеш набора шагов плана."""

        canonical_actions = [self._canonicalize_plan_action(action) for action in actions]
        serialized = json.dumps(canonical_actions, sort_keys=True, ensure_ascii=False)
        return md5(serialized.encode("utf-8")).hexdigest()

    def _canonicalize_plan_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Возвращает копию шага без временных флагов, влияющих на подтверждение."""

        if not isinstance(action, dict):
            return action

        canonical: Dict[str, Any] = {}
        for key, value in action.items():
            if key in PLAN_HASH_IGNORED_KEYS:
                continue
            canonical[key] = self._canonicalize_plan_value(value)
        return canonical

    def _canonicalize_plan_value(self, value: Any) -> Any:
        """Рекурсивно нормализует значение шага для сравнения хеша."""

        if isinstance(value, dict):
            return {key: self._canonicalize_plan_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._canonicalize_plan_value(item) for item in value]
        return value

    def _extract_action_keys(self, actions: List[Dict[str, Any]]) -> List[str]:
        """Возвращает список ключей подтверждения для набора шагов."""

        keys: List[str] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            method = action.get("method") or ""
            if not method:
                continue
            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            confirmation_key = self._build_confirmation_key(method, params)
            if confirmation_key:
                keys.append(confirmation_key)
        return keys

    def _mark_plan_steps_confirmed(self, state: AgentState, steps: List[Dict[str, Any]]) -> None:
        """Помечает шаги плана как одобренные в списке подтверждений."""

        if not steps:
            return

        approved_keys: Set[str] = set(state.plan_confirmation.get("approved_step_keys") or [])
        for step in steps:
            if not isinstance(step, dict):
                continue
            method = step.get("method") or ""
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            if not method:
                continue

            confirmation_key = self._build_confirmation_key(method, params)
            if not confirmation_key:
                continue

            record = state.confirmations.get(confirmation_key, {})
            reason = record.get("reason") or step.get("confirmation_reason")
            risky_fields = RISKY_FIELDS_BY_METHOD.get(method, {}).get("fields", set())
            if not reason:
                reason = self._build_confirmation_reason(method, params, risky_fields)

            record.update(
                {
                    "status": "approved",
                    "approved_at": _utc_iso_z(),
                    "action": step,
                    "reason": reason,
                    "description": reason,
                }
            )
            state.confirmations[confirmation_key] = record
            approved_keys.add(confirmation_key)

        if approved_keys:
            state.plan_confirmation["approved_step_keys"] = list(dict.fromkeys(approved_keys))

    def _summarize_plan_actions(self, actions: List[Dict[str, Any]]) -> str:
        """Строит краткое человеко-понятное описание плана."""

        if not actions:
            return ""

        lines: List[str] = []
        for idx, action in enumerate(actions, 1):
            method = action.get("method")
            friendly = self._friendly_method_name(method)
            comment = (action.get("comment") or "").strip()

            params = action.get("params") if isinstance(action.get("params"), dict) else {}
            extra: Optional[str] = None
            if isinstance(params, dict):
                if "id" in params:
                    extra = f"id={params.get('id')}"
                elif "ID" in params:
                    extra = f"id={params.get('ID')}"
                elif "taskId" in params:
                    extra = f"taskId={params.get('taskId')}"
                elif isinstance(params.get("fields"), dict):
                    entity_id = params["fields"].get("ENTITY_ID")
                    entity_type = params["fields"].get("ENTITY_TYPE")
                    if entity_type and entity_id:
                        extra = f"{entity_type}:{entity_id}"

            pieces = [f"{idx}) {friendly}"]
            if comment:
                pieces.append(f"— {comment}")
            if extra:
                pieces.append(f"({extra})")
            lines.append(" ".join(pieces).strip())

        return "\n".join(lines)

    def _store_last_plan(self, state: AgentState, actions: List[Dict[str, Any]], summary: str) -> None:
        """Сохраняет последний план в состоянии пользователя."""

        state.last_plan = {
            "timestamp": _utc_iso_z(),
            "summary": summary or "",
            "actions": actions or [],
        }

    def _compose_final_reply(
        self,
        assistant_from_model: str,
        plan_summary: str,
        execution_summary: str,
        errors: List[ErrorEntry],
        warnings: List[str],
    ) -> str:
        """Собирает финальный ответ: план, результаты, ошибки и рекомендации."""

        reply_parts: List[str] = []
        if plan_summary:
            reply_parts.append(f"План действий:\n{plan_summary}")

        if self.settings.mode == "shadow":
            reply_parts.append(
                "Работаю в режиме shadow — план сохранил, но запросы к Bitrix24 не выполнял."
            )
        elif execution_summary:
            reply_parts.append(execution_summary)

        sanitized_assistant = self._strip_confirmation_language(assistant_from_model)
        if sanitized_assistant:
            reply_parts.append(sanitized_assistant)

        base_reply = "\n\n".join(part.strip() for part in reply_parts if part and part.strip())
        combined_errors: List[ErrorEntry] = list(errors)
        combined_errors.extend(warnings)
        final_reply = self._append_errors_to_reply(base_reply, combined_errors)
        return final_reply.strip()

    @staticmethod
    def _strip_confirmation_language(text: str) -> str:
        """Удаляет из ответа модели просьбы подтвердить план и лишние повторения."""

        if not text:
            return ""

        lines = []
        for line in text.splitlines():
            normalized = line.strip().lower()
            if not normalized:
                lines.append(line)
                continue
            if "подтверд" in normalized or "жду подтверждения" in normalized:
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _update_plan_confirmation_state(self, state: AgentState, actions: List[Dict[str, Any]]) -> None:
        """Синхронизирует состояние плана с текущим ACTION."""

        if not actions:
            state.plan_confirmation = {}
            state.next_planned_actions = []
            return

        plan_hash = self._hash_plan(actions)
        plan_summary = self._summarize_plan_actions(actions)
        new_step_keys = self._extract_action_keys(actions)
        plan_record = state.plan_confirmation or {}

        current_hash = plan_record.get("hash")
        current_status = plan_record.get("status")
        approved_keys: Set[str] = set(plan_record.get("approved_step_keys") or [])

        if current_hash != plan_hash:
            subset_of_approved = bool(approved_keys) and set(new_step_keys).issubset(approved_keys)
            if current_status == "approved" and subset_of_approved:
                plan_record["hash"] = plan_hash
                plan_record["summary"] = plan_summary
                plan_record["steps"] = actions
                plan_record["step_keys"] = new_step_keys
                state.plan_confirmation = plan_record
            else:
                if current_status == "pending":
                    plan_record["status"] = "denied"
                    plan_record["denied_at"] = _utc_iso_z()
                state.plan_confirmation = {
                    "status": "pending",
                    "hash": plan_hash,
                    "requested_at": _utc_iso_z(),
                    "approved_at": None,
                    "denied_at": None,
                    "summary": plan_summary,
                    "steps": actions,
                    "step_keys": new_step_keys,
                    "approved_step_keys": [],
                }
        else:
            if current_status == "denied":
                plan_record["status"] = "pending"
                plan_record["requested_at"] = _utc_iso_z()
                plan_record["denied_at"] = None
                plan_record["approved_at"] = None
                plan_record["approved_step_keys"] = []
            plan_record["summary"] = plan_summary
            plan_record["steps"] = actions
            plan_record["step_keys"] = new_step_keys
            state.plan_confirmation = plan_record

        state.next_planned_actions = actions

    def _ensure_plan_prompt_appended(self, reply: str, state: AgentState) -> str:
        """Добавляет в ответ просьбу подтвердить план, если он не одобрен."""

        plan_record = state.plan_confirmation or {}
        if plan_record.get("status") != "pending":
            return reply

        summary = plan_record.get("summary") or self._summarize_plan_actions(plan_record.get("steps", []))
        prompt_parts: List[str] = []
        if summary:
            prompt_parts.append("План действий:\n" + summary)
        prompt_parts.append("Подтвердите план или напишите, что нужно изменить.")
        prompt_block = "\n\n".join(prompt_parts)
        cleaned_reply = reply.strip()
        return f"{cleaned_reply}\n\n{prompt_block}" if cleaned_reply else prompt_block

    def _update_state_from_action(self, state: AgentState, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Обновляет `agent_state` на основе выполненного действия."""

        method = action.get("method")
        if method == "crm.deal.add":
            deal_id = result.get("result")
            if deal_id:
                state.objects["current_deal_id"] = deal_id
                self._append_done_entry(state, "Создана новая сделка", {"deal_id": deal_id})
        elif method == "tasks.task.add":
            task_id = result.get("result", {}).get("task", {}).get("id")
            if task_id:
                state.objects["current_task_id"] = task_id
                self._append_done_entry(state, "Создана новая задача", {"task_id": task_id})
        elif method == "crm.deal.update":
            if result.get("result"):
                deal_id = (action.get("params") or {}).get("id")
                if deal_id:
                    state.objects["current_deal_id"] = deal_id
                    self._append_done_entry(state, "Обновлены параметры сделки", {"deal_id": deal_id})
        elif method == "crm.deal.list":
            deals = result.get("result") if isinstance(result, dict) else None
            if isinstance(deals, list):
                payload = {"count": len(deals)}
                if isinstance(result, dict) and result.get("total") is not None:
                    payload["total"] = result.get("total")
                self._append_done_entry(state, "Получен список сделок", payload)
        elif method == "crm.contact.list":
            contacts = result.get("result") if isinstance(result, dict) else None
            if isinstance(contacts, list):
                payload = {"count": len(contacts)}
                if isinstance(result, dict) and result.get("total") is not None:
                    payload["total"] = result.get("total")
                self._append_done_entry(state, "Получен список контактов", payload)
        elif method == "crm.deal.get":
            deal_data = result.get("result") if isinstance(result, dict) else None
            if isinstance(deal_data, dict):
                deal_id = deal_data.get("ID")
                if deal_id:
                    state.objects["current_deal_id"] = deal_id
                contact_id = deal_data.get("CONTACT_ID")
                if contact_id:
                    state.objects["current_contact_id"] = contact_id
                company_id = deal_data.get("COMPANY_ID")
                if company_id:
                    state.objects["current_company_id"] = company_id
                if deal_id or contact_id or company_id:
                    self._append_done_entry(
                        state,
                        "Получены данные сделки",
                        {
                            "deal_id": deal_data.get("ID"),
                            "contact_id": deal_data.get("CONTACT_ID"),
                            "company_id": deal_data.get("COMPANY_ID"),
                        },
                    )
        elif method == "crm.deal.category.list":
            categories = result.get("result") if isinstance(result, dict) else None
            if isinstance(categories, list):
                self._append_done_entry(state, "Получен список направлений продаж", {"count": len(categories)})
        elif method == "crm.deal.category.stage.list":
            stages = result.get("result") if isinstance(result, dict) else None
            if isinstance(stages, list):
                params_payload = action.get("params") or {}
                category_id = (
                    params_payload.get("id")
                    or params_payload.get("ID")
                    or params_payload.get("categoryId")
                    or params_payload.get("CATEGORY_ID")
                )
                payload = {"count": len(stages)}
                if category_id is not None:
                    payload["category_id"] = category_id
                self._append_done_entry(state, "Получен список стадий сделки", payload)
        elif method == "crm.status.list":
            statuses = result.get("result") if isinstance(result, dict) else None
            if isinstance(statuses, list):
                entity_id = ((action.get("params") or {}).get("filter") or {}).get("ENTITY_ID")
                payload = {"count": len(statuses)}
                if entity_id:
                    payload["entity_id"] = entity_id
                self._append_done_entry(state, "Получен справочник CRM", payload)
        elif method == "crm.contact.get":
            contact_data = result.get("result") if isinstance(result, dict) else None
            if isinstance(contact_data, dict):
                contact_id = contact_data.get("ID")
                if contact_id:
                    state.objects["current_contact_id"] = contact_id
                company_id = contact_data.get("COMPANY_ID")
                if company_id:
                    state.objects["current_company_id"] = company_id
                if contact_id or company_id:
                    self._append_done_entry(
                        state,
                        "Получены данные контакта",
                        {
                            "contact_id": contact_data.get("ID"),
                            "company_id": contact_data.get("COMPANY_ID"),
                        },
                    )
        elif method == "crm.company.list":
            companies = result.get("result") if isinstance(result, dict) else None
            if isinstance(companies, list):
                payload = {"count": len(companies)}
                if isinstance(result, dict) and result.get("total") is not None:
                    payload["total"] = result.get("total")
                self._append_done_entry(state, "Получен список компаний", payload)
        elif method == "crm.company.get":
            company_data = result.get("result") if isinstance(result, dict) else None
            if isinstance(company_data, dict):
                company_id = company_data.get("ID")
                if company_id:
                    state.objects["current_company_id"] = company_id
                if company_id:
                    self._append_done_entry(
                        state,
                        "Получены данные компании",
                        {"company_id": company_id},
                    )
        elif method == "crm.timeline.comment.add":
            comment_id = result.get("result")
            entity_id = (action.get("params") or {}).get("fields", {}).get("ENTITY_ID")
            if comment_id:
                self._append_done_entry(
                    state,
                    "Добавлен комментарий в таймлайн",
                    {"comment_id": comment_id, "entity_id": entity_id},
                )
        elif method == "task.commentitem.add":
            comment_id = result.get("result")
            task_id = (action.get("params") or {}).get("taskId") or (action.get("params") or {}).get("TASK_ID")
            if comment_id:
                self._append_done_entry(
                    state,
                    "Добавлен комментарий к задаче",
                    {"task_comment_id": comment_id, "task_id": task_id},
                )
        elif method == "task.checklistitem.add":
            item_id = result.get("result")
            task_id = (action.get("params") or {}).get("taskId") or (action.get("params") or {}).get("TASK_ID")
            if item_id:
                self._append_done_entry(
                    state,
                    "Добавлен пункт чек-листа",
                    {"checklist_item_id": item_id, "task_id": task_id},
                )
        elif method == "tasks.task.update":
            updated_task = result.get("result", {}).get("task", {})
            task_id = updated_task.get("id") or (action.get("params") or {}).get("taskId")
            if task_id:
                state.objects["current_task_id"] = task_id
                self._append_done_entry(state, "Обновлены параметры задачи", {"task_id": task_id})
        elif method == "crm.activity.add":
            activity_id = result.get("result")
            owner_id = (action.get("params") or {}).get("fields", {}).get("OWNER_ID")
            if activity_id:
                self._append_done_entry(
                    state,
                    "Создана активность CRM",
                    {"activity_id": activity_id, "owner_id": owner_id},
                )
        elif method == "crm.activity.list":
            activities = result.get("result") if isinstance(result, dict) else None
            total = result.get("total") if isinstance(result, dict) else None
            if isinstance(activities, list):
                payload = {"count": len(activities)}
                if total is not None:
                    payload["total"] = total
                self._append_done_entry(state, "Получен список активностей", payload)
        elif method == "tasks.task.list":
            tasks_payload = result.get("result") if isinstance(result, dict) else None
            total = result.get("total") if isinstance(result, dict) else None
            if isinstance(tasks_payload, list):
                payload = {"count": len(tasks_payload)}
                if total is not None:
                    payload["total"] = total
                self._append_done_entry(state, "Получен список задач", payload)
        elif method == "event.bind":
            if result.get("result") is True:
                event_code = (action.get("params") or {}).get("event")
                handler_url = (action.get("params") or {}).get("handler")
                binding = {"event": event_code, "handler": handler_url}
                if event_code and handler_url and binding not in state.event_bindings:
                    state.event_bindings.append(binding)
                self._append_done_entry(
                    state,
                    "Подписка на событие обновлена",
                    {"event": event_code, "handler": handler_url},
                )
        elif method == "event.unbind":
            if result.get("result") is True:
                event_code = (action.get("params") or {}).get("event")
                handler_url = (action.get("params") or {}).get("handler")
                if event_code and handler_url:
                    state.event_bindings = [
                        item
                        for item in state.event_bindings
                        if not (item.get("event") == event_code and item.get("handler") == handler_url)
                    ]
                self._append_done_entry(
                    state,
                    "Подписка на событие удалена",
                    {"event": event_code, "handler": handler_url},
                )
        elif method == "event.get":
            bindings = result.get("result")
            if isinstance(bindings, dict) and "result" in bindings:
                bindings = bindings["result"]
            if isinstance(bindings, list):
                state.event_bindings = [
                    {"event": item.get("event"), "handler": item.get("handler")}
                    for item in bindings
                    if isinstance(item, dict)
                ]
                self._append_done_entry(
                    state,
                    "Получен список подписок",
                    {"count": len(state.event_bindings)},
                )
        elif method == "sonet.group.get":
            groups = result.get("result") if isinstance(result, dict) else None
            if isinstance(groups, list):
                self._append_done_entry(state, "Получен список рабочих групп", {"count": len(groups)})
        elif method == "sonet.group.user.get":
            members = result.get("result") if isinstance(result, dict) else None
            if isinstance(members, list):
                group_id = (action.get("params") or {}).get("GROUP_ID")
                payload = {"count": len(members)}
                if group_id is not None:
                    payload["group_id"] = group_id
                self._append_done_entry(state, "Получен список участников группы", payload)
        elif method == "batch":
            try:
                commands = self._extract_batch_commands(action)
            except ValueError:
                commands = []
            summary = {"commands": [{"key": cmd.key, "method": cmd.method} for cmd in commands]}
            self._append_done_entry(state, "Выполнен пакетный вызов batch", summary)
            batch_result = (result.get("result") or {}).get("result") if isinstance(result, dict) else None
            if isinstance(batch_result, dict):
                for command in commands:
                    sub_result = batch_result.get(command.key)
                    if command.method == "event.bind" and sub_result is True:
                        faux_action = {"method": "event.bind", "params": {"event": command.get_argument("event"), "handler": command.get_argument("handler")}}
                        self._update_state_from_action(state, faux_action, {"result": True})
                    elif command.method == "event.unbind" and sub_result is True:
                        faux_action = {"method": "event.unbind", "params": {"event": command.get_argument("event"), "handler": command.get_argument("handler")}}
                        self._update_state_from_action(state, faux_action, {"result": True})
                    elif command.method == "event.get":
                        data = sub_result
                        if isinstance(data, dict) and "result" in data:
                            data = data["result"]
                        faux_action = {"method": "event.get", "params": {}}
                        self._update_state_from_action(state, faux_action, {"result": data})

        self._remove_action_from_plan(state, action)
    def _append_done_entry(self, state: AgentState, description: str, object_ids: Dict[str, Any]) -> None:
        """Добавляет запись об успешном действии в историю."""

        state.done.append(
            {
                "timestamp": _utc_iso_z(),
                "description": description,
                "object_ids": object_ids,
            }
        )

    def _remove_action_from_plan(self, state: AgentState, action: Dict[str, Any]) -> None:
        """Удаляет выполненный или отклонённый шаг из отложенного плана."""

        method = action.get("method", "")
        params = action.get("params") or {}
        if not method:
            return

        target_key = self._build_confirmation_key(method, params)
        filtered: List[Dict[str, Any]] = []
        for item in state.next_planned_actions:
            item_method = item.get("method", "")
            item_params = item.get("params") or {}
            item_key = self._build_confirmation_key(item_method, item_params)
            if item_key != target_key:
                filtered.append(item)
        state.next_planned_actions = filtered

        if state.plan_confirmation:
            steps = [
                step
                for step in state.plan_confirmation.get("steps", [])
                if self._build_confirmation_key(step.get("method", ""), step.get("params") or {}) != target_key
            ]
            state.plan_confirmation["steps"] = steps
            state.plan_confirmation["step_keys"] = self._extract_action_keys(steps)
            state.plan_confirmation["summary"] = self._summarize_plan_actions(steps)
