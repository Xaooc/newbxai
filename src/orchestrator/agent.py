"""Оркестратор, управляющий коммуникацией с моделью и Bitrix24."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple

from src.bitrix_client.client import BitrixClientError, call_bitrix
from src.logging.logger import InteractionLogger
from src.orchestrator.model_client import (
    ModelClient,
    ModelClientError,
)
from src.state.manager import AgentStateManager, AgentState

logger = logging.getLogger(__name__)


READ_METHODS = {
    "user.current",
    "user.get",
    "crm.contact.list",
    "crm.contact.get",
    "crm.company.list",
    "crm.company.get",
    "crm.deal.list",
    "crm.deal.get",
    "crm.activity.list",
    "tasks.task.list",
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
}

ALL_ALLOWED_METHODS = READ_METHODS | SAFE_CREATE_METHODS | UPDATE_METHODS

ALLOWED_METHODS_BY_MODE = {
    "shadow": set(),
    "canary": READ_METHODS | SAFE_CREATE_METHODS,
    "full": ALL_ALLOWED_METHODS,
}

RISKY_FIELDS_BY_METHOD = {
    "crm.deal.add": {"fields": {"OPPORTUNITY", "ASSIGNED_BY_ID", "STAGE_ID", "CATEGORY_ID"}},
    "crm.deal.update": {"fields": {"OPPORTUNITY", "ASSIGNED_BY_ID", "STAGE_ID", "CATEGORY_ID"}},
    "crm.activity.add": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
    "tasks.task.add": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
    "tasks.task.update": {"fields": {"RESPONSIBLE_ID", "DEADLINE"}},
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
    " crm.company.get, crm.deal.list, crm.deal.get, crm.deal.add, crm.deal.update, crm.activity.list,"
    " crm.activity.add, crm.timeline.comment.add, tasks.task.add, tasks.task.update, tasks.task.list,"
    " task.commentitem.add, task.checklistitem.add. Запрещено использовать любые иные методы.\n"
    "Перед изменением сумм, стадий, ответственных, дедлайнов указывай requires_confirmation=true и жди подтверждения."
)


@dataclass
class OrchestratorSettings:
    """Настройки оркестратора."""

    mode: str = "shadow"
    model_name: str = "gpt-5-thinking"
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

    def process_message(self, user_id: str, message: str) -> str:
        """Обрабатывает сообщение пользователя и возвращает ответ ассистента.

        Args:
            user_id: Идентификатор пользователя или сессии.
            message: Текст запроса.

        Returns:
            Ответ ассистента, который нужно показать пользователю.
        """

        state = self.state_manager.load_state(user_id)
        if message and (not state.goals or state.goals[0] != message):
            state.goals.insert(0, message)
        logger.debug("Загружено состояние", extra={"user_id": user_id, "state": state})

        model_response = self._call_model(message, state)
        self.interaction_logger.log_model_response(user_id, message, model_response)

        actions = model_response.get("ACTION", [])
        assistant_reply = model_response.get("ASSISTANT", "")

        executed_actions: List[Dict[str, Any]] = []
        errors: List[str] = []
        pending_actions: List[Dict[str, Any]] = []

        if self.settings.mode == "shadow":
            state.next_planned_actions = actions
        else:
            for action in actions:
                method = action.get("method")
                params = action.get("params") or {}
                comment = action.get("comment", "")
                http_method = action.get("http_method", "POST")
                decision = (action.get("confirmation_decision") or "").strip().lower()

                if decision == "approve" and not action.get("confirmed"):
                    action["confirmed"] = True

                if decision == "deny":
                    confirmation_key = self._build_confirmation_key(method or "", params)
                    denial_reason = action.get("confirmation_reason") or self._build_confirmation_reason(
                        method or "неизвестный метод",
                        params,
                        RISKY_FIELDS_BY_METHOD.get(method or "", {}).get("fields", set()),
                    )
                    record = state.confirmations.get(confirmation_key, {})
                    record.update(
                        {
                            "status": "denied",
                            "denied_at": datetime.utcnow().isoformat() + "Z",
                            "action": action,
                            "reason": denial_reason,
                            "description": denial_reason,
                        }
                    )
                    state.confirmations[confirmation_key] = record
                    self._remove_action_from_plan(state, action)
                    errors.append(f"Шаг {method or 'без имени'} отклонён: {denial_reason}")
                    continue

                if not method:
                    errors.append("В ACTION найден шаг без метода")
                    pending_actions.append(action)
                    continue

                if not self._is_action_allowed(method):
                    errors.append(f"Метод {method} запрещён в режиме {self.settings.mode}")
                    pending_actions.append(action)
                    continue

                confirmation_needed, confirmation_key, confirmation_reason = self._check_confirmation_needed(
                    state, action
                )
                if confirmation_needed:
                    record_status = state.confirmations.get(confirmation_key or "", {}).get("status")
                    if record_status == "denied":
                        errors.append(
                            f"Шаг {method} ранее отклонён: {confirmation_reason}. Сформулируйте новый план, если ситуация изменилась."
                        )
                        self._remove_action_from_plan(state, action)
                        continue
                    errors.append(
                        f"Требуется подтверждение: {confirmation_reason}. Ответьте, чтобы агент повторил действие."
                    )
                    pending_actions.append(action)
                    continue

                if confirmation_key:
                    record = state.confirmations.get(confirmation_key, {})
                    record["status"] = "approved"
                    record["approved_at"] = datetime.utcnow().isoformat() + "Z"
                    record["reason"] = confirmation_reason
                    record["description"] = confirmation_reason
                    state.confirmations[confirmation_key] = record

                try:
                    result = call_bitrix(method, params, http_method=http_method)
                    executed_actions.append({
                        "method": method,
                        "params": params,
                        "comment": comment,
                        "result": result,
                    })
                    self._update_state_from_action(state, action, result)
                except BitrixClientError as exc:
                    errors.append(str(exc))
                    pending_actions.append(action)

        if self.settings.mode != "shadow":
            state.next_planned_actions = pending_actions

        if errors:
            assistant_reply += "\n\n" + "\n".join(f"⚠️ {err}" for err in errors)

        self.state_manager.save_state(user_id, state)
        self.interaction_logger.log_iteration(user_id, message, model_response, state, executed_actions, errors)

        return assistant_reply or "Не удалось получить ответ от модели."

    def _call_model(self, message: str, state: AgentState) -> Dict[str, Any]:
        """Заглушка вызова GPT-5 Thinking.

        В реальной реализации здесь будет обращение к API. Пока возвращается
        заранее подготовленный шаблон ответа.
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

    def _is_action_allowed(self, method: str) -> bool:
        """Проверяет, разрешён ли метод в текущем режиме безопасности."""

        allowed = ALLOWED_METHODS_BY_MODE.get(self.settings.mode, set())
        return method in allowed

    def _check_confirmation_needed(self, state: AgentState, action: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Определяет, требуется ли подтверждение перед выполнением шага."""

        method = action.get("method", "")
        params = action.get("params") or {}
        explicit_request = action.get("requires_confirmation", False)
        confirmed_flag = action.get("confirmed", False)
        reason = action.get("confirmation_reason")

        risky_fields = RISKY_FIELDS_BY_METHOD.get(method, {}).get("fields", set())
        payload_fields: set[str] = set()
        if isinstance(params, dict):
            fields_payload = params.get("fields") if isinstance(params.get("fields"), dict) else params
            if isinstance(fields_payload, dict):
                payload_fields = {key.upper() for key in fields_payload.keys()}

        auto_request = bool(risky_fields & payload_fields)
        needs_confirmation = explicit_request or auto_request

        if not needs_confirmation:
            return False, None, None

        key = self._build_confirmation_key(method, params)
        record = state.confirmations.get(key)

        if confirmed_flag or (record and record.get("status") == "approved"):
            confirmation_text = reason or self._build_confirmation_reason(method, params, risky_fields)
            updated_record = record or {}
            updated_record.update(
                {
                    "status": "approved",
                    "approved_at": datetime.utcnow().isoformat() + "Z",
                    "action": action,
                    "reason": confirmation_text,
                    "description": confirmation_text,
                }
            )
            state.confirmations[key] = updated_record
            return False, key, updated_record["reason"]

        if record and record.get("status") == "denied":
            return True, key, record.get("reason", "операция ранее отклонена")

        confirmation_reason = reason or self._build_confirmation_reason(method, params, risky_fields)
        state.confirmations[key] = {
            "status": "requested",
            "requested_at": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "reason": confirmation_reason,
            "description": confirmation_reason,
        }
        return True, key, confirmation_reason

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

        if not risky_fields:
            return f"Необходимо подтверждение для вызова {method}"

        fields_payload = params.get("fields") if isinstance(params.get("fields"), dict) else params
        target_fields: List[str] = []
        if isinstance(fields_payload, dict):
            for field in risky_fields:
                if field in fields_payload:
                    target_fields.append(field)

        if target_fields:
            field_list = ", ".join(sorted(target_fields))
            return f"Изменение полей {field_list} через {method}"
        return f"Критичный вызов {method}"

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

        self._remove_action_from_plan(state, action)

    def _append_done_entry(self, state: AgentState, description: str, object_ids: Dict[str, Any]) -> None:
        """Добавляет запись об успешном действии в историю."""

        state.done.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
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
