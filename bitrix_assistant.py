# -*- coding: utf-8 -*-
"""
Helpers for building a Bitrix24-focused LangChain agent.
The agent exposes Bitrix REST methods as structured tools, plans multi-step
workflows, and aggregates calendar events for multiple users.
"""
from __future__ import annotations
import inspect
import json
import logging
import textwrap
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError, create_model
from bitrix24_client import Bitrix24Client, Bitrix24Error
from memory_utils import ConversationBufferMemory
load_dotenv()
logger = logging.getLogger(__name__)
@dataclass
class BitrixMethodSpec:
    """Description of a public Bitrix24Client method."""
    name: str
    signature: inspect.Signature
    summary: str
    param_docs: Dict[str, str]
class BitrixToolset:
    """Create StructuredTool objects for Bitrix24Client endpoints."""
    def __init__(self, client: Bitrix24Client) -> None:
        self._client = client
        self._method_specs = self._collect_public_methods(client)
    @staticmethod
    def _collect_public_methods(client: Bitrix24Client) -> List[BitrixMethodSpec]:
        specs: List[BitrixMethodSpec] = []
        for name, attr in inspect.getmembers(client, predicate=callable):
            if name.startswith("_") or name in {"log"}:
                continue
            try:
                signature = inspect.signature(attr)
            except (TypeError, ValueError):
                continue
            doc = inspect.getdoc(attr) or ""
            summary = textwrap.shorten(doc, width=200, placeholder="...") if doc else "Bitrix24 REST method."
            param_docs = BitrixToolset._extract_param_docs(doc)
            specs.append(BitrixMethodSpec(name=name, signature=signature, summary=summary, param_docs=param_docs))
        specs.sort(key=lambda item: item.name)
        return specs
    @staticmethod
    def _extract_param_docs(doc: str) -> Dict[str, str]:
        details: Dict[str, str] = {}
        for line in doc.splitlines():
            line = line.strip()
            if not line.startswith(":param "):
                continue
            payload = line[len(":param ") :]
            if ":" not in payload:
                continue
            param_name, description = payload.split(":", 1)
            details[param_name.strip()] = description.strip()
        return details
    def build_tools(self) -> List[StructuredTool]:
        tools: List[StructuredTool] = []
        for spec in self._method_specs:
            logger.debug("Готовлю инструмент для метода Bitrix24: %s", spec.name)
            tool = self._build_tool(spec)
            if tool is not None:
                logger.debug("Инструмент %s успешно создан.", spec.name)
                tools.append(tool)
            else:
                logger.debug("Инструмент для метода %s опущен из-за неподдерживаемой сигнатуры.", spec.name)
        return tools
    def _build_tool(self, spec: BitrixMethodSpec) -> StructuredTool | None:
        args_model = self._build_args_model(spec)
        if args_model is None:
            logger.debug("Skip method %s: unsupported signature.", spec.name)
            return None
        method = getattr(self._client, spec.name)
        description = self._compose_description(spec, args_model)
        logger.debug("Создаю StructuredTool для метода %s.", spec.name)
        def _invoke(**kwargs: Any) -> str:
            logger.debug("Инструмент %s получил вход: %s", spec.name, kwargs)
            try:
                parsed = args_model(**kwargs)
            except ValidationError as exc:
                logger.warning("Validation error in tool %s: %s", spec.name, exc)
                error_payload = self._format_response(
                    status="error",
                    method=spec.name,
                    params=kwargs,
                    payload={"type": "ValidationError", "message": exc.errors()},
                )
                logger.debug("Инструмент %s завершился ошибкой валидации: %s", spec.name, error_payload)
                return error_payload
            params = parsed.model_dump(mode="json", exclude_none=True)
            logger.debug("Инструмент %s вызывает Bitrix24 с параметрами: %s", spec.name, params)
            try:
                result = method(**params)
            except Bitrix24Error as exc:
                logger.exception("Bitrix24Error while calling %s", spec.name)
                recommendations = self._build_recommendations(
                    spec.name,
                    http_status=exc.http_status,
                    b24_code=exc.b24_code,
                    message=str(exc),
                    params=params,
                )
                error_payload = self._format_response(
                    status="error",
                    method=spec.name,
                    params=params,
                    payload={
                        "type": "Bitrix24Error",
                        "message": str(exc),
                        "http_status": exc.http_status,
                        "bitrix_code": exc.b24_code,
                        "recommendations": recommendations,
                    },
                )
                logger.debug("Инструмент %s вернул ошибку Bitrix24: %s", spec.name, error_payload)
                return error_payload
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error while calling %s", spec.name)
                recommendations = self._build_recommendations(spec.name, message=str(exc), params=params)
                error_payload = self._format_response(
                    status="error",
                    method=spec.name,
                    params=params,
                    payload={
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                        "recommendations": recommendations,
                    },
                )
                logger.debug("Инструмент %s завершился неожиданной ошибкой: %s", spec.name, error_payload)
                return error_payload
            success_payload = self._format_response(
                status="ok",
                method=spec.name,
                params=params,
                payload={"result": result},
            )
            logger.debug("Инструмент %s успешно завершён. Payload: %s", spec.name, success_payload)
            return success_payload
        return StructuredTool.from_function(
            func=_invoke,
            name=spec.name,
            description=description,
            args_schema=args_model,
            infer_schema=False,
        )
    def _build_args_model(self, spec: BitrixMethodSpec) -> type[BaseModel] | None:
        fields: Dict[str, tuple[Any, Any]] = {}
        for param in spec.signature.parameters.values():
            if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                return None
            if param.name == "self":
                continue
            annotation = param.annotation if param.annotation is not inspect._empty else Any
            description = spec.param_docs.get(param.name)
            if param.default is inspect._empty:
                default = Field(..., description=description)
            else:
                default = Field(param.default, description=description)
            fields[param.name] = (annotation, default)
        model_name = f"{spec.name.title().replace('_', '')}Input"
        args_model = create_model(model_name, **fields)  # type: ignore[arg-type]
        args_model.__module__ = __name__
        args_model.__doc__ = spec.summary
        args_model.model_rebuild()
        return args_model
    def _compose_description(self, spec: BitrixMethodSpec, args_model: type[BaseModel]) -> str:
        lines = [spec.summary, "", "Параметры:"]
        for field_name, field in args_model.model_fields.items():
            requirement = "обязательный" if field.is_required() else f"по умолчанию: {field.default!r}"
            detail = field.description or ""
            lines.append(f"- {field_name}: {detail} ({requirement})")
        lines.append("Возвращает JSON-ответ со статусом и данными Bitrix24.")
        return "\n".join(lines)
    def _format_response(
        self,
        *,
        status: str,
        method: str,
        params: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> str:
        body: Dict[str, Any] = {
            "status": status,
            "method": method,
            "requested_params": dict(params),
        }
        body.update(payload)
        return json.dumps(body, ensure_ascii=False, default=str)
    def _build_recommendations(
        self,
        method_name: str,
        *,
        http_status: Optional[int] = None,
        b24_code: Optional[str] = None,
        message: Optional[str] = None,
        params: Mapping[str, Any] | None = None,
    ) -> List[str]:
        suggestions: List[str] = []
        normalized_code = (b24_code or "").upper()
        normalized_message = (message or "").lower()
        if http_status in {401, 403} or "ACCESS_DENIED" in normalized_code:
            suggestions.append("Проверьте права доступа и корректность OAuth-токена Bitrix24.")
        if http_status == 404 or "not found" in normalized_message:
            suggestions.append("Убедитесь, что указанные идентификаторы существуют в Bitrix24.")
        if http_status in {408, 504} or "timeout" in normalized_message:
            suggestions.append("Повторите запрос позже или уменьшите объём данных, чтобы избежать таймаута.")
        if "required parameter missing" in normalized_message or "missing parameter" in normalized_message:
            suggestions.append("Заполните обязательные параметры метода Bitrix24.")
        if "limit" in normalized_message or "too many requests" in normalized_message:
            suggestions.append("Снизьте частоту запросов или используйте пакетные операции Bitrix24.")
        if not suggestions:
            suggestions.append("Изучите ответ Bitrix24, исправьте причину ошибки и повторите запрос.")
        if params:
            suggestions.append(f"Параметры запроса: {dict(params)}")
        logger.debug(
            "Рекомендации для Bitrix24 method=%s status=%s code=%s: %s",
            method_name,
            http_status,
            b24_code,
            suggestions,
        )
        return suggestions
class CalendarEventsForUsersArgs(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Дата в формате YYYY-MM-DD или ISO. Если не указана, берётся текущий день.",
    )
    user_ids: Optional[List[int]] = Field(
        None,
        description="Список ID сотрудников Bitrix24. Если не указан, используется список активных сотрудников.",
    )
    include_inactive: bool = Field(
        False,
        description="Включать ли пользователей с ACTIVE='N' (используется только если user_ids не задан).",
    )
    timezone: Optional[str] = Field(
        None,
        description="Часовой пояс, например 'Europe/Moscow'. Если не указан, используется локальный.",
    )
    max_users: int = Field(
        100,
        ge=1,
        le=500,
        description="Максимальное количество пользователей, для которых выгружаются события.",
    )
def _calendar_events_for_users(client: Bitrix24Client, args: CalendarEventsForUsersArgs) -> Dict[str, Any]:
    tz = _resolve_timezone(args.timezone)
    day_start, day_end, day_label = _normalize_day_bounds(args.date, tz)
    since_iso = day_start.isoformat()
    until_iso = day_end.isoformat()
    user_records: List[Mapping[str, Any]] = []
    truncated = False
    if args.user_ids:
        ids: List[int] = []
        seen: set[int] = set()
        for uid in args.user_ids:
            if uid is None:
                continue
            int_id = int(uid)
            if int_id in seen:
                continue
            seen.add(int_id)
            ids.append(int_id)
        if not ids:
            return {
                "status": "error",
                "method": "calendar_events_for_users",
                "message": "Параметр user_ids не содержит допустимых значений.",
            }
        try:
            filter_payload: Mapping[str, Any] = {"ID": [str(uid) for uid in ids]}
            user_records = client.user_get(
                filter=filter_payload,
                select=["ID", "NAME", "LAST_NAME", "EMAIL", "ACTIVE"],
                fetch_all=False,
            )
        except Bitrix24Error as exc:
            logger.warning("Не удалось загрузить карточки пользователей %s: %s", ids, exc)
            user_records = [{"ID": uid} for uid in ids]
    else:
        filter_payload = {"ACTIVE": "true"} if not args.include_inactive else None
        try:
            user_records = client.user_get(
                filter=filter_payload,
                select=["ID", "NAME", "LAST_NAME", "EMAIL", "ACTIVE"],
                fetch_all=True,
            )
        except Bitrix24Error as exc:
            return {
                "status": "error",
                "method": "calendar_events_for_users",
                "message": f"Не удалось получить список сотрудников: {exc}",
            }
    if not user_records:
        return {
            "status": "ok",
            "date": day_label,
            "timezone": getattr(tz, "key", str(tz)),
            "users_processed": 0,
            "events_total": 0,
            "users": [],
            "notice": "Активных сотрудников не найдено.",
        }
    if args.max_users and len(user_records) > args.max_users:
        truncated = True
        user_records = user_records[: args.max_users]
    user_map: Dict[int, Mapping[str, Any]] = {}
    target_ids: List[int] = []
    for record in user_records:
        try:
            int_id = int(record.get("ID"))
        except (TypeError, ValueError):
            continue
        user_map[int_id] = record
        target_ids.append(int_id)
    processed_users: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    total_events = 0
    for user_id in target_ids:
        try:
            events = client.calendar_event_get(
                type="user",
                ownerId=user_id,
                since=since_iso,
                until=until_iso,
            )
        except Bitrix24Error as exc:
            errors.append({"user_id": user_id, "message": str(exc), "code": exc.b24_code})
            continue
        if isinstance(events, Mapping):
            raw_events = list(events.get("events") or events.get("items") or events.get("result") or [])
        elif isinstance(events, list):
            raw_events = events
        else:
            raw_events = [events]
        summarized = [_summarize_calendar_event(item) for item in raw_events]
        total_events += len(summarized)
        processed_users.append(
            {
                "user_id": user_id,
                "user_name": _compose_user_name(user_map.get(user_id, {})),
                "events_count": len(summarized),
                "events": summarized,
            }
        )
    result_payload: Dict[str, Any] = {
        "status": "ok",
        "date": day_label,
        "timezone": getattr(tz, "key", str(tz)),
        "users_processed": len(processed_users),
        "events_total": total_events,
        "users": processed_users,
    }
    if truncated:
        result_payload["users_limit"] = args.max_users
    if errors:
        result_payload["errors"] = errors
    return result_payload
def fetch_calendar_events_for_users(
    client: Bitrix24Client,
    *,
    date: Optional[str] = None,
    user_ids: Optional[List[int]] = None,
    include_inactive: bool = False,
    timezone: Optional[str] = None,
    max_users: int = 100,
) -> Dict[str, Any]:
    args = CalendarEventsForUsersArgs(
        date=date,
        user_ids=user_ids,
        include_inactive=include_inactive,
        timezone=timezone,
        max_users=max_users,
    )
    return _calendar_events_for_users(client, args)
def _resolve_timezone(value: Optional[str]) -> ZoneInfo:
    if value:
        try:
            return ZoneInfo(value)
        except ZoneInfoNotFoundError:
            logger.warning("Не удалось распознать часовой пояс %s. Используется локальный по умолчанию.", value)
    local_tz = datetime.now().astimezone().tzinfo
    if isinstance(local_tz, ZoneInfo):
        return local_tz
    return ZoneInfo("UTC")
def _normalize_day_bounds(date_value: Optional[str], tz: ZoneInfo) -> tuple[datetime, datetime, str]:
    if date_value:
        try:
            parsed = datetime.fromisoformat(date_value)
        except ValueError:
            parsed = datetime.strptime(date_value, "%Y-%m-%d")
        if parsed.tzinfo is None:
            parsed = datetime(parsed.year, parsed.month, parsed.day, tzinfo=tz)
        else:
            parsed = parsed.astimezone(tz)
    else:
        parsed = datetime.now(tz)
    day = parsed.date()
    start = datetime.combine(day, time(0, 0, 0), tzinfo=tz)
    end = datetime.combine(day, time(23, 59, 59), tzinfo=tz)
    return start, end, day.isoformat()
def _compose_user_name(user: Mapping[str, Any]) -> str:
    parts = [
        str(user.get("NAME") or "").strip(),
        str(user.get("LAST_NAME") or "").strip(),
    ]
    name = " ".join(part for part in parts if part)
    if not name:
        email = str(user.get("EMAIL") or "").strip()
        if email:
            name = email
    if not name:
        name = f"ID {user.get('ID')}"
    return name
def _summarize_calendar_event(event: Any) -> Mapping[str, Any]:
    if not isinstance(event, Mapping):
        return {"value": event}
    keep_keys = {
        "ID",
        "EVENT_TYPE",
        "NAME",
        "DESCRIPTION",
        "DATE_FROM",
        "DATE_TO",
        "DT_SKIP_TIME",
        "TZ_FROM",
        "TZ_TO",
        "OWNER_ID",
        "CREATED_BY",
        "LOCATION",
        "MEETING_STATUS",
        "MEETING_HOST",
        "IS_MEETING",
        "COLOR",
    }
    summary: Dict[str, Any] = {key: event.get(key) for key in keep_keys if key in event}
    attendees = event.get("ATTENDEE_LIST") or event.get("ATTENDEES")
    if attendees:
        summary["attendees"] = attendees
    return summary
def _build_calendar_events_tool(client: Bitrix24Client) -> StructuredTool:
    def _run(**kwargs: Any) -> str:
        try:
            args = CalendarEventsForUsersArgs(**kwargs)
        except ValidationError as exc:
            return json.dumps(
                {
                    "status": "error",
                    "method": "calendar_events_for_users",
                    "message": "Параметры не прошли валидацию.",
                    "details": exc.errors(),
                },
                ensure_ascii=False,
            )
        payload = _calendar_events_for_users(client, args)
        return json.dumps(payload, ensure_ascii=False, default=str)
    return StructuredTool.from_function(
        func=_run,
        name="calendar_events_for_users",
        description=(
            "Получает календарные события за указанный день для выбранных сотрудников или всех активных, если user_ids не заданы."
            " По умолчанию дата берётся как 'сегодня', а часовой пояс определяется параметром 	imezone или локальными настройками портала."
        ),
        args_schema=CalendarEventsForUsersArgs,
        infer_schema=False,
    )
def build_additional_tools(client: Bitrix24Client) -> List[StructuredTool]:
    return [_build_calendar_events_tool(client)]
def _summarize_tool_capabilities(tools: Sequence[StructuredTool]) -> str:
    if not tools:
        return "нет зарегистрированных инструментов"
    lines: List[str] = []
    for tool in tools:
        description = (tool.description or "").strip()
        first_line = description.splitlines()[0].strip() if description else ""
        sanitized = first_line.replace("{", "(").replace("}", ")")
        lines.append(f"- {tool.name}: {sanitized}")
    return "\n".join(lines)
def _get_chat_history(memory: ConversationBufferMemory) -> BaseChatMessageHistory:
    chat_history = getattr(memory, "chat_memory", None)
    if isinstance(chat_history, BaseChatMessageHistory):
        return chat_history
    raise TypeError("Conversation memory must expose `chat_memory` compatible with BaseChatMessageHistory.")
@dataclass
class BitrixAgentConfig:
    llm: BaseChatModel
    client: Bitrix24Client
    ai_name: str = "Bitrix24 ассистент"
    memory: ConversationBufferMemory | None = None
    verbose: bool = False
def build_agent(
    config: BitrixAgentConfig,
    *,
    agent_kwargs: Mapping[str, Any] | None = None,
):
    toolset = BitrixToolset(config.client)
    base_tools = toolset.build_tools()
    extra_tools = build_additional_tools(config.client)
    all_tools = base_tools + extra_tools
    logger.debug("Сформирован набор инструментов: base=%s extra=%s total=%s", len(base_tools), len(extra_tools), len(all_tools))
    if not all_tools:
        raise RuntimeError("В Bitrix24Client не найдено доступных методов для инструментов.")
    capabilities = _summarize_tool_capabilities(all_tools)
    logger.debug("Краткое описание возможностей:\n%s", capabilities)
    memory = config.memory or ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_prompt = DEFAULT_SYSTEM_PROMPT
    if agent_kwargs:
        if "system_message" in agent_kwargs:
            system_prompt = str(agent_kwargs["system_message"])
        elif "system_prompt" in agent_kwargs:
            system_prompt = str(agent_kwargs["system_prompt"])
    if "{capabilities}" in system_prompt:
        system_prompt = system_prompt.replace("{capabilities}", capabilities)
    else:
        system_prompt = f"{system_prompt}\n\nКраткое описание инструментов:\n{capabilities}"
    logger.debug("Итоговый системный промпт:\n%s", system_prompt)
    agent = create_agent(
        model=config.llm,
        tools=all_tools,
        system_prompt=system_prompt,
        debug=config.verbose,
    )
    if memory:
        chat_history = _get_chat_history(memory)
        agent = RunnableWithMessageHistory(
            agent,
            get_session_history=lambda _session_id: chat_history,
            input_messages_key="input",
        )
        logger.debug("К агенту добавлена поддержка истории диалога.")
    return agent
DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    Ты — внимательный ассистент по Bitrix24.
    Основные принципы работы:
    1. Всегда сначала попытайся выполнить запрос пользователя. Если действие понятно из формулировки, немедленно приступай к его выполнению и не проси перечислить действие ещё раз. Уточняющие вопросы допустимы только тогда, когда без них невозможно завершить задачу или данные противоречат друг другу.
    2. Если запрос содержит несколько действий, выполняй их последовательно или чётко объясняй, как пользователь может разделить их на шаги.
    3. Активно используй инструменты Bitrix24: при нехватке данных попробуй получить их сам (например, `user.get` для поиска ID сотрудника, `tasks.task.list` для задач, `crm.*` для сущностей CRM).
    4. Преобразовывай естественные описания параметров в данные для API. Когда пользователь называет имя, компанию или дату, постарайся подобрать нужное значение через доступные методы поиска и только в крайнем случае проси уточнения.
    5. В ответе удерживайся в пределах 6–8 предложений или пунктов, уделяя основное внимание сделанным действиям и найденной информации.
    6. Сообщай, какие шаги ты предпринял и к каким результатам пришёл. Если выполнить запрос не получилось, честно объясни причину и предложи варианты, что сделать дальше.
    7. Если пользователь повторно формулирует задачу после твоего вопроса, считай, что все необходимые данные уже есть, и приступай к выполнению без дополнительных уточнений.
    8. Для запросов вида «какие события/встречи ...» используй инструмент `calendar_events_for_users`: `date` = сегодня (если не указано иначе), `user_ids` — список сотрудников из запроса, `include_inactive` = `False`, `timezone` — стандартный для портала. Затем коротко опиши найденные события.
    9. Не отвечай шаблоном «Что нужно сделать в Bitrix24?». Вместо этого пытайся выполнить запрос, а если это невозможно — сразу поясняй причину.
    Список доступных действий:
    {capabilities}
    """
).strip()
