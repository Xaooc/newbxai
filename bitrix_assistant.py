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
            logger.debug("Р“РѕС‚РѕРІР»СЋ РёРЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ РјРµС‚РѕРґР° Bitrix24: %s", spec.name)
            tool = self._build_tool(spec)
            if tool is not None:
                logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s СѓСЃРїРµС€РЅРѕ СЃРѕР·РґР°РЅ.", spec.name)
                tools.append(tool)
            else:
                logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ РґР»СЏ РјРµС‚РѕРґР° %s РѕРїСѓС‰РµРЅ РёР·-Р·Р° РЅРµРїРѕРґРґРµСЂР¶РёРІР°РµРјРѕР№ СЃРёРіРЅР°С‚СѓСЂС‹.", spec.name)
        return tools
    def _build_tool(self, spec: BitrixMethodSpec) -> StructuredTool | None:
        args_model = self._build_args_model(spec)
        if args_model is None:
            logger.debug("Skip method %s: unsupported signature.", spec.name)
            return None
        method = getattr(self._client, spec.name)
        description = self._compose_description(spec, args_model)
        logger.debug("РЎРѕР·РґР°СЋ StructuredTool РґР»СЏ РјРµС‚РѕРґР° %s.", spec.name)
        def _invoke(**kwargs: Any) -> str:
            logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s РїРѕР»СѓС‡РёР» РІС…РѕРґ: %s", spec.name, kwargs)
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
                logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s Р·Р°РІРµСЂС€РёР»СЃСЏ РѕС€РёР±РєРѕР№ РІР°Р»РёРґР°С†РёРё: %s", spec.name, error_payload)
                return error_payload
            params = parsed.model_dump(mode="json", exclude_none=True)
            logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s РІС‹Р·С‹РІР°РµС‚ Bitrix24 СЃ РїР°СЂР°РјРµС‚СЂР°РјРё: %s", spec.name, params)
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
                logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s РІРµСЂРЅСѓР» РѕС€РёР±РєСѓ Bitrix24: %s", spec.name, error_payload)
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
                logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s Р·Р°РІРµСЂС€РёР»СЃСЏ РЅРµРѕР¶РёРґР°РЅРЅРѕР№ РѕС€РёР±РєРѕР№: %s", spec.name, error_payload)
                return error_payload
            success_payload = self._format_response(
                status="ok",
                method=spec.name,
                params=params,
                payload={"result": result},
            )
            logger.debug("РРЅСЃС‚СЂСѓРјРµРЅС‚ %s СѓСЃРїРµС€РЅРѕ Р·Р°РІРµСЂС€С‘РЅ. Payload: %s", spec.name, success_payload)
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
        lines = [spec.summary, "", "РџР°СЂР°РјРµС‚СЂС‹:"]
        for field_name, field in args_model.model_fields.items():
            requirement = "РѕР±СЏР·Р°С‚РµР»СЊРЅС‹Р№" if field.is_required() else f"РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ: {field.default!r}"
            detail = field.description or ""
            lines.append(f"- {field_name}: {detail} ({requirement})")
        lines.append("Р’РѕР·РІСЂР°С‰Р°РµС‚ JSON-РѕС‚РІРµС‚ СЃРѕ СЃС‚Р°С‚СѓСЃРѕРј Рё РґР°РЅРЅС‹РјРё Bitrix24.")
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
            suggestions.append("РџСЂРѕРІРµСЂСЊС‚Рµ РїСЂР°РІР° РґРѕСЃС‚СѓРїР° Рё РєРѕСЂСЂРµРєС‚РЅРѕСЃС‚СЊ OAuth-С‚РѕРєРµРЅР° Bitrix24.")
        if http_status == 404 or "not found" in normalized_message:
            suggestions.append("РЈР±РµРґРёС‚РµСЃСЊ, С‡С‚Рѕ СѓРєР°Р·Р°РЅРЅС‹Рµ РёРґРµРЅС‚РёС„РёРєР°С‚РѕСЂС‹ СЃСѓС‰РµСЃС‚РІСѓСЋС‚ РІ Bitrix24.")
        if http_status in {408, 504} or "timeout" in normalized_message:
            suggestions.append("РџРѕРІС‚РѕСЂРёС‚Рµ Р·Р°РїСЂРѕСЃ РїРѕР·Р¶Рµ РёР»Рё СѓРјРµРЅСЊС€РёС‚Рµ РѕР±СЉС‘Рј РґР°РЅРЅС‹С…, С‡С‚РѕР±С‹ РёР·Р±РµР¶Р°С‚СЊ С‚Р°Р№РјР°СѓС‚Р°.")
        if "required parameter missing" in normalized_message or "missing parameter" in normalized_message:
            suggestions.append("Р—Р°РїРѕР»РЅРёС‚Рµ РѕР±СЏР·Р°С‚РµР»СЊРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹ РјРµС‚РѕРґР° Bitrix24.")
        if "limit" in normalized_message or "too many requests" in normalized_message:
            suggestions.append("РЎРЅРёР·СЊС‚Рµ С‡Р°СЃС‚РѕС‚Сѓ Р·Р°РїСЂРѕСЃРѕРІ РёР»Рё РёСЃРїРѕР»СЊР·СѓР№С‚Рµ РїР°РєРµС‚РЅС‹Рµ РѕРїРµСЂР°С†РёРё Bitrix24.")
        if not suggestions:
            suggestions.append("РР·СѓС‡РёС‚Рµ РѕС‚РІРµС‚ Bitrix24, РёСЃРїСЂР°РІСЊС‚Рµ РїСЂРёС‡РёРЅСѓ РѕС€РёР±РєРё Рё РїРѕРІС‚РѕСЂРёС‚Рµ Р·Р°РїСЂРѕСЃ.")
        if params:
            suggestions.append(f"РџР°СЂР°РјРµС‚СЂС‹ Р·Р°РїСЂРѕСЃР°: {dict(params)}")
        logger.debug(
            "Р РµРєРѕРјРµРЅРґР°С†РёРё РґР»СЏ Bitrix24 method=%s status=%s code=%s: %s",
            method_name,
            http_status,
            b24_code,
            suggestions,
        )
        return suggestions
class CalendarEventsForUsersArgs(BaseModel):
    date: Optional[str] = Field(
        None,
        description="Р”Р°С‚Р° РІ С„РѕСЂРјР°С‚Рµ YYYY-MM-DD РёР»Рё ISO. Р•СЃР»Рё РЅРµ СѓРєР°Р·Р°РЅР°, Р±РµСЂС‘С‚СЃСЏ С‚РµРєСѓС‰РёР№ РґРµРЅСЊ.",
    )
    user_ids: Optional[List[int]] = Field(
        None,
        description="РЎРїРёСЃРѕРє ID СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ Bitrix24. Р•СЃР»Рё РЅРµ СѓРєР°Р·Р°РЅ, РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ СЃРїРёСЃРѕРє Р°РєС‚РёРІРЅС‹С… СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ.",
    )
    include_inactive: bool = Field(
        False,
        description="Р’РєР»СЋС‡Р°С‚СЊ Р»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№ СЃ ACTIVE='N' (РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ С‚РѕР»СЊРєРѕ РµСЃР»Рё user_ids РЅРµ Р·Р°РґР°РЅ).",
    )
    timezone: Optional[str] = Field(
        None,
        description="Р§Р°СЃРѕРІРѕР№ РїРѕСЏСЃ, РЅР°РїСЂРёРјРµСЂ 'Europe/Moscow'. Р•СЃР»Рё РЅРµ СѓРєР°Р·Р°РЅ, РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ Р»РѕРєР°Р»СЊРЅС‹Р№.",
    )
    max_users: int = Field(
        100,
        ge=1,
        le=500,
        description="РњР°РєСЃРёРјР°Р»СЊРЅРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№, РґР»СЏ РєРѕС‚РѕСЂС‹С… РІС‹РіСЂСѓР¶Р°СЋС‚СЃСЏ СЃРѕР±С‹С‚РёСЏ.",
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
                "message": "РџР°СЂР°РјРµС‚СЂ user_ids РЅРµ СЃРѕРґРµСЂР¶РёС‚ РґРѕРїСѓСЃС‚РёРјС‹С… Р·РЅР°С‡РµРЅРёР№.",
            }
        try:
            filter_payload: Mapping[str, Any] = {"ID": [str(uid) for uid in ids]}
            user_records = client.user_get(
                filter=filter_payload,
                select=["ID", "NAME", "LAST_NAME", "EMAIL", "ACTIVE"],
                fetch_all=False,
            )
        except Bitrix24Error as exc:
            logger.warning("РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РіСЂСѓР·РёС‚СЊ РєР°СЂС‚РѕС‡РєРё РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№ %s: %s", ids, exc)
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
                "message": f"РќРµ СѓРґР°Р»РѕСЃСЊ РїРѕР»СѓС‡РёС‚СЊ СЃРїРёСЃРѕРє СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ: {exc}",
            }
    if not user_records:
        return {
            "status": "ok",
            "date": day_label,
            "timezone": getattr(tz, "key", str(tz)),
            "users_processed": 0,
            "events_total": 0,
            "users": [],
            "notice": "РђРєС‚РёРІРЅС‹С… СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ РЅРµ РЅР°Р№РґРµРЅРѕ.",
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
            logger.warning("РќРµ СѓРґР°Р»РѕСЃСЊ СЂР°СЃРїРѕР·РЅР°С‚СЊ С‡Р°СЃРѕРІРѕР№ РїРѕСЏСЃ %s. РСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ Р»РѕРєР°Р»СЊРЅС‹Р№ РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ.", value)
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
                    "message": "РџР°СЂР°РјРµС‚СЂС‹ РЅРµ РїСЂРѕС€Р»Рё РІР°Р»РёРґР°С†РёСЋ.",
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
            "РџРѕР»СѓС‡Р°РµС‚ РєР°Р»РµРЅРґР°СЂРЅС‹Рµ СЃРѕР±С‹С‚РёСЏ Р·Р° СѓРєР°Р·Р°РЅРЅС‹Р№ РґРµРЅСЊ РґР»СЏ РІС‹Р±СЂР°РЅРЅС‹С… СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ РёР»Рё РІСЃРµС… Р°РєС‚РёРІРЅС‹С…, РµСЃР»Рё user_ids РЅРµ Р·Р°РґР°РЅС‹."
            " РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РґР°С‚Р° Р±РµСЂС‘С‚СЃСЏ РєР°Рє 'СЃРµРіРѕРґРЅСЏ', Р° С‡Р°СЃРѕРІРѕР№ РїРѕСЏСЃ РѕРїСЂРµРґРµР»СЏРµС‚СЃСЏ РїР°СЂР°РјРµС‚СЂРѕРј 	imezone РёР»Рё Р»РѕРєР°Р»СЊРЅС‹РјРё РЅР°СЃС‚СЂРѕР№РєР°РјРё РїРѕСЂС‚Р°Р»Р°."
        ),
        args_schema=CalendarEventsForUsersArgs,
        infer_schema=False,
    )

@dataclass
class ToolPreparation:
    """Container with prepared StructuredTool objects and summary metadata."""
    tools: List[StructuredTool]
    capabilities: str
    base_count: int
    extra_count: int
def build_additional_tools(client: Bitrix24Client) -> List[StructuredTool]:
    return [_build_calendar_events_tool(client)]
def prepare_tools(client: Bitrix24Client) -> ToolPreparation:
    """
    Build the complete set of StructuredTool objects for the provided client and
    return both the tools and a short, human-readable summary of their
    capabilities.
    """
    toolset = BitrixToolset(client)
    base_tools = toolset.build_tools()
    extra_tools = build_additional_tools(client)
    all_tools = base_tools + extra_tools
    total_tools = len(all_tools)
    logger.debug(
        "Collected Bitrix24 toolset: base=%s extra=%s total=%s",
        len(base_tools),
        len(extra_tools),
        total_tools,
    )
    if not all_tools:
        raise RuntimeError("Bitrix24Client does not expose any public methods for tools.")
    capabilities = _summarize_tool_capabilities(all_tools)
    logger.debug("Tool capabilities summary:\n%s", capabilities)
    return ToolPreparation(
        tools=all_tools,
        capabilities=capabilities,
        base_count=len(base_tools),
        extra_count=len(extra_tools),
    )
def _summarize_tool_capabilities(tools: Sequence[StructuredTool]) -> str:
    if not tools:
        return "РЅРµС‚ Р·Р°СЂРµРіРёСЃС‚СЂРёСЂРѕРІР°РЅРЅС‹С… РёРЅСЃС‚СЂСѓРјРµРЅС‚РѕРІ"
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
    ai_name: str = "Bitrix24 Р°СЃСЃРёСЃС‚РµРЅС‚"
    memory: ConversationBufferMemory | None = None
    verbose: bool = False
def build_agent(
    config: BitrixAgentConfig,
    *,
    agent_kwargs: Mapping[str, Any] | None = None,
    preparation: ToolPreparation | None = None,
):
    prep = preparation or prepare_tools(config.client)
    all_tools = list(prep.tools)
    capabilities = prep.capabilities
    logger.debug(
        "Сформирован набор инструментов: base=%s extra=%s total=%s",
        prep.base_count,
        prep.extra_count,
        len(all_tools),
    )
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
        system_prompt = f"{system_prompt}\n\nРљСЂР°С‚РєРѕРµ РѕРїРёСЃР°РЅРёРµ РёРЅСЃС‚СЂСѓРјРµРЅС‚РѕРІ:\n{capabilities}"
    logger.debug("РС‚РѕРіРѕРІС‹Р№ СЃРёСЃС‚РµРјРЅС‹Р№ РїСЂРѕРјРїС‚:\n%s", system_prompt)
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
        logger.debug("Рљ Р°РіРµРЅС‚Сѓ РґРѕР±Р°РІР»РµРЅР° РїРѕРґРґРµСЂР¶РєР° РёСЃС‚РѕСЂРёРё РґРёР°Р»РѕРіР°.")
    return agent
DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    РўС‹ вЂ” РІРЅРёРјР°С‚РµР»СЊРЅС‹Р№ Р°СЃСЃРёСЃС‚РµРЅС‚ РїРѕ Bitrix24.
    РћСЃРЅРѕРІРЅС‹Рµ РїСЂРёРЅС†РёРїС‹ СЂР°Р±РѕС‚С‹:
    1. Р’СЃРµРіРґР° СЃРЅР°С‡Р°Р»Р° РїРѕРїС‹С‚Р°Р№СЃСЏ РІС‹РїРѕР»РЅРёС‚СЊ Р·Р°РїСЂРѕСЃ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ. Р•СЃР»Рё РґРµР№СЃС‚РІРёРµ РїРѕРЅСЏС‚РЅРѕ РёР· С„РѕСЂРјСѓР»РёСЂРѕРІРєРё, РЅРµРјРµРґР»РµРЅРЅРѕ РїСЂРёСЃС‚СѓРїР°Р№ Рє РµРіРѕ РІС‹РїРѕР»РЅРµРЅРёСЋ Рё РЅРµ РїСЂРѕСЃРё РїРµСЂРµС‡РёСЃР»РёС‚СЊ РґРµР№СЃС‚РІРёРµ РµС‰С‘ СЂР°Р·. РЈС‚РѕС‡РЅСЏСЋС‰РёРµ РІРѕРїСЂРѕСЃС‹ РґРѕРїСѓСЃС‚РёРјС‹ С‚РѕР»СЊРєРѕ С‚РѕРіРґР°, РєРѕРіРґР° Р±РµР· РЅРёС… РЅРµРІРѕР·РјРѕР¶РЅРѕ Р·Р°РІРµСЂС€РёС‚СЊ Р·Р°РґР°С‡Сѓ РёР»Рё РґР°РЅРЅС‹Рµ РїСЂРѕС‚РёРІРѕСЂРµС‡Р°С‚ РґСЂСѓРі РґСЂСѓРіСѓ.
    2. Р•СЃР»Рё Р·Р°РїСЂРѕСЃ СЃРѕРґРµСЂР¶РёС‚ РЅРµСЃРєРѕР»СЊРєРѕ РґРµР№СЃС‚РІРёР№, РІС‹РїРѕР»РЅСЏР№ РёС… РїРѕСЃР»РµРґРѕРІР°С‚РµР»СЊРЅРѕ РёР»Рё С‡С‘С‚РєРѕ РѕР±СЉСЏСЃРЅСЏР№, РєР°Рє РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РјРѕР¶РµС‚ СЂР°Р·РґРµР»РёС‚СЊ РёС… РЅР° С€Р°РіРё.
    3. РђРєС‚РёРІРЅРѕ РёСЃРїРѕР»СЊР·СѓР№ РёРЅСЃС‚СЂСѓРјРµРЅС‚С‹ Bitrix24: РїСЂРё РЅРµС…РІР°С‚РєРµ РґР°РЅРЅС‹С… РїРѕРїСЂРѕР±СѓР№ РїРѕР»СѓС‡РёС‚СЊ РёС… СЃР°Рј (РЅР°РїСЂРёРјРµСЂ, `user.get` РґР»СЏ РїРѕРёСЃРєР° ID СЃРѕС‚СЂСѓРґРЅРёРєР°, `tasks.task.list` РґР»СЏ Р·Р°РґР°С‡, `crm.*` РґР»СЏ СЃСѓС‰РЅРѕСЃС‚РµР№ CRM).
    4. РџСЂРµРѕР±СЂР°Р·РѕРІС‹РІР°Р№ РµСЃС‚РµСЃС‚РІРµРЅРЅС‹Рµ РѕРїРёСЃР°РЅРёСЏ РїР°СЂР°РјРµС‚СЂРѕРІ РІ РґР°РЅРЅС‹Рµ РґР»СЏ API. РљРѕРіРґР° РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РЅР°Р·С‹РІР°РµС‚ РёРјСЏ, РєРѕРјРїР°РЅРёСЋ РёР»Рё РґР°С‚Сѓ, РїРѕСЃС‚Р°СЂР°Р№СЃСЏ РїРѕРґРѕР±СЂР°С‚СЊ РЅСѓР¶РЅРѕРµ Р·РЅР°С‡РµРЅРёРµ С‡РµСЂРµР· РґРѕСЃС‚СѓРїРЅС‹Рµ РјРµС‚РѕРґС‹ РїРѕРёСЃРєР° Рё С‚РѕР»СЊРєРѕ РІ РєСЂР°Р№РЅРµРј СЃР»СѓС‡Р°Рµ РїСЂРѕСЃРё СѓС‚РѕС‡РЅРµРЅРёСЏ.
    5. Р’ РѕС‚РІРµС‚Рµ СѓРґРµСЂР¶РёРІР°Р№СЃСЏ РІ РїСЂРµРґРµР»Р°С… 6вЂ“8 РїСЂРµРґР»РѕР¶РµРЅРёР№ РёР»Рё РїСѓРЅРєС‚РѕРІ, СѓРґРµР»СЏСЏ РѕСЃРЅРѕРІРЅРѕРµ РІРЅРёРјР°РЅРёРµ СЃРґРµР»Р°РЅРЅС‹Рј РґРµР№СЃС‚РІРёСЏРј Рё РЅР°Р№РґРµРЅРЅРѕР№ РёРЅС„РѕСЂРјР°С†РёРё.
    6. РЎРѕРѕР±С‰Р°Р№, РєР°РєРёРµ С€Р°РіРё С‚С‹ РїСЂРµРґРїСЂРёРЅСЏР» Рё Рє РєР°РєРёРј СЂРµР·СѓР»СЊС‚Р°С‚Р°Рј РїСЂРёС€С‘Р». Р•СЃР»Рё РІС‹РїРѕР»РЅРёС‚СЊ Р·Р°РїСЂРѕСЃ РЅРµ РїРѕР»СѓС‡РёР»РѕСЃСЊ, С‡РµСЃС‚РЅРѕ РѕР±СЉСЏСЃРЅРё РїСЂРёС‡РёРЅСѓ Рё РїСЂРµРґР»РѕР¶Рё РІР°СЂРёР°РЅС‚С‹, С‡С‚Рѕ СЃРґРµР»Р°С‚СЊ РґР°Р»СЊС€Рµ.
    7. Р•СЃР»Рё РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РїРѕРІС‚РѕСЂРЅРѕ С„РѕСЂРјСѓР»РёСЂСѓРµС‚ Р·Р°РґР°С‡Сѓ РїРѕСЃР»Рµ С‚РІРѕРµРіРѕ РІРѕРїСЂРѕСЃР°, СЃС‡РёС‚Р°Р№, С‡С‚Рѕ РІСЃРµ РЅРµРѕР±С…РѕРґРёРјС‹Рµ РґР°РЅРЅС‹Рµ СѓР¶Рµ РµСЃС‚СЊ, Рё РїСЂРёСЃС‚СѓРїР°Р№ Рє РІС‹РїРѕР»РЅРµРЅРёСЋ Р±РµР· РґРѕРїРѕР»РЅРёС‚РµР»СЊРЅС‹С… СѓС‚РѕС‡РЅРµРЅРёР№.
    8. Р”Р»СЏ Р·Р°РїСЂРѕСЃРѕРІ РІРёРґР° В«РєР°РєРёРµ СЃРѕР±С‹С‚РёСЏ/РІСЃС‚СЂРµС‡Рё ...В» РёСЃРїРѕР»СЊР·СѓР№ РёРЅСЃС‚СЂСѓРјРµРЅС‚ `calendar_events_for_users`: `date` = СЃРµРіРѕРґРЅСЏ (РµСЃР»Рё РЅРµ СѓРєР°Р·Р°РЅРѕ РёРЅР°С‡Рµ), `user_ids` вЂ” СЃРїРёСЃРѕРє СЃРѕС‚СЂСѓРґРЅРёРєРѕРІ РёР· Р·Р°РїСЂРѕСЃР°, `include_inactive` = `False`, `timezone` вЂ” СЃС‚Р°РЅРґР°СЂС‚РЅС‹Р№ РґР»СЏ РїРѕСЂС‚Р°Р»Р°. Р—Р°С‚РµРј РєРѕСЂРѕС‚РєРѕ РѕРїРёС€Рё РЅР°Р№РґРµРЅРЅС‹Рµ СЃРѕР±С‹С‚РёСЏ.
    9. РќРµ РѕС‚РІРµС‡Р°Р№ С€Р°Р±Р»РѕРЅРѕРј В«Р§С‚Рѕ РЅСѓР¶РЅРѕ СЃРґРµР»Р°С‚СЊ РІ Bitrix24?В». Р’РјРµСЃС‚Рѕ СЌС‚РѕРіРѕ РїС‹С‚Р°Р№СЃСЏ РІС‹РїРѕР»РЅРёС‚СЊ Р·Р°РїСЂРѕСЃ, Р° РµСЃР»Рё СЌС‚Рѕ РЅРµРІРѕР·РјРѕР¶РЅРѕ вЂ” СЃСЂР°Р·Сѓ РїРѕСЏСЃРЅСЏР№ РїСЂРёС‡РёРЅСѓ.
    РЎРїРёСЃРѕРє РґРѕСЃС‚СѓРїРЅС‹С… РґРµР№СЃС‚РІРёР№:
    {capabilities}
    """
).strip()
