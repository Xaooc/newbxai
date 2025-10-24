"""Оркестрация LangGraph для Telegram-ассистента."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Literal, Mapping

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ...logging import bind_event, get_logger
from ..prompts import load_error_tips, load_system_prompt
from ..tools.bitrix import BitrixGateway
from ..tools.formatter import Formatter
from ..tools.memory import MemoryStore
from .llm import ActionPlan, OpenAIPlanner, PlanResponse, PlannerToolContext


class AgentState(TypedDict, total=False):
    """Описание состояния графа."""

    chat_id: str
    user_message: str
    history: List[HumanMessage | AIMessage]
    plan: List[str]
    actions: List[ActionPlan]
    action_index: int
    action_results: List[Mapping[str, Any]]
    next_action: bool
    selected_method: str
    entity: str
    params: Dict[str, Any]
    missing_fields: List[str]
    missing_details: List[Dict[str, str]]
    call_result: Mapping[str, Any]
    error: Mapping[str, Any]
    response: str
    status: Literal["clarify", "error", "success"]
    logger: Any
    memory_snapshot: Mapping[str, Any]
    pending_memory_set: Mapping[str, Any]


class AssistantOrchestrator:
    """Организует шаги ассистента через LangGraph."""

    def __init__(
        self,
        *,
        planner: OpenAIPlanner,
        gateway: BitrixGateway,
        formatter: Formatter,
        memory: MemoryStore,
    ) -> None:
        self._planner = planner
        self._gateway = gateway
        self._formatter = formatter
        self._memory = memory
        self._system_prompt = load_system_prompt()
        self._error_tips = load_error_tips()
        self._logger = get_logger(__name__)
        self._locks: Dict[str, asyncio.Semaphore] = {}
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("ingest", self._node_ingest)
        graph.add_node("plan", self._node_plan)
        graph.add_node("ensure_fields", self._node_ensure_fields)
        graph.add_node("clarify", self._node_ask_for_fields)
        graph.add_node("act", self._node_act)
        graph.add_node("handle_error", self._node_handle_error)
        graph.add_node("finalize", self._node_finalize)
        graph.set_entry_point("ingest")
        graph.add_edge("ingest", "plan")
        graph.add_edge("plan", "ensure_fields")
        graph.add_conditional_edges(
            "ensure_fields",
            self._ensure_router,
            {"ask": "clarify", "call": "act"},
        )
        graph.add_edge("clarify", "finalize")
        graph.add_conditional_edges(
            "act",
            self._act_router,
            {"error": "handle_error", "next": "ensure_fields", "ok": "finalize"},
        )
        graph.add_edge("handle_error", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    async def handle(self, chat_id: str, message: str) -> str:
        """Основная точка входа: возвращает текст ответа."""

        lock = self._locks.setdefault(chat_id, asyncio.Semaphore(1))
        async with lock:
            logger = bind_event(self._logger, chat_id=chat_id)
            logger.info("orchestrator.handle.start", message=message)
            state: AgentState = {
                "chat_id": chat_id,
                "user_message": message,
                "logger": logger,
            }
            result: AgentState = await self._graph.ainvoke(state)
            response = result.get("response", "Готово.")
            await self._memory.add_message(chat_id, "user", message)
            await self._memory.add_message(chat_id, "assistant", response)
            logger.info("orchestrator.handle.finish", status=result.get("status"), response=response)
            return response

    async def _node_ingest(self, state: AgentState) -> AgentState:
        """Собирает историю и подготавливает контекст."""

        logger = state.get("logger") or self._logger
        history_records = await self._memory.get_history(state["chat_id"])
        history: List[HumanMessage | AIMessage] = []
        for record in history_records:
            if record["role"] == "user":
                history.append(HumanMessage(content=record["content"]))
            else:
                history.append(AIMessage(content=record["content"]))
        state["history"] = history
        logger.info("node.ingest", history_len=len(history))
        return state

    async def _node_plan(self, state: AgentState) -> AgentState:
        """Запрашивает у модели план и параметры."""

        logger = state.get("logger") or self._logger
        candidates = self._gateway.search_methods(state["user_message"])["candidates"]
        history = state.get("history", [])
        tool_context = PlannerToolContext(
            gateway=self._gateway,
            memory=self._memory,
            formatter=self._formatter,
            chat_id=state["chat_id"],
        )
        plan: PlanResponse = await self._planner.plan(
            system_prompt=self._system_prompt,
            user_message=state["user_message"],
            history=history,
            candidates=candidates,
            tool_context=tool_context,
        )
        if plan.memory_get:
            memory_values = await self._memory.tool_get(state["chat_id"], plan.memory_get)
            state["memory_snapshot"] = memory_values
            logger.info("node.plan.memory.get", keys=plan.memory_get, found=list(memory_values.keys()))
            if memory_values:
                augmented_history = list(history)
                augmented_history.append(
                    AIMessage(content=f"memory.get -> {memory_values}")
                )
                plan = await self._planner.plan(
                    system_prompt=self._system_prompt,
                    user_message=state["user_message"],
                    history=augmented_history,
                    candidates=candidates,
                    tool_context=tool_context,
                )
        state["plan"] = plan.plan
        state["actions"] = plan.actions
        state["action_index"] = 0
        state["action_results"] = []
        state["next_action"] = False
        current_action = plan.actions[0] if plan.actions else ActionPlan()
        state["selected_method"] = current_action.method
        state["entity"] = current_action.entity
        state["params"] = dict(current_action.params)
        state["missing_fields"] = list(current_action.missing_fields)
        state["pending_memory_set"] = plan.memory_set
        if plan.memory_set:
            logger.info("node.plan.memory.set", fields=list(plan.memory_set.keys()))
        logger.info(
            "node.plan",
            method=plan.method,
            entity=plan.entity,
            missing=len(plan.missing_fields),
        )
        return state

    async def _node_ensure_fields(self, state: AgentState) -> AgentState:
        """Сверяет параметры с JSON-схемой и собирает недостающие поля."""

        if not state.get("selected_method"):
            state["error"] = {"code": "UNKNOWN", "message": "Модель не выбрала метод."}
            return state
        logger = state.get("logger") or self._logger
        schema = self._gateway.schema(state["selected_method"])
        params = state.get("params") or {}
        required_schema = schema.get("input_schema", {})
        missing_details = self._collect_missing_fields(params, required_schema)
        plan_missing = [field for field in state.get("missing_fields", []) if field]
        known_paths = {item["path"] for item in missing_details}
        for field in plan_missing:
            if field not in known_paths:
                missing_details.append({"path": field, "display": field})
        state["missing_fields"] = [item["path"] for item in missing_details]
        state["missing_details"] = missing_details
        logger.info("node.ensure_fields", missing=len(missing_details))
        return state

    def _ensure_router(self, state: AgentState) -> Literal["ask", "call"]:
        """Решает, нужно ли уточнять недостающие поля."""

        if state.get("missing_fields"):
            return "ask"
        return "call"

    async def _node_ask_for_fields(self, state: AgentState) -> AgentState:
        """Формирует запрос на уточнение недостающих данных."""

        details = state.get("missing_details", [])
        logger = state.get("logger") or self._logger
        if not details:
            state["response"] = "Не хватает данных. Уточните параметры."
            state["status"] = "clarify"
            logger.warning("node.clarify.empty")
            return state
        names = [item.get("display") or item["path"] for item in details]
        questions: List[str] = []
        for item in details[:3]:
            display = item.get("display") or item["path"]
            description = item.get("description")
            example = item.get("example")
            question = display
            if description:
                question += f" ({description})"
            question += "?"
            if example:
                question += f" Например: {example}."
            questions.append(question)
        missing_text = ", ".join(names)
        questions_text = "; ".join(questions)
        response = f"Не хватает: {missing_text}. Уточните: {questions_text}"
        state["response"] = response
        state["status"] = "clarify"
        logger.info("node.clarify.ask", text=response)
        return state

    async def _node_act(self, state: AgentState) -> AgentState:
        """Вызывает метод Bitrix24."""

        logger = state.get("logger") or self._logger
        if not state.get("selected_method"):
            state["error"] = {"code": "UNKNOWN", "message": "Метод не определён."}
            return state
        params = state.get("params") or {}
        attempts = 0
        backoff = 1.0
        while attempts < 3:
            attempts += 1
            logger.info("node.act.call", method=state["selected_method"], attempt=attempts)
            call_result = await asyncio.to_thread(self._gateway.call, state["selected_method"], params)
            state["call_result"] = call_result
            if call_result.get("ok", False):
                state.pop("error", None)
                logger.info("node.act.success", attempt=attempts)
                results = state.setdefault("action_results", [])
                current_action = self._current_action(state)
                results.append(
                    {
                        "method": state.get("selected_method"),
                        "entity": (current_action.entity if current_action else state.get("entity")),
                        "data": call_result.get("data"),
                    }
                )
                self._prepare_next_action(state)
                return state
            error_payload = call_result.get("error", {"code": "UNKNOWN", "message": "Неизвестная ошибка"})
            state["error"] = error_payload
            status = call_result.get("status")
            retry_after = self._extract_retry_after(call_result)
            code = str(error_payload.get("code") or "").upper()
            should_retry = status in {429, 503} or code in {"RATE_LIMIT", "NETWORK"}
            if should_retry and attempts < 3:
                delay = retry_after or backoff
                logger.warning(
                    "node.act.retry",
                    attempt=attempts,
                    delay=delay,
                    status=status,
                    code=code,
                )
                await asyncio.sleep(max(float(delay), 0))
                backoff = min(backoff * 2, 8.0)
                continue
            logger.error("node.act.failed", status=status, code=code)
            break
        return state

    def _act_router(self, state: AgentState) -> Literal["error", "ok"]:
        """Определяет, был ли вызов успешным."""

        call_result = state.get("call_result") or {}
        if call_result.get("ok"):
            if state.get("next_action"):
                return "next"
            return "ok"
        return "error"

    async def _node_handle_error(self, state: AgentState) -> AgentState:
        """Преобразует ошибку Bitrix24 в понятный ответ."""

        logger = state.get("logger") or self._logger
        error: Mapping[str, Any] = state.get("error") or {}
        call_result = state.get("call_result") or {}
        status = call_result.get("status") if isinstance(call_result, Mapping) else None
        code = str(error.get("code") or "").upper()
        message = str(error.get("message") or "Ошибка выполнения")
        details = error.get("details") if isinstance(error, Mapping) else None
        normalized_code = self._normalize_error_code(code, status)
        tips = self._tips_for_code(normalized_code, details)
        tip_text = "; ".join(tips[:3])
        state["response"] = f"Ошибка: {message}. Что сделать: {tip_text}."
        state["status"] = "error"
        logger.error("node.handle_error", code=normalized_code, message=message)
        return state

    async def _node_finalize(self, state: AgentState) -> AgentState:
        """Строит финальный ответ для пользователя."""

        logger = state.get("logger") or self._logger
        if state.get("response"):
            if state.get("pending_memory_set") and state.get("status") == "clarify":
                logger.info("node.finalize.skip_memory", reason="clarify")
            return state
        call_result = state.get("call_result", {})
        data = call_result.get("data") if isinstance(call_result, Mapping) else {}
        entity = state.get("entity", "объект")
        tool_context = PlannerToolContext(
            gateway=self._gateway,
            memory=self._memory,
            formatter=self._formatter,
            chat_id=state["chat_id"],
        )
        formatted = await tool_context.execute(
            "formatter.humanize", {"entity": entity, "data": data, "locale": "ru"}
        )
        result_text = formatted.get("result") or "Готово"
        response_parts = [f"Готово: {result_text}"]
        details = formatted.get("details")
        if isinstance(details, Iterable):
            detail_text = "; ".join(str(item) for item in details if item)
            if detail_text:
                response_parts.append(f"Детали: {detail_text}")
        if state.get("plan"):
            plan_text = "; ".join(item for item in state["plan"][:3] if item)
            if plan_text:
                response_parts.append(f"План: {plan_text}")
        next_step = formatted.get("next_step")
        if next_step:
            response_parts.append(f"Следующий шаг: {next_step}")
        state["response"] = ". ".join(response_parts).strip()
        state["status"] = "success"
        pending_memory = state.get("pending_memory_set") or {}
        if pending_memory:
            await self._memory.tool_set(state["chat_id"], pending_memory)
            logger.info("node.finalize.memory.set", fields=list(pending_memory.keys()))
        return state

    def _current_action(self, state: AgentState) -> ActionPlan | None:
        """Возвращает активный шаг плана."""

        actions = state.get("actions") or []
        index = state.get("action_index", 0)
        if 0 <= index < len(actions):
            return actions[index]
        return None

    def _prepare_next_action(self, state: AgentState) -> None:
        """Готовит данные следующего шага, если он есть."""

        actions = state.get("actions") or []
        index = state.get("action_index", 0)
        if index + 1 < len(actions):
            next_action = actions[index + 1]
            state["action_index"] = index + 1
            state["selected_method"] = next_action.method
            state["entity"] = next_action.entity
            state["params"] = dict(next_action.params)
            state["missing_fields"] = list(next_action.missing_fields)
            state["next_action"] = True
        else:
            state["next_action"] = False

    def _collect_missing_fields(
        self,
        params: Mapping[str, Any],
        schema: Mapping[str, Any],
        prefix: str = "",
    ) -> List[Dict[str, str]]:
        """Находит обязательные поля, отсутствующие в params."""

        missing: List[Dict[str, str]] = []
        required = schema.get("required", []) if isinstance(schema, Mapping) else []
        properties: Mapping[str, Any] = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
        for field in required:
            full_name = f"{prefix}.{field}" if prefix else str(field)
            property_schema = properties.get(field, {}) if isinstance(properties, Mapping) else {}
            value = params.get(field) if isinstance(params, Mapping) else None
            if value in (None, "", [], {}):
                missing.append(self._build_field_detail(full_name, property_schema))
                if isinstance(property_schema, Mapping) and property_schema.get("type") == "object":
                    child_params = value if isinstance(value, Mapping) else {}
                    missing.extend(self._collect_missing_fields(child_params, property_schema, full_name))
                continue
            if isinstance(property_schema, Mapping) and property_schema.get("type") == "object":
                child_params = value if isinstance(value, Mapping) else {}
                missing.extend(self._collect_missing_fields(child_params, property_schema, full_name))
        return missing

    @staticmethod
    def _build_field_detail(path: str, schema: Mapping[str, Any]) -> Dict[str, str]:
        """Формирует описание поля для вопроса."""

        title = schema.get("title") if isinstance(schema, Mapping) else None
        description = schema.get("description") if isinstance(schema, Mapping) else None
        example = None
        if isinstance(schema, Mapping):
            if "example" in schema:
                example = schema.get("example")
            elif "examples" in schema and isinstance(schema["examples"], list) and schema["examples"]:
                example = schema["examples"][0]
        display = f"{title} ({path})" if title else path
        detail: Dict[str, str] = {"path": path, "display": display}
        if description:
            detail["description"] = description
        if example:
            detail["example"] = str(example)
        return detail

    @staticmethod
    def _normalize_error_code(code: str, status: int | None) -> str:
        """Нормализует код ошибки Bitrix24."""

        normalized = (code or "").upper()
        if not normalized and status:
            if status == 400:
                normalized = "INVALID_ARGUMENT"
            elif status == 403:
                normalized = "ACCESS_DENIED"
            elif status == 404:
                normalized = "NOT_FOUND"
            elif status == 429:
                normalized = "RATE_LIMIT"
            elif status >= 500:
                normalized = "NETWORK"
        if normalized in {"INVALID_REQUEST", "ARGUMENT_ERROR"}:
            normalized = "INVALID_ARGUMENT"
        if normalized in {"TOO_MANY_REQUESTS", "RATE_LIMITED"}:
            normalized = "RATE_LIMIT"
        if not normalized:
            normalized = "UNKNOWN"
        return normalized

    def _tips_for_code(self, code: str, details: Any) -> List[str]:
        """Возвращает рекомендации по ошибке."""

        tips = list(self._error_tips.get(code, []))
        retry_after = None
        if isinstance(details, Mapping):
            retry_after = details.get("retry_after") or details.get("retry-after")
        if retry_after:
            tips = [f"Подождите {retry_after} сек перед повтором"] + tips
        if not tips:
            tips = [
                "Проверьте параметры запроса.",
                "Обратитесь к администратору портала, если ошибка повторяется.",
            ]
        return tips

    @staticmethod
    def _extract_retry_after(call_result: Mapping[str, Any]) -> float | None:
        """Извлекает retry_after из ответа Bitrix."""

        if "retry_after" in call_result:
            try:
                return float(call_result["retry_after"])
            except (TypeError, ValueError):  # pragma: no cover - нечисловое значение
                return None
        error = call_result.get("error")
        if isinstance(error, Mapping):
            details = error.get("details")
            if isinstance(details, Mapping):
                retry_after = details.get("retry_after") or details.get("retry-after")
                try:
                    return float(retry_after) if retry_after is not None else None
                except (TypeError, ValueError):  # pragma: no cover
                    return None
        return None
