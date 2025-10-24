"""Клиент OpenAI для планирования и генерации параметров."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Sequence

from langchain_core.messages import AIMessage, HumanMessage

try:  # pragma: no cover
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    class AsyncOpenAI:  # noqa: D401
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise RuntimeError("Библиотека openai не установлена")


if TYPE_CHECKING:  # pragma: no cover - только для подсказок типов
    from ..tools.bitrix import BitrixGateway
    from ..tools.formatter import Formatter
    from ..tools.memory import MemoryStore


@dataclass
class ActionPlan:
    """Описывает один шаг выполнения."""

    method: str = ""
    entity: str = "объект"
    params: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)


@dataclass
class PlanResponse:
    """Структура, которую ожидаем от GPT-5."""

    plan: List[str] = field(default_factory=list)
    method: str = ""
    entity: str = "объект"
    params: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    memory_get: List[str] = field(default_factory=list)
    memory_set: Dict[str, Any] = field(default_factory=dict)
    actions: List[ActionPlan] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.actions:
            self.actions = [
                ActionPlan(
                    method=self.method,
                    entity=self.entity,
                    params=self.params,
                    missing_fields=self.missing_fields,
                )
            ]

    @classmethod
    def from_json(cls, payload: str) -> "PlanResponse":
        data = json.loads(payload)
        actions_payload = []
        for item in data.get("actions", []) or []:
            if not isinstance(item, dict):
                continue
            actions_payload.append(
                ActionPlan(
                    method=item.get("method", ""),
                    entity=item.get("entity", data.get("entity", "объект")),
                    params=dict(item.get("params", {})),
                    missing_fields=list(item.get("missing_fields", [])),
                )
            )
        instance = cls(
            plan=list(data.get("plan", [])),
            method=data.get("method", ""),
            entity=data.get("entity", "объект"),
            params=dict(data.get("params", {})),
            missing_fields=list(data.get("missing_fields", [])),
            memory_get=list(data.get("memory_get", [])),
            memory_set=dict(data.get("memory_set", {})),
            actions=actions_payload,
        )
        return instance


PLANNER_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bitrix.search_methods",
            "description": "Подбор подходящих методов Bitrix24 по свободной формулировке запроса.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Свободное описание задачи пользователя.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bitrix.schema",
            "description": "Получение JSON-схемы параметров для выбранного метода Bitrix24.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Точный идентификатор метода Bitrix24.",
                    }
                },
                "required": ["method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bitrix.call",
            "description": "Выполнение метода Bitrix24 и получение результата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "description": "Метод Bitrix24 для вызова.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Параметры запроса, соответствующие схеме метода.",
                    },
                },
                "required": ["method", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "formatter.humanize",
            "description": "Преобразование данных Bitrix24 в короткий человеко-понятный ответ.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Тип сущности (crm.deal, task, контакт и т.д.).",
                    },
                    "data": {
                        "description": "Данные Bitrix24, полученные из метода.",
                        "type": ["object", "array", "string", "number", "boolean", "null"],
                    },
                    "locale": {
                        "type": "string",
                        "enum": ["ru"],
                        "description": "Локаль ответа (поддерживается только ru).",
                        "default": "ru",
                    },
                },
                "required": ["entity", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory.get",
            "description": "Чтение долговременной памяти по указанным ключам.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список ключей из долговременной памяти.",
                    }
                },
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory.set",
            "description": "Сохранение значений в долговременную память.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pairs": {
                        "type": "object",
                        "description": "Ключи и значения, которые нужно запомнить.",
                    }
                },
                "required": ["pairs"],
            },
        },
    },
]


@dataclass
class PlannerToolContext:
    """Исполнитель инструментов для GPT-5."""

    gateway: "BitrixGateway"
    memory: "MemoryStore"
    formatter: "Formatter"
    chat_id: str

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        """Выполняет инструмент и возвращает результат."""

        if name == "bitrix.search_methods":
            query = str(arguments.get("query", ""))
            return self.gateway.search_methods(query)
        if name == "bitrix.schema":
            method = str(arguments.get("method", ""))
            return self.gateway.schema(method)
        if name == "bitrix.call":
            method = str(arguments.get("method", ""))
            params = arguments.get("params")
            if not isinstance(params, Mapping):
                params = {}
            return await asyncio.to_thread(self.gateway.call, method, params)
        if name == "formatter.humanize":
            entity = str(arguments.get("entity", "объект"))
            locale = str(arguments.get("locale", "ru"))
            data = arguments.get("data")
            return self.formatter.humanize(entity, data, locale=locale)
        if name == "memory.get":
            keys = arguments.get("keys")
            if not isinstance(keys, list):
                keys = []
            return await self.memory.tool_get(self.chat_id, [str(item) for item in keys])
        if name == "memory.set":
            pairs = arguments.get("pairs")
            if not isinstance(pairs, Mapping):
                pairs = {}
            return await self.memory.tool_set(self.chat_id, pairs)
        raise ValueError(f"Неизвестный инструмент: {name}")


class OpenAIPlanner:
    """Обёртка вокруг GPT-5 tool calling."""

    def __init__(self, api_key: str, model: str = "gpt-5.0-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def plan(
        self,
        *,
        system_prompt: str,
        user_message: str,
        history: Sequence[HumanMessage | AIMessage],
        candidates: List[Dict[str, Any]],
        tool_context: PlannerToolContext,
    ) -> PlanResponse:
        """Запрашивает у модели план действий."""

        messages: List[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for item in history:
            messages.append({"role": item.type, "content": item.content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Сформируй план действий в формате JSON с ключами plan, method, entity, params, missing_fields."
                    " Учти кандидатов: "
                    f"{candidates}. Сообщение пользователя: {user_message}"
                ),
            }
        )
        while True:
            response = await self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=messages,
                tools=PLANNER_TOOLS,
                tool_choice="auto",
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments or "{}"
                    try:
                        parsed_args = json.loads(arguments)
                    except json.JSONDecodeError:
                        parsed_args = {}
                    result = await tool_context.execute(name, parsed_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                continue
            content = message.content if message.content else "{}"
            return PlanResponse.from_json(content)


__all__ = [
    "ActionPlan",
    "OpenAIPlanner",
    "PlanResponse",
    "PlannerToolContext",
    "PLANNER_TOOLS",
]
