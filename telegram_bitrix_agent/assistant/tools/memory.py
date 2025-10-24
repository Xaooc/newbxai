"""Хранилище памяти чатов с поддержкой Redis."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

try:  # pragma: no cover - Redis не обязателен в тестовой среде
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore

try:  # pragma: no cover - orjson используется если доступен
    import orjson
except ImportError:  # pragma: no cover
    import json as orjson  # type: ignore


def _dumps(value: Any) -> str:
    """Сериализует значение в строку JSON."""

    payload = orjson.dumps(value)
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


@dataclass
class MemoryStore:
    """Память в оперативной памяти процесса (по умолчанию)."""

    short_term_limit: int = 10
    _history: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _long_term: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add_message(self, chat_id: str, role: str, content: str) -> None:
        """Сохраняет сообщение пользователя или ассистента."""

        async with self._lock:
            history = self._history.setdefault(chat_id, [])
            history.append({"role": role, "content": content})
            if len(history) > self.short_term_limit:
                del history[0]

    async def get_history(self, chat_id: str) -> List[Dict[str, Any]]:
        """Возвращает историю сообщений для чата."""

        async with self._lock:
            return list(self._history.get(chat_id, []))

    async def tool_get(self, chat_id: str, keys: List[str]) -> Dict[str, Any]:
        """Возвращает значения долговременной памяти по ключам."""

        async with self._lock:
            stored = self._long_term.get(chat_id, {})
            return {key: stored[key] for key in keys if key in stored}

    async def tool_set(self, chat_id: str, pairs: Mapping[str, Any]) -> Dict[str, bool]:
        """Обновляет долговременную память."""

        async with self._lock:
            storage = self._long_term.setdefault(chat_id, {})
            storage.update(dict(pairs))
        return {"ok": True}

    async def close(self) -> None:  # pragma: no cover - в памяти действий не требуется
        """Метод совместимости для закрытия ресурсов."""

        return None


class RedisMemoryStore(MemoryStore):
    """Хранилище памяти, работающее через Redis."""

    def __init__(
        self,
        dsn: str,
        *,
        short_term_limit: int = 10,
        namespace: str = "telegram-bitrix-agent",
        redis_client: Optional[aioredis.Redis] = None,
    ) -> None:
        if aioredis is None:  # pragma: no cover - Redis может отсутствовать в тестах
            raise RuntimeError("Пакет redis не установлен, RedisMemoryStore недоступен")
        super().__init__(short_term_limit=short_term_limit)
        self._redis = redis_client or aioredis.from_url(dsn, encoding="utf-8", decode_responses=True)
        self._namespace = namespace

    def _history_key(self, chat_id: str) -> str:
        return f"{self._namespace}:{chat_id}:history"

    def _memory_key(self, chat_id: str) -> str:
        return f"{self._namespace}:{chat_id}:memory"

    async def add_message(self, chat_id: str, role: str, content: str) -> None:
        payload = _dumps({"role": role, "content": content})
        key = self._history_key(chat_id)
        await self._redis.rpush(key, payload)
        await self._redis.ltrim(key, -self.short_term_limit, -1)

    async def get_history(self, chat_id: str) -> List[Dict[str, Any]]:
        key = self._history_key(chat_id)
        raw_items = await self._redis.lrange(key, -self.short_term_limit, -1)
        history: List[Dict[str, Any]] = []
        for item in raw_items:
            try:
                history.append(orjson.loads(item))
            except ValueError:  # pragma: no cover - повреждённые записи пропускаем
                continue
        return history

    async def tool_get(self, chat_id: str, keys: List[str]) -> Dict[str, Any]:
        if not keys:
            return {}
        key = self._memory_key(chat_id)
        values = await self._redis.hmget(key, keys)
        result: Dict[str, Any] = {}
        for key_name, value in zip(keys, values):
            if value is None:
                continue
            try:
                result[key_name] = orjson.loads(value)
            except ValueError:
                result[key_name] = value
        return result

    async def tool_set(self, chat_id: str, pairs: Mapping[str, Any]) -> Dict[str, bool]:
        if not pairs:
            return {"ok": True}
        key = self._memory_key(chat_id)
        encoded = {name: _dumps(value) for name, value in pairs.items()}
        await self._redis.hset(key, mapping=encoded)
        return {"ok": True}

    async def close(self) -> None:  # pragma: no cover - зависит от наличия Redis
        await self._redis.close()


__all__ = ["MemoryStore", "RedisMemoryStore"]
