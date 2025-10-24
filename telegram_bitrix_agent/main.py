"""Точка входа для запуска Telegram Bitrix ассистента."""
from __future__ import annotations

import asyncio
from pathlib import Path

from bitrix24_client import Bitrix24Client

from .assistant.graph.llm import OpenAIPlanner
from .assistant.graph.runner import AssistantOrchestrator
from .assistant.tools.bitrix import BitrixGateway
from .assistant.tools.formatter import Formatter
from .assistant.tools.memory import MemoryStore, RedisMemoryStore
from .config import Settings
from .telegram.bot import run_webhook


async def main() -> None:
    """Инициализирует зависимости и запускает вебхук."""

    settings = Settings.from_env()
    planner = OpenAIPlanner(api_key=settings.openai_api_key)
    schemas_dir = Path(__file__).resolve().parent.parent / "schemas"
    client = Bitrix24Client(base_url=settings.bitrix_webhook)
    gateway = BitrixGateway(client, schemas_dir)
    formatter = Formatter()
    if settings.redis_dsn:
        memory = RedisMemoryStore(settings.redis_dsn)
    else:
        memory = MemoryStore()
    orchestrator = AssistantOrchestrator(
        planner=planner,
        gateway=gateway,
        formatter=formatter,
        memory=memory,
    )
    await run_webhook(settings, orchestrator)


if __name__ == "__main__":
    asyncio.run(main())
