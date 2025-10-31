"""CLI-точка входа для AI-менеджера Bitrix24."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.logging.logger import InteractionLogger
from src.orchestrator.agent import Orchestrator, OrchestratorSettings
from src.orchestrator.model_client import ModelClientError, build_default_model_client
from src.state.manager import AgentStateManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def build_orchestrator(mode: str, storage_dir: Path, log_dir: Path) -> Orchestrator:
    """Создаёт оркестратор с базовыми зависимостями."""

    state_manager = AgentStateManager(storage_dir=storage_dir)
    interaction_logger = InteractionLogger(log_dir=log_dir)
    settings = OrchestratorSettings(mode=mode)

    model_client = None
    try:
        model_client = build_default_model_client(settings.model_name)
    except ModelClientError as exc:
        logging.getLogger(__name__).warning(
            "GPT-5 Thinking недоступен: %s. Оркестратор запустится в режиме заглушки.",
            exc,
        )

    return Orchestrator(
        state_manager=state_manager,
        interaction_logger=interaction_logger,
        settings=settings,
        model_client=model_client,
    )


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI.

    Пользователь вводит идентификатор сессии и сообщение, агент возвращает ответ.
    """

    parser = argparse.ArgumentParser(description="AI-менеджер Bitrix24 (MVP)")
    parser.add_argument("user_id", help="Идентификатор пользователя или сессии")
    parser.add_argument("message", help="Сообщение для агента")
    parser.add_argument("--mode", choices=["shadow", "canary", "full"], default="shadow", help="Режим безопасности")
    parser.add_argument("--state-dir", default="./data/state", help="Каталог хранения состояний")
    parser.add_argument("--log-dir", default="./data/logs", help="Каталог логов")

    args = parser.parse_args(argv)

    orchestrator = build_orchestrator(args.mode, Path(args.state_dir), Path(args.log_dir))
    response = orchestrator.process_message(args.user_id, args.message)
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
