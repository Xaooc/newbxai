"""Утилита для оценки скорости обработки запросов оркестратором."""

from __future__ import annotations

import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app_logging.logger import InteractionLogger
from src.orchestrator.agent import Orchestrator, OrchestratorSettings
from src.state.manager import AgentStateManager


class _FakeModelClient:
    """Упрощённый клиент модели, возвращающий пустой план."""

    def __init__(self) -> None:
        self.response = (
            "THOUGHT:\n- План пуст\nACTION:\n[]\nASSISTANT:\nПроверено."
        )

    def generate(self, system_prompt: str, state_snapshot: Dict[str, object], user_message: str) -> str:  # noqa: D401
        return self.response


def _build_orchestrator(base_dir: Path) -> Orchestrator:
    state_dir = base_dir / "state"
    log_dir = base_dir / "logs"
    state_manager = AgentStateManager(storage_dir=state_dir)
    interaction_logger = InteractionLogger(log_dir=log_dir, max_bytes=1024)
    settings = OrchestratorSettings(mode="full")
    model_client = _FakeModelClient()
    return Orchestrator(
        state_manager=state_manager,
        interaction_logger=interaction_logger,
        settings=settings,
        model_client=model_client,
    )


def _run_single(orchestrator: Orchestrator, user_id: str, text: str) -> float:
    started = time.perf_counter()
    orchestrator.process_message(user_id, text)
    return time.perf_counter() - started


def run_load_test(concurrency: int = 12, iterations: int = 60) -> Dict[str, float]:
    """Запускает параллельную обработку и возвращает метрики времени."""

    with tempfile.TemporaryDirectory() as tmp:
        orchestrator = _build_orchestrator(Path(tmp))
        durations: List[float] = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for index in range(iterations):
                user_id = f"user-{index % concurrency}"
                futures.append(pool.submit(_run_single, orchestrator, user_id, f"Запрос {index}"))
            for future in futures:
                durations.append(future.result())

    durations.sort()
    avg = statistics.mean(durations)
    p95 = durations[max(0, int(len(durations) * 0.95) - 1)]
    p99 = durations[max(0, int(len(durations) * 0.99) - 1)]
    return {"avg": avg, "p95": p95, "p99": p99}


def main() -> None:
    metrics = run_load_test()
    print("Среднее время ответа: %.3f с" % metrics["avg"])
    print("95-й перцентиль: %.3f с" % metrics["p95"])
    print("99-й перцентиль: %.3f с" % metrics["p99"])


if __name__ == "__main__":  # pragma: no cover
    main()
