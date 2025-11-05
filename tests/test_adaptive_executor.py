"""Проверка адаптивного пула потоков."""

import threading
import time

import pytest

from src.adapters.adaptive_executor import AdaptiveThreadPoolExecutor


def test_adaptive_executor_scales_up(monkeypatch):
    """Пул добавляет потоки при росте очереди и низкой загрузке CPU."""

    monkeypatch.setattr("src.adapters.adaptive_executor.cpu_load_ratio", lambda: 0.0)
    executor = AdaptiveThreadPoolExecutor(min_workers=1, max_workers=3, idle_timeout=0.1)

    started_threads: list[int] = []
    release = threading.Event()

    def blocking_task() -> None:
        started_threads.append(threading.get_ident())
        release.wait(0.2)

    futures = [executor.submit(blocking_task) for _ in range(3)]
    time.sleep(0.05)

    assert len(set(started_threads)) >= 2

    release.set()
    for future in futures:
        future.result()

    executor.shutdown(wait=True)
