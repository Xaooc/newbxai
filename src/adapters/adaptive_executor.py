"""Адаптивный пул потоков для Telegram-адаптера."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
from typing import Any, Callable

from src.orchestrator.telemetry import cpu_load_ratio


class AdaptiveThreadPoolExecutor(concurrent.futures.Executor):
    """Пул потоков, динамически подстраивающий количество воркеров."""

    def __init__(
        self,
        *,
        min_workers: int,
        max_workers: int,
        cpu_threshold: float = 0.85,
        idle_timeout: float = 10.0,
    ) -> None:
        if min_workers < 1:
            raise ValueError("min_workers должен быть положительным")
        if max_workers < min_workers:
            raise ValueError("max_workers не может быть меньше min_workers")
        self._min_workers = min_workers
        self._max_workers = max_workers
        self._cpu_threshold = cpu_threshold
        self._idle_timeout = idle_timeout
        self._queue: "queue.Queue[tuple[concurrent.futures.Future[Any], Callable[[], Any]]]" = queue.Queue()
        self._threads: set[threading.Thread] = set()
        self._shutdown = False
        self._lock = threading.Lock()
        for _ in range(self._min_workers):
            self._spawn_worker()

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        if self._shutdown:
            raise RuntimeError("executor has been shut down")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()

        def callback() -> Any:
            return fn(*args, **kwargs)

        self._queue.put((future, callback))
        self._maybe_scale_up()
        return future

    def shutdown(self, wait: bool = True) -> None:  # noqa: D401
        with self._lock:
            self._shutdown = True
            for _ in range(len(self._threads)):
                self._queue.put((None, None))  # type: ignore[arg-type]
        if wait:
            for thread in list(self._threads):
                thread.join()

    def _spawn_worker(self) -> None:
        worker = threading.Thread(target=self._worker_loop, name="telegram-adaptive-worker", daemon=True)
        self._threads.add(worker)
        worker.start()

    def _worker_loop(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=self._idle_timeout)
                except queue.Empty:
                    with self._lock:
                        if self._shutdown:
                            break
                        if len(self._threads) > self._min_workers:
                            self._threads.remove(threading.current_thread())
                            break
                        else:
                            continue
                future, callback = item
                if future is None or callback is None:
                    break
                if future.set_running_or_notify_cancel():
                    try:
                        result = callback()
                    except BaseException as exc:  # noqa: BLE001
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
                self._queue.task_done()
        finally:
            with self._lock:
                self._threads.discard(threading.current_thread())

    def _maybe_scale_up(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            backlog = self._queue.qsize()
            active = len(self._threads)
            if backlog < active:
                return
            if active >= self._max_workers:
                return
            if cpu_load_ratio() > self._cpu_threshold:
                return
            self._spawn_worker()
