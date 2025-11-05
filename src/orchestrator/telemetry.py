"""Вспомогательные классы для сбора телеметрии оркестратора."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RiskWarningInfo:
    """Описание предупреждения о рискованных изменениях."""

    message: str
    method: str
    fields: Set[str] = field(default_factory=set)


class MetricsCollector:
    """Агрегирует статистику по недостающим полям и рискованным действиям."""

    def __init__(self, log_every: int = 10) -> None:
        self._missing_counter: Counter[str] = Counter()
        self._risk_counter: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._log_every = max(1, log_every)

    def record_missing_fields(
        self,
        state: "AgentState",
        raw_paths: Sequence[Tuple[object, ...]],
        *,
        path_formatter: Callable[[Tuple[object, ...]], str],
        friendly_formatter: Callable[[Tuple[object, ...]], str],
    ) -> None:
        if not raw_paths:
            return
        metrics = state.metrics.setdefault("missing_fields", {})
        friendly_names: List[str] = []
        with self._lock:
            for path in raw_paths:
                key = path_formatter(path)
                metrics[key] = metrics.get(key, 0) + 1
                self._missing_counter[key] += 1
                friendly_names.append(friendly_formatter(path))
            total = sum(self._missing_counter.values())
            if total % self._log_every == 0:
                top = ", ".join(
                    f"{name} — {count}" for name, count in self._missing_counter.most_common(5)
                )
                logger.info("Статистика недостающих полей: %s", top)
        if friendly_names:
            deduped = list(dict.fromkeys(friendly_names))
            logger.debug(
                "Запрошены дополнительные данные: %s",
                ", ".join(deduped),
            )

    def record_risk_warnings(
        self,
        state: "AgentState",
        warnings: Sequence[RiskWarningInfo],
    ) -> None:
        if not warnings:
            return
        metrics = state.metrics.setdefault("risk_warnings", {})
        with self._lock:
            for warning in warnings:
                for field_name in warning.fields or {warning.method}:
                    metrics[field_name] = metrics.get(field_name, 0) + 1
                    self._risk_counter[field_name] += 1
            total = sum(self._risk_counter.values())
            if total % self._log_every == 0:
                top = ", ".join(
                    f"{name} — {count}" for name, count in self._risk_counter.most_common(5)
                )
                logger.info("Статистика рискованных изменений: %s", top)


class BitrixErrorMonitor:
    """Следит за всплесками ошибок Bitrix24 и поднимает алерты."""

    def __init__(
        self,
        *,
        threshold: int = 5,
        interval: float = 60.0,
        alert_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._threshold = max(1, threshold)
        self._interval = max(1.0, interval)
        self._alert_callback = alert_callback
        self._events: deque[Tuple[float, str]] = deque()
        self._lock = threading.Lock()
        self._last_alert: float = 0.0

    def record_error(self, method: str, diagnostic: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._events.append((now, method))
            self._trim(now)
            if len(self._events) >= self._threshold and now - self._last_alert >= self._interval / 2:
                methods = sorted({entry[1] for entry in self._events})
                message = (
                    "Повторяющиеся ошибки Bitrix24: методы %s за последние %.0f секунд."
                    % (", ".join(methods), self._interval)
                )
                logger.error("%s Последняя ошибка: %s", message, diagnostic)
                self._last_alert = now
                if self._alert_callback:
                    try:
                        self._alert_callback(message)
                    except Exception:  # noqa: BLE001
                        logger.exception("Не удалось отправить оповещение операторам")

    def _trim(self, now: float) -> None:
        cutoff = now - self._interval
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


def cpu_load_ratio() -> float:
    """Возвращает усреднённую загрузку CPU, нормированную на число ядер."""

    try:
        load1, _load5, _load15 = os.getloadavg()
    except (AttributeError, OSError):  # pragma: no cover - платформа без getloadavg
        return 0.0
    cpus = os.cpu_count() or 1
    return max(0.0, load1 / max(1, cpus))


# Локальный импорт для аннотаций (чтобы избежать циклической зависимости).
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.state.manager import AgentState
