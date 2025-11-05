"""Утилиты нормализации пользовательского ввода."""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ_NAME = os.getenv("BITRIX_DEFAULT_TZ", "Europe/Amsterdam")
DEFAULT_TZ = ZoneInfo(DEFAULT_TZ_NAME)


@dataclass
class DeadlineParsingResult:
    """Результат парсинга срока."""

    dt: dt.datetime
    source: str


def parse_deadline(text: str, *, now: Optional[dt.datetime] = None) -> DeadlineParsingResult:
    """Преобразовать человеческое описание срока в ISO-время."""

    baseline = now or dt.datetime.now(tz=DEFAULT_TZ)
    lowered = text.lower().strip()

    if lowered.startswith("через "):
        match = re.match(r"через\s+(\d+)\s*(час|часа|часов|мин|минут|день|дня|дней)", lowered)
        if match:
            value = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("час"):
                delta = dt.timedelta(hours=value)
            elif unit.startswith("мин"):
                delta = dt.timedelta(minutes=value)
            else:
                delta = dt.timedelta(days=value)
            result_dt = baseline + delta
            return DeadlineParsingResult(dt=result_dt, source=text)

    if lowered in {"сегодня", "сегодня до конца дня"}:
        result_dt = baseline.replace(hour=18, minute=0, second=0, microsecond=0)
        return DeadlineParsingResult(dt=result_dt, source=text)

    if lowered.startswith("завтра"):
        result_dt = (baseline + dt.timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0)
        time_match = re.search(r"(\d{1,2}:\d{2})", lowered)
        if time_match:
            hours, minutes = map(int, time_match.group(1).split(":"))
            result_dt = result_dt.replace(hour=hours, minute=minutes)
        return DeadlineParsingResult(dt=result_dt, source=text)

    parsed = _parse_absolute(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=DEFAULT_TZ)
    return DeadlineParsingResult(dt=parsed, source=text)


def parse_amount(text: str, default_currency: str = "RUB") -> tuple[Decimal, str]:
    """Распознать сумму и валюту."""

    cleaned = text.replace("руб.", "").replace("₽", "").replace(" ", "").strip()
    currency = default_currency
    match = re.match(r"([0-9]+(?:[.,][0-9]+)?)\s*([A-Z]{3})?", cleaned)
    if not match:
        raise ValueError(f"Не удалось распознать сумму: {text}")
    amount_str, currency_match = match.groups()
    amount = Decimal(amount_str.replace(",", "."))
    if currency_match:
        currency = currency_match
    return amount, currency


def _parse_absolute(text: str) -> dt.datetime:
    """Попробовать разобрать абсолютное время без сторонних зависимостей."""

    raw = text.strip().replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        pass

    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(raw, fmt)
            if "H" not in fmt:
                parsed = parsed.replace(hour=18, minute=0)
            return parsed.replace(tzinfo=DEFAULT_TZ)
        except ValueError:
            continue
    raise ValueError(f"Не удалось распознать срок: {text}")
