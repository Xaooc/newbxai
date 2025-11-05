"""Тесты нормализации пользовательского ввода."""

import datetime as dt

from bitrix_agent.normalizer import DEFAULT_TZ, parse_amount, parse_deadline


def test_parse_deadline_relative_hours():
    now = dt.datetime(2024, 1, 1, 10, 0, tzinfo=DEFAULT_TZ)
    result = parse_deadline("через 2 часа", now=now)
    assert result.dt == dt.datetime(2024, 1, 1, 12, 0, tzinfo=DEFAULT_TZ)


def test_parse_deadline_tomorrow_time():
    now = dt.datetime(2024, 5, 10, 9, 0, tzinfo=DEFAULT_TZ)
    result = parse_deadline("завтра 12:30", now=now)
    assert result.dt.hour == 12 and result.dt.minute == 30
    assert result.dt.date() == dt.date(2024, 5, 11)


def test_parse_amount_with_currency():
    amount, currency = parse_amount("1 250 000 RUB")
    assert str(amount) == "1250000"
    assert currency == "RUB"
