import pytest

from telegram_bot import _is_plan_cancellation, _is_plan_confirmation, _normalize_user_text


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Да", "да"),
        ("  Да, запускай!!! ", "да запускай"),
        ("    отмена\n", "отмена"),
        ("Запускай-после проверки", "запускай после проверки"),
    ],
)
def test_normalize_user_text(source: str, expected: str) -> None:
    assert _normalize_user_text(source) == expected


@pytest.mark.parametrize(
    "phrase",
    [
        "Да",
        "ДА!",
        "Да, запускай",
        "Запускай.",
        "Выполняй пожалуйста",
        "Старт",
        "go!",
    ],
)
def test_confirmation_detection_accepts_punctuation(phrase: str) -> None:
    assert _is_plan_confirmation(phrase)


@pytest.mark.parametrize(
    "phrase",
    [
        "Нет",
        "Нет, нужен другой вариант",
        "Отмена.",
        "cancel!",
    ],
)
def test_cancellation_detection_accepts_punctuation(phrase: str) -> None:
    assert _is_plan_cancellation(phrase)
