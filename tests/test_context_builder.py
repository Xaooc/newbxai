"""Тесты для построителя текстового резюме состояния."""

from src.orchestrator.context_builder import build_state_summary


def test_summary_includes_key_sections():
    """Резюме должно включать цели, последний план и известные объекты."""

    state = {
        "goals": ["Создать сделку", "Назначить встречу"],
        "in_progress": [{"description": "Ожидаем подтверждение суммы"}],
        "done": [
            {"description": "Создана сделка"},
            {"description": "Добавлен комментарий"},
        ],
        "objects": {
            "current_deal_id": 101,
            "current_contact_id": 55,
        },
        "last_plan": {
            "summary": "1) обновить сумму сделки",
            "actions": [],
        },
    }

    summary = build_state_summary(state, limit=800)

    assert "Активные цели" in summary
    assert "Последний план" in summary
    assert "Известные объекты" in summary
    assert "101" in summary
    assert "55" in summary


def test_summary_trims_to_limit():
    """При ограничении длины текст обрезается и завершается многоточием."""

    state = {
        "goals": [f"Цель {idx}" for idx in range(10)],
        "in_progress": [],
        "confirmations": {},
        "done": [],
        "objects": {},
    }

    summary = build_state_summary(state, limit=40)
    assert len(summary) <= 41  # допускаем символ перевода строки перед многоточием
    assert summary.endswith("…")
