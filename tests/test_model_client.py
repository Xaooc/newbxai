"""Тесты для клиента модели ChatGPT."""

from src.orchestrator.model_client import ModelClient


def test_build_messages_includes_summary():
    """Пользовательское сообщение должно содержать текстовую сводку состояния."""

    client = ModelClient(model_name="gpt-4.1", api_key="test-key", base_url="https://example.com")

    state = {
        "goals": ["Показать сделку"],
        "confirmations": {},
        "done": [{"description": "Создана сделка", "timestamp": "2024-01-01T00:00:00Z"}],
        "objects": {"current_deal_id": 123},
    }

    messages = client._build_messages("system", state, "Покажи детали")

    assert len(messages) == 2
    user_content = messages[1]["content"]
    assert "Краткая сводка состояния" in user_content
    assert "Текущее состояние агента (JSON)" in user_content
    assert "Создана сделка" in user_content
    assert "123" in user_content
