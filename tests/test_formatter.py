"""Проверки форматтера ответов."""
from telegram_bitrix_agent.assistant.tools.formatter import Formatter


def test_contact_formatting() -> None:
    formatter = Formatter()
    data = {
        "ID": 12,
        "NAME": "Иван Петров",
        "PHONE": [{"VALUE": "+79990000000"}],
        "EMAIL": [{"VALUE": "ivan@example.com"}],
        "COMPANY_ID": 34,
    }
    payload = formatter.humanize("crm.contact", data)
    assert payload["result"] == "Контакт обновлён: ID 12"
    assert "Имя: Иван Петров" in payload.get("details", [])
    assert "Телефон: +79990000000" in payload.get("details", [])
    assert payload.get("next_step") == "Назначить задачу по контакту?"


def test_company_formatting() -> None:
    formatter = Formatter()
    data = {
        "ID": 55,
        "TITLE": "ООО Ромашка",
        "INDUSTRY": "Торговля",
        "PHONE": [{"VALUE": "+71234567890"}],
    }
    payload = formatter.humanize("crm.company", data)
    assert payload["result"] == "Компания обновлена: ID 55"
    assert "Название: ООО Ромашка" in payload.get("details", [])
    assert "Отрасль: Торговля" in payload.get("details", [])
    assert payload.get("next_step") == "Проверить ответственного за компанию?"


def test_lead_formatting() -> None:
    formatter = Formatter()
    data = {
        "ID": 101,
        "TITLE": "Лид по рекламе",
        "STATUS_ID": "NEW",
        "SOURCE_ID": "ADVERT",
    }
    payload = formatter.humanize("crm.lead", data)
    assert payload["result"] == "Лид обновлён: ID 101"
    assert "Статус: NEW" in payload.get("details", [])
    assert "Источник: ADVERT" in payload.get("details", [])
    assert payload.get("next_step") == "Назначить менеджера?"


def test_call_formatting() -> None:
    formatter = Formatter()
    data = {
        "ID": "call-1",
        "PHONE_NUMBER": "+78005553535",
        "DURATION": 120,
        "CALL_TYPE": "inbound",
    }
    payload = formatter.humanize("telephony.call", data)
    assert payload["result"] == "Звонок зарегистрирован: ID call-1"
    assert "Номер: +78005553535" in payload.get("details", [])
    assert "Длительность: 120 сек" in payload.get("details", [])
    assert payload.get("next_step") == "Добавить комментарий к звонку?"
