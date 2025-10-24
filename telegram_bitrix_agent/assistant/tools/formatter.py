"""Форматирование ответов в человекочитаемый вид."""
from __future__ import annotations

from typing import Any, Dict, List


class Formatter:
    """Формирует краткие человеческие ответы по данным Bitrix24."""

    def humanize(self, entity: str, data: Any, locale: str = "ru") -> Dict[str, Any]:
        """Возвращает структуру с результатом, деталями и подсказкой по следующему шагу."""

        if locale != "ru":
            raise ValueError("Поддерживается только русский язык")
        result = "Готово"
        details: List[str] = []
        next_step: str | None = None

        if entity == "crm.deal" and isinstance(data, dict):
            deal_id = data.get("ID")
            title = data.get("TITLE")
            amount = data.get("OPPORTUNITY")
            currency = data.get("CURRENCY_ID")
            stage = data.get("STAGE_ID")
            result = f"Сделка создана: ID {deal_id}" if deal_id else "Сделка обработана"
            if title:
                details.append(f"Название: {title}")
            if amount is not None:
                currency_part = f" {currency}" if currency else ""
                details.append(f"Сумма: {amount}{currency_part}")
            if stage:
                details.append(f"Стадия: {stage}")
            next_step = "Назначить ответственного?"
        elif entity in {"crm.contact", "contact"} and isinstance(data, dict):
            contact_id = data.get("ID")
            name = data.get("NAME") or data.get("FULL_NAME") or data.get("TITLE")
            result = (
                f"Контакт обновлён: ID {contact_id}" if contact_id else "Контакт обработан"
            )
            if name:
                details.append(f"Имя: {name}")
            phone = _first_value(data.get("PHONE"))
            if phone:
                details.append(f"Телефон: {phone}")
            email = _first_value(data.get("EMAIL"))
            if email:
                details.append(f"Email: {email}")
            company_id = data.get("COMPANY_ID")
            if company_id:
                details.append(f"Компания: {company_id}")
            next_step = "Назначить задачу по контакту?"
        elif entity in {"crm.company", "company"} and isinstance(data, dict):
            company_id = data.get("ID")
            title = data.get("TITLE") or data.get("NAME")
            result = (
                f"Компания обновлена: ID {company_id}" if company_id else "Компания обработана"
            )
            if title:
                details.append(f"Название: {title}")
            industry = data.get("INDUSTRY")
            if industry:
                details.append(f"Отрасль: {industry}")
            phone = _first_value(data.get("PHONE"))
            if phone:
                details.append(f"Телефон: {phone}")
            next_step = "Проверить ответственного за компанию?"
        elif entity in {"crm.lead", "lead"} and isinstance(data, dict):
            lead_id = data.get("ID")
            title = data.get("TITLE") or data.get("NAME")
            status = data.get("STATUS_ID") or data.get("STATUS")
            result = f"Лид обновлён: ID {lead_id}" if lead_id else "Лид обработан"
            if title:
                details.append(f"Название: {title}")
            if status:
                details.append(f"Статус: {status}")
            source = data.get("SOURCE_ID") or data.get("SOURCE")
            if source:
                details.append(f"Источник: {source}")
            next_step = "Назначить менеджера?"
        elif entity in {"telephony.call", "call"} and isinstance(data, dict):
            call_id = data.get("ID") or data.get("CALL_ID")
            phone = data.get("PHONE_NUMBER") or data.get("PHONE")
            duration = data.get("DURATION") or data.get("CALL_DURATION")
            direction = data.get("CALL_TYPE") or data.get("DIRECTION")
            result = f"Звонок зарегистрирован: ID {call_id}" if call_id else "Звонок обработан"
            if phone:
                details.append(f"Номер: {phone}")
            if duration is not None:
                details.append(f"Длительность: {duration} сек")
            if direction:
                details.append(f"Тип: {direction}")
            next_step = "Добавить комментарий к звонку?"
        elif entity == "task" and isinstance(data, dict):
            task_id = data.get("ID")
            title = data.get("TITLE") or "Задача"
            deadline = data.get("DEADLINE")
            result = f"Задача обновлена: ID {task_id}" if task_id else f"Задача: {title}"
            details.append(f"Название: {title}")
            details.append(f"Дедлайн: {deadline or 'не задан'}")
            next_step = "Проверить ответственного?"
        elif isinstance(data, list):
            result = f"Найдено элементов: {len(data)}"
            preview = []
            for item in data[:3]:
                if isinstance(item, dict):
                    label = item.get("TITLE") or item.get("NAME") or item.get("ID")
                    if label is not None:
                        preview.append(str(label))
            if preview:
                details.append(f"Примеры: {', '.join(preview)}")
        elif isinstance(data, dict):
            obj_id = data.get("ID")
            result = f"Объект {entity} обработан" if not obj_id else f"Объект {entity}: ID {obj_id}"
            for key in ("TITLE", "NAME", "LOGIN"):
                if key in data:
                    details.append(f"{key}: {data[key]}")
        else:
            result = "Готово"

        payload: Dict[str, Any] = {"result": result}
        if details:
            payload["details"] = details
        if next_step:
            payload["next_step"] = next_step
        return payload


def _first_value(field: Any) -> str | None:
    """Возвращает первое значение из списка полей Bitrix."""

    if isinstance(field, list) and field:
        value = field[0]
        if isinstance(value, dict):
            return value.get("VALUE") or value.get("value")
        return str(value)
    if isinstance(field, dict):
        return field.get("VALUE") or field.get("value")
    if isinstance(field, str):
        return field
    return None
