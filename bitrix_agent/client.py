"""Клиент для вызова REST-методов Bitrix24 через входящий вебхук."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from .config import BitrixConfig
from .tools import bitrix_batch, bitrix_call


class BitrixWebhookClient:
    """Обёртка над REST-методами Bitrix24, допустимыми для автономного агента."""

    def __init__(self, *, config: BitrixConfig | None = None) -> None:
        self.config = config or BitrixConfig.from_env()

    # --- Методы работы с пользователями ---
    def user_current(self) -> Mapping[str, Any]:
        """Получить информацию о пользователе вебхука."""

        response = bitrix_call("user.current", config=self.config)
        return response.get("result", {})

    def user_get(self, *, filter: Mapping[str, Any] | None = None,
                 order: Mapping[str, Any] | None = None,
                 select: Iterable[str] | None = None) -> Iterable[Mapping[str, Any]]:
        """Получить список пользователей по фильтру."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        response = bitrix_call("user.get", params, config=self.config)
        return response.get("result", [])

    # --- Пакетные запросы ---
    def batch(self, cmd: Mapping[str, str], *, halt: bool = False) -> Mapping[str, Any]:
        """Выполнить batch-вызов до 50 команд."""

        return bitrix_batch(cmd, halt=halt, config=self.config)

    # --- Подписки на события ---
    def event_bind(self, event: str, handler: str, *, auth_type: str | None = None) -> bool:
        """Зарегистрировать обработчик события."""

        params: Dict[str, Any] = {"event": event, "handler": handler}
        if auth_type:
            params["auth_type"] = auth_type
        response = bitrix_call("event.bind", params, config=self.config)
        return bool(response.get("result"))

    def event_get(self) -> Iterable[Mapping[str, Any]]:
        """Получить список активных подписок."""

        response = bitrix_call("event.get", config=self.config)
        return response.get("result", [])

    def event_unbind(self, event: str, handler: str) -> bool:
        """Удалить подписку на событие."""

        params = {"event": event, "handler": handler}
        response = bitrix_call("event.unbind", params, config=self.config)
        return bool(response.get("result"))

    # --- Сделки CRM ---
    def crm_deal_add(self, fields: Mapping[str, Any], *, params: Mapping[str, Any] | None = None) -> int:
        """Создать сделку и вернуть её ID."""

        payload: Dict[str, Any] = {"fields": dict(fields)}
        if params:
            payload["params"] = dict(params)
        response = bitrix_call("crm.deal.add", payload, config=self.config)
        return int(response.get("result"))

    def crm_deal_get(self, deal_id: int) -> Mapping[str, Any]:
        """Получить данные сделки по ID."""

        response = bitrix_call("crm.deal.get", {"id": int(deal_id)}, config=self.config)
        return response.get("result", {})

    def crm_deal_list(self, *, filter: Mapping[str, Any] | None = None,
                      order: Mapping[str, Any] | None = None,
                      select: Iterable[str] | None = None,
                      start: int | None = None) -> Mapping[str, Any]:
        """Получить список сделок с поддержкой пагинации."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        if start is not None:
            params["start"] = start
        return bitrix_call("crm.deal.list", params, config=self.config)

    def crm_deal_update(self, deal_id: int, fields: Mapping[str, Any], *, params: Mapping[str, Any] | None = None) -> bool:
        """Обновить сделку по ID."""

        payload: Dict[str, Any] = {"id": int(deal_id), "fields": dict(fields)}
        if params:
            payload["params"] = dict(params)
        response = bitrix_call("crm.deal.update", payload, config=self.config)
        return bool(response.get("result"))

    # --- Категории и стадии сделок (устаревшие методы) ---
    def crm_deal_category_list(self, *, filter: Mapping[str, Any] | None = None) -> Iterable[Mapping[str, Any]]:
        """Получить список направлений продаж."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        response = bitrix_call("crm.deal.category.list", params, config=self.config)
        return response.get("result", [])

    def crm_deal_category_stage_list(self, category_id: int | str) -> Iterable[Mapping[str, Any]]:
        """Получить стадии для указанной категории сделки."""

        response = bitrix_call("crm.deal.category.stage.list", {"id": category_id}, config=self.config)
        return response.get("result", [])

    # --- Справочники CRM ---
    def crm_status_list(self, *, filter: Mapping[str, Any], order: Mapping[str, Any] | None = None,
                        select: Iterable[str] | None = None) -> Iterable[Mapping[str, Any]]:
        """Получить элементы справочника CRM."""

        params: Dict[str, Any] = {"filter": dict(filter)}
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        response = bitrix_call("crm.status.list", params, config=self.config)
        return response.get("result", [])

    # --- Активности CRM ---
    def crm_activity_list(self, *, filter: Mapping[str, Any] | None = None,
                          order: Mapping[str, Any] | None = None,
                          select: Iterable[str] | None = None,
                          start: int | None = None) -> Mapping[str, Any]:
        """Получить список активностей CRM."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        if start is not None:
            params["start"] = start
        return bitrix_call("crm.activity.list", params, config=self.config)

    def crm_activity_add(self, fields: Mapping[str, Any]) -> int:
        """Добавить CRM-активность."""

        response = bitrix_call("crm.activity.add", {"fields": dict(fields)}, config=self.config)
        return int(response.get("result"))

    # --- Комментарии таймлайна ---
    def crm_timeline_comment_add(self, fields: Mapping[str, Any]) -> int:
        """Добавить комментарий в таймлайн."""

        response = bitrix_call("crm.timeline.comment.add", {"fields": dict(fields)}, config=self.config)
        return int(response.get("result"))

    # --- Контакты CRM ---
    def crm_contact_list(self, *, filter: Mapping[str, Any] | None = None,
                         order: Mapping[str, Any] | None = None,
                         select: Iterable[str] | None = None,
                         start: int | None = None) -> Mapping[str, Any]:
        """Получить список контактов."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        if start is not None:
            params["start"] = start
        return bitrix_call("crm.contact.list", params, config=self.config)

    def crm_contact_get(self, contact_id: int) -> Mapping[str, Any]:
        """Получить контакт по ID."""

        response = bitrix_call("crm.contact.get", {"id": int(contact_id)}, config=self.config)
        return response.get("result", {})

    # --- Компании CRM ---
    def crm_company_list(self, *, filter: Mapping[str, Any] | None = None,
                         order: Mapping[str, Any] | None = None,
                         select: Iterable[str] | None = None,
                         start: int | None = None) -> Mapping[str, Any]:
        """Получить список компаний."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        if start is not None:
            params["start"] = start
        return bitrix_call("crm.company.list", params, config=self.config)

    def crm_company_get(self, company_id: int) -> Mapping[str, Any]:
        """Получить компанию по ID."""

        response = bitrix_call("crm.company.get", {"id": int(company_id)}, config=self.config)
        return response.get("result", {})

    # --- Задачи ---
    def tasks_task_add(self, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        """Создать задачу и вернуть структуру с данными."""

        response = bitrix_call("tasks.task.add", {"fields": dict(fields)}, config=self.config)
        return response.get("result", {}).get("task", {})

    def tasks_task_update(self, task_id: int, fields: Mapping[str, Any]) -> Mapping[str, Any]:
        """Обновить задачу и вернуть обновлённые поля."""

        payload = {"taskId": int(task_id), "fields": dict(fields)}
        response = bitrix_call("tasks.task.update", payload, config=self.config)
        return response.get("result", {}).get("task", {})

    def tasks_task_list(self, *, filter: Mapping[str, Any] | None = None,
                        order: Mapping[str, Any] | None = None,
                        select: Iterable[str] | None = None,
                        start: int | None = None) -> Mapping[str, Any]:
        """Получить список задач."""

        params: Dict[str, Any] = {}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        if start is not None:
            params["start"] = start
        return bitrix_call("tasks.task.list", params, config=self.config)

    def task_commentitem_add(self, task_id: int, message: str) -> int:
        """Добавить комментарий к задаче."""

        payload = {"TASK_ID": int(task_id), "fields": {"POST_MESSAGE": message}}
        response = bitrix_call("task.commentitem.add", payload, config=self.config)
        return int(response.get("result"))

    def task_checklistitem_add(self, task_id: int, fields: Mapping[str, Any]) -> int:
        """Добавить пункт чек-листа задачи."""

        payload = {"TASK_ID": int(task_id), "fields": dict(fields)}
        response = bitrix_call("task.checklistitem.add", payload, config=self.config)
        return int(response.get("result"))

    # --- Рабочие группы ---
    def sonet_group_get(self, *, group_id: int | None = None,
                        filter: Mapping[str, Any] | None = None,
                        order: Mapping[str, Any] | None = None,
                        select: Iterable[str] | None = None) -> Iterable[Mapping[str, Any]]:
        """Получить список групп или конкретную группу."""

        params: Dict[str, Any] = {}
        if group_id is not None:
            params["ID"] = group_id
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        response = bitrix_call("sonet.group.get", params, config=self.config)
        return response.get("result", [])

    def sonet_group_user_get(self, group_id: int, *, filter: Mapping[str, Any] | None = None,
                             order: Mapping[str, Any] | None = None,
                             select: Iterable[str] | None = None) -> Iterable[Mapping[str, Any]]:
        """Получить список участников группы."""

        params: Dict[str, Any] = {"GROUP_ID": int(group_id)}
        if filter:
            params["filter"] = dict(filter)
        if order:
            params["order"] = dict(order)
        if select:
            params["select"] = list(select)
        response = bitrix_call("sonet.group.user.get", params, config=self.config)
        return response.get("result", [])
