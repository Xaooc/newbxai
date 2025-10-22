
# -*- coding: utf-8 -*-
"""
bitrix24_client.py — минимальная, прикладная библиотека для Bitrix24 REST (облако).

Поддержаны методы из запроса пользователя:
  - access.name
  - app.info
  - app.option.get
  - app.option.set
  - batch
  - calendar.accessibility.get
  - calendar.event.add
  - calendar.event.delete
  - calendar.event.get
  - calendar.event.get.nearest
  - calendar.event.getbyid
  - calendar.event.update
  - calendar.meeting.params.set
  - calendar.meeting.status.get
  - calendar.meeting.status.set
  - calendar.resource.add
  - calendar.resource.booking.list
  - calendar.resource.delete
  - calendar.resource.list
  - calendar.resource.update

Архитектура:
  - Работает и с вебхуком (base_url вида https://<portal>/rest/<user>/<code>/)
    и с OAuth (base_url вида https://<portal>/rest/ + token=? в query).
  - Единая приватная функция _request(...) нормализует вызовы, проверяет HTTP-статус,
    распаковывает JSON и поднимает исключения с текстом из Bitrix24, если error присутствует.

Зависимости: requests (pip install requests)

Пример инициализации (вебхук):
    client = Bitrix24Client(base_url="https://portal.bitrix24.ru/rest/1/WEBHOOK/")

Пример инициализации (OAuth-токен):
    client = Bitrix24Client(base_url="https://portal.bitrix24.ru/rest/", oauth_token="XXXX")

Все докстринги ниже содержат краткое описание, список параметров, пример вызова и
пример успешного ответа (200 OK). Примеры ориентированы на облако Bitrix24.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

import requests


__all__ = ["Bitrix24Client", "Bitrix24Error"]


class Bitrix24Error(RuntimeError):
    """Исключение высокого уровня для ошибок Bitrix24 REST."""
    def __init__(self, message: str, *, http_status: Optional[int] = None, b24_code: Optional[str] = None):
        self.http_status = http_status
        self.b24_code = b24_code
        super().__init__(message)


class Bitrix24Client:
    """
    Клиент Bitrix24 REST (облако).

    :param base_url: Базовый URL. Для вебхука — заканчивающийся слэшем path к webhook, напр.:
                     "https://portal.bitrix24.ru/rest/1/WEBHOOK/".
                     Для OAuth — обычно "https://portal.bitrix24.ru/rest/".
    :param oauth_token: Если используется OAuth, укажите токен. Будет добавлен в query как auth=<token>,
                        если явно не указан в params.
    :param timeout: Таймаут HTTP-запросов в секундах.
    :param session: Необязательная requests.Session для переиспользования соединений.
    """
    def __init__(
        self,
        base_url: str,
        oauth_token: Optional[str] = None,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url.endswith("/"):
            base_url += "/"
        self.base_url = base_url
        self.oauth_token = oauth_token
        self.timeout = timeout
        self.session = session or requests.Session()
        self.log = logging.getLogger(self.__class__.__name__)

    # -----------------------------
    # НИЗКОУРОВНЕВЫЙ ВСПОМОГАТЕЛЬНЫЙ МЕТОД
    # -----------------------------
    def _request(
        self,
        method_name: str,
        http_method: str = "GET",
        *,
        params: Optional[MutableMapping[str, Any]] = None,
        data: Optional[MutableMapping[str, Any]] = None,
        json_body: Optional[Any] = None,
        with_metadata: bool = False,
    ) -> Any:
        """
        Универсальный запрос к REST: <base_url>/<method_name>

        Добавляет auth=<oauth_token> в query, если oauth_token задан и auth не передан явно.
        Поддерживает GET/POST. Возвращает значение ключа "result" из ответа при 200 OK и отсутствии error.
        В случае ошибок поднимает Bitrix24Error с деталями.
        """
        url = f"{self.base_url}{method_name}"
        params = dict(params or {})
        if self.oauth_token and "auth" not in params:
            params["auth"] = self.oauth_token

        http_method = http_method.upper()
        if http_method not in {"GET", "POST"}:
            raise ValueError("http_method должен быть GET или POST")

        try:
            if http_method == "GET":
                resp = self.session.get(url, params=params, timeout=self.timeout)
            else:
                # Для сложных структур Bitrix24 часто удобнее JSON-POST
                if json_body is not None:
                    resp = self.session.post(url, params=params, json=json_body, timeout=self.timeout)
                else:
                    resp = self.session.post(url, params=params, data=data, timeout=self.timeout)
        except requests.RequestException as e:
            raise Bitrix24Error(f"Сетевая ошибка: {e}") from e

        http_status = resp.status_code
        text = resp.text
        # Пытаемся распарсить JSON
        try:
            payload = resp.json()
        except ValueError:
            raise Bitrix24Error(f"Ожидался JSON, получено: {text[:500]}", http_status=http_status)

        # Обработка ошибок Bitrix24 (error / error_description) и HTTP
        if http_status != 200 or "error" in payload:
            raise Bitrix24Error(
                payload.get("error_description") or payload.get("error") or f"HTTP {http_status}",
                http_status=http_status,
                b24_code=payload.get("error"),
            )
        if with_metadata:
            return payload
        return payload.get("result")

    # -----------------------------
    # ОБЩИЕ МЕТОДЫ
    # -----------------------------
    def access_name(self, access: Iterable[str]) -> Mapping[str, str]:
        """
        access.name — Получение названий прав доступа (ACL).

        :param access: Список кодов прав ("AU" и т.п.).
        :return: Словарь {код: человекочитаемое название}.

        Пример:
            client.access_name(["AU"])

        Пример успешного ответа (200):
            {"AU": "Все пользователи"}
        """
        params = {}
        # Bitrix24 принимает ACCESS[], ACCESS[0]... ACCESS[n]
        for i, code in enumerate(access):
            params[f"ACCESS[{i}]"] = code
        return self._request("access.name", "GET", params=params)

    def user_current(self) -> Mapping[str, Any]:
        """
        user.current — сведения о текущем пользователе.

        :return: Словарь с полями профиля пользователя, от имени которого выполняется запрос.

        Пример:
            info = client.user_current()
        """
        return self._request("user.current", "GET")

    def user_get(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        fetch_all: bool = True,
    ) -> List[Mapping[str, Any]]:
        """
        user.get — получить пользователей Bitrix24.

        :param filter: Фильтры (например, {"ACTIVE": "true"}).
        :param select: Набор полей для выборки.
        :param order: Сортировка ({"ID": "ASC"}).
        :param start: Смещение пагинации Bitrix24.
        :param fetch_all: Собирать все страницы результата.
        :return: Список словарей с данными пользователей.
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("user.get", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from user.get", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break
        return collected

        # app.info
    def app_info(self) -> Mapping[str, Any]:
        """
        app.info — Информация о приложении и портале.

        :return: Объект с данными о приложении/портале (ID, CODE, MEMBER_ID, язык, часовой пояс и т.д.).

        Пример:
            info = client.app_info()
        """
        return self._request("app.info", "GET")

    def app_option_get(self, option: Optional[str] = None) -> Any:
        """
        app.option.get — Получить сохранённые настройки приложения.

        :param option: Ключ настройки. Если None — вернуть все настройки.
        :return: Значение по ключу или объект всех опций.

        Пример:
            client.app_option_get("mySetting")
            client.app_option_get()

        Пример успешного ответа (200):
            # один ключ
            "some value"
            # все ключи
            {"mySetting": "some value", "anotherSetting": "123"}
        """
        params: Dict[str, Any] = {}
        if option:
            params["option"] = option
        return self._request("app.option.get", "GET", params=params)

    def app_option_set(self, options: Mapping[str, Any]) -> bool:
        """
        app.option.set — Сохранить настройки приложения.

        :param options: Ассоциативный объект {ключ: значение}.
        :return: True при успехе.

        Пример:
            client.app_option_set({"data": "value", "data2": "value2"})

        Пример успешного ответа (200):
            True
        """
        body = {"options": dict(options)}
        return bool(self._request("app.option.set", "POST", json_body=body))

    def batch(self, cmd: Mapping[str, Union[str, Mapping[str, Any]]], *, halt: int = 0) -> Mapping[str, Any]:
        """
        batch — Пакетное выполнение до 50 REST-команд.

        :param cmd: Словарь подзапросов {"q1": "crm.contact.get?id=2", "q2": {...}}.
                    Допускается строка с query либо объект {"method": "...", "params": {...}}.
        :param halt: 0 или 1. При 1 выполнение прерывается на первой ошибке.
        :return: Полная структура результата batch (result, result_error, result_time и т.д.).

        Пример:
            result = client.batch({"q1": "crm.contact.get?id=2", "q2": "crm.contact.get?id=4"})
        """
        # Bitrix24 понимает и form-encoding с cmd[q1]=..., и JSON c {"cmd": {...}}
        body = {"cmd": cmd, "halt": halt}
        return self._request("batch", "POST", json_body=body)

    # -----------------------------
    # События
    # -----------------------------
    def event_bind(
        self,
        event: str,
        handler: str,
        *,
        auth_type: Optional[str] = None,
        additional_fields: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """
        event.bind — регистрация обработчика события.

        :param event: Код события (например, onCrmLeadAdd).
        :param handler: URL обработчика, который должен получать уведомления.
        :param auth_type: Тип авторизации обработчика (webhook, event, server и т.д.).
        :param additional_fields: Дополнительные поля Bitrix24 (comment, type и пр.).
        :return: True при успешной регистрации.
        """
        payload: Dict[str, Any] = {"event": event, "handler": handler}
        if auth_type:
            payload["auth_type"] = auth_type
        if additional_fields:
            payload.update(dict(additional_fields))
        return bool(self._request("event.bind", "POST", data=payload))

    def event_get(self, filter: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        """
        event.get — список зарегистрированных обработчиков событий.

        :param filter: Необязательные критерии (event, handler).
        :return: Перечень словарей с событиями и URL обработчиков.
        """
        params = dict(filter or {})
        return self._request("event.get", "GET", params=params or None)

    def event_unbind(
        self,
        event: str,
        handler: str,
        *,
        auth_type: Optional[str] = None,
    ) -> bool:
        """
        event.unbind — удаление обработчика события.

        :param event: Код события, от которого нужно отписаться.
        :param handler: URL обработчика, который требуется отвязать.
        :param auth_type: Тип авторизации (если использовался при event.bind).
        :return: True при успешной отвязке.
        """
        payload: Dict[str, Any] = {"event": event, "handler": handler}
        if auth_type:
            payload["auth_type"] = auth_type
        return bool(self._request("event.unbind", "POST", data=payload))

    # -----------------------------
    # CRM — сделки и справочники
    # -----------------------------

    def crm_deal_add(
        self,
        fields: Mapping[str, Any],
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        crm.deal.add — создание новой сделки.

        :param fields: Поля сделки (TITLE, STAGE_ID, CATEGORY_ID, CONTACT_ID и т.п.).
        :param params: Дополнительные параметры (например, {"REGISTER_SONET_EVENT": "Y"}).
        :return: Идентификатор созданной сделки.
        """
        body: Dict[str, Any] = {"fields": dict(fields)}
        if params:
            body["params"] = dict(params)
        return int(self._request("crm.deal.add", "POST", json_body=body))

    def crm_deal_get(
        self,
        id: Union[int, str],
        *,
        select: Optional[Iterable[str]] = None,
    ) -> Mapping[str, Any]:
        """
        crm.deal.get — получение сделки по ID.

        :param id: Идентификатор сделки.
        :param select: Явный список полей, которые нужно вернуть.
        :return: Словарь с данными сделки.
        """
        body: Dict[str, Any] = {"id": int(id)}
        if select:
            body["select"] = list(select)
        return self._request("crm.deal.get", "POST", json_body=body)

    def crm_deal_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        *,
        fetch_all: bool = True,
    ) -> Union[
        List[Mapping[str, Any]],
        Tuple[List[Mapping[str, Any]], Optional[int], Optional[int]],
    ]:
        """
        crm.deal.list — выборка сделок с фильтрацией.

        :param filter: Условия отбора (TITLE, STAGE_ID, >DATE_CREATE и т.п.).
        :param select: Поля сделки, которые необходимо вернуть.
        :param order: Сортировка (например, {"ID": "DESC"}).
        :param start: Смещение для постраничной выборки Bitrix24.
        :param fetch_all: Получать все страницы (True) или вернуть только первую (False).
        :return: Список сделок. При fetch_all=False дополнительно возвращает (items, next, total).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        total: Optional[int] = None

        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("crm.deal.list", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from crm.deal.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            total = payload.get("total", total)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start, total

    def crm_deal_update(
        self,
        id: Union[int, str],
        fields: Mapping[str, Any],
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """
        crm.deal.update — обновление существующей сделки.

        :param id: Идентификатор сделки.
        :param fields: Поля для изменения (TITLE, STAGE_ID, OPPORTUNITY и др.).
        :param params: Дополнительные параметры (REGISTER_SONET_EVENT и пр.).
        :return: True, если обновление прошло успешно.
        """
        body: Dict[str, Any] = {"id": int(id), "fields": dict(fields)}
        if params:
            body["params"] = dict(params)
        return bool(self._request("crm.deal.update", "POST", json_body=body))

    def crm_deal_category_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        order: Optional[Mapping[str, Any]] = None,
    ) -> List[Mapping[str, Any]]:
        """
        crm.deal.category.list — перечень направлений сделок (устаревший метод).

        :param filter: Критерии отбора категорий.
        :param order: Сортировка категорий.
        :return: Список направлений (воронок) CRM.
        """
        body: Dict[str, Any] = {}
        if filter:
            body["filter"] = dict(filter)
        if order:
            body["order"] = dict(order)
        return self._request("crm.deal.category.list", "POST", json_body=body or None)

    def crm_deal_category_stage_list(
        self,
        category_id: Union[int, str],
        *,
        filter: Optional[Mapping[str, Any]] = None,
    ) -> List[Mapping[str, Any]]:
        """
        crm.deal.category.stage.list — стадии сделок внутри направления (устаревший метод).

        :param category_id: Идентификатор категории/воронки.
        :param filter: Дополнительная фильтрация стадий.
        :return: Список стадий с кодами STATUS_ID и их атрибутами.
        """
        body: Dict[str, Any] = {"id": int(category_id)}
        if filter:
            body["filter"] = dict(filter)
        return self._request("crm.deal.category.stage.list", "POST", json_body=body)

    def crm_status_list(
        self,
        entity_id: str,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        fetch_all: bool = True,
    ) -> Union[List[Mapping[str, Any]], Tuple[List[Mapping[str, Any]], Optional[int]]]:
        """
        crm.status.list — элементы CRM-справочников (статусы, отрасли и т.п.).

        :param entity_id: Код справочника (например, DEAL_STAGE, INDUSTRY).
        :param filter: Дополнительные критерии (STATUS_ID, CATEGORY_ID и др.).
        :param select: Список полей, которые требуется вернуть.
        :param order: Параметры сортировки.
        :param start: Смещение для постраничного вывода.
        :param fetch_all: Получать все страницы автоматически (True) или только одну (False).
        :return: Список элементов справочника. При fetch_all=False возвращает (items, next).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start

        while True:
            body: Dict[str, Any] = {"filter": {"ENTITY_ID": entity_id}}
            if filter:
                body["filter"].update(dict(filter))
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("crm.status.list", "POST", json_body=body, with_metadata=True)
            items = payload.get("result") or []
            if not isinstance(items, list):
                raise Bitrix24Error("Unexpected response format from crm.status.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(items)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start

    def crm_activity_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        *,
        fetch_all: bool = True,
    ) -> Union[
        List[Mapping[str, Any]],
        Tuple[List[Mapping[str, Any]], Optional[int], Optional[int]],
    ]:
        """
        crm.activity.list — список дел/активностей CRM.

        :param filter: Критерии отбора активностей (OWNER_TYPE_ID, COMPLETED и др.).
        :param select: Поля активностей для возврата.
        :param order: Сортировка (например, {"ID": "DESC"}).
        :param start: Смещение для постраничной загрузки.
        :param fetch_all: True — получить все страницы; False — только одну страницу.
        :return: Список активностей. При fetch_all=False возвращает (items, next, total).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        total: Optional[int] = None

        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("crm.activity.list", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from crm.activity.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            total = payload.get("total", total)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start, total

    def crm_activity_add(self, fields: Mapping[str, Any]) -> int:
        """
        crm.activity.add — создание новой активности (дела) CRM.

        :param fields: Поля активности (OWNER_TYPE_ID, OWNER_ID, TYPE_ID, SUBJECT и др.).
        :return: Идентификатор созданной активности.
        """
        body = {"fields": dict(fields)}
        return int(self._request("crm.activity.add", "POST", json_body=body))

    def crm_timeline_comment_add(
        self,
        entity_type: str,
        entity_id: Union[int, str],
        comment: str,
        *,
        additional_fields: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        crm.timeline.comment.add — добавление комментария в таймлайн CRM.

        :param entity_type: Тип сущности (deal, lead, contact, company).
        :param entity_id: Идентификатор элемента CRM.
        :param comment: Текст комментария.
        :param additional_fields: Дополнительные поля (FILES, AUTHOR_ID и пр.).
        :return: Идентификатор созданного комментария.
        """
        fields_payload: Dict[str, Any] = {
            "ENTITY_TYPE": entity_type,
            "ENTITY_ID": int(entity_id),
            "COMMENT": comment,
        }
        if additional_fields:
            fields_payload.update(dict(additional_fields))
        body = {"fields": fields_payload}
        return int(self._request("crm.timeline.comment.add", "POST", json_body=body))

    def crm_contact_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        *,
        fetch_all: bool = True,
    ) -> Union[
        List[Mapping[str, Any]],
        Tuple[List[Mapping[str, Any]], Optional[int], Optional[int]],
    ]:
        """
        crm.contact.list — список контактов CRM.

        :param filter: Фильтр по полям контактов (NAME, PHONE, COMPANY_ID и т.п.).
        :param select: Набор полей для возврата (например, ["ID", "NAME", "PHONE"]).
        :param order: Сортировка ({"LAST_NAME": "ASC"}).
        :param start: Смещение постраничной выдачи.
        :param fetch_all: True — собрать все страницы, False — только первую.
        :return: Список контактов. При fetch_all=False возвращает (items, next, total).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        total: Optional[int] = None

        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("crm.contact.list", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from crm.contact.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            total = payload.get("total", total)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start, total

    def crm_contact_get(
        self,
        id: Union[int, str],
        *,
        select: Optional[Iterable[str]] = None,
    ) -> Mapping[str, Any]:
        """
        crm.contact.get — получение контакта по ID.

        :param id: Идентификатор контакта.
        :param select: Список полей для возврата.
        :return: Словарь с данными контакта.
        """
        body: Dict[str, Any] = {"id": int(id)}
        if select:
            body["select"] = list(select)
        return self._request("crm.contact.get", "POST", json_body=body)

    def crm_company_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        *,
        fetch_all: bool = True,
    ) -> Union[
        List[Mapping[str, Any]],
        Tuple[List[Mapping[str, Any]], Optional[int], Optional[int]],
    ]:
        """
        crm.company.list — список компаний CRM.

        :param filter: Критерии отбора (TITLE, COMPANY_TYPE, HAS_PHONE и др.).
        :param select: Поля компании, которые нужно получить.
        :param order: Порядок сортировки результатов.
        :param start: Смещение для постраничного вывода.
        :param fetch_all: True — получать все страницы, False — только первую.
        :return: Список компаний. При fetch_all=False возвращает (items, next, total).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        total: Optional[int] = None

        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("crm.company.list", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from crm.company.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            total = payload.get("total", total)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start, total

    def crm_company_get(
        self,
        id: Union[int, str],
        *,
        select: Optional[Iterable[str]] = None,
    ) -> Mapping[str, Any]:
        """
        crm.company.get — получение компании по ID.

        :param id: Идентификатор компании.
        :param select: Набор полей для возврата.
        :return: Словарь с данными компании.
        """
        body: Dict[str, Any] = {"id": int(id)}
        if select:
            body["select"] = list(select)
        return self._request("crm.company.get", "POST", json_body=body)

    # -----------------------------
    # Задачи и комментарии
    # -----------------------------

    def tasks_task_add(
        self,
        fields: Mapping[str, Any],
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        tasks.task.add — создание новой задачи.

        :param fields: Поля задачи (TITLE, RESPONSIBLE_ID, DEADLINE и др.).
        :param params: Дополнительные параметры (например, {"UPDATE_DEADLINE": "Y"}).
        :return: Полный словарь задачи, возвращаемый Bitrix24.
        """
        body: Dict[str, Any] = {"fields": dict(fields)}
        if params:
            body["params"] = dict(params)
        result = self._request("tasks.task.add", "POST", json_body=body)
        if isinstance(result, dict) and "task" in result and isinstance(result["task"], dict):
            return result["task"]
        return result

    def tasks_task_update(
        self,
        task_id: Union[int, str],
        fields: Mapping[str, Any],
        *,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """
        tasks.task.update — обновление существующей задачи.

        :param task_id: Идентификатор задачи.
        :param fields: Поля для изменения.
        :param params: Дополнительные параметры Bitrix24.
        :return: Обновлённая задача из ответа REST.
        """
        body: Dict[str, Any] = {"taskId": int(task_id), "fields": dict(fields)}
        if params:
            body["params"] = dict(params)
        result = self._request("tasks.task.update", "POST", json_body=body)
        if isinstance(result, dict) and "task" in result and isinstance(result["task"], dict):
            return result["task"]
        return result

    def tasks_task_list(
        self,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        *,
        fetch_all: bool = True,
    ) -> Union[
        List[Mapping[str, Any]],
        Tuple[List[Mapping[str, Any]], Optional[int], Optional[int]],
    ]:
        """
        tasks.task.list — выборка задач по фильтру.

        :param filter: Фильтр задач (RESPONSIBLE_ID, STATUS, >=DEADLINE и др.).
        :param select: Поля задачи, которые нужно вернуть.
        :param order: Сортировка ({"ID": "DESC"} и т.п.).
        :param start: Смещение постраничного вывода.
        :param fetch_all: True — собрать все страницы; False — вернуть первую страницу вместе с next и total.
        :return: Список задач. При fetch_all=False возвращает (items, next, total).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start
        total: Optional[int] = None

        while True:
            body: Dict[str, Any] = {}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("tasks.task.list", "POST", json_body=body or None, with_metadata=True)
            payload_result = payload.get("result") or {}
            if not isinstance(payload_result, dict):
                raise Bitrix24Error("Unexpected response format from tasks.task.list", b24_code="UNEXPECTED_RESULT")
            tasks = payload_result.get("tasks") or []
            if not isinstance(tasks, list):
                raise Bitrix24Error("Unexpected tasks array in tasks.task.list", b24_code="UNEXPECTED_RESULT")
            collected.extend(tasks)
            total = payload.get("total", payload_result.get("total", total))
            next_start = payload_result.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start, total

    def task_commentitem_add(
        self,
        task_id: Union[int, str],
        message: str,
        *,
        additional_fields: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        task.commentitem.add — добавление комментария в задачу.

        :param task_id: Идентификатор задачи.
        :param message: Текст комментария (поддерживает BBCode).
        :param additional_fields: Дополнительные атрибуты комментария (FILES, AUTHOR_ID и др.).
        :return: Идентификатор созданного комментария.
        """
        fields_payload: Dict[str, Any] = {"POST_MESSAGE": message}
        if additional_fields:
            fields_payload.update(dict(additional_fields))
        body = {"TASK_ID": int(task_id), "fields": fields_payload}
        return int(self._request("task.commentitem.add", "POST", json_body=body))

    def task_checklistitem_add(
        self,
        task_id: Union[int, str],
        title: str,
        *,
        is_complete: Optional[bool] = None,
        sort_index: Optional[int] = None,
        parent_id: Optional[int] = None,
        responsible_id: Optional[int] = None,
        additional_fields: Optional[Mapping[str, Any]] = None,
    ) -> int:
        """
        task.checklistitem.add — добавление пункта чек-листа задачи.

        :param task_id: Идентификатор задачи.
        :param title: Текст пункта чек-листа.
        :param is_complete: Признак выполнения пункта.
        :param sort_index: Индекс сортировки (чем меньше, тем выше).
        :param parent_id: Родительский пункт для вложенности.
        :param responsible_id: Ответственный за пункт чек-листа.
        :param additional_fields: Прочие параметры Bitrix24.
        :return: Идентификатор созданного пункта чек-листа.
        """
        fields_payload: Dict[str, Any] = {"TITLE": title}
        if is_complete is not None:
            fields_payload["IS_COMPLETE"] = "Y" if is_complete else "N"
        if sort_index is not None:
            fields_payload["SORT_INDEX"] = int(sort_index)
        if parent_id is not None:
            fields_payload["PARENT_ID"] = int(parent_id)
        if responsible_id is not None:
            fields_payload["RESPONSIBLE_ID"] = int(responsible_id)
        if additional_fields:
            fields_payload.update(dict(additional_fields))

        body = {"TASK_ID": int(task_id), "fields": fields_payload}
        return int(self._request("task.checklistitem.add", "POST", json_body=body))

    # -----------------------------
    # Рабочие группы (проекты)
    # -----------------------------

    def sonet_group_get(
        self,
        group_id: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
        *,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        order: Optional[Mapping[str, Any]] = None,
        start: Optional[int] = None,
        fetch_all: bool = True,
    ) -> Union[List[Mapping[str, Any]], Tuple[List[Mapping[str, Any]], Optional[int]]]:
        """
        sonet.group.get — получение рабочих групп (проектов).

        :param group_id: Идентификатор или последовательность ID групп.
        :param filter: Фильтр по полям группы (NAME, OWNER_ID, PROJECT и др.).
        :param select: Список полей для возврата.
        :param order: Сортировка ({"NAME": "ASC"} и др.).
        :param start: Смещение постраничной загрузки.
        :param fetch_all: True — вернуть все страницы; False — только первую.
        :return: Список групп. При fetch_all=False возвращает (items, next).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start

        while True:
            body: Dict[str, Any] = {}
            if group_id is not None:
                if isinstance(group_id, (list, tuple, set)):
                    ids = [int(value) for value in group_id]
                else:
                    ids = [int(group_id)]
                body["ID"] = ids if len(ids) != 1 else ids[0]
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if order:
                body["order"] = dict(order)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("sonet.group.get", "POST", json_body=body or None, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from sonet.group.get", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start

    def sonet_group_user_get(
        self,
        group_id: Union[int, str],
        *,
        filter: Optional[Mapping[str, Any]] = None,
        select: Optional[Iterable[str]] = None,
        start: Optional[int] = None,
        fetch_all: bool = True,
    ) -> Union[List[Mapping[str, Any]], Tuple[List[Mapping[str, Any]], Optional[int]]]:
        """
        sonet.group.user.get — участники рабочей группы.

        :param group_id: Идентификатор группы.
        :param filter: Фильтр по участникам (ROLE, AUTO_MEMBER и др.).
        :param select: Дополнительные поля для возврата.
        :param start: Смещение постраничной выдачи.
        :param fetch_all: True — вернуть всех участников; False — только первую страницу.
        :return: Список участников. При fetch_all=False возвращает (items, next).
        """
        collected: List[Mapping[str, Any]] = []
        next_start: Optional[int] = start

        while True:
            body: Dict[str, Any] = {"GROUP_ID": int(group_id)}
            if filter:
                body["filter"] = dict(filter)
            if select:
                body["select"] = list(select)
            if next_start is not None:
                body["start"] = int(next_start)

            payload = self._request("sonet.group.user.get", "POST", json_body=body, with_metadata=True)
            result = payload.get("result") or []
            if not isinstance(result, list):
                raise Bitrix24Error("Unexpected response format from sonet.group.user.get", b24_code="UNEXPECTED_RESULT")
            collected.extend(result)
            next_start = payload.get("next")
            if not fetch_all or next_start is None:
                break

        if fetch_all:
            return collected
        return collected, next_start
    # КАЛЕНДАРЬ
    # -----------------------------
    def calendar_accessibility_get(self, users: Iterable[int], since: str, until: str) -> Mapping[str, Any]:
        """
        calendar.accessibility.get — Занятость пользователей.

        :param users: Идентификаторы пользователей.
        :param since: Начало периода, ISO "YYYY-MM-DDThh:mm:ss".
        :param until: Конец периода, ISO "YYYY-MM-DDThh:mm:ss".
        :return: Словарь {userId: [события...]}

        Пример:
            client.calendar_accessibility_get([1,2,3], "2025-01-18T09:00:00", "2025-01-18T18:00:00")
        """
        params: Dict[str, Any] = {"from": since, "to": until}
        for i, uid in enumerate(users):
            params[f"users[{i}]"] = int(uid)
        return self._request("calendar.accessibility.get", "GET", params=params)

    def calendar_event_add(self, fields: Mapping[str, Any]) -> int:
        """
        calendar.event.add — Создать событие/встречу.

        :param fields: Поля события. Частые:
            type ("user"/"group"/"company"), ownerId, name, description, from, to,
            skip_time, location, attendees, is_meeting ("Y"/"N"),
            host, meeting{notify,allowInvite,hideGuests,...}, remind[{type,count}],
            section, importance, color, text_color, rrule.
        :return: ID созданного события.

        Пример:
            event_id = client.calendar_event_add({
                "type": "user", "ownerId": 1, "name": "Встреча",
                "from": "2025-01-15T10:00:00", "to": "2025-01-15T11:00:00",
                "is_meeting": "Y", "attendees": [2,3,4],
                "remind": [{"type":"min", "count":15}]
            })
        """
        return int(self._request("calendar.event.add", "POST", json_body=dict(fields)))

    def calendar_event_get(
        self,
        *,
        type: Optional[str] = None,
        ownerId: Optional[int] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        id: Optional[int] = None,
    ) -> Any:
        """
        calendar.event.get — Список событий или одно событие по id.

        :param type: Тип календаря ("user"/"group"/"company") для выборки списка.
        :param ownerId: Владелец календаря.
        :param since: Начало периода (ISO).
        :param until: Конец периода (ISO).
        :param id: Если указан — вернёт конкретное событие.
        :return: Массив событий либо объект события.

        Пример (список за период):
            client.calendar_event_get(type="user", ownerId=1,
                                      since="2025-01-01T00:00:00", until="2025-01-31T23:59:59")
        Пример (одно событие):
            client.calendar_event_get(id=200)
        """
        params: Dict[str, Any] = {}
        if id is not None:
            params["id"] = int(id)
        else:
            if type is not None:
                params["type"] = type
            if ownerId is not None:
                params["ownerId"] = int(ownerId)
            if since is not None:
                params["from"] = since
            if until is not None:
                params["to"] = until
        return self._request("calendar.event.get", "GET", params=params)

    def calendar_event_get_nearest(self, *, type: str, ownerId: int, days: Optional[int] = None) -> Any:
        """
        calendar.event.get.nearest — Ближайшие события.

        :param type: Тип календаря.
        :param ownerId: Владелец.
        :param days: Горизонт в днях. По умолчанию ~60 в облаке.
        :return: Массив ближайших событий.

        Пример:
            client.calendar_event_get_nearest(type="user", ownerId=1, days=7)
        """
        params: Dict[str, Any] = {"type": type, "ownerId": int(ownerId)}
        if days is not None:
            params["days"] = int(days)
        return self._request("calendar.event.get.nearest", "GET", params=params)

    def calendar_event_getbyid(self, id: int) -> Mapping[str, Any]:
        """
        calendar.event.getbyid — Событие по ID.

        :param id: Идентификатор события.
        :return: Полный объект события.

        Пример:
            client.calendar_event_getbyid(200)
        """
        return self._request("calendar.event.getbyid", "GET", params={"id": int(id)})

    def calendar_event_update(self, id: int, fields: Mapping[str, Any]) -> bool:
        """
        calendar.event.update — Обновить событие.

        :param id: Идентификатор события.
        :param fields: Поля для изменения. Для повторяющихся событий используйте
                       recurrence_mode ("this"/"next"/"all") и current_date_from.
        :return: True при успехе.

        Пример:
            client.calendar_event_update(200, {
                "name": "Встреча (перенесено)",
                "from": "2025-01-15T11:00:00", "to": "2025-01-15T12:00:00"
            })
        """
        body = dict(fields)
        body["id"] = int(id)
        return bool(self._request("calendar.event.update", "POST", json_body=body))

    def calendar_event_delete(self, id: int) -> bool:
        """
        calendar.event.delete — Удалить событие.

        :param id: Идентификатор события.
        :return: True при успехе.

        Пример:
            client.calendar_event_delete(201)
        """
        return bool(self._request("calendar.event.delete", "GET", params={"id": int(id)}))

    def calendar_meeting_params_set(self, id: int, meeting: Mapping[str, Any]) -> bool:
        """
        calendar.meeting.params.set — Изменить параметры встречи.

        :param id: Идентификатор встречи.
        :param meeting: Параметры: notify(True/False), reinvite, allowInvite, hideGuests и т.п.
        :return: True при успехе.

        Пример:
            client.calendar_meeting_params_set(200, {"allowInvite": False, "hideGuests": True})
        """
        body = {"id": int(id), "meeting": dict(meeting)}
        return bool(self._request("calendar.meeting.params.set", "POST", json_body=body))

    def calendar_meeting_status_get(self, id: int) -> List[Mapping[str, Any]]:
        """
        calendar.meeting.status.get — Статусы участников встречи.

        :param id: Идентификатор встречи.
        :return: Список объектов {USER_ID, STATUS}.

        Пример:
            client.calendar_meeting_status_get(200)
        """
        return self._request("calendar.meeting.status.get", "GET", params={"id": int(id)})

    def calendar_meeting_status_set(self, id: int, status: str) -> bool:
        """
        calendar.meeting.status.set — Установить статус участия.

        :param id: Идентификатор встречи.
        :param status: "Y" — принять, "N" — отклонить.
        :return: True при успехе.

        Пример:
            client.calendar_meeting_status_set(200, "N")
        """
        params = {"id": int(id), "status": status}
        return bool(self._request("calendar.meeting.status.set", "GET", params=params))

    # -----------------------------
    # РЕСУРСЫ КАЛЕНДАРЯ
    # -----------------------------
    def calendar_resource_add(self, fields: Mapping[str, Any]) -> int:
        """
        calendar.resource.add — Создать ресурс для бронирования.

        :param fields: Поля, напр.: NAME, DESCRIPTION, CAPACITY, ACTIVE("Y"/"N").
        :return: ID созданного ресурса.

        Пример:
            rid = client.calendar_resource_add({
                "NAME": "Переговорная А",
                "DESCRIPTION": "Зал на 10 человек",
                "CAPACITY": 10,
                "ACTIVE": "Y"
            })
        """
        body = {"fields": dict(fields)}
        return int(self._request("calendar.resource.add", "POST", json_body=body))

    def calendar_resource_update(self, id: int, fields: Mapping[str, Any]) -> bool:
        """
        calendar.resource.update — Обновить ресурс.

        :param id: ID ресурса.
        :param fields: Изменяемые поля (NAME, DESCRIPTION, CAPACITY, ACTIVE).
        :return: True при успехе.

        Пример:
            client.calendar_resource_update(5, {"NAME": "Переговорная (ремонт)", "ACTIVE": "N"})
        """
        body = {"id": int(id), "fields": dict(fields)}
        return bool(self._request("calendar.resource.update", "POST", json_body=body))

    def calendar_resource_delete(self, id: int) -> bool:
        """
        calendar.resource.delete — Удалить ресурс.

        :param id: ID ресурса.
        :return: True при успехе.

        Пример:
            client.calendar_resource_delete(5)
        """
        return bool(self._request("calendar.resource.delete", "GET", params={"id": int(id)}))

    def calendar_resource_list(self, filter: Optional[Mapping[str, Any]] = None) -> List[Mapping[str, Any]]:
        """
        calendar.resource.list — Список ресурсов.

        :param filter: Необязательный фильтр, например {"ACTIVE":"Y"}.
        :return: Массив объектов ресурсов.

        Пример:
            client.calendar_resource_list()
            client.calendar_resource_list({"ACTIVE": "Y"})
        """
        params: Dict[str, Any] = {}
        if filter:
            # Передаём как JSON в POST для предсказуемости
            return self._request("calendar.resource.list", "POST", json_body={"filter": dict(filter)})
        return self._request("calendar.resource.list", "GET", params=params)

    def calendar_resource_booking_list(self, filter: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        """
        calendar.resource.booking.list — Список бронирований ресурсов по фильтру.

        :param filter: Критерии, напр.: {"RESOURCE_ID": 6, "FROM": "...", "TO": "...", "USER_ID": 1}.
        :return: Массив событий-бронирований.

        Пример:
            client.calendar_resource_booking_list({
                "RESOURCE_ID": 6,
                "FROM": "2025-09-01T00:00:00",
                "TO": "2025-09-30T23:59:59"
            })
        """
        body = {"filter": dict(filter)}
        return self._request("calendar.resource.booking.list", "POST", json_body=body)


if __name__ == "__main__":
    # Небольшая демонстрация построения клиента и печати версии портала.
    # ВАЖНО: замените значения на свои. Вызовы ниже выполнят реальные HTTP-запросы.
    # Пример с вебхуком:
    # client = Bitrix24Client(base_url="https://portal.bitrix24.ru/rest/1/WEBHOOK/")
    # print(client.app_info())

    # Пример с OAuth-токеном:
    # client = Bitrix24Client(base_url="https://portal.bitrix24.ru/rest/", oauth_token="OAUTH_TOKEN")
    # print(client.access_name(["AU"]))
    pass
