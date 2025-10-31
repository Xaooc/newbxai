# 02. Методы Bitrix24

## Базовый вебхук
* Все вызовы выполняются через URL `https://portal.magnitmedia.ru/rest/132/1s0mz4mw8d42bfvk/`.
* URL может быть переопределён переменной окружения `BITRIX_WEBHOOK_URL`, но по умолчанию используется указанный портал.
* Модель получает только логические имена методов и параметры, сам URL скрывается.

## Разрешённые методы (кратко)
| Метод | Назначение | HTTP по умолчанию | Обязательные параметры | Рискованные поля/действия |
|-------|------------|-------------------|------------------------|---------------------------|
| `user.current` | Информация о владельце вебхука | GET | — | — |
| `user.get` | Поиск пользователей | GET | `filter`/`order`/`select` по необходимости | Нет |
| `crm.contact.list` | Список контактов | GET | `filter` (опционально) | Нет |
| `crm.contact.get` | Контакт по ID | GET | `id` | Нет |
| `crm.company.list` | Список компаний | GET | `filter` (опционально) | Нет |
| `crm.company.get` | Компания по ID | GET | `id` | Нет |
| `crm.deal.list` | Список сделок | GET | `filter`/`select`/`start` | Нет |
| `crm.deal.get` | Сделка по ID | GET | `id` | Нет |
| `crm.deal.add` | Создание сделки | POST | `fields.TITLE` минимум | `OPPORTUNITY`, `ASSIGNED_BY_ID`, `STAGE_ID`, `CATEGORY_ID` |
| `crm.deal.update` | Обновление сделки | POST | `id`, `fields` | Любое изменение суммы, стадии, ответственного |
| `crm.deal.category.list` | Список направлений продаж | GET | — | Нет |
| `crm.deal.category.stage.list` | Список стадий по направлению | GET | `id` или `categoryId` | Нет |
| `crm.status.list` | Элементы справочников CRM | GET | `filter.ENTITY_ID` | Нет |
| `crm.activity.list` | Список активностей | GET | `filter.OWNER_TYPE_ID` (рекомендуется) | Нет |
| `crm.activity.add` | Создание активности | POST | `fields`: `OWNER_TYPE_ID`, `OWNER_ID`, `TYPE_ID`, `SUBJECT` | `RESPONSIBLE_ID`, `DEADLINE` |
| `crm.timeline.comment.add` | Комментарий в таймлайн | POST | `fields`: `ENTITY_ID`, `ENTITY_TYPE`, `COMMENT` | Содержимое текста |
| `tasks.task.add` | Создание задачи | POST | `fields`: `TITLE`, `RESPONSIBLE_ID`, `DESCRIPTION` | `RESPONSIBLE_ID`, `DEADLINE` |
| `tasks.task.update` | Обновление задачи | POST | `taskId` (`id`), `fields` | Ответственный, дедлайн |
| `tasks.task.list` | Список задач | GET | `filter` (опционально) | Нет |
| `task.commentitem.add` | Комментарий к задаче | POST | `taskId`/`TASK_ID`, `fields.POST_MESSAGE` | Текст |
| `task.checklistitem.add` | Пункт чек-листа | POST | `taskId`/`TASK_ID`, `fields.TITLE` | Нет |
| `sonet.group.get` | Информация о рабочих группах | GET | `ID`/`filter` (опционально) | Нет |
| `sonet.group.user.get` | Список участников группы | GET | `GROUP_ID` | Нет |
| `batch` | Пакетный вызов до 50 команд | POST | `cmd` (dict/array), `halt` (опционально) | Наследует ограничения вложенных методов |
| `event.bind` | Привязка обработчика события | POST | `event`, `handler` | Требует подтверждения; меняет конфигурацию |
| `event.get` | Список подписок | GET | — | Нет |
| `event.unbind` | Удаление обработчика события | POST | `event`, `handler` | Требует подтверждения; меняет конфигурацию |
## Детализация по методам

### user.current
* Возвращает объект текущего пользователя, от чьего имени выполняется вебхук.
* Используется для диагностики прав и отображения имени в отчётах.
* Пример ответа: `{"ID": "1", "NAME": "Антон", ...}`.

### user.get
* Поддерживает параметры `filter`, `order`, `select`.
* При поиске по имени используем фильтр `{"NAME": "Олег"}`; допускаются операторы (`%`, `!`).
* Для массовых выборок рекомендуется ограничивать поля `select`, чтобы уменьшить размер ответа.

### crm.contact.list / crm.contact.get
* HTTP-метод по умолчанию — `GET`. Запросы допускают `POST`, но предпочтительнее `GET` для чтения.
* Поля `PHONE` и `EMAIL` множественные — в `select` явно указывать `"PHONE"`, `"EMAIL"`.
* Поиск по номеру: `{"PHONE": "+74951234567"}`. Допускаются операторы (`%`, `!`, `>=` и др.).
* Ответ списка возвращает максимум 50 контактов за вызов и поле `total` при наличии следующей страницы. Для получения телефонов и e-mail Bitrix возвращает массив объектов с `VALUE` и `VALUE_TYPE`.
* Метод `get` возвращает полный объект контакта, включая пользовательские поля `UF_*`, множественные поля коммуникаций и флаги (`HAS_PHONE`, `HAS_EMAIL`).

### crm.company.list / crm.company.get
* По умолчанию используется `GET`. Можно передавать фильтры по типу (`COMPANY_TYPE`), отрасли (`INDUSTRY`), наличию телефона (`HAS_PHONE`) и т.д.
* Для получения телефонов и e-mail необходим `select`: `"PHONE"`, `"EMAIL"`.
* Метод `get` возвращает все поля компании, включая множественные и пользовательские (`UF_*`), а также финансовые показатели, если они заполнены.

### crm.deal.list
* HTTP-метод по умолчанию — `GET`, допускается `POST`.
* Максимум 50 записей за вызов; используем `start` (0, 50, 100, ...) для постраничного чтения. Поле `next` в ответе указывает смещение следующей страницы.
* Частые фильтры: `{"ASSIGNED_BY_ID": <id>}`, `{"%TITLE": "Продажа"}`, `{"CATEGORY_ID": 5}`, `{"CURRENCY_ID": "RUB"}`.
* В `select` можно запросить `"*"` для всех полей или конкретные (`"ID"`, `"TITLE"`, `"STAGE_ID"`, `"OPPORTUNITY"`, `"CURRENCY_ID"`).
* Ответ содержит массив сделок, каждая запись включает основные поля и UTM-метки при их наличии.

### crm.deal.get
* Возвращает объект сделки с полями `ID`, `TITLE`, `TYPE_ID`, `STAGE_ID`, `OPPORTUNITY`, `CURRENCY_ID`, `COMPANY_ID`, `CONTACT_ID`, датами и множеством системных флагов (`OPENED`, `CLOSED`, `STAGE_SEMANTIC_ID`).
* Используется для проверки перед обновлением: позволяет подтвердить текущую сумму, ответственного и стадии.

### crm.deal.add
* Минимально необходимо поле `fields.TITLE`. Без `STAGE_ID` и `CATEGORY_ID` сделка создаётся на стартовой стадии воронки 0.
* Допустимые поля: `TYPE_ID`, `CATEGORY_ID`, `STAGE_ID`, `COMPANY_ID`, `CONTACT_ID`, `ASSIGNED_BY_ID`, `OPPORTUNITY`, `CURRENCY_ID`, даты (`BEGINDATE`, `CLOSEDATE`) и любые пользовательские поля.
* `params.REGISTER_SONET_EVENT` (по умолчанию `"Y"`) определяет, добавлять ли запись в живую ленту о создании сделки.
* Рискованные поля: `OPPORTUNITY`, `CURRENCY_ID`, `ASSIGNED_BY_ID`, `STAGE_ID`, `CATEGORY_ID` — требуют подтверждения перед выполнением.

### crm.deal.update
* Требует параметр `id` (целое) и объект `fields` с изменяемыми значениями.
* Возвращает `true` при успехе. Меняет только переданные поля.
* Перед изменением суммы (`OPPORTUNITY`), стадии (`STAGE_ID`), воронки (`CATEGORY_ID`) или ответственного (`ASSIGNED_BY_ID`) необходимо подтверждение пользователя.

### crm.deal.category.list
* Возвращает направления продаж (воронки). Ответ — массив объектов с полями `ID`, `NAME`, `SORT`, `IS_DEFAULT`.
* Метод устаревший, но остаётся доступным; новые реализации используют `crm.category.*`.

### crm.deal.category.stage.list
* Возвращает стадии для заданной категории. Требует `id` (или `categoryId`).
* Каждая стадия содержит `STATUS_ID`, `NAME`, `SORT`, `SEMANTICS` (`P` — в процессе, `S` — успешная, `F` — провальная).

### crm.status.list
* Универсальный метод для справочников CRM.
* Обязателен фильтр `filter.ENTITY_ID` (например, `"DEAL_STAGE"`, `"INDUSTRY"`, `"SOURCE"`). Можно добавлять `CATEGORY_ID`, `STATUS_ID`.
* Ответ — массив элементов со значениями `STATUS_ID`, `NAME`, `SORT`, `SYSTEM`, `COLOR` и др.

### crm.activity.list
* Рекомендуется задавать `filter.OWNER_TYPE_ID` и `filter.OWNER_ID`, чтобы ограничить выборку (например, все дела сделки или контакта). Без фильтра список может быть очень большим.
* Можно использовать фильтры по статусу (`COMPLETED`), дедлайну (`>DEADLINE`), типу (`TYPE_ID`).
* В `select` доступны поля `COMMUNICATIONS`, `RESPONSIBLE_ID`, даты и описание.

### crm.activity.add
* Требуемые поля: `OWNER_TYPE_ID` (лид=1, сделка=2, контакт=3, компания=4), `OWNER_ID`, `TYPE_ID`, `SUBJECT`.
* Дополнительные поля: `START_TIME`, `END_TIME`, `RESPONSIBLE_ID`, `PRIORITY`, `DESCRIPTION`, `COMMUNICATIONS`, `PROVIDER_ID`, `PROVIDER_TYPE_ID`.
* Время указывается в ISO8601. Поля `RESPONSIBLE_ID` и дедлайн требуют подтверждения.

### crm.timeline.comment.add
* Обязательные поля: `ENTITY_ID` (ID сущности), `ENTITY_TYPE` (строка в нижнем регистре: `deal`, `lead`, `contact`, `company`), `COMMENT` (текст).
* Возвращает ID комментария. Используется для добавления заметок в таймлайн.

### tasks.task.add
* Требуемые поля: `TITLE`, `DESCRIPTION`, `RESPONSIBLE_ID`.
* Дополнительные: `CREATED_BY`, `DEADLINE`, `START_DATE_PLAN`, `END_DATE_PLAN`, `PRIORITY`, `AUDITORS`, `ACCOMPLICES`, `GROUP_ID`, `TAGS`, `UF_*`.
* Ответ возвращает объект задачи в `result.task`.

### tasks.task.update
* Требуется `taskId` (или `id`) и объект `fields` с изменяемыми полями.
* Может менять `TITLE`, `DESCRIPTION`, `RESPONSIBLE_ID`, `DEADLINE`, пользовательские поля и др.
* Изменение ответственного и дедлайна требует подтверждения.

### tasks.task.list
* По умолчанию возвращает максимум 50 задач; используйте `start` для постраничного чтения.
* Распространённые фильтры: `RESPONSIBLE_ID`, `CREATED_BY`, `STATUS`, `>=DEADLINE`, `GROUP_ID`, `!RESPONSIBLE_ID`.
* `select` позволяет запросить дополнительные поля, включая `UF_CRM_TASK`.
* Ответ содержит `result` (список задач) и `total`.

### task.commentitem.add
* Требует `taskId` (или `TASK_ID`) и `fields.POST_MESSAGE`.
* Возвращает ID созданного комментария (ID записи форума задач).

### task.checklistitem.add
* Требует `taskId` (или `TASK_ID`) и `fields.TITLE`.
* Дополнительно можно указать `SORT_INDEX`, `PARENT_ID`, `IS_COMPLETE`, `RESPONSIBLE_ID`.

### sonet.group.get
* Возвращает данные рабочих групп (проектов). Можно указать `ID` или `filter` (например, по `NAME`, `OWNER_ID`).
* Ответ содержит поля `ID`, `NAME`, `DESCRIPTION`, `OWNER_ID`, `DATE_CREATE`, `PROJECT`, `CLOSED`, `NUMBER_OF_MEMBERS` и др.

### sonet.group.user.get
* Требует `GROUP_ID`.
* Возвращает список участников с полями `USER_ID`, `ROLE` (`A` — владелец, `E` — руководитель/эксперт, `M` — участник), опционально `AUTO_MEMBER`.

## Общие требования
* Все вызовы выполняются через обёртку `call_bitrix` из модуля `bitrix_client`.
* Агент валидирует обязательные поля до вызова и сообщает пользователю, если данных не хватает.
* При ошибке Bitrix агент передаёт пользователю код и описание (`error` / `error_description`).
* Ожидаемый успешный ответ содержит поле `result` и метаданные `time`.

### batch
* Позволяет сгруппировать до 50 подзапросов Bitrix24 за один HTTP-вызов.
* `cmd` — словарь или массив строк вида `ключ: "crm.deal.get?id=123"`. Метод в строке определяется префиксом до `?`.
* Оркестратор проверяет каждый подзапрос: разрешён ли метод, не изменяет ли он критичные поля без подтверждения. При нарушении пакет блокируется.
* Ответ содержит `result.result` с данными по ключам и `result.result_error` для ошибок подзапросов. Поле `result.total` отсутствует, поэтому количество элементов фиксируется в истории состояний вручную.

### event.bind
* Регистрирует обработчик события (`event`, например `onCrmDealAdd`) на заданный URL `handler`.
* Всегда требует подтверждения пользователя и доступен только в режиме `full`.
* Успешный ответ `{ "result": true }` заставляет оркестратор добавить подписку в `agent_state.event_bindings`.

### event.get
* Возвращает список активных подписок вебхука.
* Доступен в режимах `canary` и `full`.
* Результат — массив объектов `{ "event": ..., "handler": ... }`, которым замещается `agent_state.event_bindings`.

### event.unbind
* Удаляет ранее созданную подписку (`event`, `handler`).
* Всегда требует подтверждения пользователя и доступен только в режиме `full`.
* При успехе `{ "result": true }` оркестратор удаляет запись из `agent_state.event_bindings`.
