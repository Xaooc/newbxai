# 02. Методы Bitrix24

## Базовый вебхук
* Все вызовы выполняются через URL `https://portal.magnitmedia.ru/rest/132/1s0mz4mw8d42bfvk/`.
* URL может быть переопределён переменной окружения `BITRIX_WEBHOOK_URL`, но по умолчанию используется указанный портал.
* Модель получает только логические имена методов и параметры, сам URL скрывается.

## Разрешённые методы (кратко)
| Метод | Назначение | Обязательные параметры | Рискованные поля/действия |
|-------|------------|------------------------|---------------------------|
| `user.current` | Информация о владельце вебхука | — | — |
| `user.get` | Поиск пользователей | `filter`/`order`/`select` по необходимости | Нет |
| `crm.contact.list` | Список контактов | `filter` (опционально) | Нет |
| `crm.contact.get` | Контакт по ID | `id` | Нет |
| `crm.company.list` | Список компаний | `filter` (опционально) | Нет |
| `crm.company.get` | Компания по ID | `id` | Нет |
| `crm.deal.list` | Список сделок | `filter`/`select`/`start` | Нет |
| `crm.deal.get` | Сделка по ID | `id` | Нет |
| `crm.deal.add` | Создание сделки | `fields.TITLE` минимум | `OPPORTUNITY`, `ASSIGNED_BY_ID`, `STAGE_ID` |
| `crm.deal.update` | Обновление сделки | `id`, `fields` | Любое изменение суммы, стадии, ответственного |
| `crm.activity.list` | Список активностей | `filter` | Нет |
| `crm.activity.add` | Создание активности | `fields`: `OWNER_TYPE_ID`, `OWNER_ID`, `TYPE_ID`, `SUBJECT` | `RESPONSIBLE_ID`, `DEADLINE` |
| `crm.timeline.comment.add` | Комментарий в таймлайн | `fields`: `ENTITY_ID`, `ENTITY_TYPE`, `COMMENT` | Содержимое текста |
| `tasks.task.add` | Создание задачи | `fields`: `TITLE`, `RESPONSIBLE_ID`, `DESCRIPTION` | `RESPONSIBLE_ID`, `DEADLINE` |
| `tasks.task.update` | Обновление задачи | `taskId`, `fields` | Ответственный, дедлайн |
| `tasks.task.list` | Список задач | `filter` | Нет |
| `task.commentitem.add` | Комментарий к задаче | `taskId`, `fields.POST_MESSAGE` | Текст |
| `task.checklistitem.add` | Пункт чек-листа | `taskId`, `fields.TITLE` | Нет |

Дополнительные методы (`batch`, `event.*`, `crm.status.list`, `sonet.group.*`) описаны в документации, но пока **не разрешены**. Их можно включать только после отдельного согласования и обновления спецификации.

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
* Поля `PHONE` и `EMAIL` множественные — в `select` явно указывать `"PHONE"`, `"EMAIL"`.
* Поиск по номеру: `{"PHONE": "+74951234567"}`.
* Метод `get` возвращает полный объект, включая пользовательские поля `UF_*`.

### crm.company.list / crm.company.get
* Аналогично контактам, множественные поля требуют явного `select`.
* Поля `COMPANY_TYPE`, `INDUSTRY`, `EMPLOYEES` помогают сегментировать клиентов.

### crm.deal.list
* Максимум 50 записей за вызов; используем `start` для постраничного чтения.
* Частые фильтры: `{"ASSIGNED_BY_ID": <id>}`, `{"%TITLE": "Продажа"}`, `{"CATEGORY_ID": 5}`.
* Для анализа прогресса можно запрашивать `STAGE_ID`, `OPPORTUNITY`, `DATE_CREATE`.

### crm.deal.get
* Возвращает все поля сделки, в том числе пользовательские и UTM-метки.
* Используется для проверки перед обновлением, чтобы не потерять данные.

### crm.deal.add
* Обязателен `fields.TITLE`; при множественных воронках задаём `CATEGORY_ID` и `STAGE_ID`.
* Рискованные поля: `OPPORTUNITY`, `CURRENCY_ID`, `ASSIGNED_BY_ID` — требуют подтверждения в режиме `full`.
* Пример тела запроса:
  ```json
  {
    "fields": {
      "TITLE": "Продажа оборудования",
      "CONTACT_ID": 123,
      "ASSIGNED_BY_ID": 7,
      "OPPORTUNITY": 150000,
      "CURRENCY_ID": "RUB"
    }
  }
  ```

### crm.deal.update
* Меняет только указанные поля. Перед изменением суммы или стадии требуется подтверждение.
* Для фиксации подтверждения модель должна указать `"confirmed": true` в шаге ACTION.

### crm.activity.list
* Требует `OWNER_TYPE_ID` (лид=1, сделка=2, контакт=3, компания=4) и при необходимости `OWNER_ID`.
* Для актуальных дел используем `{"COMPLETED": "N"}`.

### crm.activity.add
* Стандартные типы: `1` — встреча, `2` — звонок, `3` — email.
* `START_TIME`/`END_TIME` задаём в ISO8601; при звонках добавляем `COMMUNICATIONS` с телефоном.
* Рискованные поля: `RESPONSIBLE_ID`, `DEADLINE` — требуют подтверждения в режиме `full`.

### crm.timeline.comment.add
* `ENTITY_TYPE` — строчные значения (`"deal"`, `"lead"`, `"contact"`, `"company"`).
* Текст комментария должен быть согласован с пользователем.

### tasks.task.add
* Минимум: `TITLE`, `RESPONSIBLE_ID`, `DESCRIPTION`.
* `DEADLINE` — формат `YYYY-MM-DD HH:MM:SS` либо ISO8601 в часовом поясе портала.
* Ответ содержит `result.task.id`, который сохраняем в состоянии.

### tasks.task.update
* URL-параметр `taskId` обязателен.
* Изменение ответственного или дедлайна — рискованное действие и требует подтверждения.

### tasks.task.list
* Фильтры: `RESPONSIBLE_ID`, `CREATED_BY`, `STATUS`, `>=DEADLINE`, `!RESPONSIBLE_ID` и т.д.
* Ответ содержит `total` и `result` с объектами задач.

### task.commentitem.add
* `fields.POST_MESSAGE` принимает текст комментария (поддерживает BBCode).
* Возвращает числовой ID комментария.

### task.checklistitem.add
* Добавляет пункт в чек-лист задачи. Для управления порядком используем `SORT_INDEX`.

## Общие требования
* Все вызовы выполняются через обёртку `call_bitrix` из модуля `bitrix_client`.
* Агент валидирует обязательные поля до вызова и сообщает пользователю, если данных не хватает.
* При ошибке Bitrix агент передаёт пользователю код и описание (`error` / `error_description`).
* Ожидаемый успешный ответ содержит поле `result` и метаданные `time`.
