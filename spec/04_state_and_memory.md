# 04. Состояние и память агента

## Формат `agent_state`
```json
{
  "goals": ["Создать сделку", "Назначить звонок"],
  "done": [
    {
      "timestamp": "2024-05-01T10:00:00Z",
      "description": "Создана сделка",
      "object_ids": {"deal_id": 123}
    }
  ],
  "in_progress": [
    {
      "description": "Ожидаем подтверждения суммы",
      "requested_at": "2024-05-01T10:05:00Z"
    }
  ],
  "objects": {
    "current_deal_id": 123,
    "current_contact_id": 456,
    "current_company_id": null,
    "current_task_id": 789
  },
  "next_planned_actions": [
    {
      "method": "crm.deal.update",
      "params": {"id": 123, "fields": {"OPPORTUNITY": 100000}},
      "requires_confirmation": true
    }
  ],
  "confirmations": {
    "deal_123_opportunity": {
      "status": "requested",
      "requested_at": "2024-05-01T10:05:00Z",
      "description": "Изменить сумму сделки 123 до 100000",
      "action": {
        "method": "crm.deal.update",
        "params": {"id": 123, "fields": {"OPPORTUNITY": 100000}},
        "requires_confirmation": true
      }
    },
    "task_456_deadline": {
      "status": "denied",
      "requested_at": "2024-05-02T09:00:00Z",
      "denied_at": "2024-05-02T09:05:00Z",
      "description": "Перенести дедлайн задачи 456",
      "reason": "Пользователь отклонил перенос",
      "action": {
        "method": "tasks.task.update",
        "params": {"taskId": 456, "fields": {"DEADLINE": "2024-05-05"}},
        "confirmation_decision": "deny"
      }
    }
  },
  "event_bindings": [
    {
      "event": "onCrmDealAdd",
      "handler": "https://example.test/hook"
    }
  ]
}
```

## Правила обновления
* `goals` — список целей, сформированных из пользовательских запросов. Новые цели добавляются в начало.
* `done` — истории завершённых действий с временными метками и ID сущностей.
* `in_progress` — текущие шаги, требующие данных или подтверждения.
* `objects` — последние активные сущности для контекста. Обновляются при успешных вызовах Bitrix.
  * После `crm.deal.get` фиксируются `current_deal_id`, а также `current_contact_id` и `current_company_id`, если они присутствуют в ответе сделки.
  * После `crm.deal.list`, `crm.contact.list`, `crm.company.list`, `crm.activity.list`, `tasks.task.list`, `crm.deal.category.list`, `crm.deal.category.stage.list`, `crm.status.list`, `sonet.group.get`, `sonet.group.user.get` в историю `done` добавляется запись с количеством элементов, чтобы пользователь видел объём данных.
  * После `crm.contact.get` обновляется `current_contact_id`; при наличии `COMPANY_ID` синхронизируется `current_company_id`.
  * После `crm.company.get` поле `current_company_id` обновляется значением ID компании.
* `next_planned_actions` — сохраняет план при режиме shadow или при ожидании подтверждения.
* `confirmations` — отслеживает статусы запросов подтверждения (`requested`/`approved`/`denied`), временные метки `requested_at`/`approved_at`/`denied_at`, причину и копию шага. При `approved` шаг автоматически повторяется при следующем появлении с `confirmed: true`; при `denied` шаг игнорируется, пока модель не сформирует новый запрос.
* `event_bindings` — актуальный список подписок на события Bitrix24. Обновляется после `event.get`, добавления и удаления подписок; используется, чтобы модель понимала, какие вебхуки уже настроены.
* Все поля с временными метками сериализуются в формате ISO 8601 с суффиксом `Z`, полученными из timezone-aware UTC-времени.

## Хранение
* Состояние хранится в файловой системе в формате JSON (один файл на пользователя/сессию).
* Оркестратор читает состояние при старте диалога и сохраняет после каждой итерации.

## Передача в модель
* В системный промпт включается сериализованное состояние, в том числе ожидающие подтверждения.
* При больших объёмах состояния допускается сокращение: последние 5 записей `done`, актуальные `objects`, открытые `in_progress` и первые 5 подтверждений со статусом `requested`.
