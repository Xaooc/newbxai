# Bitrix24 GPT-5 Telegram Assistant

Телеграм-бот на базе LangChain и встроенного клиента `Bitrix24Client`. Ассистент умеет выполнять операции Bitrix24 REST и отвечать на последующие вопросы в естественной форме.

## Возможности
- Единый инструмент `bitrix_api`, который открывает для модели все публичные методы `Bitrix24Client`.

## Поддерживаемые REST-методы
Перечень ключевых методов Bitrix24, уже обёрнутых в `Bitrix24Client`. Подробные docstring и типизированные сигнатуры помогают агенту LangChain выбирать подходящие инструменты автоматически.

- Пользователи и сервис: `access.name`, `user.current`, `user.get`, `app.info`, `app.option.get`, `app.option.set`, `batch`.
- События: `event.bind`, `event.get`, `event.unbind`.
- CRM сделки и справочники: `crm.deal.add`, `crm.deal.get`, `crm.deal.list`, `crm.deal.update`, `crm.deal.category.list`, `crm.deal.category.stage.list`, `crm.status.list`.
- CRM активности и таймлайн: `crm.activity.add`, `crm.activity.list`, `crm.timeline.comment.add`.
- CRM контакты и компании: `crm.contact.list`, `crm.contact.get`, `crm.company.list`, `crm.company.get`.
- Задачи: `tasks.task.add`, `tasks.task.update`, `tasks.task.list`, `task.commentitem.add`, `task.checklistitem.add`.
- Рабочие группы: `sonet.group.get`, `sonet.group.user.get`.
- Календарь: `calendar.accessibility.get`, `calendar.event.add`, `calendar.event.get`, `calendar.event.get.nearest`, `calendar.event.getbyid`, `calendar.event.update`, `calendar.event.delete`, `calendar.meeting.params.set`, `calendar.meeting.status.get`, `calendar.meeting.status.set`, `calendar.resource.add`, `calendar.resource.update`, `calendar.resource.delete`, `calendar.resource.list`, `calendar.resource.booking.list`.

## Требования
- Python 3.10+
- Перед запуском установите зависимости:
  ```bash
  pip install python-telegram-bot langchain langchain-openai openai requests python-dotenv
  ```

## Быстрый старт
1. Скопируйте файл `.env.example` в `.env` и заполните значения.
   ```bash
   cp .env.example .env
   # далее отредактируйте .env и укажите свои токены
   ```
2. Запустите бота:
   ```bash
   python telegram_bot.py
   ```

## Переменные окружения
| Имя | Описание |
| --- | --- |
| `TELEGRAM_BOT_TOKEN` | Токен бота от BotFather. |
| `TELEGRAM_ALLOWED_USER_IDS` | Список разрешённых ID пользователей через запятую (например, `444761925`). Пустое значение разрешает всех. |
| `BITRIX_BASE_URL` | URL вебхука или REST-портала Bitrix24 (например, `https://portal.bitrix24.ru/rest/1/WEBHOOK/`). |
| `BITRIX_OAUTH_TOKEN` | Необязательный OAuth-токен при работе через REST-портал. |
| `OPENAI_API_KEY` | API-ключ для модели GPT-5 (совместимой с OpenAI). |
| `OPENAI_MODEL` | Имя модели (по умолчанию `gpt-5.0-mini`, можно изменить под свой стенд). |
| `OPENAI_TEMPERATURE` | Необязательная температура генерации, по умолчанию `0`. |
| `OPENAI_MAX_OUTPUT_TOKENS` | Необязательный предел на количество токенов в ответе. |
| `BITRIX_TIMEOUT` | Необязательный таймаут HTTP-запросов к Bitrix24 в секундах. |
| `LANGCHAIN_VERBOSE` | Необязательный флаг подробных логов агента (`true/false`). |

## Расширение ассистента
1. Добавляйте новые методы в `Bitrix24Client` — инструмент обнаружит их автоматически.
2. Отредактируйте `DEFAULT_SYSTEM_PROMPT` в `bitrix_assistant.py`, если нужно изменить стиль общения или сценарии.
3. Расширяйте `AgentManager`, чтобы подключить постоянное хранение, аналитику или очереди фоновых задач.
