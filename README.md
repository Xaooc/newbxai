# Telegram Bitrix ассистент

Проект реализует Telegram-бота на основе GPT-5 tool calling и LangGraph. Ассистент помогает выполнять операции в Bitrix24, выбирает методы, проверяет схемы входных данных и формирует краткие ответы.

## Основные компоненты

- `telegram_bitrix_agent/assistant/graph` — граф LangGraph с узлами `plan → ensure_fields → act → finalize`.
- `telegram_bitrix_agent/assistant/tools` — адаптеры для Bitrix24, форматтер ответов и хранилище памяти.
- `telegram_bitrix_agent/telegram` — aiogram-вебхук и обработчики.
- `schemas/` — JSON-схемы методов Bitrix24 и индекс.
- `scripts/build_catalog.py` — генерация индекса по схемам.
- `tests/` — интеграционные проверки сценариев.

## Запуск

1. Установите зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Скопируйте `.env.example` в `.env` и заполните значения (в том числе `REDIS_DSN`, если требуется устойчивое хранилище памяти).
3. Задайте переменные окружения (или используйте `.env`):
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - `BITRIX_WEBHOOK_URL`
   - `TELEGRAM_WEBHOOK_URL` (опционально)
   - `REDIS_DSN` (опционально, например `redis://localhost:6379/0`)
4. Запустите вебхук:

```bash
python -m telegram_bitrix_agent.main
```

### Docker

```bash
docker build -t telegram-bitrix-agent .
docker run --env-file .env -p 8000:8000 telegram-bitrix-agent
```

## Тесты

```bash
pytest -q
```
