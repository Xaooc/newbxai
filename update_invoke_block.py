from pathlib import Path

path = Path("telegram_bot.py")
text = path.read_text(encoding="utf-8")
needle = "    try:\n        logger.debug(\"�������� ������ ��� ���� %s\", chat_id)\n        response = await loop.run_in_executor(\n            None,\n            lambda: session.agent.invoke(\n                {\"input\": text},\n                config={\"configurable\": {\"session_id\": str(chat_id)}},\n            ),\n        )\n        logger.info(\"�������� ������ (�����) ��� ���� %s: %s\", chat_id, _format_for_log(response))\n"
replacement = (
    "    try:\n"
    "        logger.debug(\"Начинаю вызов агента для чата %s.\", chat_id)\n"
    "        logger.debug(\"Сообщение пользователя: %s\", _format_for_log(text))\n"
    "        response = await loop.run_in_executor(\n"
    "            None,\n"
    "            lambda: session.agent.invoke(\n"
    "                {\"input\": text},\n"
    "                config={\"configurable\": {\"session_id\": str(chat_id)}},\n"
    "            ),\n"
    "        )\n"
    "        logger.info(\"Агент завершил выполнение для чата %s: %s\", chat_id, _format_for_log(response))\n"
)
if needle not in text:
    raise SystemExit("needle not found")
text = text.replace(needle, replacement, 1)
path.write_text(text, encoding="utf-8")
