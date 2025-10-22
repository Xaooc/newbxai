from pathlib import Path

path = Path("telegram_bot.py")
text = path.read_text(encoding="utf-8")
needle = "    if shortcut_reply:\n        await message.reply_text(shortcut_reply)\n        return\n"
replacement = (
    "    if shortcut_reply:\n"
    "        logger.debug(\"Отправляю пользователю ответ календарного шортката.\")\n"
    "        await message.reply_text(shortcut_reply)\n"
    "        return\n"
)
if needle not in text:
    raise SystemExit("needle not found")
text = text.replace(needle, replacement, 1)
path.write_text(text, encoding="utf-8")
