from pathlib import Path

path = Path("telegram_bot.py")
text = path.read_text(encoding="utf-8")
old = "            agent = build_agent(\n                BitrixAgentConfig(\n                    llm=self._llm,\n                    client=self._client,\n                    memory=memory,\n                    verbose=self._verbose,\n                )\n            )\n            self._sessions[chat_id] = ChatSession(agent=agent, memory=memory)\n            logger.debug(\"������� ����� ������ ������ ��� ���� %s\", chat_id)\n        return self._sessions[chat_id]\n\n"
new = "            logger.debug(\"Создаю нового агента для чата %s\", chat_id)\n            agent = build_agent(\n                BitrixAgentConfig(\n                    llm=self._llm,\n                    client=self._client,\n                    memory=memory,\n                    verbose=self._verbose,\n                )\n            )\n            self._sessions[chat_id] = ChatSession(agent=agent, memory=memory)\n            logger.debug(\"Сессия для чата %s создана и сохранена\", chat_id)\n        else:\n            logger.debug(\"Использую существующую сессию для чата %s\", chat_id)\n        return self._sessions[chat_id]\n\n"
if old not in text:
    raise SystemExit("original block not matched")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
