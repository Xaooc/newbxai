"""
Telegram-бот для Bitrix24 на базе LangChain.

Выполняет команды пользователей, перенаправляя их в Bitrix24 через LangChain-агента,
и логирует каждый промежуточный этап для диагностирования.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime_t import datetime
from textwrap import indent
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - совместимость со старыми версиями LangChain
    from langchain.chat_models import ChatOpenAI  # type: ignore[misc]

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from memory_utils import ConversationBufferMemory

from bitrix24_client import Bitrix24Client
from bitrix_assistant import BitrixAgentConfig, build_agent, fetch_calendar_events_for_users


logger = logging.getLogger(__name__)
load_dotenv()

# --- Контроль доступа по списку разрешённых пользователей --------------------
ALLOWED_USER_IDS: Set[int] = set()
_raw_allowed = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
if _raw_allowed:
    for chunk in _raw_allowed.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ALLOWED_USER_IDS.add(int(chunk))
        except ValueError:
            logger.warning(
                "Игнорирую некорректный ID '%s' в TELEGRAM_ALLOWED_USER_IDS", chunk
            )


def _is_user_allowed(update: Update) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)


async def _reply_unauthorized(update: Update) -> None:
    user = update.effective_user
    logger.info(
        "Отклонён запрос неавторизованного пользователя: id=%s username=%s",
        getattr(user, "id", None),
        getattr(user, "username", None),
    )
    await update.effective_message.reply_text(
        "Извините, доступ к этому боту ограничен. Пожалуйста, обратитесь к администратору."
    )


# --- Создание зависимостей ---------------------------------------------------

def create_llm() -> ChatOpenAI:
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.0-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1024"))
    logger.debug(
        "Инициализирую LLM: model=%s temperature=%s max_tokens=%s",
        model_name,
        temperature,
        max_tokens,
    )
    return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)


def create_bitrix_client() -> Bitrix24Client:
    base_url = os.getenv("BITRIX_BASE_URL")
    if not base_url:
        raise RuntimeError("Не задан BITRIX_BASE_URL — невозможно создать Bitrix24Client.")
    timeout = int(os.getenv("BITRIX_TIMEOUT", "30"))
    if os.getenv("BITRIX_OAUTH_TOKEN"):
        logger.warning(
            "Переменная BITRIX_OAUTH_TOKEN задана, но будет проигнорирована — используем только вебхук из BITRIX_BASE_URL."
        )
    logger.debug("Создаю Bitrix24Client через вебхук: base_url=%s timeout=%s", base_url, timeout)
    return Bitrix24Client(base_url=base_url, oauth_token=None, timeout=timeout)


@dataclass
class ChatSession:
    agent: Any
    memory: ConversationBufferMemory


# --- Утилиты обработки сообщений --------------------------------------------

def _message_content_to_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content") or item.get("value")
                if text:
                    pieces.append(str(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if text:
                    pieces.append(str(text))
        if pieces:
            return "\n".join(pieces)
    return str(content)


def _extract_from_messages(messages: Any) -> str:
    if not isinstance(messages, Sequence):
        return ""
    collected: List[str] = []
    for item in messages:
        if isinstance(item, (AIMessage, HumanMessage, BaseMessage)):
            text = _message_content_to_text(item)
        elif isinstance(item, Mapping):
            text = str(item.get("content") or item.get("text") or "")
        else:
            text = str(item)
        text = text.strip()
        if text:
            collected.append(text)
    return "\n".join(collected)


def _extract_agent_output(response: Any) -> str:
    if isinstance(response, dict):
        for key in ("output", "final_output"):
            value = response.get(key)
            if value:
                return str(value)
        if "return_values" in response:
            nested = _extract_agent_output(response["return_values"])
            if nested:
                return nested
        if "messages" in response:
            nested = _extract_from_messages(response["messages"])
            if nested:
                return nested
        message = response.get("message")
        if isinstance(message, str) and message.strip():
            return message
        return ""
    if isinstance(response, (AIMessage, BaseMessage)):
        return _message_content_to_text(response)
    if isinstance(response, Sequence):
        nested = _extract_from_messages(response)
        if nested:
            return nested
    if isinstance(response, str):
        return response
    return ""


def _format_for_log(value: Any, limit: int = 600) -> str:
    text = repr(value)
    if len(text) > limit:
        return f"{text[:limit]}… (len={len(text)})"
    return text


_CLARIFICATION_PATTERNS = (
    "что нужно сделать в bitrix24",
    "уточните одно действие",
    "опишите коротко действие",
    "назовите действие",
)


def _looks_like_clarification(text: str) -> bool:
    normalized = text.lower()
    return any(pattern in normalized for pattern in _CLARIFICATION_PATTERNS)


def _rewind_last_turn(session: ChatSession) -> None:
    history = getattr(session.memory, "chat_memory", None)
    if history is None:
        logger.debug("Откат шага невозможен: chat_memory отсутствует.")
        return
    messages = getattr(history, "messages", None)
    if not isinstance(messages, list):
        logger.debug("Откат шага невозможен: неизвестный тип messages (%s).", type(messages).__name__)
        return
    removed: List[str] = []
    if messages and isinstance(messages[-1], AIMessage):
        removed.append(f"AI: {_message_content_to_text(messages.pop())}")
    if messages and isinstance(messages[-1], HumanMessage):
        removed.append(f"Human: {_message_content_to_text(messages.pop())}")
    if removed:
        logger.debug("Удалены последние сообщения из памяти: %s", _format_for_log(removed))
    else:
        logger.debug("Откат шага: удалять было нечего.")


# --- Обработка календаря -----------------------------------------------------

_CALENDAR_QUERY_KEYWORDS = ("какие", "встреч", "событи", "меропр")
_CALENDAR_CREATE_KEYWORDS = ("создат", "добав", "назнач", "создаё", "создай")


def _should_auto_fetch_calendar(text: str) -> bool:
    normalized = text.lower()
    if "календар" not in normalized and "встреч" not in normalized and "событ" not in normalized:
        return False
    if any(token in normalized for token in _CALENDAR_CREATE_KEYWORDS):
        return False
    return any(token in normalized for token in _CALENDAR_QUERY_KEYWORDS)


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    cleaned = value.strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        for pattern in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                return datetime.strptime(cleaned, pattern)
            except ValueError:
                continue
    return None


def _format_date_label(value: Optional[str]) -> str:
    parsed = _parse_iso_datetime(value)
    if not parsed:
        return "сегодня"
    return parsed.strftime("%d.%m.%Y")


def _format_event_entry(event: Mapping[str, Any]) -> str:
    title = str(event.get("title") or event.get("NAME") or "Без названия").strip()
    start = _parse_iso_datetime(event.get("start") or event.get("DATE_FROM"))
    end = _parse_iso_datetime(event.get("end") or event.get("DATE_TO"))
    time_part = ""
    if start and end:
        if start.date() == end.date():
            time_part = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
        else:
            time_part = f"{start.strftime('%d.%m %H:%M')} → {end.strftime('%d.%m %H:%M')}"
    elif start:
        time_part = start.strftime("%d.%m %H:%M")
    attendees = event.get("attendees") or event.get("USER_IDS")
    attendees_part = f"; участники: {attendees}" if attendees else ""
    if time_part:
        return f"{title} — {time_part}{attendees_part}"
    return f"{title}{attendees_part}"


def _format_calendar_events_message(data: Mapping[str, Any]) -> str:
    status = data.get("status")
    if status != "ok":
        message = data.get("message") or "не удалось получить события"
        return f"Не удалось получить события календаря: {message}"

    date_text = _format_date_label(data.get("date"))
    users = [user for user in data.get("users", []) if user.get("events")]
    total = data.get("events_total", 0)
    if total == 0 or not users:
        return f"На {date_text} запланированных событий не найдено."

    lines = [f"События на {date_text}:"]
    max_users = 4
    max_events = 3
    for user in users[:max_users]:
        name = (str(user.get("user_name") or "")).strip() or f"ID {user.get('user_id')}"
        events = user.get("events") or []
        lines.append(f"{name}:")
        for event in events[:max_events]:
            lines.append(f"- {_format_event_entry(event)}")
        extra = len(events) - max_events
        if extra > 0:
            lines.append(f"- … ещё {extra} событие(й)")
    extra_users = len(users) - max_users
    if extra_users > 0:
        lines.append(f"Всего пользователей с событиями: {len(users)} (показаны первые {max_users}).")
    if data.get("errors"):
        lines.append("Некоторые календари вернуть не удалось — проверьте логи.")
    return "\n".join(lines)


def _handle_calendar_shortcut(manager: "AgentManager", text: str) -> Optional[str]:
    if not _should_auto_fetch_calendar(text):
        return None
    logger.debug("Запрос распознан как календарный шорткат: %s", _format_for_log(text))
    try:
        payload = fetch_calendar_events_for_users(manager.client)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка при получении календарных событий", exc_info=exc)
        return f"Не удалось получить календарь: {exc}"
    logger.debug("Ответ Bitrix24 по календарю: %s", _format_for_log(payload))
    return _format_calendar_events_message(payload)


# --- Управление сессиями агента ---------------------------------------------


class AgentManager:
    def __init__(self, llm: ChatOpenAI, client: Bitrix24Client, verbose: bool = False) -> None:
        self._llm = llm
        self._client = client
        self._verbose = verbose
        self._sessions: Dict[int, ChatSession] = {}

    @property
    def client(self) -> Bitrix24Client:
        return self._client

    def get_session(self, chat_id: int) -> ChatSession:
        if chat_id not in self._sessions:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            logger.debug("Создаю нового агента для чата %s.", chat_id)
            agent = build_agent(
                BitrixAgentConfig(
                    llm=self._llm,
                    client=self._client,
                    memory=memory,
                    verbose=self._verbose,
                )
            )
            self._sessions[chat_id] = ChatSession(agent=agent, memory=memory)
            logger.debug("Сессия для чата %s создана и сохранена.", chat_id)
        else:
            logger.debug("Использую сохранённую сессию для чата %s.", chat_id)
        return self._sessions[chat_id]

    def reset(self, chat_id: int) -> None:
        if chat_id in self._sessions:
            logger.debug("Удаляю сессию для чата %s по запросу пользователя.", chat_id)
            del self._sessions[chat_id]


# --- Telegram-команды --------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return
    await update.effective_message.reply_text(
        "Привет! Я ассистент по Bitrix24. Опишите задачу, и я постараюсь выполнить её автоматически."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return
    await update.effective_message.reply_text(
        "Я принимаю ваши инструкции и выполняю их через Bitrix24.\n"
        "Пример: 'Покажи задачи на сегодня по Иванову' или 'Создай встречу на 15:00'."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return
    chat_id = update.effective_chat.id
    manager: AgentManager = context.application.bot_data["agent_manager"]
    manager.reset(chat_id)
    await update.effective_message.reply_text(
        "История диалога сброшена. Можем начинать заново!"
    )


# --- Главный обработчик сообщений -------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return

    manager: AgentManager = context.application.bot_data["agent_manager"]
    message = update.effective_message
    chat_id = update.effective_chat.id
    text = message.text or ""

    if not text.strip():
        await message.reply_text("Пожалуйста, пришлите текстовое сообщение.")
        return

    user = update.effective_user
    logger.info(
        "Новое сообщение: chat_id=%s user_id=%s username=%s text=%s",
        chat_id,
        getattr(user, "id", None),
        getattr(user, "username", None),
        _format_for_log(text),
    )

    loop = asyncio.get_running_loop()
    try:
        shortcut_reply = await loop.run_in_executor(
            None,
            lambda: _handle_calendar_shortcut(manager, text),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка при обработке шортката", exc_info=exc)
        shortcut_reply = f"Не удалось обработать календарный запрос: {exc}"

    if shortcut_reply:
        logger.debug("Ответ сформирован шорткатом, отправляю пользователю.")
        await message.reply_text(shortcut_reply)
        return

    session = manager.get_session(chat_id)

    try:
        logger.debug("Начинаю вызов агента для чата %s.", chat_id)
        logger.debug("Сообщение пользователя: %s", _format_for_log(text))
        response = await loop.run_in_executor(
            None,
            lambda: session.agent.invoke(
                {"input": text},
                config={"configurable": {"session_id": str(chat_id)}},
            ),
        )
        logger.info("Агент завершил выполнение для чата %s: %s", chat_id, _format_for_log(response))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Сбой при вызове агента: chat_id=%s", chat_id)
        await message.reply_text(
            "Не удалось выполнить запрос из-за внутренней ошибки. "
            "Проверьте соединение или повторите попытку позже."
            f" Подробности: {exc}"
        )
        return

    answer = _extract_agent_output(response).strip()

    if _looks_like_clarification(answer):
        logger.debug(
            "Ответ похож на просьбу уточнить действие; запускаю повторную попытку: chat_id=%s",
            chat_id,
        )
        _rewind_last_turn(session)
        forced_text = (
            f"{text}\n\n"
            "(Служебное сообщение: выполни указанное действие без дополнительных уточнений. "
            "Если не хватает данных, постарайся получить их через доступные инструменты Bitrix24. "
            "Если выполнить нельзя, объясни причину.)"
        )
        logger.debug("Повторный запрос к агенту: %s", _format_for_log(forced_text))
        try:
            fallback_response = await loop.run_in_executor(
                None,
                lambda: session.agent.invoke(
                    {"input": forced_text},
                    config={"configurable": {"session_id": str(chat_id)}},
                ),
            )
            logger.info(
                "Повторная попытка агента для чата %s завершена: %s",
                chat_id,
                _format_for_log(fallback_response),
            )
            fallback_answer = _extract_agent_output(fallback_response).strip()
            if fallback_answer:
                logger.debug("Использую ответ, полученный из повторной попытки.")
                answer = fallback_answer
                response = fallback_response
        except Exception as exc:  # noqa: BLE001
            logger.exception("Не удалось выполнить повторный вызов агента: chat_id=%s", chat_id)
            answer = (
                "Не получилось получить ответ без уточнений — возникла ошибка. "
                f"Подробности: {exc}"
            )

    if not answer:
        logger.warning(
            "Агент вернул пустой ответ: chat_id=%s тип=%s",
            chat_id,
            type(response).__name__,
        )
        answer = (
            "Не удалось сформировать понятный ответ. "
            "Попробуйте переформулировать запрос или добавить больше деталей."
        )

    logger.info("Отправляю ответ пользователю: chat_id=%s text=%s", chat_id, _format_for_log(answer))
    await message.reply_text(answer)


# --- Построение приложения ---------------------------------------------------


def build_application() -> Application:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN — бот не может быть запущен.")

    verbose = os.getenv("LANGCHAIN_VERBOSE", "false").lower() in {"1", "true", "yes"}

    llm = create_llm()
    client = create_bitrix_client()
    manager = AgentManager(llm=llm, client=client, verbose=verbose)

    application = ApplicationBuilder().token(token).build()
    application.bot_data["agent_manager"] = manager

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return application


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Запуск Telegram-бота Bitrix24...")
    application = build_application()
    application.run_polling()


if __name__ == "__main__":
    main()
