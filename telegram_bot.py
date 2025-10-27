"""
Telegram-бот для Bitrix24 на базе LangChain.

Выполняет команды пользователей, перенаправляя их в Bitrix24 через LangChain-агента,
и логирует каждый промежуточный этап для диагностирования.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from textwrap import dedent
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import re
import json

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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from memory_utils import ConversationBufferMemory

from bitrix24_client import Bitrix24Client
from bitrix_assistant import (
    BitrixAgentConfig,
    ToolPreparation,
    build_agent,
    prepare_tools,
)
from telegram_bitrix_agent.assistant.planning import (
    ActionPlan,
    PlanGenerator,
    PlanGenerationError,
    PlanStep,
)


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
    await _safe_reply_text(
        update.effective_message,
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
    pending_plan: ActionPlan | None = None


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


PLAN_STEP_MARKER = "[INTERNAL-PLAN]"
_MAX_TELEGRAM_CHARS = 3500


def _chunk_text(text: str, limit: int = _MAX_TELEGRAM_CHARS) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    chunks: List[str] = []
    cursor = 0
    length = len(cleaned)
    while cursor < length:
        upper = min(cursor + limit, length)
        split = upper
        if upper < length:
            newline_pos = cleaned.rfind("\n", cursor, upper)
            space_pos = cleaned.rfind(" ", cursor, upper)
            candidate = max(newline_pos, space_pos)
            if candidate > cursor:
                split = candidate
        chunk = cleaned[cursor:split].rstrip()
        if chunk:
            chunks.append(chunk)
        cursor = split
        while cursor < length and cleaned[cursor] in {"\n", " "}:
            cursor += 1
    return chunks or [cleaned]


async def _safe_reply_text(message, text: str) -> None:
    for chunk in _chunk_text(text):
        await message.reply_text(chunk)


_CONFIRM_KEYWORDS = (
    "да",
    "ок",
    "okay",
    "окей",
    "подтверждаю",
    "подтвердить",
    "выполняй",
    "начинай",
    "запускай",
    "go",
    "поехали",
    "старт",
)
_CANCEL_KEYWORDS = (
    "нет",
    "не надо",
    "отмена",
    "отменить",
    "стоп",
    "остановись",
    "cancel",
    "отбой",
)
_PLAN_HISTORY_LIMIT = 4
_EXECUTION_SUMMARY_SYSTEM_PROMPT = dedent(
    """
    Ты помощник, который преобразует результаты инструментов Bitrix24 в понятный ответ для пользователя.
    Дай человеку только те факты, которые помогают закрыть его изначальный запрос.
    Игнорируй технические детали, идентификаторы и JSON, если о них не просили.
    Будь точным: если данных недостаточно, скажи об этом и предложи, что можно сделать.
    """
).strip()

_STEP_OUTPUT_LIMIT = 400


_TEXT_SANITIZE_REGEX = re.compile(r"[^\w]+", flags=re.UNICODE)


def _normalize_user_text(text: str) -> str:
    normalized = (text or "").lower().replace("ё", "е")
    normalized = _TEXT_SANITIZE_REGEX.sub(" ", normalized)
    return normalized.strip()


def _is_plan_confirmation(text: str) -> bool:
    cleaned = _normalize_user_text(text)
    if not cleaned:
        return False
    return any(cleaned == keyword or cleaned.startswith(f"{keyword} ") for keyword in _CONFIRM_KEYWORDS)


def _is_plan_cancellation(text: str) -> bool:
    cleaned = _normalize_user_text(text)
    if not cleaned:
        return False
    return any(cleaned == keyword or cleaned.startswith(f"{keyword} ") for keyword in _CANCEL_KEYWORDS)


def _shorten_for_user(text: str, limit: int = _STEP_OUTPUT_LIMIT) -> str:
    sanitized = text.replace("\r", " ").replace("\n", " ").strip()
    if len(sanitized) <= limit:
        return sanitized
    return f"{sanitized[:limit].rstrip()}…"


def _get_recent_history_for_plan(session: ChatSession, limit: int = _PLAN_HISTORY_LIMIT) -> List[str]:
    try:
        payload = session.memory.load_memory_variables({})
    except Exception:  # noqa: BLE001
        return []
    messages = payload.get(session.memory.memory_key) or payload.get("chat_history") or []
    if not isinstance(messages, Sequence):
        return []
    snippets: List[str] = []
    for message in messages[-limit * 2 :]:
        if isinstance(message, HumanMessage):
            prefix = "Пользователь"
        elif isinstance(message, AIMessage):
            prefix = "Ассистент"
        elif isinstance(message, BaseMessage):
            prefix = getattr(message, "type", "Сообщение")
        else:
            continue
        text = _message_content_to_text(message)
        if not text or text.startswith(PLAN_STEP_MARKER):
            continue
        snippets.append(f"{prefix}: {text.strip()}")
    return snippets[-limit * 2 :]


def _build_step_prompt(plan: ActionPlan, step: PlanStep) -> str:
    plan_lines = []
    for item in plan.steps:
        postfix = f" (инструмент: {item.tool})" if item.tool else ""
        plan_lines.append(f"{item.number}. {item.title}{postfix}")
    plan_overview = "\n".join(plan_lines)
    details = step.details or "Без дополнительных пояснений."
    preferred_tool = step.tool or "Подбери подходящий метод самостоятельно."
    return dedent(
        f"""
        {PLAN_STEP_MARKER} Выполняем согласованный план для пользователя.

        Запрос пользователя: {plan.request}

        Согласованный план:
        {plan_overview}

        Текущий шаг {step.number}: {step.title}
        Детали шага: {details}
        Предпочтительный инструмент: {preferred_tool}

        Выполни только этот шаг. Используй инструменты Bitrix24 по необходимости и не запрашивай повторного подтверждения у пользователя. После выполнения предоставь краткий отчет о результате.
        """
    ).strip()


def _format_execution_report(plan: ActionPlan, results: Sequence[Tuple[PlanStep, str]]) -> str:
    lines: List[str] = []
    lines.append("План выполнен. Итоги по шагам:")
    for step, output in results:
        summary = _shorten_for_user(output) if output else "Шаг выполнен."
        lines.append(f"{step.number}. {step.title} — {summary}")
    if plan.notes:
        lines.append("")
        lines.append("Примечания к плану:")
        for note in plan.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)


class PlanExecutionError(RuntimeError):
    def __init__(self, step: PlanStep, message: str) -> None:
        super().__init__(message)
        self.step = step




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


def _build_calendar_shortcut_plan(text: str) -> Optional[ActionPlan]:
    if not _should_auto_fetch_calendar(text):
        return None

    logger.debug("Сформирована заготовка плана для календарного запроса: %s", _format_for_log(text))
    steps = [
        PlanStep(
            number=1,
            title="Проверю запрос",
            details="Пойму из сообщения дату и сотрудников, а также уточню недостающие детали.",
            tool=None,
        ),
        PlanStep(
            number=2,
            title="Запрошу события в Bitrix24",
            details="Использую инструмент calendar_events_for_users, чтобы получить список встреч на выбранный день.",
            tool="calendar_events_for_users",
        ),
        PlanStep(
            number=3,
            title="Подготовлю ответ",
            details="Соберу краткую сводку найденных событий и выделю ключевые моменты.",
            tool=None,
        ),
    ]
    raw_payload = {
        "plan_summary": "Проверю календарь и подготовлю сводку по встречам.",
        "steps": [
            {"id": step.number, "title": step.title, "tool": step.tool, "details": step.details}
            for step in steps
        ],
        "notes": [
            "Если нужна другая дата или сотрудники, напиши об этом до подтверждения плана."
        ],
    }
    return ActionPlan(
        request=text.strip(),
        steps=steps,
        summary="Покажу встречи сотрудников на выбранный день.",
        raw_source=json.dumps(raw_payload, ensure_ascii=False),
        notes=list(raw_payload["notes"]),
    )



# --- Управление сессиями агента ---------------------------------------------


class AgentManager:
    def __init__(self, llm: ChatOpenAI, client: Bitrix24Client, verbose: bool = False) -> None:
        self._llm = llm
        self._client = client
        self._verbose = verbose
        self._sessions: Dict[int, ChatSession] = {}
        self._tool_preparation: ToolPreparation = prepare_tools(client)
        self._plan_generator = PlanGenerator(
            llm=self._llm,
            capabilities=self._tool_preparation.capabilities,
        )

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
                ),
                preparation=self._tool_preparation,
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

    def generate_plan(self, chat_id: int, session: ChatSession, request: str) -> ActionPlan:
        history = _get_recent_history_for_plan(session)
        plan = self._plan_generator.generate(request, history=history)
        session.pending_plan = plan
        logger.debug("Сформирован план для чата %s", chat_id)
        logger.debug("Черновик плана: %s", _format_for_log(plan.raw_source))
        return plan

    def execute_plan(self, chat_id: int, session: ChatSession, plan: ActionPlan) -> List[Tuple[PlanStep, str]]:
        results: List[Tuple[PlanStep, str]] = []
        for step in plan.steps:
            prompt = _build_step_prompt(plan, step)
            logger.debug("Выполняю шаг %s для чата %s: %s", step.number, chat_id, _format_for_log(prompt))
            response = session.agent.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": str(chat_id)}},
            )
            output = _extract_agent_output(response).strip()
            logger.debug("Результат шага %s для чата %s: %s", step.number, chat_id, _format_for_log(output))
            if _looks_like_clarification(output):
                raise PlanExecutionError(step, output)
            results.append((step, output))
        return results

    def build_execution_reply(self, chat_id: int, plan: ActionPlan, results: Sequence[Tuple[PlanStep, str]]) -> str:
        summary = self._summarize_results(chat_id, plan, results)
        if summary:
            return summary
        return _format_execution_report(plan, results)

    def _summarize_results(self, chat_id: int, plan: ActionPlan, results: Sequence[Tuple[PlanStep, str]]) -> str:
        if not results:
            logger.debug("No step results to summarize: chat_id=%s", chat_id)
            return ""
        steps_blocks = []
        for step, output in results:
            raw_output = (output or "").strip() or "—"
            step_lines = [
                f"Шаг {step.number}: {step.title}",
                f"Инструмент: {step.tool or 'не использовался'}",
                "Сырые данные:",
                raw_output,
            ]
            steps_blocks.append("\n".join(step_lines))
        steps_text = "\n\n".join(steps_blocks)
        notes_text = ""
        if plan.notes:
            notes_lines = "\n".join(f"- {note}" for note in plan.notes)
            notes_text = f"\n\nЗаметки плана:\n{notes_lines}"
        user_prompt = dedent(
            f"""
            Запрос пользователя: {plan.request}

            Цель плана: {plan.summary or 'не указана'}.

            Результаты шагов:
            {steps_text}{notes_text}

            Сформулируй короткий и понятный итоговый ответ. Покажи только то,
            что помогает пользователю решить исходную задачу. Не включай
            идентификаторы, поля или подробности, о которых не просили.
            Если данных недостаточно, скажи об этом и предложи следующий шаг.
            """
        ).strip()
        try:
            response = self._llm.invoke(
                [
                    SystemMessage(content=_EXECUTION_SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception:
            logger.exception("Failed to build execution summary: chat_id=%s", chat_id)
            return ""
        summary_text = _extract_agent_output(response).strip()
        if not summary_text:
            logger.debug("LLM returned empty execution summary: chat_id=%s", chat_id)
        else:
            logger.debug("Execution summary for chat %s: %s", chat_id, _format_for_log(summary_text))
        return summary_text



# --- Telegram-команды --------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return
    await _safe_reply_text(
        update.effective_message,
        "Привет! Я ассистент по Bitrix24. Опишите задачу, и я постараюсь выполнить её автоматически."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_user_allowed(update):
        await _reply_unauthorized(update)
        return
    await _safe_reply_text(
        update.effective_message,
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
    await _safe_reply_text(
        update.effective_message,
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
        await _safe_reply_text(message, "Please send a text message.")
        return

    user = update.effective_user
    logger.info(
        "Incoming message: chat_id=%s user_id=%s username=%s text=%s",
        chat_id,
        getattr(user, "id", None),
        getattr(user, "username", None),
        _format_for_log(text),
    )

    loop = asyncio.get_running_loop()
    session = manager.get_session(chat_id)
    replacing_plan = False

    if session.pending_plan:
        if _is_plan_confirmation(text):
            plan_to_execute = session.pending_plan
            session.pending_plan = None
            await _safe_reply_text(message, "Starting the approved plan. This might take a few moments.")
            try:
                results = await loop.run_in_executor(
                    None,
                    lambda: manager.execute_plan(chat_id, session, plan_to_execute),
                )
            except PlanExecutionError as exc:
                logger.info("Plan step %s needs clarification: chat_id=%s", exc.step.number, chat_id)
                session.pending_plan = plan_to_execute
                clarification = (
                    f"Step {exc.step.number} \"{exc.step.title}\" requires clarification: {exc}. "
                    "Please answer the question or provide the missing data, then confirm the plan again.",
                )
                await _safe_reply_text(message, " ".join(clarification))
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("Approved plan execution failed: chat_id=%s", chat_id)
                session.pending_plan = plan_to_execute
                await _safe_reply_text(
                    message,
                    "Could not finish the approved plan. Try again later or adjust the request. "
            f"\nDetails: {exc}",
                )
                return
            summary = manager.build_execution_reply(chat_id, plan_to_execute, results)
            logger.info("Plan finished for chat %s", chat_id)
            await _safe_reply_text(message, summary or "Plan completed, но не удалось сформировать ответ.")
            return
        if _is_plan_cancellation(text):
            session.pending_plan = None
            await _safe_reply_text(message, "Plan cancelled. Describe a new task and I will prepare another plan.")
            return
        logger.debug("New request received while a plan is pending: chat_id=%s", chat_id)
        session.pending_plan = None
        replacing_plan = True

    shortcut_plan = _build_calendar_shortcut_plan(text)
    if shortcut_plan:
        session.pending_plan = shortcut_plan
        response_text = shortcut_plan.format_for_user()
        if replacing_plan:
            response_text = "Обновлённый план для последнего запроса:\n\n" + response_text
        await _safe_reply_text(message, response_text)
        return

    try:
        plan = await loop.run_in_executor(
            None,
            lambda: manager.generate_plan(chat_id, session, text),
        )
    except PlanGenerationError as exc:
        logger.warning("Plan generation failed: chat_id=%s reason=%s", chat_id, exc)
        await _safe_reply_text(
            message,
            "Failed to build a plan. Try refining the request or break the task into smaller steps."
            f"\nDetails: {exc}",
        )
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected planning error: chat_id=%s", chat_id)
        await _safe_reply_text(
            message,
            "A technical error occurred while planning. Please try again later.",
        )
        return

    logger.info("Plan prepared for chat %s", chat_id)
    response_text = plan.format_for_user()
    if replacing_plan:
        response_text = "Updated plan for the latest request:\n\n" + response_text
    await _safe_reply_text(message, response_text)


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

