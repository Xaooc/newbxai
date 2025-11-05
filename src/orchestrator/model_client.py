"""Клиент для обращения к ChatGPT (модели семейства GPT-4.1)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.orchestrator.context_builder import build_state_summary

logger = logging.getLogger(__name__)


DEFAULT_MODEL_NAME = "gpt-4.1"
MODEL_ENV_VAR = "OPENAI_MODEL"

METHODS_CHEATSHEET = (
    "user.current — данные владельца вебхука.\n"
    "user.get — поиск сотрудников по имени, отделу, e-mail; не использовать для клиентов.\n"
    "crm.contact.list — поиск клиентов CRM по имени, телефону, e-mail.\n"
    "crm.contact.get — карточка клиента по ID.\n"
    "crm.company.list — поиск компаний по отрасли, ответственному, признакам.\n"
    "crm.company.get — карточка компании по ID.\n"
    "crm.deal.list — список сделок (фильтры: менеджер, стадия, направление, сумма).\n"
    "crm.deal.get — карточка сделки по ID.\n"
    "crm.deal.add — создание сделки (TITLE обязателен; при изменении суммы или ответственного напомни пользователю проверить данные).\n"
    "crm.deal.update — обновление сделки (id + fields).\n"
    "crm.deal.category.list — воронки продаж.\n"
    "crm.deal.category.stage.list — стадии выбранной воронки.\n"
    "crm.status.list — элементы справочников CRM (ENTITY_ID обязателен).\n"
    "crm.activity.list — список дел (фильтры по OWNER_TYPE_ID/OWNER_ID рекомендуется).\n"
    "crm.activity.add — создать дело (OWNER_TYPE_ID, OWNER_ID, TYPE_ID, SUBJECT).\n"
    "crm.timeline.comment.add — добавить комментарий к сущности CRM.\n"
    "tasks.task.add — создать задачу (TITLE, DESCRIPTION, RESPONSIBLE_ID).\n"
    "tasks.task.update — изменить задачу (taskId + fields).\n"
    "tasks.task.list — поиск задач по фильтрам (RESPONSIBLE_ID, STATUS и др.).\n"
    "task.commentitem.add — комментарий к задаче (taskId, POST_MESSAGE).\n"
    "task.checklistitem.add — пункт чек-листа (taskId, TITLE).\n"
    "sonet.group.get — сведения о рабочей группе/проекте.\n"
    "sonet.group.user.get — участники группы (GROUP_ID).\n"
    "batch — пакет до 50 команд (cmd обязательный, наследует ограничения).\n"
    "event.bind — подписка на событие (event, handler, обязательно предупреди, что меняются вебхуки).\n"
    "event.get — перечень активных подписок.\n"
    "event.unbind — снять подписку (event, handler, обязательно предупреди, что уведомление отключено)."
)


class ModelClientError(Exception):
    """Исключение, возникающее при ошибке общения с моделью."""


@dataclass
class ModelClient:
    """Минимальный HTTP-клиент для вызова ChatGPT."""

    model_name: str = DEFAULT_MODEL_NAME
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504)
    temperature: float = 1

    def __post_init__(self) -> None:
        if self.api_key is None:
            raise ModelClientError("ChatGPT API key is missing (set OPENAI_API_KEY)")

        self.timeout = self._normalize_timeout(self.timeout)
        if self.timeout <= 0:
            raise ModelClientError("Timeout must be greater than 0 seconds")

        if self.max_retries < 0:
            logger.warning("OPENAI_MAX_RETRIES=%s is negative; falling back to 0.", self.max_retries)
            self.max_retries = 0

        if self.retry_backoff_factor < 0:
            logger.warning(
                "OPENAI_RETRY_BACKOFF=%s is negative; falling back to 0.",
                self.retry_backoff_factor,
            )
            self.retry_backoff_factor = 0.0

        self._session = self._build_session()


    def generate(
        self,
        system_prompt: str,
        state_snapshot: Dict[str, Any],
        user_message: str,
    ) -> str:
        """Запрашивает у модели ответ с блоками THOUGHT/ACTION/ASSISTANT."""

        url = self._build_url("/chat/completions")
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": self._build_messages(system_prompt, state_snapshot, user_message),
        }

        logger.info(
            "Запрос к модели",
            extra={"model": self.model_name, "payload": payload},
        )

        logger.debug("Отправляем запрос в модель", extra={"url": url, "model": self.model_name})

        try:
            response = self._session.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ModelClientError(f"Ошибка сети при обращении к модели: {exc}") from exc
        logger.info(
            "Ответ от модели (HTTP)",
            extra={"model": self.model_name, "status_code": response.status_code},
        )

        if response.status_code != 200:
            raise ModelClientError(
                f"Модель вернула статус {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise ModelClientError(
                f"Ответ модели не является корректным JSON: {response.text}"
            ) from exc
        logger.info(
            "Ответ от модели (JSON)",
            extra={"model": self.model_name, "response": data},
        )
        content = self._extract_content(data)
        logger.info(
            "Ответ от модели (контент)",
            extra={"model": self.model_name, "content": content},
        )
        if not content:
            raise ModelClientError("Пустой ответ от модели")
        return content

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            redirect=0,
            backoff_factor=self.retry_backoff_factor,
            status_forcelist=self.retry_statuses,
            allowed_methods=frozenset({"POST"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @staticmethod
    def _normalize_timeout(timeout: float) -> float:
        if isinstance(timeout, (int, float)):
            return float(timeout)
        try:
            return float(timeout)
        except (TypeError, ValueError) as exc:
            raise ModelClientError(f"Timeout value must be convertible to float, got {timeout!r}") from exc

    def _build_messages(
        self,
        system_prompt: str,
        state_snapshot: Dict[str, Any],
        user_message: str,
    ) -> List[Dict[str, str]]:
        summary = build_state_summary(state_snapshot)
        state_json = json.dumps(state_snapshot, ensure_ascii=False, indent=2)
        state_block = (
            "Краткая сводка состояния:\n"
            f"{summary}\n\n"
            "Текущее состояние агента (JSON):\n"
            f"```json\n{state_json}\n```"
        )
        methods_block = "Описание доступных методов:\n" + METHODS_CHEATSHEET
        user_payload = (
            f"{state_block}\n\n"
            f"{methods_block}\n\n"
            f"Запрос пользователя: {user_message}\n\n"
            "Сформируй ответ строго с блоками THOUGHT:/ACTION:/ASSISTANT:."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_payload,
            },
        ]

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    @staticmethod
    def _extract_content(payload: Dict[str, Any]) -> Optional[str]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content
        return None


def resolve_model_name(preferred_name: Optional[str] = None) -> str:
    """Определяет имя модели, учитывая окружение и переданное значение."""

    if preferred_name:
        candidate = preferred_name.strip()
        if candidate:
            return candidate

    env_value = os.getenv(MODEL_ENV_VAR, "").strip()
    if env_value:
        return env_value

    return DEFAULT_MODEL_NAME


def build_default_model_client(model_name: Optional[str] = None) -> ModelClient:
    """Создаёт клиент, используя переменные окружения."""

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    resolved_model_name = resolve_model_name(model_name)
    timeout = _read_float_env("OPENAI_TIMEOUT", default=60.0, minimum=1.0)
    max_retries = _read_int_env("OPENAI_MAX_RETRIES", default=3, minimum=0)
    backoff = _read_float_env("OPENAI_RETRY_BACKOFF", default=1.0, minimum=0.0)
    return ModelClient(
        model_name=resolved_model_name,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_factor=backoff,
    )


def _read_float_env(name: str, default: float, minimum: Optional[float] = None) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    candidate = raw_value.strip()
    if not candidate:
        return default
    try:
        value = float(candidate)
    except ValueError:
        logger.warning("%s=%s is not a valid float. Using %s.", name, raw_value, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is below minimum. Using %s.", name, raw_value, default)
        return default
    return value


def _read_int_env(name: str, default: int, minimum: Optional[int] = None) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    candidate = raw_value.strip()
    if not candidate:
        return default
    try:
        value = int(candidate)
    except ValueError:
        logger.warning("%s=%s is not a valid int. Using %s.", name, raw_value, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%s is below minimum. Using %s.", name, raw_value, default)
        return default
    return value
