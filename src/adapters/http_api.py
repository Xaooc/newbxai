"""HTTP API-адаптер для взаимодействия с оркестратором."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from src.app import build_orchestrator
from src.orchestrator.agent import Orchestrator

logger = logging.getLogger(__name__)


@dataclass
class HttpAdapterConfig:
    """Настройки HTTP API-адаптера."""

    host: str = "0.0.0.0"
    port: int = 8080
    mode: str = "shadow"
    state_dir: Path = field(default_factory=lambda: Path("./data/state"))
    log_dir: Path = field(default_factory=lambda: Path("./data/logs"))
    allowed_tokens: Tuple[str, ...] = field(default_factory=tuple)
    auth_header: str = "X-API-Key"
    max_failed_attempts: int = 5
    audit_window_seconds: int = 900
    block_duration_seconds: int = 900


def _utc_now() -> datetime:
    """Возвращает текущее время в UTC с таймзоной."""

    return datetime.now(timezone.utc)


class AuthAttemptTracker:
    """Отслеживает неудачные попытки аутентификации и блокировки по IP."""

    def __init__(
        self,
        *,
        max_attempts: int,
        window_seconds: int,
        block_seconds: int,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts должен быть положительным")
        if window_seconds <= 0 or block_seconds <= 0:
            raise ValueError("Параметры window_seconds и block_seconds должны быть положительными")
        self.max_attempts = max_attempts
        self.window = window_seconds
        self.block_seconds = block_seconds
        self._now = now_provider or _utc_now
        self._failures: Dict[str, list[datetime]] = {}
        self._blocked_until: Dict[str, datetime] = {}

    def register_failure(self, identifier: str) -> Tuple[int, Optional[datetime]]:
        """Регистрирует неудачную попытку и возвращает счётчик и время блокировки."""

        now = self._now()
        self._purge_expired(identifier, now)
        attempts = self._failures.setdefault(identifier, [])
        attempts.append(now)
        block_until: Optional[datetime] = None
        if len(attempts) >= self.max_attempts:
            block_until = now + timedelta(seconds=self.block_seconds)
            self._blocked_until[identifier] = block_until
        return len(attempts), block_until

    def register_success(self, identifier: str) -> None:
        """Сбрасывает счётчик неудачных попыток и снимает блокировку."""

        self._failures.pop(identifier, None)
        self._blocked_until.pop(identifier, None)

    def is_blocked(self, identifier: str) -> Optional[datetime]:
        """Возвращает время окончания блокировки, если она активна."""

        now = self._now()
        blocked_until = self._blocked_until.get(identifier)
        if blocked_until and blocked_until <= now:
            self.register_success(identifier)
            return None
        return blocked_until

    def _purge_expired(self, identifier: str, now: datetime) -> None:
        """Удаляет устаревшие записи об ошибках вне окна наблюдения."""

        attempts = self._failures.get(identifier)
        if not attempts:
            return
        threshold = now - timedelta(seconds=self.window)
        self._failures[identifier] = [moment for moment in attempts if moment >= threshold]


def _normalize_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Возвращает кортеж непустых токенов без лишних пробелов."""

    return tuple(sorted({token.strip() for token in tokens if token and token.strip()}))


def is_token_allowed(provided: str, allowed_tokens: Tuple[str, ...]) -> bool:
    """Проверяет, входит ли токен в список разрешённых."""

    normalized = _normalize_tokens(allowed_tokens)
    return provided.strip() in normalized


class AgentHttpRequestHandler(BaseHTTPRequestHandler):
    """Обработчик HTTP-запросов, взаимодействующий с оркестратором."""

    orchestrator: Orchestrator
    config: HttpAdapterConfig
    orchestrator_cache: Dict[Tuple[str, str, str], Orchestrator]
    auth_tracker: AuthAttemptTracker

    def do_POST(self) -> None:  # noqa: N802 (совместимость с BaseHTTPRequestHandler)
        """Обрабатывает POST-запросы к эндпоинту `/chat`."""

        if self.path != "/chat":
            self._send_json(404, {"error": "endpoint_not_found", "message": "Эндпоинт не найден"})
            return

        blocked_until = self._blocked_until()
        if blocked_until:
            self._send_blocked_response(blocked_until)
            return

        if not self._is_authorized():
            blocked_until = self._blocked_until()
            if blocked_until:
                self._send_blocked_response(blocked_until)
            else:
                self._send_json(
                    401,
                    {"error": "unauthorized", "message": "Недействительный или отсутствующий API-токен"},
                    headers={"WWW-Authenticate": self.config.auth_header},
                )
            return

        length_header = self.headers.get("Content-Length", "0")
        try:
            content_length = int(length_header)
        except ValueError:
            self._send_json(400, {"error": "invalid_length", "message": "Некорректный заголовок Content-Length"})
            return

        body = self.rfile.read(content_length) if content_length > 0 else b""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json", "message": "Не удалось разобрать тело запроса как JSON"})
            return

        user_id = payload.get("user_id")
        message = payload.get("message")

        if not user_id or not isinstance(user_id, str):
            self._send_json(400, {"error": "missing_user_id", "message": "Поле user_id обязательно и должно быть строкой"})
            return
        if not message or not isinstance(message, str):
            self._send_json(400, {"error": "missing_message", "message": "Поле message обязательно и должно быть строкой"})
            return

        try:
            orchestrator = self._resolve_orchestrator(payload)
            assistant_reply = orchestrator.process_message(user_id, message)
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Ошибка обработки запроса", extra={"path": self.path})
            self._send_json(500, {"error": "internal_error", "message": str(exc)})
            return

        self._send_json(200, {"assistant": assistant_reply})

    def log_message(self, format: str, *args) -> None:  # noqa: A003 (совместимость)
        """Переопределяем стандартный вывод в stderr и используем logging."""

        logger.info("HTTP %s - %s", self.address_string(), format % args)

    def _resolve_orchestrator(self, payload: Dict[str, object]) -> Orchestrator:
        """Возвращает оркестратор, учитывая переопределения из запроса."""

        mode = payload.get("mode") if isinstance(payload.get("mode"), str) else self.config.mode
        state_dir = payload.get("state_dir") if isinstance(payload.get("state_dir"), str) else str(self.config.state_dir)
        log_dir = payload.get("log_dir") if isinstance(payload.get("log_dir"), str) else str(self.config.log_dir)

        key = (mode, state_dir, log_dir)
        orchestrator = self.orchestrator_cache.get(key)
        if orchestrator is not None:
            return orchestrator

        logger.info(
            "Создаём новый оркестратор для HTTP-запроса", extra={"mode": mode, "state_dir": state_dir, "log_dir": log_dir}
        )
        orchestrator = build_orchestrator(mode, Path(state_dir), Path(log_dir))
        self.orchestrator_cache[key] = orchestrator
        return orchestrator

    def _is_authorized(self) -> bool:
        """Проверяет наличие корректного API-токена."""

        tokens = _normalize_tokens(self.config.allowed_tokens)
        if not tokens:
            logger.error(
                "HTTP API отклоняет запрос: не настроены токены доступа",
                extra={"remote": self.address_string()},
            )
            return False

        provided = self.headers.get(self.config.auth_header) or ""
        remote = self._remote_ip()
        if is_token_allowed(provided, tokens):
            if hasattr(self, "auth_tracker"):
                self.auth_tracker.register_success(remote)
            return True

        attempts = 1
        block_until: Optional[datetime] = None
        if hasattr(self, "auth_tracker"):
            attempts, block_until = self.auth_tracker.register_failure(remote)
        extra: Dict[str, object] = {"remote": self.address_string(), "attempts": attempts}
        if block_until:
            extra["blocked_until"] = block_until.isoformat()
            logger.error("HTTP API: IP заблокирован из-за повторных ошибок", extra=extra)
        else:
            logger.warning("HTTP API: попытка доступа с неверным токеном", extra=extra)
        return False

    def _blocked_until(self) -> Optional[datetime]:
        """Возвращает время разблокировки, если адрес заблокирован."""

        if not hasattr(self, "auth_tracker"):
            return None
        return self.auth_tracker.is_blocked(self._remote_ip())

    def _remote_ip(self) -> str:
        """Возвращает IP-адрес клиента."""

        return self.client_address[0] if self.client_address else "unknown"

    def _send_blocked_response(self, blocked_until: datetime) -> None:
        """Возвращает ответ о временной блокировке клиента."""

        retry_after = max(1, int((blocked_until - _utc_now()).total_seconds()))
        payload = {
            "error": "too_many_requests",
            "message": "Доступ временно заблокирован из-за повторных ошибок аутентификации",
            "blocked_until": blocked_until.isoformat().replace("+00:00", "Z"),
        }
        self._send_json(429, payload, headers={"Retry-After": str(retry_after)})

    def _send_json(self, status: int, payload: Dict[str, object], *, headers: Dict[str, str] | None = None) -> None:
        """Отправляет JSON-ответ клиенту."""

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_http_server(config: HttpAdapterConfig | None = None) -> None:
    """Запускает HTTP API-сервер."""

    active_config = config or HttpAdapterConfig()
    if not active_config.allowed_tokens:
        env_tokens = tuple(token.strip() for token in os.getenv("HTTP_API_TOKENS", "").split(","))
        active_config.allowed_tokens = _normalize_tokens(env_tokens)

    if not active_config.allowed_tokens:
        raise RuntimeError(
            "Не заданы разрешённые токены HTTP API. Укажите их в конфигурации или переменной окружения HTTP_API_TOKENS."
        )

    base_orchestrator = build_orchestrator(active_config.mode, active_config.state_dir, active_config.log_dir)

    class Handler(AgentHttpRequestHandler):
        pass

    Handler.orchestrator = base_orchestrator
    Handler.config = active_config
    Handler.orchestrator_cache = {
        (active_config.mode, str(active_config.state_dir), str(active_config.log_dir)): base_orchestrator
    }
    Handler.auth_tracker = AuthAttemptTracker(
        max_attempts=active_config.max_failed_attempts,
        window_seconds=active_config.audit_window_seconds,
        block_seconds=active_config.block_duration_seconds,
    )

    server = HTTPServer((active_config.host, active_config.port), Handler)
    logger.info(
        "HTTP API запущен", extra={
            "host": active_config.host,
            "port": active_config.port,
            "mode": active_config.mode,
            "state_dir": str(active_config.state_dir),
            "log_dir": str(active_config.log_dir),
        }
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Остановка HTTP API по сигналу KeyboardInterrupt")
    finally:
        server.server_close()
        logger.info("HTTP API остановлен")
