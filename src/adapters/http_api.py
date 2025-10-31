"""HTTP API-адаптер для взаимодействия с оркестратором."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Tuple

from src.main_entrypoint import build_orchestrator
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

    def do_POST(self) -> None:  # noqa: N802 (совместимость с BaseHTTPRequestHandler)
        """Обрабатывает POST-запросы к эндпоинту `/chat`."""

        if self.path != "/chat":
            self._send_json(404, {"error": "endpoint_not_found", "message": "Эндпоинт не найден"})
            return

        if not self._is_authorized():
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
        if is_token_allowed(provided, tokens):
            return True

        logger.warning(
            "HTTP API: попытка доступа с неверным токеном",
            extra={"remote": self.address_string()},
        )
        return False

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
