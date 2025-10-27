"""Проверки загрузки настроек из окружения."""
from __future__ import annotations

from telegram_bitrix_agent import config


def test_settings_from_env_reads_dotenv(monkeypatch, tmp_path) -> None:
    """Убеждаемся, что настройки подхватываются из файла .env."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        """
TELEGRAM_BOT_TOKEN=test-token
OPENAI_API_KEY=test-key
BITRIX_WEBHOOK_URL=https://example.com/webhook
        """.strip()
    )

    for key in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "BITRIX_WEBHOOK_URL"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(config, "DEFAULT_DOTENV_PATH", env_path)

    settings = config.Settings.from_env()

    assert settings.bot_token == "test-token"
    assert settings.openai_api_key == "test-key"
    assert settings.bitrix_webhook == "https://example.com/webhook"
