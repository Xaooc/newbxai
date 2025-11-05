"""Пакет с автономным агентом Bitrix24."""

from .agent import BitrixAutonomousAgent
from .client import BitrixWebhookClient

__all__ = ["BitrixAutonomousAgent", "BitrixWebhookClient"]
