"""Вспомогательные классы для выгрузки архивов логов."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .logger import ArchiveUploader, ArchiveUploadError

logger = logging.getLogger(__name__)

try:  # pragma: no cover - зависимость опциональная
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # noqa: BLE001 - любые проблемы с импортом должны обрабатываться одинаково
    boto3 = None
    BotoCoreError = ClientError = Exception


def _to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class S3ArchiveUploader(ArchiveUploader):
    """Загрузчик архивов в Amazon S3 или совместимые хранилища."""

    bucket: str
    prefix: str = ""
    delete_after_upload: bool = False
    client_kwargs: Optional[Dict[str, Any]] = None
    client: Any | None = None

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket обязателен для загрузчика S3")
        if self.client is None:
            if boto3 is None:
                raise RuntimeError(
                    "boto3 не установлен. Установите boto3 или передайте готовый клиент в S3ArchiveUploader",
                )
            self.client = boto3.client("s3", **(self.client_kwargs or {}))
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"

    def upload(
        self,
        archive_path: Path,
        *,
        user_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        if not archive_path.exists():
            raise ArchiveUploadError(f"Архив {archive_path} не найден для загрузки")

        key = f"{self.prefix}{archive_path.name}" if self.prefix else archive_path.name
        metadata = {}
        if user_id:
            metadata["user_id"] = user_id
        if timestamp:
            metadata["log_timestamp"] = timestamp

        extra_args: Dict[str, Any] = {}
        if metadata:
            extra_args["Metadata"] = metadata

        try:
            if extra_args:
                self.client.upload_file(str(archive_path), self.bucket, key, ExtraArgs=extra_args)
            else:
                self.client.upload_file(str(archive_path), self.bucket, key)
        except (BotoCoreError, ClientError) as exc:  # pragma: no cover - ошибки SDK сложно стабильно воспроизвести
            raise ArchiveUploadError(f"Ошибка загрузки архива в S3: {exc}") from exc

        if self.delete_after_upload:
            try:
                archive_path.unlink()
            except FileNotFoundError:
                logger.debug("Архив %s уже удалён локально", archive_path.name)


def build_archive_uploader_from_env() -> Optional[ArchiveUploader]:
    """Строит загрузчик архивов на основе переменных окружения."""

    bucket = os.getenv("LOG_ARCHIVE_S3_BUCKET")
    if not bucket:
        return None

    prefix = os.getenv("LOG_ARCHIVE_S3_PREFIX", "")
    delete_after_upload = _to_bool(os.getenv("LOG_ARCHIVE_S3_DELETE_LOCAL", "false"))
    region = os.getenv("LOG_ARCHIVE_S3_REGION")

    client_kwargs: Dict[str, Any] = {}
    if region:
        client_kwargs["region_name"] = region

    try:
        uploader = S3ArchiveUploader(
            bucket=bucket,
            prefix=prefix,
            delete_after_upload=delete_after_upload,
            client_kwargs=client_kwargs or None,
        )
    except RuntimeError as exc:
        logger.warning("Не удалось создать загрузчик S3: %s", exc)
        return None

    return uploader


__all__ = ["S3ArchiveUploader", "build_archive_uploader_from_env"]
