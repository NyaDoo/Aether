from __future__ import annotations

import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.models.database import ApiKey, RequestCandidate, User

from .gateway_contract import GatewayStreamReportRequest, GatewaySyncReportRequest
from .gateway_reporting_common import _gateway_module


def _build_gateway_sync_telemetry_writer(
    *,
    db: Session,
    request_id: str,
    user_id: str,
    api_key_id: str,
    fallback_telemetry: Any,
) -> Any:
    from src.config.settings import config
    from src.services.system.config import SystemConfigService
    from src.services.usage.telemetry_writer import DbTelemetryWriter, QueueTelemetryWriter

    if config.usage_queue_enabled and user_id and api_key_id:
        try:
            log_level = SystemConfigService.get_request_record_level(db).value
            sensitive_headers = SystemConfigService.get_sensitive_headers(db) or []
            max_request_body_size = int(
                SystemConfigService.get_config(db, "max_request_body_size", 5242880) or 0
            )
            max_response_body_size = int(
                SystemConfigService.get_config(db, "max_response_body_size", 5242880) or 0
            )
            return QueueTelemetryWriter(
                request_id=request_id,
                user_id=user_id,
                api_key_id=api_key_id,
                log_level=log_level,
                sensitive_headers=sensitive_headers,
                max_request_body_size=max_request_body_size,
                max_response_body_size=max_response_body_size,
            )
        except Exception:
            return DbTelemetryWriter(fallback_telemetry)

    return DbTelemetryWriter(fallback_telemetry)


async def _dispatch_gateway_sync_telemetry(
    *,
    telemetry_writer: Any,
    operation: str,
    **kwargs: Any,
) -> None:
    gateway_module = _gateway_module()
    submitter = getattr(telemetry_writer, operation, None)
    if not callable(submitter):
        raise AttributeError(f"Telemetry writer missing operation: {operation}")

    if bool(getattr(telemetry_writer, "supports_background_submission", lambda: False)()):
        request_id = str(getattr(telemetry_writer, "request_id", "") or "unknown")

        async def _run_in_background() -> None:
            try:
                await submitter(**kwargs)
            except Exception as exc:
                logger.warning(
                    "[gateway] background telemetry submission failed: request_id={}, operation={}, error={}",
                    request_id,
                    operation,
                    exc,
                )

        task = gateway_module.safe_create_task(_run_in_background())
        if task is None:
            await submitter(**kwargs)
        return

    await submitter(**kwargs)


async def _schedule_gateway_sync_telemetry(
    *,
    background_tasks: BackgroundTasks | None,
    telemetry_writer: Any,
    operation: str,
    **kwargs: Any,
) -> None:
    gateway_module = _gateway_module()
    if background_tasks is not None:
        background_tasks.add_task(
            gateway_module._dispatch_gateway_sync_telemetry,
            telemetry_writer=telemetry_writer,
            operation=operation,
            **kwargs,
        )
        return

    await gateway_module._dispatch_gateway_sync_telemetry(
        telemetry_writer=telemetry_writer,
        operation=operation,
        **kwargs,
    )


def _build_gateway_usage_metadata(
    *,
    request_metadata: dict[str, Any] | None = None,
    response_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    metadata: dict[str, Any] | None = None
    if request_metadata:
        metadata = dict(request_metadata)
        if response_metadata:
            metadata.setdefault("response", response_metadata)
    elif response_metadata:
        metadata = dict(response_metadata)
    return metadata
