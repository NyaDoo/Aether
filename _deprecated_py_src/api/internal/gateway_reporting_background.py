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

_INLINE_SYNC_REPORT_KINDS = {
    "openai_video_create_sync_success",
    "openai_video_remix_sync_success",
    "gemini_video_create_sync_success",
}


def _gateway_sync_report_requires_inline(payload: GatewaySyncReportRequest) -> bool:
    return payload.report_kind in _INLINE_SYNC_REPORT_KINDS


async def _apply_gateway_sync_report(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()

    if payload.report_kind == "gemini_files_store_mapping":
        from src.api.public.gemini_files import _maybe_store_file_mapping_from_payload

        await _maybe_store_file_mapping_from_payload(
            status_code=payload.status_code,
            headers=dict(payload.headers or {}),
            content_bytes=gateway_module._extract_gateway_report_body_bytes(payload),
            file_key_id=str(payload.report_context.get("file_key_id") or "") or None,
            user_id=str(payload.report_context.get("user_id") or "") or None,
        )
    elif payload.report_kind == "gemini_files_delete_mapping" and payload.status_code < 300:
        from src.services.gemini_files_mapping import delete_file_key_mapping

        file_name = str(payload.report_context.get("file_name") or "").strip()
        if file_name:
            await delete_file_key_mapping(file_name)
    elif payload.report_kind == "openai_chat_sync_success":
        await gateway_module._record_gateway_openai_chat_sync_success(payload, db=db)
    elif payload.report_kind == "claude_chat_sync_success":
        await gateway_module._record_gateway_passthrough_chat_sync_success(payload, db=db)
    elif payload.report_kind == "gemini_chat_sync_success":
        await gateway_module._record_gateway_passthrough_chat_sync_success(payload, db=db)
    elif payload.report_kind == "openai_chat_sync_error":
        await gateway_module._record_gateway_chat_sync_failure(payload, db=db)
    elif payload.report_kind == "claude_chat_sync_error":
        await gateway_module._record_gateway_chat_sync_failure(payload, db=db)
    elif payload.report_kind == "gemini_chat_sync_error":
        await gateway_module._record_gateway_chat_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_cli_sync_success":
        await gateway_module._record_gateway_passthrough_cli_sync_success(payload, db=db)
    elif payload.report_kind == "claude_cli_sync_success":
        await gateway_module._record_gateway_passthrough_cli_sync_success(payload, db=db)
    elif payload.report_kind == "gemini_cli_sync_success":
        await gateway_module._record_gateway_passthrough_cli_sync_success(payload, db=db)
    elif payload.report_kind == "openai_video_delete_sync_success":
        await gateway_module._record_gateway_openai_video_delete_sync_success(payload, db=db)
    elif payload.report_kind == "openai_video_cancel_sync_success":
        await gateway_module._record_gateway_openai_video_cancel_sync_success(payload, db=db)
    elif payload.report_kind == "gemini_video_cancel_sync_success":
        await gateway_module._record_gateway_gemini_video_cancel_sync_success(payload, db=db)
    elif payload.report_kind == "openai_video_create_sync_success":
        await gateway_module._record_gateway_openai_video_create_sync_success(payload, db=db)
    elif payload.report_kind == "openai_video_remix_sync_success":
        await gateway_module._record_gateway_openai_video_remix_sync_success(payload, db=db)
    elif payload.report_kind == "gemini_video_create_sync_success":
        await gateway_module._record_gateway_gemini_video_create_sync_success(payload, db=db)
    elif payload.report_kind == "openai_cli_sync_error":
        await gateway_module._record_gateway_cli_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_compact_sync_error":
        await gateway_module._record_gateway_cli_sync_failure(payload, db=db)
    elif payload.report_kind == "claude_cli_sync_error":
        await gateway_module._record_gateway_cli_sync_failure(payload, db=db)
    elif payload.report_kind == "gemini_cli_sync_error":
        await gateway_module._record_gateway_cli_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_video_create_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_video_remix_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)
    elif payload.report_kind == "gemini_video_create_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_video_delete_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)
    elif payload.report_kind == "openai_video_cancel_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)
    elif payload.report_kind == "gemini_video_cancel_sync_error":
        await gateway_module._record_gateway_video_sync_failure(payload, db=db)


async def _run_gateway_sync_report_background(
    payload: GatewaySyncReportRequest,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    context = dict(payload.report_context or {})
    candidate = None
    try:
        candidate = gateway_module._ensure_gateway_request_candidate(
            db=db,
            report_context=context,
            trace_id=payload.trace_id,
            initial_status="pending",
        )
        await gateway_module._apply_gateway_sync_report(payload, db=db)
    except Exception as exc:
        logger.warning("gateway background sync report failed: {}", exc)
    finally:
        try:
            gateway_module._mark_gateway_sync_candidate_terminal_state(
                db=db,
                candidate=candidate,
                payload=payload,
            )
        except Exception as exc:
            logger.warning("gateway sync candidate finalize failed: {}", exc)


def _close_gateway_session(db: Session) -> None:
    try:
        db.close()
    except Exception as exc:
        logger.warning("gateway finalize session close failed: {}", exc)


def _resolve_gateway_background_db(app: Any | None) -> tuple[Any, Any | None]:
    gateway_module = _gateway_module()
    overrides = getattr(app, "dependency_overrides", None)
    if isinstance(overrides, dict):
        override = overrides.get(gateway_module.get_db)
        if callable(override):
            override_value = override()
            if inspect.isgenerator(override_value):
                generator = override_value
                db = next(generator)

                def _cleanup_override_generator() -> None:
                    try:
                        next(generator)
                    except StopIteration:
                        return
                    except Exception as exc:
                        logger.warning(
                            "gateway background override generator cleanup failed: {}",
                            exc,
                        )

                return db, _cleanup_override_generator
            return override_value, None

    db = gateway_module.create_session()
    return db, lambda: gateway_module._close_gateway_session(db)


async def _run_gateway_sync_report_background_with_session(
    payload: GatewaySyncReportRequest,
    app: Any | None,
) -> None:
    db, cleanup = _resolve_gateway_background_db(app)
    try:
        await _run_gateway_sync_report_background(payload, db)
    finally:
        if cleanup is not None:
            cleanup()


async def _apply_gateway_stream_report(
    payload: GatewayStreamReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()

    if payload.report_kind == "openai_chat_stream_success":
        await gateway_module._record_gateway_openai_chat_stream_success(payload, db=db)
    elif payload.report_kind == "claude_chat_stream_success":
        await gateway_module._record_gateway_passthrough_chat_stream_success(payload, db=db)
    elif payload.report_kind == "gemini_chat_stream_success":
        await gateway_module._record_gateway_passthrough_chat_stream_success(payload, db=db)
    elif payload.report_kind == "openai_cli_stream_success":
        await gateway_module._record_gateway_passthrough_cli_stream_success(payload, db=db)
    elif payload.report_kind == "claude_cli_stream_success":
        await gateway_module._record_gateway_passthrough_cli_stream_success(payload, db=db)
    elif payload.report_kind == "gemini_cli_stream_success":
        await gateway_module._record_gateway_passthrough_cli_stream_success(payload, db=db)


async def _run_gateway_stream_report_background(
    payload: GatewayStreamReportRequest,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    try:
        gateway_module._ensure_gateway_request_candidate(
            db=db,
            report_context=dict(payload.report_context or {}),
            trace_id=payload.trace_id,
            initial_status="streaming",
        )
        await gateway_module._apply_gateway_stream_report(payload, db=db)
    except Exception as exc:
        logger.warning("gateway background stream report failed: {}", exc)


async def _run_gateway_stream_report_background_with_session(
    payload: GatewayStreamReportRequest,
    app: Any | None,
) -> None:
    db, cleanup = _resolve_gateway_background_db(app)
    try:
        await _run_gateway_stream_report_background(payload, db)
    finally:
        if cleanup is not None:
            cleanup()


async def _run_gateway_video_finalize_submitted_background(
    *,
    db: Session,
    request_id: str,
    provider_name: str,
    provider_id: str | None,
    provider_endpoint_id: str | None,
    provider_api_key_id: str | None,
    response_time_ms: int,
    status_code: int,
    endpoint_api_format: str | None,
    provider_request_headers: dict[str, Any],
    response_headers: dict[str, Any],
    response_body: dict[str, Any],
) -> None:
    from src.services.usage.service import UsageService

    try:
        UsageService.finalize_submitted(
            db,
            request_id=request_id,
            provider_name=provider_name,
            provider_id=provider_id,
            provider_endpoint_id=provider_endpoint_id,
            provider_api_key_id=provider_api_key_id,
            response_time_ms=response_time_ms,
            status_code=status_code,
            endpoint_api_format=endpoint_api_format,
            provider_request_headers=provider_request_headers,
            response_headers=response_headers,
            response_body=response_body,
        )
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("gateway background video finalize_submitted failed: {}", exc)
