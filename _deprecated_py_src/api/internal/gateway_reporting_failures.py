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


def _resolve_gateway_failure_adapter(client_api_format: str, *, cli: bool) -> Any | None:
    if cli:
        from src.api.handlers.claude_cli import ClaudeCliAdapter
        from src.api.handlers.gemini_cli import GeminiCliAdapter
        from src.api.handlers.openai_cli import OpenAICliAdapter, OpenAICompactAdapter

        if client_api_format == "openai:compact":
            return OpenAICompactAdapter()
        if client_api_format == "claude:cli":
            return ClaudeCliAdapter()
        if client_api_format == "gemini:cli":
            return GeminiCliAdapter()
        if client_api_format == "openai:cli":
            return OpenAICliAdapter()
        return None

    from src.api.handlers.claude import ClaudeChatAdapter
    from src.api.handlers.gemini import GeminiChatAdapter
    from src.api.handlers.openai import OpenAIChatAdapter

    if client_api_format == "claude:chat":
        return ClaudeChatAdapter()
    if client_api_format == "gemini:chat":
        return GeminiChatAdapter()
    if client_api_format == "openai:chat":
        return OpenAIChatAdapter()
    return None


async def _record_gateway_sync_failure(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
    cli: bool,
) -> None:
    from src.api.handlers.base.stream_context import is_format_converted
    from src.api.handlers.base.utils import filter_proxy_response_headers

    gateway_module = _gateway_module()
    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    if not user_id or not api_key_id:
        return

    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    provider_api_format = str(context.get("provider_api_format") or "").strip().lower()
    adapter = _resolve_gateway_failure_adapter(client_api_format, cli=cli)
    if adapter is None:
        return

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        return

    request_id = str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]).strip()
    model = str(context.get("model") or "unknown").strip() or "unknown"
    handler = adapter._create_handler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=request_id,
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    request_metadata: dict[str, Any] = {
        "gateway_direct_executor": True,
        "phase": "3c_trial",
    }
    proxy_info = context.get("proxy_info")
    if isinstance(proxy_info, dict):
        request_metadata["proxy"] = proxy_info

    response_time_ms = 0
    if isinstance(payload.telemetry, dict):
        raw_elapsed_ms = payload.telemetry.get("elapsed_ms")
        try:
            response_time_ms = max(int(raw_elapsed_ms or 0), 0)
        except (TypeError, ValueError):
            response_time_ms = 0

    client_response_headers = filter_proxy_response_headers(dict(payload.headers or {}))
    client_response_headers["content-type"] = "application/json"
    telemetry_writer = gateway_module._build_gateway_sync_telemetry_writer(
        db=db,
        request_id=request_id,
        user_id=user_id,
        api_key_id=api_key_id,
        fallback_telemetry=handler.telemetry,
    )
    await gateway_module._dispatch_gateway_sync_telemetry(
        telemetry_writer=telemetry_writer,
        operation="record_failure",
        provider=str(context.get("provider_name") or "unknown"),
        model=model,
        response_time_ms=response_time_ms,
        status_code=payload.status_code,
        error_message=gateway_module._extract_gateway_sync_error_message(payload),
        request_headers=dict(context.get("original_headers") or {}),
        request_body=dict(context.get("original_request_body") or {}),
        provider_request_body=context.get("provider_request_body"),
        is_stream=False,
        api_format=client_api_format,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        provider_request_headers=dict(context.get("provider_request_headers") or {}),
        response_headers=dict(payload.headers or {}),
        client_response_headers=client_response_headers,
        provider_id=str(context.get("provider_id") or "") or None,
        provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
        provider_api_key_id=str(context.get("key_id") or "") or None,
        endpoint_api_format=provider_api_format or None,
        has_format_conversion=is_format_converted(provider_api_format, client_api_format),
        target_model=str(context.get("mapped_model") or "") or None,
        metadata=request_metadata,
    )


async def _record_gateway_chat_sync_failure(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    await _record_gateway_sync_failure(payload, db=db, cli=False)


async def _record_gateway_cli_sync_failure(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    await _record_gateway_sync_failure(payload, db=db, cli=True)


async def _record_gateway_video_sync_failure(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    _ = (payload, db)
