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


class _GatewayReportStreamContext:
    async def __aenter__(self) -> _GatewayReportStreamContext:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


async def _iter_gateway_report_body_chunks(body_bytes: bytes) -> Any:
    yield body_bytes


async def _record_gateway_openai_chat_stream_success(
    payload: GatewayStreamReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.base.stream_context import StreamContext
    from src.api.handlers.base.stream_processor import StreamProcessor
    from src.api.handlers.base.stream_telemetry import StreamTelemetryRecorder
    from src.api.handlers.openai import OpenAIChatAdapter
    from src.services.system.config import SystemConfigService

    gateway_module = _gateway_module()
    if payload.status_code >= 400:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    if not user_id or not api_key_id:
        return

    body_bytes = gateway_module._extract_gateway_report_body_bytes(payload)
    if not body_bytes:
        return

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        return

    adapter = OpenAIChatAdapter()
    handler = adapter._create_handler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    ctx = StreamContext(
        model=str(context.get("model") or "unknown"),
        api_format=handler.allowed_api_formats[0],
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )
    ctx.request_id = handler.request_id
    ctx.client_api_format = str(context.get("client_api_format") or "openai:chat")
    ctx.provider_type = str(context.get("provider_name") or "openai")
    ctx.update_provider_info(
        provider_name=str(context.get("provider_name") or "openai"),
        provider_id=str(context.get("provider_id") or ""),
        endpoint_id=str(context.get("endpoint_id") or ""),
        key_id=str(context.get("key_id") or ""),
        provider_api_format=str(context.get("provider_api_format") or "openai:chat"),
    )
    ctx.mapped_model = str(context.get("mapped_model") or "") or None
    ctx.provider_request_headers = dict(context.get("provider_request_headers") or {})
    ctx.provider_request_body = context.get("provider_request_body")
    ctx.response_headers = dict(payload.headers or {})
    ctx.status_code = payload.status_code
    ctx.record_parsed_chunks = SystemConfigService.should_log_body(db)
    if str(context.get("candidate_id") or "").strip():
        ctx.attempt_id = str(context.get("candidate_id"))

    proxy_info = context.get("proxy_info")
    if isinstance(proxy_info, dict):
        ctx.proxy_info = dict(proxy_info)
    ctx.set_proxy_timing(ctx.response_headers)

    telemetry = payload.telemetry if isinstance(payload.telemetry, dict) else {}
    ttfb_ms = telemetry.get("ttfb_ms")
    if ttfb_ms is not None:
        try:
            ctx.first_byte_time_ms = max(int(ttfb_ms), 0)
        except (TypeError, ValueError):
            ctx.first_byte_time_ms = None
    if ctx.proxy_info is not None and ctx.first_byte_time_ms is not None:
        ctx.set_ttfb_ms(ctx.first_byte_time_ms)

    stream_processor = StreamProcessor(
        request_id=ctx.request_id,
        default_parser=handler.parser,
        on_streaming_start=None,
    )
    stream = stream_processor.create_response_stream(
        ctx,
        _iter_gateway_report_body_chunks(body_bytes),
        _GatewayReportStreamContext(),
        start_time=time.time(),
    )
    async for _chunk in stream:
        pass

    elapsed_ms = telemetry.get("elapsed_ms")
    try:
        response_elapsed_ms = max(int(elapsed_ms or 0), 0)
    except (TypeError, ValueError):
        response_elapsed_ms = 0
    request_start_time = (
        time.time() - (response_elapsed_ms / 1000.0) if response_elapsed_ms > 0 else time.time()
    )

    telemetry_recorder = StreamTelemetryRecorder(
        request_id=ctx.request_id,
        user_id=str(user.id),
        api_key_id=str(api_key.id),
        client_ip="127.0.0.1",
        format_id=handler.FORMAT_ID,
    )
    await telemetry_recorder.record_stream_stats(
        ctx,
        dict(context.get("original_headers") or {}),
        dict(context.get("original_request_body") or {}),
        request_start_time,
    )


async def _record_gateway_passthrough_chat_stream_success(
    payload: GatewayStreamReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.base.stream_context import StreamContext
    from src.api.handlers.base.stream_processor import StreamProcessor
    from src.api.handlers.base.stream_telemetry import StreamTelemetryRecorder
    from src.api.handlers.claude import ClaudeChatAdapter
    from src.api.handlers.gemini import GeminiChatAdapter
    from src.api.handlers.openai import OpenAIChatAdapter
    from src.services.system.config import SystemConfigService

    gateway_module = _gateway_module()
    if payload.status_code >= 400:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    if not user_id or not api_key_id or not client_api_format:
        return

    body_bytes = gateway_module._extract_gateway_report_body_bytes(payload)
    if not body_bytes:
        return

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        return

    if client_api_format == "claude:chat":
        adapter = ClaudeChatAdapter()
    elif client_api_format == "gemini:chat":
        adapter = GeminiChatAdapter()
    elif client_api_format == "openai:chat":
        adapter = OpenAIChatAdapter()
    else:
        return

    handler = adapter._create_handler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    ctx = StreamContext(
        model=str(context.get("model") or "unknown"),
        api_format=handler.allowed_api_formats[0],
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )
    ctx.request_id = handler.request_id
    ctx.client_api_format = client_api_format
    ctx.provider_type = str(context.get("provider_name") or "unknown")
    ctx.update_provider_info(
        provider_name=str(context.get("provider_name") or "unknown"),
        provider_id=str(context.get("provider_id") or ""),
        endpoint_id=str(context.get("endpoint_id") or ""),
        key_id=str(context.get("key_id") or ""),
        provider_api_format=str(context.get("provider_api_format") or client_api_format),
    )
    ctx.mapped_model = str(context.get("mapped_model") or "") or None
    ctx.provider_request_headers = dict(context.get("provider_request_headers") or {})
    ctx.provider_request_body = context.get("provider_request_body")
    ctx.response_headers = dict(payload.headers or {})
    ctx.status_code = payload.status_code
    ctx.record_parsed_chunks = SystemConfigService.should_log_body(db)
    if str(context.get("candidate_id") or "").strip():
        ctx.attempt_id = str(context.get("candidate_id"))

    proxy_info = context.get("proxy_info")
    if isinstance(proxy_info, dict):
        ctx.proxy_info = dict(proxy_info)
    ctx.set_proxy_timing(ctx.response_headers)

    telemetry = payload.telemetry if isinstance(payload.telemetry, dict) else {}
    ttfb_ms = telemetry.get("ttfb_ms")
    if ttfb_ms is not None:
        try:
            ctx.first_byte_time_ms = max(int(ttfb_ms), 0)
        except (TypeError, ValueError):
            ctx.first_byte_time_ms = None
    if ctx.proxy_info is not None and ctx.first_byte_time_ms is not None:
        ctx.set_ttfb_ms(ctx.first_byte_time_ms)

    stream_processor = StreamProcessor(
        request_id=ctx.request_id,
        default_parser=handler.parser,
        on_streaming_start=None,
    )
    stream = stream_processor.create_response_stream(
        ctx,
        _iter_gateway_report_body_chunks(body_bytes),
        _GatewayReportStreamContext(),
        start_time=time.time(),
    )
    async for _chunk in stream:
        pass

    elapsed_ms = telemetry.get("elapsed_ms")
    try:
        response_elapsed_ms = max(int(elapsed_ms or 0), 0)
    except (TypeError, ValueError):
        response_elapsed_ms = 0
    request_start_time = (
        time.time() - (response_elapsed_ms / 1000.0) if response_elapsed_ms > 0 else time.time()
    )

    telemetry_recorder = StreamTelemetryRecorder(
        request_id=ctx.request_id,
        user_id=str(user.id),
        api_key_id=str(api_key.id),
        client_ip="127.0.0.1",
        format_id=handler.FORMAT_ID,
    )
    await telemetry_recorder.record_stream_stats(
        ctx,
        dict(context.get("original_headers") or {}),
        dict(context.get("original_request_body") or {}),
        request_start_time,
    )


async def _record_gateway_passthrough_cli_stream_success(
    payload: GatewayStreamReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.base.stream_context import StreamContext
    from src.api.handlers.claude_cli import ClaudeCliAdapter
    from src.api.handlers.gemini_cli import GeminiCliAdapter
    from src.api.handlers.openai_cli import OpenAICliAdapter
    from src.services.system.config import SystemConfigService

    gateway_module = _gateway_module()
    if payload.status_code >= 400:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    if not user_id or not api_key_id or not client_api_format:
        return

    body_bytes = gateway_module._extract_gateway_report_body_bytes(payload)
    if not body_bytes:
        return

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        return

    if client_api_format == "claude:cli":
        adapter = ClaudeCliAdapter()
    elif client_api_format == "gemini:cli":
        adapter = GeminiCliAdapter()
    elif client_api_format == "openai:cli":
        adapter = OpenAICliAdapter()
    else:
        return

    handler = adapter.HANDLER_CLASS(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
        allowed_api_formats=adapter.allowed_api_formats,
        adapter_detector=adapter.detect_capability_requirements,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    ctx = StreamContext(
        model=str(context.get("model") or "unknown"),
        api_format=handler.primary_api_format,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        request_id=handler.request_id,
        user_id=user.id,
        api_key_id=api_key.id,
    )
    ctx.client_api_format = client_api_format
    ctx.provider_type = str(context.get("provider_name") or "unknown")
    ctx.update_provider_info(
        provider_name=str(context.get("provider_name") or "unknown"),
        provider_id=str(context.get("provider_id") or ""),
        endpoint_id=str(context.get("endpoint_id") or ""),
        key_id=str(context.get("key_id") or ""),
        provider_api_format=str(context.get("provider_api_format") or client_api_format),
    )
    ctx.mapped_model = str(context.get("mapped_model") or "") or None
    ctx.provider_request_headers = dict(context.get("provider_request_headers") or {})
    ctx.provider_request_body = context.get("provider_request_body")
    ctx.response_headers = dict(payload.headers or {})
    ctx.status_code = payload.status_code
    ctx.record_parsed_chunks = SystemConfigService.should_log_body(db)
    if str(context.get("candidate_id") or "").strip():
        ctx.attempt_id = str(context.get("candidate_id"))

    proxy_info = context.get("proxy_info")
    if isinstance(proxy_info, dict):
        ctx.proxy_info = dict(proxy_info)
    ctx.set_proxy_timing(ctx.response_headers)

    telemetry = payload.telemetry if isinstance(payload.telemetry, dict) else {}
    ttfb_ms = telemetry.get("ttfb_ms")
    if ttfb_ms is not None:
        try:
            ctx.first_byte_time_ms = max(int(ttfb_ms), 0)
        except (TypeError, ValueError):
            ctx.first_byte_time_ms = None
    if ctx.proxy_info is not None and ctx.first_byte_time_ms is not None:
        ctx.set_ttfb_ms(ctx.first_byte_time_ms)

    elapsed_ms = telemetry.get("elapsed_ms")
    try:
        response_elapsed_ms = max(int(elapsed_ms or 0), 0)
    except (TypeError, ValueError):
        response_elapsed_ms = 0
    if response_elapsed_ms > 0:
        handler.start_time = time.time() - (response_elapsed_ms / 1000.0)

    stream = handler._create_response_stream_with_prefetch(
        ctx,
        _iter_gateway_report_body_chunks(body_bytes),
        _GatewayReportStreamContext(),
        [],
    )
    async for _chunk in stream:
        pass

    await handler._record_stream_stats(
        ctx,
        dict(context.get("original_headers") or {}),
        dict(context.get("original_request_body") or {}),
    )
