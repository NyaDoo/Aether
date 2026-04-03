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


def _postprocess_gateway_report_provider_response(
    context: dict[str, Any],
    provider_response_json: dict[str, Any],
) -> None:
    if str(context.get("envelope_name") or "").strip().lower() != "antigravity:v1internal":
        return

    try:
        from src.services.provider.adapters.antigravity.envelope import (
            _inject_claude_tool_ids_response,
            cache_thought_signatures,
        )

        model = str(context.get("mapped_model") or context.get("model") or "")
        _inject_claude_tool_ids_response(provider_response_json, model)
        cache_thought_signatures(model, provider_response_json)
    except Exception:
        return


async def _record_gateway_openai_chat_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.openai import OpenAIChatAdapter

    if payload.status_code >= 400:
        return
    if not isinstance(payload.body_json, dict) or payload.body_json.get("error") is not None:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    if not user_id or not api_key_id:
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

    provider_response_json = dict(payload.body_json)
    _postprocess_gateway_report_provider_response(context, provider_response_json)
    client_response_json = (
        dict(payload.client_body_json) if isinstance(payload.client_body_json, dict) else None
    )
    response_json = handler._normalize_response(client_response_json or provider_response_json)
    usage_info = handler._extract_usage(response_json)
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

    await handler.telemetry.record_success(
        provider=str(context.get("provider_name") or "openai"),
        model=str(context.get("model") or "unknown"),
        input_tokens=int(usage_info.get("input_tokens", 0) or 0),
        output_tokens=int(usage_info.get("output_tokens", 0) or 0),
        response_time_ms=response_time_ms,
        status_code=payload.status_code,
        request_headers=dict(context.get("original_headers") or {}),
        request_body=dict(context.get("original_request_body") or {}),
        response_headers=dict(payload.headers or {}),
        client_response_headers=dict(payload.headers or {}),
        response_body=provider_response_json if client_response_json else response_json,
        client_response_body=response_json if client_response_json else None,
        provider_request_headers=dict(context.get("provider_request_headers") or {}),
        provider_request_body=context.get("provider_request_body"),
        is_stream=False,
        provider_id=str(context.get("provider_id") or "") or None,
        provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
        provider_api_key_id=str(context.get("key_id") or "") or None,
        api_format=str(context.get("client_api_format") or "openai:chat"),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        endpoint_api_format=str(context.get("provider_api_format") or "") or None,
        has_format_conversion=client_response_json is not None,
        target_model=str(context.get("mapped_model") or "") or None,
        request_metadata=request_metadata,
    )


async def _record_gateway_passthrough_chat_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.claude import ClaudeChatAdapter
    from src.api.handlers.gemini import GeminiChatAdapter
    from src.api.handlers.openai import OpenAIChatAdapter

    if payload.status_code >= 400:
        return
    if not isinstance(payload.body_json, dict) or payload.body_json.get("error") is not None:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    if not user_id or not api_key_id or not client_api_format:
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

    provider_response_json = dict(payload.body_json)
    _postprocess_gateway_report_provider_response(context, provider_response_json)
    response_json = handler._normalize_response(provider_response_json)
    usage_info = handler._extract_usage(response_json)
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

    await handler.telemetry.record_success(
        provider=str(context.get("provider_name") or "unknown"),
        model=str(context.get("model") or "unknown"),
        input_tokens=int(usage_info.get("input_tokens", 0) or 0),
        output_tokens=int(usage_info.get("output_tokens", 0) or 0),
        response_time_ms=response_time_ms,
        status_code=payload.status_code,
        request_headers=dict(context.get("original_headers") or {}),
        request_body=dict(context.get("original_request_body") or {}),
        response_headers=dict(payload.headers or {}),
        client_response_headers=dict(payload.headers or {}),
        response_body=response_json,
        provider_request_headers=dict(context.get("provider_request_headers") or {}),
        provider_request_body=context.get("provider_request_body"),
        cache_creation_tokens=int(
            usage_info.get("cache_creation_input_tokens", 0)
            or usage_info.get("cache_creation_tokens", 0)
            or 0
        ),
        cache_read_tokens=int(
            usage_info.get("cache_read_input_tokens", 0)
            or usage_info.get("cache_read_tokens", 0)
            or 0
        ),
        is_stream=False,
        provider_id=str(context.get("provider_id") or "") or None,
        provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
        provider_api_key_id=str(context.get("key_id") or "") or None,
        api_format=client_api_format,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        endpoint_api_format=str(context.get("provider_api_format") or "") or None,
        has_format_conversion=False,
        target_model=str(context.get("mapped_model") or "") or None,
        request_metadata=request_metadata,
    )


async def _record_gateway_passthrough_cli_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    from src.api.handlers.claude_cli import ClaudeCliAdapter
    from src.api.handlers.gemini_cli import GeminiCliAdapter
    from src.api.handlers.openai_cli import OpenAICliAdapter, OpenAICompactAdapter

    if payload.status_code >= 400:
        return
    if not isinstance(payload.body_json, dict) or payload.body_json.get("error") is not None:
        return

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    client_api_format = str(context.get("client_api_format") or "openai:cli").strip().lower()
    if not user_id or not api_key_id:
        return

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        return

    if client_api_format == "openai:compact":
        adapter = OpenAICompactAdapter()
    elif client_api_format == "claude:cli":
        adapter = ClaudeCliAdapter()
    elif client_api_format == "gemini:cli":
        adapter = GeminiCliAdapter()
    else:
        adapter = OpenAICliAdapter()

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

    provider_response_json = dict(payload.body_json)
    _postprocess_gateway_report_provider_response(context, provider_response_json)
    client_response_json = (
        dict(payload.client_body_json) if isinstance(payload.client_body_json, dict) else None
    )
    response_json = dict(client_response_json or provider_response_json)
    usage_info = handler.parser.extract_usage_from_response(response_json)
    response_metadata = handler._extract_response_metadata(response_json)
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

    await handler.telemetry.record_success(
        provider=str(context.get("provider_name") or "openai"),
        model=str(context.get("model") or "unknown"),
        input_tokens=int(usage_info.get("input_tokens", 0) or 0),
        output_tokens=int(usage_info.get("output_tokens", 0) or 0),
        response_time_ms=response_time_ms,
        status_code=payload.status_code,
        request_headers=dict(context.get("original_headers") or {}),
        request_body=dict(context.get("original_request_body") or {}),
        response_headers=dict(payload.headers or {}),
        client_response_headers=dict(payload.headers or {}),
        response_body=provider_response_json if client_response_json else response_json,
        client_response_body=response_json if client_response_json else None,
        provider_request_headers=dict(context.get("provider_request_headers") or {}),
        provider_request_body=context.get("provider_request_body"),
        cache_creation_tokens=int(usage_info.get("cache_creation_tokens", 0) or 0),
        cache_read_tokens=int(usage_info.get("cache_read_tokens", 0) or 0),
        is_stream=False,
        provider_id=str(context.get("provider_id") or "") or None,
        provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
        provider_api_key_id=str(context.get("key_id") or "") or None,
        api_format=client_api_format,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        endpoint_api_format=str(context.get("provider_api_format") or "") or None,
        has_format_conversion=client_response_json is not None,
        target_model=str(context.get("mapped_model") or "") or None,
        response_metadata=response_metadata if response_metadata else None,
        request_metadata=request_metadata,
    )


async def _record_gateway_openai_video_delete_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_openai_video_delete_sync(payload, db=db)


async def _record_gateway_openai_video_cancel_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_openai_video_cancel_sync(payload, db=db)


async def _record_gateway_gemini_video_cancel_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_gemini_video_cancel_sync(payload, db=db)


async def _record_gateway_openai_video_create_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_openai_video_create_sync(payload, db=db)


async def _record_gateway_openai_video_remix_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_openai_video_remix_sync(payload, db=db)


async def _record_gateway_gemini_video_create_sync_success(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    await gateway_module._finalize_gateway_gemini_video_create_sync(payload, db=db)
