from __future__ import annotations

import base64
import inspect
import time
import uuid
from typing import Any

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.database import create_session, get_db
from src.models.database import ApiKey, User

from .gateway_contract import GatewaySyncReportRequest
from .gateway_finalize_common import _gateway_module


async def _run_gateway_cli_sync_finalize_background(
    payload: GatewaySyncReportRequest,
    db: Session,
) -> None:
    gateway_module = _gateway_module()
    try:
        await gateway_module._finalize_gateway_cli_sync(
            payload,
            db=db,
            background_tasks=None,
            allow_fast_path=False,
        )
    except Exception as exc:
        logger.warning("gateway background cli finalize failed: {}", exc)


async def _run_gateway_cli_sync_finalize_background_with_session(
    payload: GatewaySyncReportRequest,
) -> None:
    gateway_module = _gateway_module()
    db = create_session()
    try:
        await gateway_module._run_gateway_cli_sync_finalize_background(payload, db)
    finally:
        gateway_module._close_gateway_session(db)


async def _finalize_gateway_cli_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
    background_tasks: BackgroundTasks | None = None,
    allow_fast_path: bool = True,
) -> Response:
    from src.api.handlers.base.parsers import get_parser_for_format
    from src.api.handlers.base.stream_context import is_format_converted
    from src.api.handlers.base.utils import (
        build_json_response_for_client,
        filter_proxy_response_headers,
        get_format_converter_registry,
        resolve_client_accept_encoding,
    )
    from src.api.handlers.claude_cli import ClaudeCliAdapter
    from src.api.handlers.gemini_cli import GeminiCliAdapter
    from src.api.handlers.openai_cli import OpenAICliAdapter, OpenAICompactAdapter
    from src.core.exceptions import EmbeddedErrorException
    from src.models.database import Provider, ProviderAPIKey, ProviderEndpoint
    from src.services.scheduling.schemas import ProviderCandidate

    gateway_module = _gateway_module()
    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    provider_id = str(context.get("provider_id") or "").strip()
    endpoint_id = str(context.get("endpoint_id") or "").strip()
    key_id = str(context.get("key_id") or "").strip()
    request_id = str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]).strip()
    model = str(context.get("model") or "unknown").strip() or "unknown"
    provider_api_format = str(context.get("provider_api_format") or "").strip().lower()
    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    if not all([user_id, api_key_id, provider_id, endpoint_id, key_id, client_api_format]):
        raise HTTPException(status_code=400, detail="Missing gateway CLI finalize context")

    if allow_fast_path and background_tasks is not None:
        fast_response = await gateway_module._maybe_build_gateway_core_sync_fast_success_response(
            payload
        )
        if fast_response is not None:
            background_tasks.add_task(
                gateway_module._run_gateway_cli_sync_finalize_background,
                payload.model_copy(deep=True),
                db,
            )
            return fast_response

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    endpoint = db.query(ProviderEndpoint).filter(ProviderEndpoint.id == endpoint_id).first()
    key = db.query(ProviderAPIKey).filter(ProviderAPIKey.id == key_id).first()
    if not user or not api_key or not provider or not endpoint or not key:
        raise HTTPException(status_code=400, detail="Invalid gateway CLI finalize context")

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
        request_id=request_id,
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

    original_headers = dict(context.get("original_headers") or {})
    original_request_body = dict(context.get("original_request_body") or {})
    mapped_model = str(context.get("mapped_model") or "").strip() or None
    candidate = ProviderCandidate(
        provider=provider,
        endpoint=endpoint,
        key=key,
        mapping_matched_model=mapped_model,
        needs_conversion=is_format_converted(provider_api_format, client_api_format),
        provider_api_format=provider_api_format,
    )
    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=original_headers,
        query_params=None,
        client_api_format=client_api_format,
        provider_api_format=provider_api_format,
        fallback_model=model,
        mapped_model=mapped_model,
        client_is_stream=False,
        needs_conversion=bool(candidate.needs_conversion),
        output_limit=None,
    )

    response_json: dict[str, Any] | None = None
    provider_response_json: dict[str, Any] | None = None
    if upstream_request.upstream_is_stream and payload.body_base64 and payload.status_code < 400:
        try:
            response_json = await handler._aggregate_upstream_stream_sync_response(
                body_bytes=gateway_module._extract_gateway_report_body_bytes(payload),
                provider_api_format=provider_api_format,
                client_api_format=client_api_format,
                provider_name=str(provider.name),
                provider_type=str(getattr(provider, "provider_type", "") or "").lower(),
                model=model,
                request_id=request_id,
                envelope=upstream_request.envelope,
            )
        except EmbeddedErrorException as exc:
            payload = payload.model_copy(
                update={
                    "status_code": int(exc.error_code or 400),
                    "body_json": gateway_module._build_gateway_embedded_error_payload(exc),
                    "body_base64": None,
                }
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Invalid upstream CLI stream response",
            ) from exc

    provider_error_parser = get_parser_for_format(provider_api_format or client_api_format or "")
    is_error_response = payload.status_code >= 400
    if isinstance(payload.body_json, dict):
        try:
            is_error_response = is_error_response or provider_error_parser.is_error_response(
                dict(payload.body_json)
            )
        except Exception:
            is_error_response = is_error_response or payload.body_json.get("error") is not None

    client_accept_encoding = resolve_client_accept_encoding(original_headers, None)
    if is_error_response:
        error_status_code = gateway_module._resolve_gateway_sync_error_status_code(
            payload,
            provider_parser=provider_error_parser,
        )
        error_payload = gateway_module._build_gateway_sync_error_payload(
            payload,
            client_api_format=client_api_format,
            provider_api_format=provider_api_format or client_api_format,
            needs_conversion=bool(candidate.needs_conversion),
        )
        client_response_headers = filter_proxy_response_headers(dict(payload.headers or {}))
        client_response_headers["content-type"] = "application/json"
        client_response = build_json_response_for_client(
            status_code=error_status_code,
            content=error_payload,
            headers=client_response_headers,
            client_accept_encoding=client_accept_encoding,
        )

        request_metadata: dict[str, Any] = {
            "gateway_direct_executor": True,
            "phase": "3c_trial",
        }
        proxy_info = context.get("proxy_info")
        if isinstance(proxy_info, dict):
            request_metadata["proxy"] = proxy_info
        telemetry_writer = gateway_module._build_gateway_sync_telemetry_writer(
            db=db,
            request_id=request_id,
            user_id=user_id,
            api_key_id=api_key_id,
            fallback_telemetry=handler.telemetry,
        )

        response_time_ms = 0
        if isinstance(payload.telemetry, dict):
            raw_elapsed_ms = payload.telemetry.get("elapsed_ms")
            try:
                response_time_ms = max(int(raw_elapsed_ms or 0), 0)
            except (TypeError, ValueError):
                response_time_ms = 0

        await gateway_module._schedule_gateway_sync_telemetry(
            background_tasks=background_tasks,
            telemetry_writer=telemetry_writer,
            operation="record_failure",
            provider=str(context.get("provider_name") or provider.name or "unknown"),
            model=model,
            response_time_ms=response_time_ms,
            status_code=error_status_code,
            error_message=gateway_module._extract_gateway_sync_error_message(payload),
            request_headers=original_headers,
            request_body=original_request_body,
            provider_request_body=context.get("provider_request_body"),
            is_stream=False,
            api_format=client_api_format,
            api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
            endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
            provider_request_headers=dict(context.get("provider_request_headers") or {}),
            response_headers=dict(payload.headers or {}),
            client_response_headers=dict(client_response.headers),
            provider_id=str(context.get("provider_id") or "") or None,
            provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
            provider_api_key_id=str(context.get("key_id") or "") or None,
            endpoint_api_format=provider_api_format or None,
            has_format_conversion=is_format_converted(provider_api_format, client_api_format),
            target_model=mapped_model,
            metadata=request_metadata,
        )
        return client_response

    if response_json is None:
        if not isinstance(payload.body_json, dict):
            raise HTTPException(status_code=502, detail="Invalid upstream CLI response")

        response_json = dict(payload.body_json)
        if upstream_request.envelope:
            response_json = upstream_request.envelope.unwrap_response(response_json)
            upstream_request.envelope.postprocess_unwrapped_response(
                model=model, data=response_json
            )
        if candidate.needs_conversion:
            provider_response_json = dict(response_json)
            registry = get_format_converter_registry()
            response_json = registry.convert_response(
                response_json,
                provider_api_format,
                client_api_format,
                requested_model=model,
            )
        response_json = handler._normalize_response(response_json)

    extract_usage = getattr(handler, "_extract_usage", None)
    if callable(extract_usage):
        usage_info = extract_usage(response_json)
    else:
        parser = getattr(handler, "parser", None)
        parser_extract_usage = getattr(parser, "extract_usage_from_response", None)
        usage_info = parser_extract_usage(response_json) if callable(parser_extract_usage) else {}
    extract_response_metadata = getattr(handler, "_extract_response_metadata", None)
    response_metadata = (
        extract_response_metadata(response_json) if callable(extract_response_metadata) else None
    )

    client_response_headers = filter_proxy_response_headers(dict(payload.headers or {}))
    client_response_headers["content-type"] = "application/json"
    client_response = build_json_response_for_client(
        status_code=payload.status_code,
        content=response_json,
        headers=client_response_headers,
        client_accept_encoding=client_accept_encoding,
    )

    request_metadata: dict[str, Any] = {
        "gateway_direct_executor": True,
        "phase": "3c_trial",
    }
    proxy_info = context.get("proxy_info")
    if isinstance(proxy_info, dict):
        request_metadata["proxy"] = proxy_info
    telemetry_writer = gateway_module._build_gateway_sync_telemetry_writer(
        db=db,
        request_id=request_id,
        user_id=user_id,
        api_key_id=api_key_id,
        fallback_telemetry=handler.telemetry,
    )

    response_time_ms = 0
    if isinstance(payload.telemetry, dict):
        raw_elapsed_ms = payload.telemetry.get("elapsed_ms")
        try:
            response_time_ms = max(int(raw_elapsed_ms or 0), 0)
        except (TypeError, ValueError):
            response_time_ms = 0

    await gateway_module._schedule_gateway_sync_telemetry(
        background_tasks=background_tasks,
        telemetry_writer=telemetry_writer,
        operation="record_success",
        provider=str(context.get("provider_name") or provider.name or "unknown"),
        model=model,
        input_tokens=int(usage_info.get("input_tokens", 0) or 0),
        output_tokens=int(usage_info.get("output_tokens", 0) or 0),
        response_time_ms=response_time_ms,
        status_code=payload.status_code,
        request_headers=original_headers,
        request_body=original_request_body,
        response_headers=dict(payload.headers or {}),
        client_response_headers=dict(client_response.headers),
        response_body=provider_response_json or response_json,
        client_response_body=response_json if provider_response_json else None,
        provider_request_headers=dict(context.get("provider_request_headers") or {}),
        provider_request_body=context.get("provider_request_body"),
        is_stream=False,
        provider_id=str(context.get("provider_id") or "") or None,
        provider_endpoint_id=str(context.get("endpoint_id") or "") or None,
        provider_api_key_id=str(context.get("key_id") or "") or None,
        api_format=client_api_format,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
        endpoint_api_format=provider_api_format or None,
        has_format_conversion=is_format_converted(provider_api_format, client_api_format),
        target_model=mapped_model,
        metadata=gateway_module._build_gateway_usage_metadata(
            request_metadata=request_metadata,
            response_metadata=response_metadata if response_metadata else None,
        ),
    )

    return client_response
