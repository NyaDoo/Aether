from __future__ import annotations

import base64
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.core.api_format.headers import extract_client_api_key_for_endpoint_with_query
from src.core.api_format.metadata import get_auth_config_for_endpoint
from src.core.crypto import crypto_service
from src.core.http_compression import normalize_content_encoding
from src.core.logger import logger
from src.database import create_session, get_db
from src.models.database import ApiKey, RequestCandidate, User
from src.services.auth.service import AuthService
from src.utils.async_utils import safe_create_task

from .common import ensure_loopback
from .gateway_contract import (
    _GEMINI_FILES_DOWNLOAD_ROUTE_RE,
    _GEMINI_FILES_RESOURCE_ROUTE_RE,
    _GEMINI_MODEL_OPERATION_CANCEL_RE,
    _GEMINI_OPERATION_CANCEL_RE,
    _GEMINI_SYNC_ROUTE_RE,
    _GEMINI_VIDEO_CREATE_ROUTE_RE,
    _GEMINI_VIDEO_MODEL_OPERATION_ANY_RE,
    _OPENAI_VIDEO_CANCEL_ROUTE_RE,
    _OPENAI_VIDEO_CONTENT_ROUTE_RE,
    _OPENAI_VIDEO_REMIX_ROUTE_RE,
    _OPENAI_VIDEO_TASK_ROUTE_RE,
    CONTROL_ACTION_HEADER,
    CONTROL_ACTION_PROXY_PUBLIC,
    CONTROL_EXECUTED_HEADER,
    GatewayAuthContext,
    GatewayExecuteRequest,
    GatewayExecutionDecisionResponse,
    GatewayExecutionPlanResponse,
    GatewayResolveRequest,
    GatewayRouteDecision,
    GatewayStreamReportRequest,
    GatewaySyncReportRequest,
    classify_gateway_route,
)


class _GatewayProxy:
    def __getattr__(self, name: str) -> Any:
        from . import gateway as gateway_module

        return getattr(gateway_module, name)


gateway_module = _GatewayProxy()


async def _build_cli_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_provider_formats: set[str],
    plan_kind: str,
    report_kind: str,
    log_label: str,
) -> GatewayExecutionPlanResponse | None:
    from loguru import logger

    from src.api.handlers.base.cli_adapter_base import CliAdapterBase
    from src.config.settings import config
    from src.services.provider.transport import redact_url_for_log
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, CliAdapterBase):
        return None
    if not gateway_module._is_stream_request_payload(payload.body_json, path_params):
        return None

    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params=path_params,
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    original_request_body = await context.ensure_json_body_async()
    if context.path_params:
        original_request_body = adapter._merge_path_params(
            original_request_body,
            context.path_params,
        )

    handler = adapter.HANDLER_CLASS(
        db=db,
        user=user,
        api_key=api_key,
        request_id=context.request_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        start_time=context.start_time,
        allowed_api_formats=adapter.allowed_api_formats,
        adapter_detector=adapter.detect_capability_requirements,
        perf_metrics=context.extra.get("perf"),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    model = handler.extract_model_from_request(original_request_body, context.path_params)
    client_api_format = handler.primary_api_format
    capability_requirements = handler._resolve_capability_requirements(
        model_name=model,
        request_headers=context.original_headers,
        request_body=original_request_body,
    )
    preferred_key_ids = await handler._resolve_preferred_key_ids(
        model_name=model,
        request_body=original_request_body,
    )

    candidate = await gateway_module._select_gateway_direct_candidate(
        db=db,
        redis_client=getattr(handler, "redis", None),
        api_format=str(client_api_format),
        model_name=str(model or "unknown"),
        user_api_key=api_key,
        request_id=context.request_id,
        is_stream=True,
        capability_requirements=capability_requirements or None,
        preferred_key_ids=preferred_key_ids or None,
        request_body=original_request_body,
    )
    if candidate is None:
        return None

    provider = candidate.provider
    endpoint = candidate.endpoint
    key = candidate.key
    provider_api_format = str(endpoint.api_format or "")
    mapped_model = candidate.mapping_matched_model if candidate else None
    if not mapped_model:
        mapped_model = await handler._get_mapped_model(
            source_model=str(model or "unknown"),
            provider_id=str(provider.id),
        )
    needs_conversion = bool(getattr(candidate, "needs_conversion", False))

    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=context.original_headers,
        query_params=context.query_params,
        client_api_format=str(client_api_format),
        provider_api_format=provider_api_format,
        fallback_model=str(model or "unknown"),
        mapped_model=mapped_model,
        client_is_stream=True,
        needs_conversion=needs_conversion,
        output_limit=candidate.output_limit if candidate else None,
    )

    effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    stream_proxy_info = await resolve_proxy_info_async(effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
    proxy_url: str | None = None
    if effective_proxy and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy)

    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        method="POST",
        url=upstream_request.url,
        headers=dict(upstream_request.headers),
        body=build_execution_plan_body(
            upstream_request.payload,
            content_type=str(upstream_request.headers.get("content-type") or "").strip() or None,
        ),
        stream=True,
        provider_api_format=provider_api_format,
        client_api_format=str(client_api_format),
        model_name=str(model or ""),
        content_type=str(upstream_request.headers.get("content-type") or "").strip() or None,
        content_encoding=context.client_content_encoding,
        proxy=ExecutionProxySnapshot.from_proxy_info(
            stream_proxy_info,
            proxy_url=proxy_url,
            mode_override="tunnel" if is_tunnel_delegate else None,
            node_id_override=(
                str(delegate_cfg.get("node_id") or "").strip() or None
                if is_tunnel_delegate
                else None
            ),
        ),
        tls_profile=upstream_request.tls_profile,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=int(config.http_connect_timeout * 1000),
            read_ms=int(config.http_read_timeout * 1000),
            write_ms=int(config.http_write_timeout * 1000),
            pool_ms=int(config.http_pool_timeout * 1000),
            total_ms=None,
        ),
    )

    if (
        not upstream_request.upstream_is_stream
        or gateway_module._stream_executor_requires_python_rewrite(
            envelope=upstream_request.envelope,
            needs_conversion=needs_conversion,
            provider_api_format=provider_api_format,
            client_api_format=str(client_api_format),
        )
        or not is_remote_execution_runtime_contract_eligible(contract)
        or provider_api_format != str(client_api_format)
        or provider_api_format not in expected_provider_formats
    ):
        return None

    logger.debug(
        "[gateway] {} stream direct executor candidate accepted: path={} provider={} url={}",
        log_label,
        payload.path,
        provider.name,
        redact_url_for_log(upstream_request.url),
    )

    return GatewayExecutionPlanResponse(
        action="executor_stream",
        plan_kind=plan_kind,
        plan=contract.to_payload(),
        report_kind=report_kind,
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "model": str(model or "unknown"),
            "provider_name": str(contract.provider_name or "unknown"),
            "provider_id": str(contract.provider_id or ""),
            "endpoint_id": str(contract.endpoint_id or ""),
            "key_id": str(contract.key_id or ""),
            "candidate_id": str(contract.candidate_id or ""),
            "provider_api_format": str(contract.provider_api_format or ""),
            "client_api_format": str(contract.client_api_format or ""),
            "mapped_model": mapped_model,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(upstream_request.headers),
            "provider_request_body": upstream_request.payload,
            "proxy_info": stream_proxy_info,
            "has_envelope": upstream_request.envelope is not None,
            "envelope_name": gateway_module._gateway_report_context_envelope_name(
                upstream_request.envelope
            ),
            "needs_conversion": bool(needs_conversion),
        },
    )


async def _build_cli_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_client_formats: set[str],
    plan_kind: str,
    report_kind: str,
    finalize_kind: str,
    log_label: str,
) -> GatewayExecutionPlanResponse | None:
    from loguru import logger

    from src.api.handlers.base.cli_adapter_base import CliAdapterBase
    from src.config.settings import config
    from src.services.provider.transport import redact_url_for_log
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, CliAdapterBase):
        return None
    if gateway_module._is_stream_request_payload(payload.body_json, path_params):
        return None

    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params=path_params,
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    original_request_body = await context.ensure_json_body_async()
    if context.path_params:
        original_request_body = adapter._merge_path_params(
            original_request_body,
            context.path_params,
        )
    if decision.route_kind == "compact":
        original_request_body.pop("stream", None)

    handler = adapter.HANDLER_CLASS(
        db=db,
        user=user,
        api_key=api_key,
        request_id=context.request_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        start_time=context.start_time,
        allowed_api_formats=adapter.allowed_api_formats,
        adapter_detector=adapter.detect_capability_requirements,
        perf_metrics=context.extra.get("perf"),
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    model = handler.extract_model_from_request(original_request_body, context.path_params)
    client_api_format = handler.primary_api_format
    capability_requirements = handler._resolve_capability_requirements(
        model_name=model,
        request_headers=context.original_headers,
        request_body=original_request_body,
    )
    preferred_key_ids = await handler._resolve_preferred_key_ids(
        model_name=model,
        request_body=original_request_body,
    )

    candidate = await gateway_module._select_gateway_direct_candidate(
        db=db,
        redis_client=getattr(handler, "redis", None),
        api_format=str(client_api_format),
        model_name=str(model or "unknown"),
        user_api_key=api_key,
        request_id=context.request_id,
        is_stream=False,
        capability_requirements=capability_requirements or None,
        preferred_key_ids=preferred_key_ids or None,
        request_body=original_request_body,
    )
    if candidate is None:
        return None

    provider = candidate.provider
    endpoint = candidate.endpoint
    key = candidate.key
    provider_api_format = str(endpoint.api_format or "")
    mapped_model = candidate.mapping_matched_model if candidate else None
    if not mapped_model:
        mapped_model = await handler._get_mapped_model(
            source_model=str(model or "unknown"),
            provider_id=str(provider.id),
        )
    needs_conversion = bool(getattr(candidate, "needs_conversion", False))

    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=context.original_headers,
        query_params=context.query_params,
        client_api_format=str(client_api_format),
        provider_api_format=provider_api_format,
        fallback_model=str(model or "unknown"),
        mapped_model=mapped_model,
        client_is_stream=False,
        needs_conversion=needs_conversion,
        output_limit=candidate.output_limit if candidate else None,
    )

    _effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    sync_proxy_info = await resolve_proxy_info_async(_effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(_effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
    proxy_url: str | None = None
    if _effective_proxy and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(_effective_proxy)

    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        method="POST",
        url=upstream_request.url,
        headers=dict(upstream_request.headers),
        body=build_execution_plan_body(
            upstream_request.payload,
            content_type=str(upstream_request.headers.get("content-type") or "").strip() or None,
        ),
        stream=upstream_request.upstream_is_stream,
        provider_api_format=provider_api_format,
        client_api_format=str(client_api_format),
        model_name=str(model or ""),
        content_type=str(upstream_request.headers.get("content-type") or "").strip() or None,
        content_encoding=context.client_content_encoding,
        proxy=ExecutionProxySnapshot.from_proxy_info(
            sync_proxy_info,
            proxy_url=proxy_url,
            mode_override="tunnel" if is_tunnel_delegate else None,
            node_id_override=(
                str(delegate_cfg.get("node_id") or "").strip() or None
                if is_tunnel_delegate
                else None
            ),
        ),
        tls_profile=upstream_request.tls_profile,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=int(config.http_connect_timeout * 1000),
            read_ms=int(config.http_read_timeout * 1000),
            write_ms=int(config.http_write_timeout * 1000),
            pool_ms=int(config.http_pool_timeout * 1000),
            total_ms=int((provider.request_timeout or config.http_request_timeout) * 1000),
        ),
    )

    normalized_client_api_format = str(client_api_format)
    if (
        not is_remote_execution_runtime_contract_eligible(contract)
        or normalized_client_api_format not in expected_client_formats
    ):
        return None

    selected_report_kind = (
        finalize_kind
        if (
            upstream_request.upstream_is_stream
            or needs_conversion
            or upstream_request.envelope is not None
            or provider_api_format != normalized_client_api_format
        )
        else report_kind
    )

    logger.debug(
        "[gateway] {} sync direct executor candidate accepted: path={} provider={} url={}",
        log_label,
        payload.path,
        provider.name,
        redact_url_for_log(upstream_request.url),
    )

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind=plan_kind,
        plan=contract.to_payload(),
        report_kind=selected_report_kind,
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "model": str(model or "unknown"),
            "provider_name": str(contract.provider_name or "unknown"),
            "provider_id": str(contract.provider_id or ""),
            "endpoint_id": str(contract.endpoint_id or ""),
            "key_id": str(contract.key_id or ""),
            "provider_api_format": str(contract.provider_api_format or ""),
            "client_api_format": str(contract.client_api_format or ""),
            "mapped_model": mapped_model,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(upstream_request.headers),
            "provider_request_body": upstream_request.payload,
            "proxy_info": sync_proxy_info,
            "has_envelope": upstream_request.envelope is not None,
            "envelope_name": gateway_module._gateway_report_context_envelope_name(
                upstream_request.envelope
            ),
            "needs_conversion": bool(needs_conversion),
        },
    )


async def _build_openai_cli_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    is_compact = decision.route_kind == "compact"
    return await _build_cli_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_provider_formats={"openai:compact" if is_compact else "openai:cli"},
        plan_kind="openai_compact_stream" if is_compact else "openai_cli_stream",
        report_kind="openai_cli_stream_success",
        log_label="openai compact" if is_compact else "openai cli",
    )


async def _build_claude_cli_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_cli_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_provider_formats={"claude:cli"},
        plan_kind="claude_cli_stream",
        report_kind="claude_cli_stream_success",
        log_label="claude cli",
    )


async def _build_gemini_cli_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_cli_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_provider_formats={"gemini:cli"},
        plan_kind="gemini_cli_stream",
        report_kind="gemini_cli_stream_success",
        log_label="gemini cli",
    )


async def _build_openai_cli_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_cli_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_client_formats={"openai:cli", "openai:compact"},
        plan_kind="openai_compact_sync" if decision.route_kind == "compact" else "openai_cli_sync",
        report_kind="openai_cli_sync_success",
        finalize_kind=(
            "openai_compact_sync_finalize"
            if decision.route_kind == "compact"
            else "openai_cli_sync_finalize"
        ),
        log_label="openai cli",
    )


async def _build_claude_cli_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_cli_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_client_formats={"claude:cli"},
        plan_kind="claude_cli_sync",
        report_kind="claude_cli_sync_success",
        finalize_kind="claude_cli_sync_finalize",
        log_label="claude cli",
    )


async def _build_gemini_cli_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_cli_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_client_formats={"gemini:cli"},
        plan_kind="gemini_cli_sync",
        report_kind="gemini_cli_sync_success",
        finalize_kind="gemini_cli_sync_finalize",
        log_label="gemini cli",
    )
