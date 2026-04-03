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


async def _build_cli_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_api_format: str,
    decision_kind: str,
    report_kind: str,
    finalize_kind: str,
) -> GatewayExecutionDecisionResponse | None:
    from src.api.handlers.base.cli_adapter_base import CliAdapterBase
    from src.config.settings import config
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
        should_bypass_remote_execution_runtime_url,
    )

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, CliAdapterBase):
        return None
    if gateway_module._is_stream_request_payload(payload.body_json, path_params):
        return None
    if not isinstance(payload.body_json, dict):
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
    client_api_format = str(handler.primary_api_format or "").strip().lower()
    if client_api_format != expected_api_format:
        return None

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
        api_format=client_api_format,
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
    provider_api_format = str(endpoint.api_format or client_api_format or "").strip().lower()

    mapped_model = candidate.mapping_matched_model if candidate else None
    if not mapped_model:
        mapped_model = await handler._get_mapped_model(
            source_model=str(model or "unknown"),
            provider_id=str(provider.id),
        )

    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=context.original_headers,
        query_params=context.query_params,
        client_api_format=client_api_format,
        provider_api_format=provider_api_format,
        fallback_model=str(model or "unknown"),
        mapped_model=mapped_model,
        client_is_stream=False,
        needs_conversion=bool(getattr(candidate, "needs_conversion", False)),
        output_limit=getattr(candidate, "output_limit", None),
    )
    if should_bypass_remote_execution_runtime_url(
        upstream_request.url,
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
    ):
        return None
    upstream_is_stream = bool(upstream_request.upstream_is_stream)
    provider_request_body = dict(upstream_request.payload or {})
    provider_request_headers = {
        str(k).lower(): str(v)
        for k, v in dict(upstream_request.headers or {}).items()
        if str(k).strip() and str(v).strip()
    }
    auth_header, auth_value = gateway_module._extract_gateway_upstream_auth(
        provider_request_headers,
        provider_api_format=provider_api_format,
        key=key,
    )
    prompt_cache_key = str(provider_request_body.get("prompt_cache_key") or "").strip() or None

    effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    proxy_info = await resolve_proxy_info_async(effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
    effective_proxy_for_contract = effective_proxy
    if not effective_proxy_for_contract or not effective_proxy_for_contract.get("enabled", True):
        effective_proxy_for_contract = await get_system_proxy_config_async()
    proxy_url: str | None = None
    if effective_proxy_for_contract and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy_for_contract)
    proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
        proxy_info,
        proxy_url=proxy_url,
        mode_override="tunnel" if is_tunnel_delegate else None,
        node_id_override=(
            str(delegate_cfg.get("node_id") or "").strip() or None if is_tunnel_delegate else None
        ),
    )

    request_timeout = provider.request_timeout or config.http_request_timeout
    timeouts = ExecutionPlanTimeouts(
        connect_ms=int(config.http_connect_timeout * 1000),
        read_ms=int(config.http_read_timeout * 1000),
        write_ms=int(config.http_write_timeout * 1000),
        pool_ms=int(config.http_pool_timeout * 1000),
        total_ms=int(request_timeout * 1000),
    )
    requires_finalize = (
        upstream_is_stream
        or bool(getattr(candidate, "needs_conversion", False))
        or provider_api_format != client_api_format
        or upstream_request.envelope is not None
    )
    has_envelope = upstream_request.envelope is not None
    needs_conversion = bool(getattr(candidate, "needs_conversion", False))
    decision_extra_headers: dict[str, str] = {}
    decision_provider_request_headers = provider_request_headers or None
    decision_provider_request_body = provider_request_body
    report_context = {
        "user_id": str(user.id),
        "api_key_id": str(api_key.id),
        "request_id": str(context.request_id),
        "candidate_id": str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        "model": str(model or "unknown"),
        "provider_name": str(provider.name or "unknown"),
        "provider_id": str(provider.id),
        "endpoint_id": str(endpoint.id),
        "key_id": str(key.id),
        "provider_api_format": provider_api_format,
        "client_api_format": client_api_format,
        "mapped_model": str(mapped_model or "").strip() or None,
        "original_headers": dict(context.original_headers),
        "original_request_body": original_request_body,
        "proxy_info": proxy_info,
        "has_envelope": has_envelope,
        "envelope_name": gateway_module._gateway_report_context_envelope_name(
            upstream_request.envelope
        ),
        "needs_conversion": needs_conversion,
    }
    report_context["provider_request_headers"] = provider_request_headers
    report_context["provider_request_body"] = provider_request_body

    return GatewayExecutionDecisionResponse(
        action="executor_sync_decision",
        decision_kind=decision_kind,
        request_id=str(context.request_id),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        upstream_base_url=str(getattr(endpoint, "base_url", "") or "").strip(),
        upstream_url=str(upstream_request.url or "").strip() or None,
        auth_header=str(auth_header or "").strip() or "authorization",
        auth_value=str(auth_value or "").strip(),
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
        model_name=str(model or "unknown"),
        mapped_model=str(mapped_model or "").strip() or None,
        prompt_cache_key=prompt_cache_key,
        extra_headers=decision_extra_headers,
        provider_request_headers=decision_provider_request_headers,
        provider_request_body=decision_provider_request_body,
        content_type=(
            str(provider_request_headers.get("content-type") or "").strip() or "application/json"
        ),
        proxy=gateway_module._serialize_gateway_sync_proxy(proxy_snapshot),
        tls_profile=str(upstream_request.tls_profile or "").strip() or None,
        timeouts=gateway_module._serialize_gateway_sync_timeouts(timeouts),
        upstream_is_stream=upstream_is_stream or None,
        report_kind=finalize_kind if requires_finalize else report_kind,
        report_context=report_context,
        auth_context=auth_context,
    )


async def _build_cli_stream_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_api_format: str,
    decision_kind: str,
    report_kind: str,
) -> GatewayExecutionDecisionResponse | None:
    from src.api.handlers.base.cli_adapter_base import CliAdapterBase
    from src.config.settings import config
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
        should_bypass_remote_execution_runtime_url,
    )

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, CliAdapterBase):
        return None
    if not gateway_module._is_stream_request_payload(payload.body_json, path_params):
        return None
    if not isinstance(payload.body_json, dict):
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
    client_api_format = str(handler.primary_api_format or "").strip().lower()
    if client_api_format != expected_api_format:
        return None

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
        api_format=client_api_format,
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
    provider_api_format = str(endpoint.api_format or client_api_format or "").strip().lower()
    if provider_api_format != expected_api_format and not (
        expected_api_format in {"openai:cli", "openai:compact"}
        and provider_api_format in {"claude:cli", "gemini:cli"}
    ):
        return None

    mapped_model = candidate.mapping_matched_model if candidate else None
    if not mapped_model:
        mapped_model = await handler._get_mapped_model(
            source_model=str(model or "unknown"),
            provider_id=str(provider.id),
        )

    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=context.original_headers,
        query_params=context.query_params,
        client_api_format=client_api_format,
        provider_api_format=provider_api_format,
        fallback_model=str(model or "unknown"),
        mapped_model=mapped_model,
        client_is_stream=True,
        needs_conversion=bool(getattr(candidate, "needs_conversion", False)),
        output_limit=getattr(candidate, "output_limit", None),
    )
    if should_bypass_remote_execution_runtime_url(
        upstream_request.url,
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
    ):
        return None
    if not bool(upstream_request.upstream_is_stream):
        return None
    if gateway_module._stream_executor_requires_python_rewrite(
        envelope=upstream_request.envelope,
        needs_conversion=bool(getattr(candidate, "needs_conversion", False)),
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
    ):
        return None

    provider_request_body = dict(upstream_request.payload or {})
    provider_request_headers = {
        str(k).lower(): str(v)
        for k, v in dict(upstream_request.headers or {}).items()
        if str(k).strip() and str(v).strip()
    }
    auth_header, auth_value = gateway_module._extract_gateway_upstream_auth(
        provider_request_headers,
        provider_api_format=provider_api_format,
        key=key,
    )
    prompt_cache_key = str(provider_request_body.get("prompt_cache_key") or "").strip() or None

    effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    proxy_info = await resolve_proxy_info_async(effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
    effective_proxy_for_contract = effective_proxy
    if not effective_proxy_for_contract or not effective_proxy_for_contract.get("enabled", True):
        effective_proxy_for_contract = await get_system_proxy_config_async()
    proxy_url: str | None = None
    if effective_proxy_for_contract and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy_for_contract)
    proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
        proxy_info,
        proxy_url=proxy_url,
        mode_override="tunnel" if is_tunnel_delegate else None,
        node_id_override=(
            str(delegate_cfg.get("node_id") or "").strip() or None if is_tunnel_delegate else None
        ),
    )

    timeouts = ExecutionPlanTimeouts(
        connect_ms=int(config.http_connect_timeout * 1000),
        read_ms=int(config.http_read_timeout * 1000),
        write_ms=int(config.http_write_timeout * 1000),
        pool_ms=int(config.http_pool_timeout * 1000),
        total_ms=None,
    )
    has_envelope = upstream_request.envelope is not None
    needs_conversion = bool(getattr(candidate, "needs_conversion", False))
    decision_extra_headers: dict[str, str] = {}
    decision_provider_request_headers = provider_request_headers or None
    decision_provider_request_body = provider_request_body
    report_context = {
        "user_id": str(user.id),
        "api_key_id": str(api_key.id),
        "request_id": str(context.request_id),
        "candidate_id": str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        "model": str(model or "unknown"),
        "provider_name": str(provider.name or "unknown"),
        "provider_id": str(provider.id),
        "endpoint_id": str(endpoint.id),
        "key_id": str(key.id),
        "provider_api_format": provider_api_format,
        "client_api_format": client_api_format,
        "mapped_model": str(mapped_model or "").strip() or None,
        "original_headers": dict(context.original_headers),
        "original_request_body": original_request_body,
        "proxy_info": proxy_info,
        "has_envelope": has_envelope,
        "envelope_name": gateway_module._gateway_report_context_envelope_name(
            upstream_request.envelope
        ),
        "needs_conversion": needs_conversion,
    }
    report_context["provider_request_headers"] = provider_request_headers
    report_context["provider_request_body"] = provider_request_body

    return GatewayExecutionDecisionResponse(
        action="executor_stream_decision",
        decision_kind=decision_kind,
        request_id=str(context.request_id),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        upstream_base_url=str(getattr(endpoint, "base_url", "") or "").strip(),
        upstream_url=str(upstream_request.url or "").strip() or None,
        auth_header=str(auth_header or "").strip() or "authorization",
        auth_value=str(auth_value or "").strip(),
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
        model_name=str(model or "unknown"),
        mapped_model=str(mapped_model or "").strip() or None,
        prompt_cache_key=prompt_cache_key,
        extra_headers=decision_extra_headers,
        provider_request_headers=decision_provider_request_headers,
        provider_request_body=decision_provider_request_body,
        content_type=(
            str(provider_request_headers.get("content-type") or "").strip() or "application/json"
        ),
        proxy=gateway_module._serialize_gateway_sync_proxy(proxy_snapshot),
        tls_profile=str(upstream_request.tls_profile or "").strip() or None,
        timeouts=gateway_module._serialize_gateway_sync_timeouts(timeouts),
        report_kind=report_kind,
        report_context=report_context,
        auth_context=auth_context,
    )


async def _build_openai_cli_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    from src.api.handlers.base.cli_adapter_base import CliAdapterBase
    from src.config.settings import config
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
    )

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, CliAdapterBase):
        return None
    if gateway_module._is_stream_request_payload(payload.body_json, path_params):
        return None
    if not isinstance(payload.body_json, dict):
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
    client_api_format = str(handler.primary_api_format or "").strip().lower()
    if client_api_format not in {"openai:cli", "openai:compact"}:
        return None

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
        api_format=client_api_format,
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
    provider_api_format = str(endpoint.api_format or client_api_format or "").strip().lower()

    mapped_model = candidate.mapping_matched_model if candidate else None
    if not mapped_model:
        mapped_model = await handler._get_mapped_model(
            source_model=str(model or "unknown"),
            provider_id=str(provider.id),
        )

    upstream_request = await handler._build_upstream_request(
        provider=provider,
        endpoint=endpoint,
        key=key,
        request_body=dict(original_request_body),
        original_headers=context.original_headers,
        query_params=context.query_params,
        client_api_format=client_api_format,
        provider_api_format=provider_api_format,
        fallback_model=str(model or "unknown"),
        mapped_model=mapped_model,
        client_is_stream=False,
        needs_conversion=bool(getattr(candidate, "needs_conversion", False)),
        output_limit=getattr(candidate, "output_limit", None),
    )
    upstream_is_stream = bool(upstream_request.upstream_is_stream)
    provider_request_body = dict(upstream_request.payload or {})
    provider_request_headers = {
        str(k).lower(): str(v)
        for k, v in dict(upstream_request.headers or {}).items()
        if str(k).strip() and str(v).strip()
    }
    prompt_cache_key = str(provider_request_body.get("prompt_cache_key") or "").strip() or None
    auth_header = ""
    auth_value = ""
    for header_name, header_value in provider_request_headers.items():
        if header_name in {"authorization", "x-api-key", "x-goog-api-key"}:
            auth_header = header_name
            auth_value = header_value
            break
    if not auth_header:
        auth_header, auth_type = get_auth_config_for_endpoint(provider_api_format)
        decrypted_key = crypto_service.decrypt(key.api_key)
        auth_value = f"Bearer {decrypted_key}" if auth_type == "bearer" else decrypted_key

    effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    proxy_info = await resolve_proxy_info_async(effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
    effective_proxy_for_contract = effective_proxy
    if not effective_proxy_for_contract or not effective_proxy_for_contract.get("enabled", True):
        effective_proxy_for_contract = await get_system_proxy_config_async()
    proxy_url: str | None = None
    if effective_proxy_for_contract and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy_for_contract)
    proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
        proxy_info,
        proxy_url=proxy_url,
        mode_override="tunnel" if is_tunnel_delegate else None,
        node_id_override=(
            str(delegate_cfg.get("node_id") or "").strip() or None if is_tunnel_delegate else None
        ),
    )

    request_timeout = provider.request_timeout or config.http_request_timeout
    timeouts = ExecutionPlanTimeouts(
        connect_ms=int(config.http_connect_timeout * 1000),
        read_ms=int(config.http_read_timeout * 1000),
        write_ms=int(config.http_write_timeout * 1000),
        pool_ms=int(config.http_pool_timeout * 1000),
        total_ms=int(request_timeout * 1000),
    )
    requires_finalize = (
        upstream_is_stream
        or bool(getattr(candidate, "needs_conversion", False))
        or provider_api_format != client_api_format
        or upstream_request.envelope is not None
    )
    has_envelope = upstream_request.envelope is not None
    needs_conversion = bool(getattr(candidate, "needs_conversion", False))
    decision_extra_headers: dict[str, str] = {}
    decision_provider_request_headers = provider_request_headers or None
    decision_provider_request_body = provider_request_body
    report_context = {
        "user_id": str(user.id),
        "api_key_id": str(api_key.id),
        "request_id": str(context.request_id),
        "candidate_id": str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        "model": str(model or "unknown"),
        "provider_name": str(provider.name or "unknown"),
        "provider_id": str(provider.id),
        "endpoint_id": str(endpoint.id),
        "key_id": str(key.id),
        "provider_api_format": provider_api_format,
        "client_api_format": client_api_format,
        "mapped_model": str(mapped_model or "").strip() or None,
        "original_headers": dict(context.original_headers),
        "original_request_body": original_request_body,
        "proxy_info": proxy_info,
        "has_envelope": has_envelope,
        "envelope_name": gateway_module._gateway_report_context_envelope_name(
            upstream_request.envelope
        ),
        "needs_conversion": needs_conversion,
    }
    report_context["provider_request_headers"] = provider_request_headers
    report_context["provider_request_body"] = provider_request_body

    is_compact = decision.route_kind == "compact"
    return GatewayExecutionDecisionResponse(
        action="executor_sync_decision",
        decision_kind="openai_compact_sync" if is_compact else "openai_cli_sync",
        request_id=str(context.request_id),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        upstream_base_url=str(getattr(endpoint, "base_url", "") or "").strip(),
        upstream_url=str(upstream_request.url or "").strip() or None,
        auth_header=str(auth_header or "").strip() or "authorization",
        auth_value=str(auth_value or "").strip(),
        provider_api_format=provider_api_format,
        client_api_format=client_api_format,
        model_name=str(model or "unknown"),
        mapped_model=str(mapped_model or "").strip() or None,
        prompt_cache_key=prompt_cache_key,
        extra_headers=decision_extra_headers,
        provider_request_headers=decision_provider_request_headers,
        provider_request_body=decision_provider_request_body,
        content_type=(
            str(provider_request_headers.get("content-type") or "").strip() or "application/json"
        ),
        proxy=gateway_module._serialize_gateway_sync_proxy(proxy_snapshot),
        tls_profile=str(upstream_request.tls_profile or "").strip() or None,
        timeouts=gateway_module._serialize_gateway_sync_timeouts(timeouts),
        upstream_is_stream=upstream_is_stream or None,
        report_kind=(
            ("openai_compact_sync_finalize" if is_compact else "openai_cli_sync_finalize")
            if requires_finalize
            else "openai_cli_sync_success"
        ),
        report_context=report_context,
        auth_context=auth_context,
    )


async def _build_openai_cli_stream_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    is_compact = decision.route_kind == "compact"
    return await _build_cli_stream_decision(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="openai:compact" if is_compact else "openai:cli",
        decision_kind="openai_compact_stream" if is_compact else "openai_cli_stream",
        report_kind="openai_cli_stream_success",
    )


async def _build_claude_cli_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    return await _build_cli_sync_decision(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="claude:cli",
        decision_kind="claude_cli_sync",
        report_kind="claude_cli_sync_success",
        finalize_kind="claude_cli_sync_finalize",
    )


async def _build_claude_cli_stream_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    return await _build_cli_stream_decision(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="claude:cli",
        decision_kind="claude_cli_stream",
        report_kind="claude_cli_stream_success",
    )


async def _build_gemini_cli_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    return await _build_cli_sync_decision(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="gemini:cli",
        decision_kind="gemini_cli_sync",
        report_kind="gemini_cli_sync_success",
        finalize_kind="gemini_cli_sync_finalize",
    )


async def _build_gemini_cli_stream_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    return await _build_cli_stream_decision(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="gemini:cli",
        decision_kind="gemini_cli_stream",
        report_kind="gemini_cli_stream_success",
    )
