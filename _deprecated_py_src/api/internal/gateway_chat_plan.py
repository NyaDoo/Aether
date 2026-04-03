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


async def _build_chat_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_api_format: str,
    plan_kind: str,
    report_kind: str,
    finalize_kind: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.base.chat_adapter_base import ChatAdapterBase
    from src.api.handlers.base.chat_sync_executor import ChatSyncExecutor
    from src.services.task.request_state import MutableRequestBodyState

    if str(payload.method or "").strip().upper() != "POST":
        return None

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if not isinstance(adapter, ChatAdapterBase):
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
    request_obj = adapter._validate_request_body(original_request_body, context.path_params)
    if isinstance(request_obj, JSONResponse):
        return None

    handler = adapter._create_handler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=context.request_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        start_time=context.start_time,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    await handler._convert_request(request_obj)
    model = handler.extract_model_from_request(
        original_request_body,
        context.path_params,
    )
    api_format = handler.allowed_api_formats[0]
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
        api_format=str(api_format),
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

    sync_executor = ChatSyncExecutor(handler)
    prepared_plan = await sync_executor._build_sync_execution_plan(
        candidate.provider,
        candidate.endpoint,
        candidate.key,
        candidate,
        model=str(model or "unknown"),
        api_format=api_format,
        original_headers=context.original_headers,
        request_state=MutableRequestBodyState(original_request_body),
        query_params=context.query_params,
        client_content_encoding=context.client_content_encoding,
    )

    contract_provider_api_format = (
        str(prepared_plan.contract.provider_api_format or "").strip().lower()
    )
    contract_client_api_format = str(prepared_plan.contract.client_api_format or "").strip().lower()
    if not prepared_plan.remote_eligible or contract_client_api_format != expected_api_format:
        return None

    selected_report_kind = (
        finalize_kind
        if (
            prepared_plan.upstream_is_stream
            or prepared_plan.needs_conversion
            or prepared_plan.envelope is not None
            or contract_provider_api_format != expected_api_format
        )
        else report_kind
    )

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind=plan_kind,
        plan=prepared_plan.contract.to_payload(),
        report_kind=selected_report_kind,
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "model": str(model or "unknown"),
            "provider_name": str(prepared_plan.contract.provider_name or "unknown"),
            "provider_id": str(prepared_plan.contract.provider_id or ""),
            "endpoint_id": str(prepared_plan.contract.endpoint_id or ""),
            "key_id": str(prepared_plan.contract.key_id or ""),
            "provider_api_format": str(prepared_plan.contract.provider_api_format or ""),
            "client_api_format": str(prepared_plan.contract.client_api_format or ""),
            "mapped_model": sync_executor._ctx.mapped_model_result,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(prepared_plan.headers),
            "provider_request_body": prepared_plan.payload,
            "proxy_info": prepared_plan.proxy_info,
            "has_envelope": prepared_plan.envelope is not None,
            "envelope_name": gateway_module._gateway_report_context_envelope_name(
                prepared_plan.envelope
            ),
            "needs_conversion": bool(prepared_plan.needs_conversion),
        },
    )


async def _build_openai_chat_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="openai:chat",
        plan_kind="openai_chat_sync",
        report_kind="openai_chat_sync_success",
        finalize_kind="openai_chat_sync_finalize",
    )


async def _build_chat_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
    expected_api_format: str,
    plan_kind: str,
    report_kind: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.base.chat_adapter_base import ChatAdapterBase
    from src.config.settings import config
    from src.core.api_format.headers import set_accept_if_absent
    from src.services.provider.auth import get_provider_auth
    from src.services.provider.transport import build_provider_url
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
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
    if not isinstance(adapter, ChatAdapterBase):
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
    request_obj = adapter._validate_request_body(original_request_body, context.path_params)
    if isinstance(request_obj, JSONResponse):
        return None

    handler = adapter._create_handler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=context.request_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        start_time=context.start_time,
        api_family=adapter.API_FAMILY.value if adapter.API_FAMILY else None,
        endpoint_kind=adapter.ENDPOINT_KIND.value if adapter.ENDPOINT_KIND else None,
    )

    converted_request = await handler._convert_request(request_obj)
    model = getattr(converted_request, "model", original_request_body.get("model", "unknown"))
    api_format = handler.allowed_api_formats[0]
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
        api_format=str(api_format),
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
    provider_api_format = str(endpoint.api_format or api_format or "")
    prep = await handler._prepare_provider_request(
        model=str(model or "unknown"),
        provider=provider,
        endpoint=endpoint,
        key=key,
        working_request_body=dict(original_request_body),
        original_headers=context.original_headers,
        client_api_format=str(api_format),
        provider_api_format=provider_api_format,
        candidate=candidate,
        client_is_stream=True,
    )
    provider_api_format = str(prep.provider_api_format or "")

    auth_info = prep.auth_info or await get_provider_auth(endpoint, key)
    provider_payload, provider_headers = handler._request_builder.build(
        prep.request_body,
        context.original_headers,
        endpoint,
        key,
        is_stream=prep.upstream_is_stream,
        extra_headers=prep.extra_headers if prep.extra_headers else None,
        pre_computed_auth=auth_info.as_tuple() if auth_info else None,
        envelope=prep.envelope,
        provider_api_format=prep.provider_api_format,
    )
    if prep.upstream_is_stream:
        set_accept_if_absent(provider_headers)
    upstream_url = build_provider_url(
        endpoint,
        query_params=context.query_params,
        path_params={"model": prep.url_model},
        is_stream=prep.upstream_is_stream,
        key=key,
        decrypted_auth_config=auth_info.decrypted_auth_config if auth_info else None,
    )

    effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
    proxy_info = await resolve_proxy_info_async(effective_proxy)
    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    effective_proxy_for_contract = effective_proxy
    if not effective_proxy_for_contract or not effective_proxy_for_contract.get("enabled", True):
        effective_proxy_for_contract = await get_system_proxy_config_async()
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))

    proxy_url: str | None = None
    if effective_proxy_for_contract and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy_for_contract)

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
        url=upstream_url,
        headers=dict(provider_headers),
        body=build_execution_plan_body(
            provider_payload,
            content_type=str(provider_headers.get("content-type") or "").strip() or None,
        ),
        stream=True,
        provider_api_format=provider_api_format,
        client_api_format=str(api_format),
        model_name=str(model or ""),
        content_type=str(provider_headers.get("content-type") or "").strip() or None,
        content_encoding=context.client_content_encoding,
        proxy=ExecutionProxySnapshot.from_proxy_info(
            proxy_info,
            proxy_url=proxy_url,
            mode_override="tunnel" if is_tunnel_delegate else None,
            node_id_override=(
                str(delegate_cfg.get("node_id") or "").strip() or None
                if is_tunnel_delegate
                else None
            ),
        ),
        tls_profile=prep.tls_profile,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=int(config.http_connect_timeout * 1000),
            read_ms=int(config.http_read_timeout * 1000),
            write_ms=int(config.http_write_timeout * 1000),
            pool_ms=int(config.http_pool_timeout * 1000),
            total_ms=None,
        ),
    )

    if (
        not prep.upstream_is_stream
        or gateway_module._stream_executor_requires_python_rewrite(
            envelope=prep.envelope,
            needs_conversion=bool(prep.needs_conversion),
            provider_api_format=provider_api_format,
            client_api_format=str(contract.client_api_format or ""),
        )
        or not is_remote_execution_runtime_contract_eligible(contract)
        or provider_api_format.strip().lower() != expected_api_format
        or str(contract.client_api_format or "").strip().lower() != expected_api_format
    ):
        return None

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
            "mapped_model": prep.mapped_model,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(provider_headers),
            "provider_request_body": provider_payload,
            "proxy_info": proxy_info,
            "has_envelope": prep.envelope is not None,
            "envelope_name": gateway_module._gateway_report_context_envelope_name(prep.envelope),
            "needs_conversion": bool(prep.needs_conversion),
        },
    )


async def _build_openai_chat_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="openai:chat",
        plan_kind="openai_chat_stream",
        report_kind="openai_chat_stream_success",
    )


async def _build_claude_chat_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="claude:chat",
        plan_kind="claude_chat_stream",
        report_kind="claude_chat_stream_success",
    )


async def _build_gemini_chat_stream_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_stream_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="gemini:chat",
        plan_kind="gemini_chat_stream",
        report_kind="gemini_chat_stream_success",
    )


async def _build_claude_chat_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="claude:chat",
        plan_kind="claude_chat_sync",
        report_kind="claude_chat_sync_success",
        finalize_kind="claude_chat_sync_finalize",
    )


async def _build_gemini_chat_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    return await _build_chat_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
        expected_api_format="gemini:chat",
        plan_kind="gemini_chat_sync",
        report_kind="gemini_chat_sync_success",
        finalize_kind="gemini_chat_sync_finalize",
    )
