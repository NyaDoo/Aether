from __future__ import annotations

import base64
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

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


def _build_video_sync_decision_from_plan(
    *,
    planned: GatewayExecutionPlanResponse | None,
    decision_kind: str,
    auth_context: GatewayAuthContext,
) -> GatewayExecutionDecisionResponse | None:
    if planned is None:
        return None

    plan = planned.plan if isinstance(planned.plan, dict) else {}
    if not plan:
        return None

    body = plan.get("body")
    provider_request_body: dict[str, Any] | None = None
    if isinstance(body, dict):
        json_body = body.get("json_body")
        if isinstance(json_body, dict):
            provider_request_body = json_body

    upstream_url = str(plan.get("url") or "").strip()
    upstream_base_url = ""
    if upstream_url:
        parsed = urlsplit(upstream_url)
        if parsed.scheme and parsed.netloc:
            upstream_base_url = f"{parsed.scheme}://{parsed.netloc}"

    provider_request_headers = plan.get("headers")
    if not isinstance(provider_request_headers, dict):
        provider_request_headers = None

    timeouts = plan.get("timeouts")
    if not isinstance(timeouts, dict):
        timeouts = None

    proxy = plan.get("proxy")
    if not isinstance(proxy, dict):
        proxy = None

    provider_request_method = str(plan.get("method") or "").strip() or None
    content_type = (
        str(
            plan.get("content_type")
            or (
                provider_request_headers.get("content-type")
                if isinstance(provider_request_headers, dict)
                else ""
            )
            or ""
        ).strip()
        or None
    )

    return GatewayExecutionDecisionResponse(
        action="executor_sync_decision",
        decision_kind=decision_kind,
        request_id=str(plan.get("request_id") or ""),
        candidate_id=str(plan.get("candidate_id") or "").strip() or None,
        provider_name=str(plan.get("provider_name") or ""),
        provider_id=str(plan.get("provider_id") or ""),
        endpoint_id=str(plan.get("endpoint_id") or ""),
        key_id=str(plan.get("key_id") or ""),
        upstream_base_url=upstream_base_url,
        upstream_url=upstream_url or None,
        provider_request_method=provider_request_method,
        auth_header="",
        auth_value="",
        provider_api_format=str(plan.get("provider_api_format") or ""),
        client_api_format=str(plan.get("client_api_format") or ""),
        model_name=str(plan.get("model_name") or ""),
        provider_request_headers=provider_request_headers,
        provider_request_body=provider_request_body,
        content_type=content_type,
        proxy=proxy,
        tls_profile=str(plan.get("tls_profile") or "").strip() or None,
        timeouts=timeouts,
        report_kind=planned.report_kind,
        report_context=planned.report_context,
        auth_context=auth_context,
    )


async def _build_openai_video_create_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.base.request_builder import apply_body_rules
    from src.api.handlers.openai.video_adapter import OpenAIVideoAdapter
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format import make_signature_key
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if (
        str(payload.method or "").strip().upper() != "POST"
        or str(payload.path or "").strip() != "/v1/videos"
    ):
        return None
    if payload.body_base64:
        return None

    adapter = OpenAIVideoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    original_request_body = await context.ensure_json_body_async()
    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=context.request_id,
        client_ip=context.client_ip,
        user_agent=context.user_agent,
        start_time=context.start_time,
        allowed_api_formats=adapter.allowed_api_formats,
    )

    internal_request = handler._normalizer.video_request_to_internal(original_request_body)
    candidate = await gateway_module._select_gateway_direct_candidate(
        db=db,
        redis_client=None,
        api_format=handler.FORMAT_ID,
        model_name=str(internal_request.model or "unknown"),
        user_api_key=api_key,
        request_id=context.request_id,
        is_stream=False,
        capability_requirements=None,
        preferred_key_ids=None,
        request_body=original_request_body,
    )
    if candidate is None:
        return None

    provider_api_format = make_signature_key(
        str(getattr(candidate.endpoint, "api_family", "")).strip().lower(),
        str(getattr(candidate.endpoint, "endpoint_kind", "")).strip().lower(),
    )
    if provider_api_format != handler.FORMAT_ID or bool(
        getattr(candidate, "needs_conversion", False)
    ):
        return None

    upstream_key, endpoint, provider_key, _auth_info = await handler._resolve_upstream_key(
        candidate
    )
    request_body = dict(original_request_body)
    if "seconds" in request_body and request_body["seconds"] is not None:
        request_body["seconds"] = str(request_body["seconds"])

    endpoint_body_rules = getattr(endpoint, "body_rules", None)
    if endpoint_body_rules:
        request_body = apply_body_rules(
            request_body,
            endpoint_body_rules,
            original_body=original_request_body,
        )

    upstream_url = handler._build_upstream_url(endpoint.base_url)
    headers = handler._build_upstream_headers(
        context.original_headers,
        upstream_key,
        endpoint,
        body=request_body,
        original_body=original_request_body,
    )

    content_type = str(headers.get("content-type") or "").strip() or "application/json"
    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(candidate.provider.name),
        provider_id=str(candidate.provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(provider_key.id),
        method="POST",
        url=upstream_url,
        headers=dict(headers),
        body=build_execution_plan_body(request_body, content_type=content_type),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(internal_request.model or ""),
        content_type=content_type,
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="openai_video_create_sync",
        plan=contract.to_payload(),
        report_kind="openai_video_create_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "model": str(internal_request.model or "unknown"),
            "provider_name": str(candidate.provider.name),
            "provider_id": str(candidate.provider.id),
            "endpoint_id": str(endpoint.id),
            "key_id": str(provider_key.id),
            "provider_api_format": provider_api_format,
            "client_api_format": handler.FORMAT_ID,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(headers),
            "provider_request_body": request_body,
        },
    )


async def _build_openai_video_create_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_openai_video_create_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        decision=decision,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="openai_video_create_sync",
        auth_context=auth_context,
    )


async def _build_openai_video_remix_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.base.request_builder import apply_body_rules
    from src.api.handlers.openai.video_adapter import OpenAIVideoAdapter
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format import make_signature_key
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.core.crypto import crypto_service
    from src.models.database import Provider
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST" or not task_id or payload.body_base64:
        return None

    adapter = OpenAIVideoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={"task_id": task_id},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    original_request_body = await context.ensure_json_body_async()
    try:
        handler = OpenAIVideoHandler(
            db=db,
            user=user,
            api_key=api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=adapter.allowed_api_formats,
        )
        original_task = handler._get_task(task_id)
        if original_task.status != VideoStatus.COMPLETED.value:
            return None
        if not original_task.external_task_id:
            return None

        endpoint, key = handler._get_endpoint_and_key(original_task)
        if not getattr(key, "api_key", None):
            return None
        provider = db.query(Provider).filter(Provider.id == original_task.provider_id).first()
        if provider is None:
            return None

        upstream_key = crypto_service.decrypt(key.api_key)
        upstream_url = handler._build_upstream_url(
            endpoint.base_url,
            f"{original_task.external_task_id}/remix",
        )

        request_body = dict(original_request_body)
        if "seconds" in request_body and request_body["seconds"] is not None:
            request_body["seconds"] = str(request_body["seconds"])

        endpoint_body_rules = getattr(endpoint, "body_rules", None)
        if endpoint_body_rules:
            request_body = apply_body_rules(
                request_body,
                endpoint_body_rules,
                original_body=original_request_body,
            )

        headers = handler._build_upstream_headers(
            context.original_headers,
            upstream_key,
            endpoint,
            body=request_body,
            original_body=original_request_body,
        )

        provider_api_format = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )
        if provider_api_format != handler.FORMAT_ID:
            return None
    except HTTPException:
        return None
    except Exception:
        return None

    content_type = str(headers.get("content-type") or "").strip() or "application/json"
    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=None,
        provider_name=str(provider.name),
        provider_id=str(provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(key.id),
        method="POST",
        url=upstream_url,
        headers=dict(headers),
        body=build_execution_plan_body(request_body, content_type=content_type),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(getattr(original_task, "model", "") or ""),
        content_type=content_type,
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="openai_video_remix_sync",
        plan=contract.to_payload(),
        report_kind="openai_video_remix_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "task_id": task_id,
            "model": str(getattr(original_task, "model", "") or ""),
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
        },
    )


async def _build_openai_video_remix_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_openai_video_remix_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        task_id=task_id,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="openai_video_remix_sync",
        auth_context=auth_context,
    )


async def _build_gemini_video_create_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    model: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.base.request_builder import apply_body_rules
    from src.api.handlers.gemini.video_adapter import GeminiVeoAdapter
    from src.api.handlers.gemini.video_handler import GeminiVeoHandler
    from src.core.api_format import make_signature_key
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST" or not model or payload.body_base64:
        return None

    adapter = GeminiVeoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={"model": model},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    original_request_body = await context.ensure_json_body_async()
    request_with_model = {**original_request_body, "model": model}
    try:
        handler = GeminiVeoHandler(
            db=db,
            user=user,
            api_key=api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=adapter.allowed_api_formats,
        )
        internal_request = handler._normalizer.video_request_to_internal(request_with_model)
        candidate = await gateway_module._select_gateway_direct_candidate(
            db=db,
            redis_client=None,
            api_format=handler.FORMAT_ID,
            model_name=str(internal_request.model or "unknown"),
            user_api_key=api_key,
            request_id=context.request_id,
            is_stream=False,
            capability_requirements=None,
            preferred_key_ids=None,
            request_body=request_with_model,
        )
        if candidate is None:
            return None

        provider_api_format = make_signature_key(
            str(getattr(candidate.endpoint, "api_family", "")).strip().lower(),
            str(getattr(candidate.endpoint, "endpoint_kind", "")).strip().lower(),
        )
        if provider_api_format != handler.FORMAT_ID or bool(
            getattr(candidate, "needs_conversion", False)
        ):
            return None

        upstream_key, endpoint, provider_key, auth_info = await handler._resolve_upstream_key(
            candidate
        )
        endpoint_body_rules = getattr(endpoint, "body_rules", None)
        request_body = dict(original_request_body)
        if endpoint_body_rules:
            request_body = apply_body_rules(
                request_body,
                endpoint_body_rules,
                original_body=original_request_body,
            )

        upstream_url = handler._build_upstream_url(endpoint.base_url, internal_request.model)
        headers = handler._build_upstream_headers(
            context.original_headers,
            upstream_key,
            endpoint,
            auth_info,
            body=request_body,
            original_body=original_request_body,
        )
    except HTTPException:
        return None
    except Exception:
        return None

    content_type = str(headers.get("content-type") or "").strip() or "application/json"
    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=str(
            getattr(candidate, "request_candidate_id", "") or getattr(candidate, "id", "") or ""
        )
        or None,
        provider_name=str(candidate.provider.name),
        provider_id=str(candidate.provider.id),
        endpoint_id=str(endpoint.id),
        key_id=str(provider_key.id),
        method="POST",
        url=upstream_url,
        headers=dict(headers),
        body=build_execution_plan_body(request_body, content_type=content_type),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(internal_request.model or ""),
        content_type=content_type,
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="gemini_video_create_sync",
        plan=contract.to_payload(),
        report_kind="gemini_video_create_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "request_id": str(context.request_id),
            "model": str(internal_request.model or model),
            "provider_name": str(candidate.provider.name),
            "provider_id": str(candidate.provider.id),
            "endpoint_id": str(endpoint.id),
            "key_id": str(provider_key.id),
            "provider_api_format": provider_api_format,
            "client_api_format": handler.FORMAT_ID,
            "original_headers": dict(context.original_headers),
            "original_request_body": original_request_body,
            "provider_request_headers": dict(headers),
            "provider_request_body": request_body,
        },
    )


async def _build_gemini_video_create_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    model: str,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_gemini_video_create_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        model=model,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="gemini_video_create_sync",
        auth_context=auth_context,
    )


async def _build_openai_video_cancel_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.openai.video_adapter import OpenAIVideoAdapter
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format import make_signature_key
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.core.crypto import crypto_service
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST" or not task_id:
        return None

    adapter = OpenAIVideoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={"task_id": task_id, "action": "cancel"},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    try:
        handler = OpenAIVideoHandler(
            db=db,
            user=user,
            api_key=api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=adapter.allowed_api_formats,
        )
        task = handler._get_task(task_id)
        if task.status in {
            VideoStatus.COMPLETED.value,
            VideoStatus.FAILED.value,
            VideoStatus.CANCELLED.value,
            VideoStatus.EXPIRED.value,
        }:
            return None
        if not task.external_task_id:
            return None

        endpoint, key = handler._get_endpoint_and_key(task)
        if not getattr(key, "api_key", None):
            return None
        upstream_key = crypto_service.decrypt(key.api_key)
        upstream_url = handler._build_upstream_url(endpoint.base_url, task.external_task_id)
        headers = handler._build_upstream_headers(
            context.original_headers,
            upstream_key,
            endpoint,
        )
        provider_api_format = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )
        if provider_api_format != handler.FORMAT_ID:
            return None
    except HTTPException:
        return None
    except Exception:
        return None

    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=None,
        provider_name="openai",
        provider_id=str(getattr(endpoint, "provider_id", "") or ""),
        endpoint_id=str(getattr(endpoint, "id", "") or ""),
        key_id=str(getattr(key, "id", "") or ""),
        method="DELETE",
        url=upstream_url,
        headers=dict(headers),
        body=ExecutionPlanBody(),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(getattr(task, "model", "") or ""),
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="openai_video_cancel_sync",
        plan=contract.to_payload(),
        report_kind="openai_video_cancel_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "task_id": task_id,
        },
    )


async def _build_openai_video_cancel_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_openai_video_cancel_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        task_id=task_id,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="openai_video_cancel_sync",
        auth_context=auth_context,
    )


async def _build_openai_video_delete_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.openai.video_adapter import OpenAIVideoAdapter
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format import make_signature_key
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.core.crypto import crypto_service
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "DELETE" or not task_id:
        return None

    adapter = OpenAIVideoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={"task_id": task_id},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    try:
        handler = OpenAIVideoHandler(
            db=db,
            user=user,
            api_key=api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=adapter.allowed_api_formats,
        )
        task = handler._get_task(task_id)
        if task.status not in (VideoStatus.COMPLETED.value, VideoStatus.FAILED.value):
            return None
        if not task.external_task_id:
            return None

        endpoint, key = handler._get_endpoint_and_key(task)
        if not getattr(key, "api_key", None):
            return None
        upstream_key = crypto_service.decrypt(key.api_key)
        upstream_url = handler._build_upstream_url(endpoint.base_url, task.external_task_id)
        headers = handler._build_upstream_headers(
            context.original_headers,
            upstream_key,
            endpoint,
        )
        provider_api_format = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )
        if provider_api_format != handler.FORMAT_ID:
            return None
    except HTTPException:
        return None
    except Exception:
        return None

    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=None,
        provider_name="openai",
        provider_id=str(getattr(endpoint, "provider_id", "") or ""),
        endpoint_id=str(getattr(endpoint, "id", "") or ""),
        key_id=str(getattr(key, "id", "") or ""),
        method="DELETE",
        url=upstream_url,
        headers=dict(headers),
        body=ExecutionPlanBody(),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(getattr(task, "model", "") or ""),
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="openai_video_delete_sync",
        plan=contract.to_payload(),
        report_kind="openai_video_delete_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "task_id": task_id,
        },
    )


async def _build_openai_video_delete_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_openai_video_delete_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        task_id=task_id,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="openai_video_delete_sync",
        auth_context=auth_context,
    )


async def _build_gemini_video_cancel_sync_plan(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionPlanResponse | None:
    from src.api.handlers.gemini.video_adapter import GeminiVeoAdapter
    from src.api.handlers.gemini.video_handler import GeminiVeoHandler
    from src.core.api_format import make_signature_key
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.services.provider.auth import get_provider_auth
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
        is_remote_execution_runtime_contract_eligible,
    )

    if str(payload.method or "").strip().upper() != "POST" or not task_id:
        return None

    adapter = GeminiVeoAdapter()
    user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
    gateway_request = gateway_module._build_gateway_forward_request(
        request=request, payload=payload
    )
    pipeline = gateway_module.get_pipeline()
    await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

    context = gateway_module._build_gateway_request_context(
        request=gateway_request,
        payload=payload,
        db=db,
        user=user,
        api_key=api_key,
        adapter=adapter,
        path_params={"task_id": task_id, "action": "cancel"},
        balance_remaining=auth_context.balance_remaining,
    )
    authorize_result = adapter.authorize(context)
    if hasattr(authorize_result, "__await__"):
        await authorize_result

    try:
        handler = GeminiVeoHandler(
            db=db,
            user=user,
            api_key=api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=adapter.allowed_api_formats,
        )
        task = handler._get_task_by_external_id(task_id)
        if task.status in {
            VideoStatus.COMPLETED.value,
            VideoStatus.FAILED.value,
            VideoStatus.CANCELLED.value,
            VideoStatus.EXPIRED.value,
        }:
            return None
        if not task.external_task_id:
            return None

        endpoint, key = handler._get_endpoint_and_key(task)
        if not getattr(key, "api_key", None):
            return None
        upstream_key = crypto_service.decrypt(key.api_key)
        auth_info = await get_provider_auth(endpoint, key)
        operation_name = str(task.external_task_id or "").strip()
        if not (operation_name.startswith("operations/") or operation_name.startswith("models/")):
            operation_name = f"operations/{operation_name}"
        upstream_url = handler._build_cancel_url(endpoint.base_url, operation_name)
        request_body: dict[str, Any] = {}
        headers = handler._build_upstream_headers(
            context.original_headers,
            upstream_key,
            endpoint,
            auth_info,
            body=request_body,
            original_body=request_body,
        )
        provider_api_format = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )
        if provider_api_format != handler.FORMAT_ID:
            return None
    except HTTPException:
        return None
    except Exception:
        return None

    content_type = str(headers.get("content-type") or "").strip() or "application/json"
    contract = ExecutionPlan(
        request_id=str(context.request_id or ""),
        candidate_id=None,
        provider_name="gemini",
        provider_id=str(getattr(endpoint, "provider_id", "") or ""),
        endpoint_id=str(getattr(endpoint, "id", "") or ""),
        key_id=str(getattr(key, "id", "") or ""),
        method="POST",
        url=upstream_url,
        headers=dict(headers),
        body=build_execution_plan_body(request_body, content_type=content_type),
        stream=False,
        provider_api_format=provider_api_format,
        client_api_format=handler.FORMAT_ID,
        model_name=str(getattr(task, "model", "") or ""),
        content_type=content_type,
        content_encoding=context.client_content_encoding,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    if not is_remote_execution_runtime_contract_eligible(contract):
        return None

    return GatewayExecutionPlanResponse(
        action="executor_sync",
        plan_kind="gemini_video_cancel_sync",
        plan=contract.to_payload(),
        report_kind="gemini_video_cancel_sync_finalize",
        report_context={
            "user_id": str(user.id),
            "api_key_id": str(api_key.id),
            "task_id": task_id,
        },
    )


async def _build_gemini_video_cancel_sync_decision(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    task_id: str,
) -> GatewayExecutionDecisionResponse | None:
    planned = await _build_gemini_video_cancel_sync_plan(
        request=request,
        payload=payload,
        db=db,
        auth_context=auth_context,
        task_id=task_id,
    )
    return _build_video_sync_decision_from_plan(
        planned=planned,
        decision_kind="gemini_video_cancel_sync",
        auth_context=auth_context,
    )


async def _build_openai_video_content_stream_plan(
    *,
    request: Request,
    db: Session,
    user: User,
    user_api_key: ApiKey,
    task_id: str,
) -> dict[str, Any]:
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
    )

    request_id = str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)
    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=user_api_key,
        request_id=request_id,
        client_ip=request.client.host if request.client else "127.0.0.1",
        user_agent=str(request.headers.get("user-agent") or "aether-gateway"),
        start_time=time.time(),
    )

    task = handler._get_task(task_id)

    if task.status in (
        VideoStatus.PENDING.value,
        VideoStatus.SUBMITTED.value,
        VideoStatus.QUEUED.value,
        VideoStatus.PROCESSING.value,
    ):
        raise HTTPException(
            status_code=202,
            detail=f"Video is still processing (status: {task.status})",
        )
    if task.status == VideoStatus.FAILED.value:
        raise HTTPException(
            status_code=422,
            detail=f"Video generation failed: {task.error_message or 'Unknown error'}",
        )
    if task.status == VideoStatus.CANCELLED.value:
        raise HTTPException(status_code=404, detail="Video task was cancelled")

    query_params = dict(request.query_params)
    variant = str(query_params.get("variant") or "video")
    if variant not in {"video", "thumbnail", "spritesheet"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant '{variant}'. Must be one of: video, thumbnail, spritesheet",
        )

    provider_id = ""
    endpoint_id = ""
    key_id = ""

    if variant == "video" and task.video_url and str(task.video_url).startswith("http"):
        upstream_url = str(task.video_url)
        headers: dict[str, str] = {}
    else:
        if not task.external_task_id:
            raise HTTPException(status_code=500, detail="Task missing external_task_id")
        endpoint, key = handler._get_endpoint_and_key(task)
        if not key.api_key:
            raise HTTPException(status_code=500, detail="Provider key not configured")
        try:
            upstream_key = crypto_service.decrypt(key.api_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to decrypt provider key") from exc

        content_path = f"{task.external_task_id}/content"
        if variant != "video":
            content_path = f"{content_path}?variant={variant}"
        upstream_url = handler._build_upstream_url(endpoint.base_url, content_path)
        headers = handler._build_upstream_headers(dict(request.headers), upstream_key, endpoint)
        provider_id = str(getattr(endpoint, "provider_id", "") or "")
        endpoint_id = str(getattr(endpoint, "id", "") or "")
        key_id = str(getattr(key, "id", "") or "")

    plan = ExecutionPlan(
        request_id=request_id,
        candidate_id=None,
        provider_name="openai",
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        key_id=key_id,
        method="GET",
        url=upstream_url,
        headers=headers,
        body=ExecutionPlanBody(),
        stream=True,
        provider_api_format="openai:video",
        client_api_format="openai:video",
        model_name=str(getattr(task, "model", "") or ""),
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=None,
        ),
    )
    return plan.to_payload()


async def _build_openai_video_content_stream_decision(
    *,
    request: Request,
    db: Session,
    user: User,
    user_api_key: ApiKey,
    task_id: str,
) -> GatewayExecutionDecisionResponse:
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format.conversion.internal_video import VideoStatus

    request_id = str(getattr(request.state, "request_id", "") or uuid.uuid4().hex)
    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=user_api_key,
        request_id=request_id,
        client_ip=request.client.host if request.client else "127.0.0.1",
        user_agent=str(request.headers.get("user-agent") or "aether-gateway"),
        start_time=time.time(),
    )

    task = handler._get_task(task_id)

    if task.status in (
        VideoStatus.PENDING.value,
        VideoStatus.SUBMITTED.value,
        VideoStatus.QUEUED.value,
        VideoStatus.PROCESSING.value,
    ):
        raise HTTPException(
            status_code=202,
            detail=f"Video is still processing (status: {task.status})",
        )
    if task.status == VideoStatus.FAILED.value:
        raise HTTPException(
            status_code=422,
            detail=f"Video generation failed: {task.error_message or 'Unknown error'}",
        )
    if task.status == VideoStatus.CANCELLED.value:
        raise HTTPException(status_code=404, detail="Video task was cancelled")

    query_params = dict(request.query_params)
    variant = str(query_params.get("variant") or "video")
    if variant not in {"video", "thumbnail", "spritesheet"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variant '{variant}'. Must be one of: video, thumbnail, spritesheet",
        )

    provider_id = ""
    endpoint_id = ""
    key_id = ""

    if variant == "video" and task.video_url and str(task.video_url).startswith("http"):
        upstream_url = str(task.video_url)
        headers: dict[str, str] = {}
    else:
        if not task.external_task_id:
            raise HTTPException(status_code=500, detail="Task missing external_task_id")
        endpoint, key = handler._get_endpoint_and_key(task)
        if not key.api_key:
            raise HTTPException(status_code=500, detail="Provider key not configured")
        try:
            upstream_key = crypto_service.decrypt(key.api_key)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Failed to decrypt provider key") from exc

        content_path = f"{task.external_task_id}/content"
        if variant != "video":
            content_path = f"{content_path}?variant={variant}"
        upstream_url = handler._build_upstream_url(endpoint.base_url, content_path)
        headers = handler._build_upstream_headers(dict(request.headers), upstream_key, endpoint)
        provider_id = str(getattr(endpoint, "provider_id", "") or "")
        endpoint_id = str(getattr(endpoint, "id", "") or "")
        key_id = str(getattr(key, "id", "") or "")

    return GatewayExecutionDecisionResponse(
        action="executor_stream_decision",
        decision_kind="openai_video_content",
        request_id=request_id,
        provider_name="openai",
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        key_id=key_id,
        upstream_base_url="",
        upstream_url=upstream_url,
        auth_header="",
        auth_value="",
        provider_api_format="openai:video",
        client_api_format="openai:video",
        model_name=str(getattr(task, "model", "") or ""),
        provider_request_headers=headers,
        timeouts={
            "connect_ms": 30_000,
            "read_ms": 300_000,
            "write_ms": 300_000,
            "pool_ms": 30_000,
        },
    )
