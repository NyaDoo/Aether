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


async def _build_gemini_files_download_stream_plan(
    *,
    request: Request,
    db: Session,
    user: User,
    user_api_key: ApiKey,
    file_id: str,
) -> dict[str, Any]:
    from src.api.public.gemini_files import (
        GEMINI_FILES_BASE_URL,
        UpstreamContext,
        _build_upstream_headers,
        _build_upstream_url,
        _enrich_upstream_context_proxy,
        _find_video_task_by_id,
        _resolve_files_model_name,
        _select_provider_candidate,
        resolve_provider_proxy,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
    )

    proxy_snapshot = None
    provider_id = ""
    endpoint_id = ""
    file_key_id = ""

    if file_id.startswith("aev_"):
        short_id = file_id[4:]
        upstream_key, video_url = await _find_video_task_by_id(db, short_id, user.id)
        if not upstream_key or not video_url:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": 404,
                        "message": f"Video not found or not ready: {file_id}",
                        "status": "NOT_FOUND",
                    }
                },
            )
        upstream_url = video_url

        try:
            from src.services.proxy_node.resolver import (
                build_proxy_url_async,
                get_system_proxy_config_async,
                resolve_delegate_config_async,
                resolve_proxy_info_async,
            )

            system_proxy = await get_system_proxy_config_async()
            delegate_cfg = await resolve_delegate_config_async(system_proxy)
            proxy_url: str | None = None
            if system_proxy and not (delegate_cfg and delegate_cfg.get("tunnel")):
                proxy_url = await build_proxy_url_async(system_proxy)
            proxy_info = await resolve_proxy_info_async(system_proxy)
            proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
                proxy_info,
                proxy_url=proxy_url,
                mode_override="tunnel" if delegate_cfg and delegate_cfg.get("tunnel") else None,
                node_id_override=(
                    str(delegate_cfg.get("node_id") or "").strip() or None
                    if delegate_cfg and delegate_cfg.get("tunnel")
                    else None
                ),
            )
        except Exception:
            proxy_snapshot = None
    else:
        model_name = _resolve_files_model_name(db, user_api_key, user)
        if not model_name:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": 503,
                        "message": "No available model for Gemini Files API routing",
                        "status": "UNAVAILABLE",
                    }
                },
            )

        candidate = await _select_provider_candidate(
            db,
            user_api_key,
            model_name,
            require_files_capability=True,
        )
        if not candidate:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": 503,
                        "message": "No available Gemini key with 'gemini_files' capability enabled",
                        "status": "UNAVAILABLE",
                    }
                },
            )

        try:
            upstream_key = crypto_service.decrypt(candidate.key.api_key)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": 500,
                        "message": "Failed to decrypt provider key",
                        "status": "INTERNAL",
                    }
                },
            ) from exc

        regular_ctx = UpstreamContext(
            upstream_key=upstream_key,
            base_url=candidate.endpoint.base_url or GEMINI_FILES_BASE_URL,
            file_key_id=str(candidate.key.id),
            user_id=str(user.id),
            provider_id=str(candidate.provider.id),
            endpoint_id=str(candidate.endpoint.id),
            provider_proxy=resolve_provider_proxy(endpoint=candidate.endpoint, key=candidate.key),
            key_proxy=(
                candidate.key.proxy
                if isinstance(getattr(candidate.key, "proxy", None), dict)
                else None
            ),
        )
        ctx = await _enrich_upstream_context_proxy(regular_ctx)
        file_key_id = ctx.file_key_id
        provider_id = ctx.provider_id
        endpoint_id = ctx.endpoint_id
        proxy_snapshot = ctx.proxy_snapshot
        file_name = f"files/{file_id}" if not file_id.startswith("files/") else file_id
        upstream_url = _build_upstream_url(
            ctx.base_url,
            f"/v1beta/{file_name}:download",
            dict(request.query_params),
        )

    headers = _build_upstream_headers(dict(request.headers), upstream_key)
    plan = ExecutionPlan(
        request_id=str(getattr(request.state, "request_id", "") or uuid.uuid4().hex),
        candidate_id=None,
        provider_name="gemini",
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        key_id=str(file_key_id or ""),
        method="GET",
        url=upstream_url,
        headers=headers,
        body=ExecutionPlanBody(),
        stream=True,
        provider_api_format="gemini:files",
        client_api_format="gemini:files",
        model_name="gemini-files",
        proxy=proxy_snapshot,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=None,
        ),
    )
    return plan.to_payload()


async def _build_gemini_files_download_stream_decision(
    *,
    request: Request,
    db: Session,
    user: User,
    user_api_key: ApiKey,
    file_id: str,
) -> GatewayExecutionDecisionResponse:
    from src.api.public.gemini_files import (
        GEMINI_FILES_BASE_URL,
        UpstreamContext,
        _build_upstream_headers,
        _build_upstream_url,
        _enrich_upstream_context_proxy,
        _find_video_task_by_id,
        _resolve_files_model_name,
        _select_provider_candidate,
        resolve_provider_proxy,
    )

    proxy_snapshot = None
    provider_id = ""
    endpoint_id = ""
    file_key_id = ""

    if file_id.startswith("aev_"):
        short_id = file_id[4:]
        upstream_key, video_url = await _find_video_task_by_id(db, short_id, user.id)
        if not upstream_key or not video_url:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "code": 404,
                        "message": f"Video not found or not ready: {file_id}",
                        "status": "NOT_FOUND",
                    }
                },
            )
        upstream_url = video_url

        try:
            from src.services.proxy_node.resolver import (
                build_proxy_url_async,
                get_system_proxy_config_async,
                resolve_delegate_config_async,
                resolve_proxy_info_async,
            )
            from src.services.request.execution_runtime_plan import ExecutionProxySnapshot

            system_proxy = await get_system_proxy_config_async()
            delegate_cfg = await resolve_delegate_config_async(system_proxy)
            proxy_url: str | None = None
            if system_proxy and not (delegate_cfg and delegate_cfg.get("tunnel")):
                proxy_url = await build_proxy_url_async(system_proxy)
            proxy_info = await resolve_proxy_info_async(system_proxy)
            proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
                proxy_info,
                proxy_url=proxy_url,
                mode_override="tunnel" if delegate_cfg and delegate_cfg.get("tunnel") else None,
                node_id_override=(
                    str(delegate_cfg.get("node_id") or "").strip() or None
                    if delegate_cfg and delegate_cfg.get("tunnel")
                    else None
                ),
            )
        except Exception:
            proxy_snapshot = None
    else:
        model_name = _resolve_files_model_name(db, user_api_key, user)
        if not model_name:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": 503,
                        "message": "No available model for Gemini Files API routing",
                        "status": "UNAVAILABLE",
                    }
                },
            )

        candidate = await _select_provider_candidate(
            db,
            user_api_key,
            model_name,
            require_files_capability=True,
        )
        if not candidate:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "code": 503,
                        "message": "No available Gemini key with 'gemini_files' capability enabled",
                        "status": "UNAVAILABLE",
                    }
                },
            )

        try:
            upstream_key = crypto_service.decrypt(candidate.key.api_key)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "code": 500,
                        "message": "Failed to decrypt provider key",
                        "status": "INTERNAL",
                    }
                },
            ) from exc

        regular_ctx = UpstreamContext(
            upstream_key=upstream_key,
            base_url=candidate.endpoint.base_url or GEMINI_FILES_BASE_URL,
            file_key_id=str(candidate.key.id),
            user_id=str(user.id),
            provider_id=str(candidate.provider.id),
            endpoint_id=str(candidate.endpoint.id),
            provider_proxy=resolve_provider_proxy(endpoint=candidate.endpoint, key=candidate.key),
            key_proxy=(
                candidate.key.proxy
                if isinstance(getattr(candidate.key, "proxy", None), dict)
                else None
            ),
        )
        ctx = await _enrich_upstream_context_proxy(regular_ctx)
        file_key_id = ctx.file_key_id
        provider_id = ctx.provider_id
        endpoint_id = ctx.endpoint_id
        proxy_snapshot = ctx.proxy_snapshot
        file_name = f"files/{file_id}" if not file_id.startswith("files/") else file_id
        upstream_url = _build_upstream_url(
            ctx.base_url,
            f"/v1beta/{file_name}:download",
            dict(request.query_params),
        )

    headers = _build_upstream_headers(dict(request.headers), upstream_key)
    return GatewayExecutionDecisionResponse(
        action="executor_stream_decision",
        decision_kind="gemini_files_download",
        request_id=str(getattr(request.state, "request_id", "") or uuid.uuid4().hex),
        provider_name="gemini",
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        key_id=str(file_key_id or ""),
        upstream_base_url="",
        upstream_url=upstream_url,
        auth_header="",
        auth_value="",
        provider_api_format="gemini:files",
        client_api_format="gemini:files",
        model_name="gemini-files",
        provider_request_headers=headers,
        proxy=asdict(proxy_snapshot) if proxy_snapshot else None,
        timeouts={
            "connect_ms": 30_000,
            "read_ms": 300_000,
            "write_ms": 300_000,
            "pool_ms": 30_000,
        },
    )


async def _build_gemini_files_get_sync_plan(
    *,
    request: Request,
    db: Session,
    file_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized_file_name = (
        file_name if str(file_name or "").startswith("files/") else f"files/{file_name}"
    )
    return await _build_gemini_files_proxy_sync_plan(
        request=request,
        db=db,
        method="GET",
        upstream_path=f"/v1beta/{normalized_file_name}",
    )


async def _build_gemini_files_proxy_sync_decision(
    *,
    request: Request,
    db: Session,
    method: str,
    upstream_path: str,
    decision_kind: str,
    report_kind: str,
    report_context: dict[str, Any] | None = None,
) -> GatewayExecutionDecisionResponse:
    from src.api.public.gemini_files import (
        _build_upstream_headers,
        _build_upstream_url,
        _enrich_upstream_context_proxy,
        _resolve_upstream_context,
    )

    ctx = await _resolve_upstream_context(request, db)
    ctx = await _enrich_upstream_context_proxy(ctx)

    upstream_url = _build_upstream_url(
        ctx.base_url,
        upstream_path,
        dict(request.query_params),
    )
    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)
    merged_report_context = {
        "file_key_id": str(ctx.file_key_id or ""),
        "user_id": str(ctx.user_id or ""),
    }
    if report_context:
        merged_report_context.update(report_context)

    return GatewayExecutionDecisionResponse(
        action="executor_sync_decision",
        decision_kind=decision_kind,
        request_id=str(getattr(request.state, "request_id", "") or uuid.uuid4().hex),
        provider_name="gemini",
        provider_id=str(ctx.provider_id or ""),
        endpoint_id=str(ctx.endpoint_id or ""),
        key_id=str(ctx.file_key_id or ""),
        upstream_base_url=str(ctx.base_url or ""),
        upstream_url=upstream_url,
        auth_header="",
        auth_value="",
        provider_api_format="gemini:files",
        client_api_format="gemini:files",
        model_name="gemini-files",
        provider_request_headers=headers,
        proxy=asdict(ctx.proxy_snapshot) if ctx.proxy_snapshot else None,
        timeouts={
            "connect_ms": 30_000,
            "read_ms": 300_000,
            "write_ms": 300_000,
            "pool_ms": 30_000,
            "total_ms": 300_000,
        },
        report_kind=report_kind,
        report_context=merged_report_context,
    )


async def _build_gemini_files_proxy_sync_plan(
    *,
    request: Request,
    db: Session,
    method: str,
    upstream_path: str,
    is_upload: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.api.public.gemini_files import (
        _build_upstream_headers,
        _build_upstream_url,
        _enrich_upstream_context_proxy,
        _resolve_upstream_context,
    )
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
    )

    ctx = await _resolve_upstream_context(request, db)
    ctx = await _enrich_upstream_context_proxy(ctx)

    upstream_url = _build_upstream_url(
        ctx.base_url,
        upstream_path,
        dict(request.query_params),
        is_upload=is_upload,
    )
    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)
    body_bytes = await request.body()
    content_type = str(headers.get("content-type") or "").strip() or None
    plan = ExecutionPlan(
        request_id=str(getattr(request.state, "request_id", "") or uuid.uuid4().hex),
        candidate_id=None,
        provider_name="gemini",
        provider_id=str(ctx.provider_id or ""),
        endpoint_id=str(ctx.endpoint_id or ""),
        key_id=str(ctx.file_key_id or ""),
        method=method.upper(),
        url=upstream_url,
        headers=headers,
        body=(
            build_execution_plan_body(body_bytes, content_type=content_type)
            if body_bytes
            else build_execution_plan_body(None)
        ),
        stream=False,
        provider_api_format="gemini:files",
        client_api_format="gemini:files",
        model_name="gemini-files",
        proxy=ctx.proxy_snapshot,
        timeouts=ExecutionPlanTimeouts(
            connect_ms=30_000,
            read_ms=300_000,
            write_ms=300_000,
            pool_ms=30_000,
            total_ms=300_000,
        ),
    )
    return plan.to_payload(), {
        "file_key_id": str(ctx.file_key_id or ""),
        "user_id": str(ctx.user_id or ""),
    }
