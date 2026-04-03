from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from src.database import get_db

from . import gateway as gateway_impl
from .gateway_contract import (
    CONTROL_EXECUTED_HEADER,
    GatewayAuthContextRequest,
    GatewayExecuteRequest,
    GatewayResolveRequest,
    GatewayStreamReportRequest,
    GatewaySyncReportRequest,
    classify_gateway_route,
)

router = APIRouter(
    prefix="/api/internal/gateway",
    tags=["Internal - Gateway"],
    include_in_schema=False,
)


def _ensure_legacy_internal_gateway_request(request: Request) -> Response | None:
    if gateway_impl._request_allows_legacy_chat_cli_internal_gateway(request):
        return None
    return gateway_impl._build_retired_internal_gateway_response()


@router.post("/resolve")
async def resolve_gateway_route(
    request: Request, payload: GatewayResolveRequest
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    decision.auth_context = await gateway_impl._resolve_auth_context(payload, decision)
    return JSONResponse(status_code=200, content=decision.model_dump(exclude_none=True))


@router.post("/auth-context")
async def resolve_gateway_auth_context(
    request: Request,
    payload: GatewayAuthContextRequest,
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    auth_context = await gateway_impl._resolve_auth_context_signature(
        headers=payload.headers,
        query_string=payload.query_string,
        auth_endpoint_signature=payload.auth_endpoint_signature,
    )
    return JSONResponse(status_code=200, content={"auth_context": auth_context})


@router.post("/decision-sync")
async def decide_gateway_sync(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    if not gateway_impl._allows_legacy_chat_cli_internal_route(request, decision):
        return gateway_impl._build_retired_internal_gateway_response()
    auth_context = await gateway_impl._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )

    if auth_context is None or not auth_context.access_allowed:
        return JSONResponse(status_code=200, content={"action": "fallback_plan"})

    try:
        resolved = await gateway_impl._build_gateway_sync_decision_response(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    if resolved is None:
        body: dict[str, object] = {"action": "fallback_plan"}
        if payload.auth_context is None:
            body["auth_context"] = auth_context.model_dump(exclude_none=True)
        return JSONResponse(status_code=200, content=body)

    if payload.auth_context is not None:
        resolved.auth_context = None

    return JSONResponse(status_code=200, content=resolved.model_dump(exclude_none=True))


@router.post("/decision-stream")
async def decide_gateway_stream(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    if not gateway_impl._allows_legacy_chat_cli_internal_route(request, decision):
        return gateway_impl._build_retired_internal_gateway_response()
    auth_context = await gateway_impl._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )

    if auth_context is None or not auth_context.access_allowed:
        return JSONResponse(status_code=200, content={"action": "fallback_plan"})

    try:
        resolved = await gateway_impl._build_gateway_stream_decision_response(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    if resolved is None:
        body: dict[str, object] = {"action": "fallback_plan"}
        if payload.auth_context is None:
            body["auth_context"] = auth_context.model_dump(exclude_none=True)
        return JSONResponse(status_code=200, content=body)

    if payload.auth_context is not None:
        resolved.auth_context = None

    return JSONResponse(status_code=200, content=resolved.model_dump(exclude_none=True))


@router.post("/execute-sync")
async def execute_gateway_sync(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    return await gateway_impl._execute_gateway_control_request(
        request=request,
        payload=payload,
        db=db,
        require_stream=False,
    )


@router.post("/execute-stream")
async def execute_gateway_stream(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    return await gateway_impl._execute_gateway_control_request(
        request=request,
        payload=payload,
        db=db,
        require_stream=True,
    )


@router.post("/plan-stream")
async def plan_gateway_stream(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    if not gateway_impl._allows_legacy_chat_cli_internal_route(request, decision):
        return gateway_impl._build_retired_internal_gateway_response()
    try:
        planned = await gateway_impl._build_gateway_stream_plan_response(
            request=request,
            payload=payload,
            db=db,
        )
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    if planned is None:
        response = await gateway_impl._execute_gateway_control_request(
            request=request,
            payload=payload,
            db=db,
            require_stream=True,
        )
        if gateway_impl._is_gateway_control_executed_response(response):
            return response
        return gateway_impl._build_proxy_public_fallback_response()

    if payload.auth_context is not None:
        planned.auth_context = None

    return JSONResponse(status_code=200, content=planned.model_dump(exclude_none=True))


@router.post("/plan-sync")
async def plan_gateway_sync(
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session = Depends(get_db),
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    if not gateway_impl._allows_legacy_chat_cli_internal_route(request, decision):
        return gateway_impl._build_retired_internal_gateway_response()
    try:
        planned = await gateway_impl._build_gateway_sync_plan_response(
            request=request,
            payload=payload,
            db=db,
        )
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    if planned is None:
        response = await gateway_impl._execute_gateway_control_request(
            request=request,
            payload=payload,
            db=db,
            require_stream=False,
        )
        if gateway_impl._is_gateway_control_executed_response(response):
            return response
        return gateway_impl._build_proxy_public_fallback_response()

    if payload.auth_context is not None:
        planned.auth_context = None

    return JSONResponse(status_code=200, content=planned.model_dump(exclude_none=True))


@router.post("/report-sync")
async def report_gateway_sync(
    request: Request,
    payload: GatewaySyncReportRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    if not gateway_impl._allows_legacy_chat_cli_report_route(request, payload.report_kind):
        return gateway_impl._build_retired_internal_gateway_response()
    payload_copy = payload.model_copy(deep=True)
    if gateway_impl._gateway_sync_report_requires_inline(payload_copy):
        db, cleanup = gateway_impl._resolve_gateway_background_db(getattr(request, "app", None))
        try:
            await gateway_impl._run_gateway_sync_report_background(payload_copy, db)
        finally:
            if cleanup is not None:
                cleanup()
        return JSONResponse(status_code=200, content={"ok": True})
    background_tasks.add_task(
        gateway_impl._run_gateway_sync_report_background_with_session,
        payload_copy,
        getattr(request, "app", None),
    )
    return JSONResponse(status_code=200, content={"ok": True})


@router.post("/finalize-sync")
async def finalize_gateway_sync(
    request: Request,
    payload: GatewaySyncReportRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    if not gateway_impl._allows_legacy_chat_cli_report_route(request, payload.report_kind):
        return gateway_impl._build_retired_internal_gateway_response()
    fast_response = await gateway_impl._maybe_build_gateway_core_sync_fast_success_response(payload)
    if fast_response is not None:
        if payload.report_kind in {
            "openai_chat_sync_finalize",
            "claude_chat_sync_finalize",
            "gemini_chat_sync_finalize",
        }:
            background_tasks.add_task(
                gateway_impl._run_gateway_chat_sync_finalize_background_with_session,
                payload.model_copy(deep=True),
            )
        elif payload.report_kind in {
            "openai_cli_sync_finalize",
            "openai_compact_sync_finalize",
            "claude_cli_sync_finalize",
            "gemini_cli_sync_finalize",
        }:
            background_tasks.add_task(
                gateway_impl._run_gateway_cli_sync_finalize_background_with_session,
                payload.model_copy(deep=True),
            )
        fast_response.headers[CONTROL_EXECUTED_HEADER] = "true"
        return fast_response

    db, cleanup = gateway_impl._resolve_gateway_finalize_db(request)
    try:
        response = await gateway_impl._finalize_gateway_sync_response(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    except Exception:
        if cleanup is not None:
            cleanup()
        raise
    if cleanup is not None:
        background_tasks.add_task(cleanup)
    response.headers[CONTROL_EXECUTED_HEADER] = "true"
    return response


@router.post("/report-stream")
async def report_gateway_stream(
    request: Request,
    payload: GatewayStreamReportRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    gateway_impl.ensure_loopback(request)
    retired = _ensure_legacy_internal_gateway_request(request)
    if retired is not None:
        return retired
    if not gateway_impl._allows_legacy_chat_cli_report_route(request, payload.report_kind):
        return gateway_impl._build_retired_internal_gateway_response()
    background_tasks.add_task(
        gateway_impl._run_gateway_stream_report_background_with_session,
        payload.model_copy(deep=True),
        getattr(request, "app", None),
    )
    return JSONResponse(status_code=200, content={"ok": True})
