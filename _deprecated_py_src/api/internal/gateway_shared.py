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

LEGACY_CHAT_CLI_CONTROL_EXECUTE_FALLBACK_HEADER = "x-aether-control-execute-fallback"
LEGACY_CHAT_CLI_INTERNAL_GATEWAY_HEADER = "x-aether-legacy-internal-gateway"
_LEGACY_CHAT_CLI_INTERNAL_GATEWAY_TRUE_VALUES = {"1", "true", "yes", "on"}
_LEGACY_CHAT_CLI_REPORT_KIND_PREFIXES = (
    "openai_chat_",
    "claude_chat_",
    "gemini_chat_",
    "openai_cli_",
    "openai_compact_",
    "claude_cli_",
    "gemini_cli_",
)


def _is_legacy_chat_cli_control_route(decision: GatewayRouteDecision) -> bool:
    return decision.route_class == "ai_public" and decision.route_kind in {
        "chat",
        "cli",
        "compact",
    }


def _request_allows_legacy_chat_cli_internal_gateway(request: Request) -> bool:
    for key, value in request.headers.items():
        normalized_key = str(key or "").strip().lower()
        if normalized_key not in {
            LEGACY_CHAT_CLI_INTERNAL_GATEWAY_HEADER,
            LEGACY_CHAT_CLI_CONTROL_EXECUTE_FALLBACK_HEADER,
        }:
            continue
        return (
            str(value or "").strip().lower()
            in _LEGACY_CHAT_CLI_INTERNAL_GATEWAY_TRUE_VALUES
        )
    return False


def _allows_legacy_chat_cli_internal_route(
    request: Request,
    decision: GatewayRouteDecision,
) -> bool:
    if not _is_legacy_chat_cli_control_route(decision):
        return True
    return _request_allows_legacy_chat_cli_internal_gateway(request)


def _is_legacy_chat_cli_report_kind(report_kind: str | None) -> bool:
    normalized = str(report_kind or "").strip().lower()
    if not normalized:
        return False
    return normalized.startswith(_LEGACY_CHAT_CLI_REPORT_KIND_PREFIXES)


def _allows_legacy_chat_cli_report_route(request: Request, report_kind: str | None) -> bool:
    if not _is_legacy_chat_cli_report_kind(report_kind):
        return True
    return _request_allows_legacy_chat_cli_internal_gateway(request)


def _build_retired_internal_gateway_response() -> JSONResponse:
    return JSONResponse(
        status_code=410,
        content={"detail": "legacy internal gateway route removed; use public proxy"},
    )


def _allows_legacy_chat_cli_control_execute(
    request: Request,
    payload: GatewayExecuteRequest,
    decision: GatewayRouteDecision,
) -> bool:
    if not _is_legacy_chat_cli_control_route(decision):
        return True
    if not _allows_legacy_chat_cli_internal_route(request, decision):
        return False

    for key, value in (payload.headers or {}).items():
        if str(key or "").strip().lower() != LEGACY_CHAT_CLI_CONTROL_EXECUTE_FALLBACK_HEADER:
            continue
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
    return False


async def _execute_gateway_control_request(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    require_stream: bool,
) -> Response:
    gateway_module.ensure_loopback(request)
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    if _is_gemini_files_route(decision):
        return await _execute_gateway_files_control_request(
            request=request,
            payload=payload,
            db=db,
            require_stream=require_stream,
        )
    if not _allows_legacy_chat_cli_control_execute(request, payload, decision):
        return gateway_module._build_retired_internal_gateway_response()

    adapter, path_params = gateway_module._resolve_gateway_sync_adapter(decision, payload.path)
    if adapter is None:
        return gateway_module._build_proxy_public_fallback_response()

    is_stream_request = _is_stream_request_payload(payload.body_json, path_params)
    if is_stream_request != require_stream:
        return gateway_module._build_proxy_public_fallback_response()

    auth_context = await gateway_module._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )
    if auth_context is None or not auth_context.access_allowed:
        return gateway_module._build_proxy_public_fallback_response()

    try:
        effective_request = (
            gateway_module._build_gateway_forward_request(request=request, payload=payload)
            if _is_video_route(decision)
            else request
        )
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(effective_request, db, user, api_key)
        context = gateway_module._build_gateway_request_context(
            request=effective_request,
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
        response = await adapter.handle(context)
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    response.headers[CONTROL_EXECUTED_HEADER] = "true"
    return response


def _is_gateway_control_executed_response(response: Response) -> bool:
    return str(response.headers.get(CONTROL_EXECUTED_HEADER) or "").strip().lower() == "true"


async def _execute_gateway_files_control_request(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    require_stream: bool,
) -> Response:
    if require_stream:
        return gateway_module._build_proxy_public_fallback_response()

    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    auth_context = await gateway_module._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )
    if auth_context is None or not auth_context.access_allowed:
        return gateway_module._build_proxy_public_fallback_response()

    try:
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        gateway_request = gateway_module._build_gateway_forward_request(
            request=request, payload=payload
        )
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
        response = await _dispatch_gateway_files_handler(gateway_request, payload)
    except HTTPException as exc:
        headers = dict(exc.headers or {})
        headers[CONTROL_EXECUTED_HEADER] = "true"
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=headers,
        )

    response.headers[CONTROL_EXECUTED_HEADER] = "true"
    return response


def _is_gemini_files_route(decision: GatewayRouteDecision) -> bool:
    return (
        decision.route_class == "ai_public"
        and decision.route_family == "gemini"
        and decision.route_kind == "files"
    )


def _is_video_route(decision: GatewayRouteDecision) -> bool:
    return decision.route_class == "ai_public" and decision.route_kind == "video"


async def _dispatch_gateway_files_handler(
    request: Request,
    payload: GatewayExecuteRequest,
) -> Response:
    from src.api.public.gemini_files import (
        delete_file,
        download_file,
        get_file,
        list_files,
        upload_file,
    )

    method = str(payload.method or "").strip().upper()
    path = str(payload.path or "").strip()

    if method == "POST" and path == "/upload/v1beta/files":
        return await upload_file(request)

    if method == "GET" and path == "/v1beta/files":
        query_params = gateway_module._parse_query_string(payload.query_string)
        page_size: int | None = None
        if query_params.get("pageSize") not in {None, ""}:
            try:
                page_size = int(str(query_params["pageSize"]))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid pageSize") from exc
        return await list_files(
            request,
            pageSize=page_size,
            pageToken=query_params.get("pageToken"),
        )

    download_match = _GEMINI_FILES_DOWNLOAD_ROUTE_RE.match(path)
    if method == "GET" and download_match:
        return await download_file(download_match.group("file_id"), request)

    resource_match = _GEMINI_FILES_RESOURCE_ROUTE_RE.match(path)
    if resource_match:
        file_name = resource_match.group("file_name")
        if method == "GET":
            return await get_file(file_name, request)
        if method == "DELETE":
            return await delete_file(file_name, request)

    return gateway_module._build_proxy_public_fallback_response()


def _extract_gemini_path_params(path: str) -> dict[str, Any]:
    match = _GEMINI_SYNC_ROUTE_RE.match(str(path or "").strip())
    if not match:
        return {}
    action = str(match.group("action") or "").strip()
    return {
        "model": str(match.group("model") or "").strip(),
        "stream": action == "streamGenerateContent",
    }


def _extract_openai_video_path_params(path: str) -> dict[str, Any]:
    normalized_path = str(path or "").strip()
    for route_re, extra in (
        (_OPENAI_VIDEO_CANCEL_ROUTE_RE, {"action": "cancel"}),
        (_OPENAI_VIDEO_REMIX_ROUTE_RE, {}),
        (_OPENAI_VIDEO_CONTENT_ROUTE_RE, {}),
        (_OPENAI_VIDEO_TASK_ROUTE_RE, {}),
    ):
        match = route_re.match(normalized_path)
        if match:
            params = {"task_id": str(match.group("task_id") or "").strip()}
            params.update(extra)
            return params
    return {}


def _extract_gemini_video_path_params(path: str) -> dict[str, Any]:
    normalized_path = str(path or "").strip()
    create_match = _GEMINI_VIDEO_CREATE_ROUTE_RE.match(normalized_path)
    if create_match:
        return {"model": str(create_match.group("model") or "").strip()}

    model_operation_match = _GEMINI_VIDEO_MODEL_OPERATION_ANY_RE.match(normalized_path)
    if model_operation_match:
        operation_name = (
            f"models/{str(model_operation_match.group('model') or '').strip()}/operations/"
            f"{str(model_operation_match.group('operation_id') or '').strip()}"
        )
        params = {"task_id": operation_name}
        if model_operation_match.group("cancel"):
            params["action"] = "cancel"
        return params

    if normalized_path == "/v1beta/operations":
        return {}

    if normalized_path.startswith("/v1beta/operations/"):
        operation_id = normalized_path[len("/v1beta/operations/") :].strip()
        if operation_id.endswith(":cancel"):
            return {"task_id": operation_id[: -len(":cancel")], "action": "cancel"}
        return {"task_id": operation_id}

    return {}


def _is_stream_request_payload(
    body_json: dict[str, Any],
    path_params: dict[str, Any] | None = None,
) -> bool:
    if bool((path_params or {}).get("stream")):
        return True
    return bool(body_json.get("stream"))


def _load_gateway_auth_models(
    db: Session,
    auth_context: GatewayAuthContext,
) -> tuple[User, ApiKey]:
    user = db.query(User).filter(User.id == auth_context.user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == auth_context.api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    if not user.is_active or user.is_deleted:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    if not api_key.is_active:
        raise HTTPException(status_code=401, detail="无效的API密钥")
    if api_key.is_locked and not api_key.is_standalone:
        raise HTTPException(status_code=403, detail="该密钥已被管理员锁定，请联系管理员")
    if str(api_key.user_id) != str(user.id):
        raise HTTPException(status_code=401, detail="无效的API密钥")
    return user, api_key


def _build_gateway_request_context(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    user: User,
    api_key: ApiKey,
    adapter: Any,
    path_params: dict[str, Any],
    balance_remaining: float | None,
) -> ApiRequestContext:
    request_id = str(
        payload.trace_id or getattr(request.state, "request_id", "") or uuid.uuid4().hex[:8]
    )
    request.state.request_id = request_id
    request.state.user_id = user.id
    request.state.api_key_id = api_key.id
    request.state.prefetched_balance_remaining = balance_remaining

    original_headers = {str(k): str(v) for k, v in (payload.headers or {}).items()}
    client_accept_encoding = str(original_headers.get("accept-encoding") or "").strip() or None
    raw_body = gateway_module._extract_gateway_raw_body(payload)
    return ApiRequestContext(
        request=request,
        db=db,
        user=user,
        api_key=api_key,
        request_id=request_id,
        start_time=time.time(),
        request_method=str(payload.method or request.method or "GET").upper(),
        request_path=str(payload.path or request.url.path or "/"),
        client_ip=request.client.host if request.client else "127.0.0.1",
        user_agent=str(original_headers.get("user-agent") or "unknown"),
        original_headers=original_headers,
        query_params=gateway_module._parse_query_string(payload.query_string),
        request_content_type=str(original_headers.get("content-type") or "").strip() or None,
        raw_body=raw_body,
        json_body=(dict(payload.body_json) if payload.body_json else None),
        balance_remaining=balance_remaining,
        mode=getattr(getattr(adapter, "mode", None), "value", "standard"),
        api_format_hint=(
            adapter.allowed_api_formats[0]
            if getattr(adapter, "allowed_api_formats", None)
            else None
        ),
        path_params=dict(path_params or {}),
        client_content_encoding=normalize_content_encoding(
            original_headers.get("content-encoding")
        ),
        client_accept_encoding=client_accept_encoding,
    )


def _build_gateway_forward_request(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
) -> Request:
    body = gateway_module._decode_gateway_body(payload)
    header_items = [
        (str(key).encode("latin-1"), str(value).encode("latin-1"))
        for key, value in (payload.headers or {}).items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": str(payload.method or "GET").upper(),
        "scheme": "http",
        "path": str(payload.path or "/"),
        "raw_path": str(payload.path or "/").encode("utf-8"),
        "query_string": str(payload.query_string or "").encode("utf-8"),
        "headers": header_items,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 80),
        "app": request.app,
        "state": {"request_id": str(payload.trace_id or uuid.uuid4().hex[:8])},
    }

    received = False

    async def receive() -> dict[str, object]:
        nonlocal received
        if received:
            return {"type": "http.request", "body": b"", "more_body": False}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _build_proxy_public_fallback_response() -> JSONResponse:
    return JSONResponse(
        status_code=409,
        content={"action": CONTROL_ACTION_PROXY_PUBLIC},
        headers={CONTROL_ACTION_HEADER: CONTROL_ACTION_PROXY_PUBLIC},
    )


def _stream_executor_requires_python_rewrite(
    *,
    envelope: Any = None,
    needs_conversion: bool,
    provider_api_format: str | None = None,
    client_api_format: str | None = None,
) -> bool:
    provider_api_format = str(provider_api_format or "").strip().lower()
    client_api_format = str(client_api_format or "").strip().lower()
    if needs_conversion:
        if (
            envelope is None
            and (
                (
                    provider_api_format in {"claude:chat", "gemini:chat"}
                    and client_api_format == "openai:chat"
                )
                or (
                    provider_api_format in {"claude:cli", "gemini:cli"}
                    and client_api_format in {"openai:cli", "openai:compact"}
                )
            )
        ) or (
            str(getattr(envelope, "name", "") or "").strip().lower() == "antigravity:v1internal"
            and (
                (provider_api_format == "gemini:chat" and client_api_format == "openai:chat")
                or (
                    provider_api_format == "gemini:cli"
                    and client_api_format in {"openai:cli", "openai:compact"}
                )
            )
        ):
            return False
        return True
    if envelope is None:
        return False
    try:
        requires_rewrite = bool(envelope.force_stream_rewrite())
    except Exception:
        return True
    if not requires_rewrite:
        return False

    envelope_name = str(getattr(envelope, "name", "") or "").strip().lower()
    if (
        envelope_name == "antigravity:v1internal"
        and provider_api_format == client_api_format
        and provider_api_format in {"gemini:chat", "gemini:cli"}
    ):
        return False
    if (
        envelope_name == "kiro:generateassistantresponse"
        and provider_api_format == "claude:cli"
        and client_api_format == "claude:cli"
    ):
        return False
    return True


def _serialize_gateway_sync_proxy(proxy: Any) -> dict[str, Any] | None:
    if proxy is None:
        return None

    raw = asdict(proxy)
    return {key: value for key, value in raw.items() if value is not None}


def _serialize_gateway_sync_timeouts(timeouts: Any) -> dict[str, Any] | None:
    if timeouts is None:
        return None

    raw = asdict(timeouts)
    return {key: value for key, value in raw.items() if value is not None}


def _extract_gateway_upstream_auth(
    provider_request_headers: dict[str, str],
    *,
    provider_api_format: str,
    key: Any,
) -> tuple[str, str]:
    normalized_headers = {
        str(header_name).strip().lower(): str(header_value).strip()
        for header_name, header_value in (provider_request_headers or {}).items()
        if str(header_name).strip() and str(header_value).strip()
    }
    for header_name in ("authorization", "x-api-key", "x-goog-api-key"):
        header_value = normalized_headers.get(header_name)
        if header_value:
            return header_name, header_value

    auth_header, auth_type = get_auth_config_for_endpoint(provider_api_format)
    decrypted_key = crypto_service.decrypt(key.api_key)
    auth_value = f"Bearer {decrypted_key}" if auth_type == "bearer" else decrypted_key
    return str(auth_header or "").strip() or "authorization", auth_value
