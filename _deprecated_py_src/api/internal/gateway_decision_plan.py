from __future__ import annotations

from typing import Any
from urllib.parse import parse_qsl

from fastapi import Request
from sqlalchemy.orm import Session

from src.core.api_format.headers import extract_client_api_key_for_endpoint_with_query
from src.services.auth.service import AuthService

from .gateway_contract import (
    _GEMINI_FILES_DOWNLOAD_ROUTE_RE,
    _GEMINI_FILES_RESOURCE_ROUTE_RE,
    _GEMINI_MODEL_OPERATION_CANCEL_RE,
    _GEMINI_OPERATION_CANCEL_RE,
    _GEMINI_VIDEO_CREATE_ROUTE_RE,
    _OPENAI_VIDEO_CANCEL_ROUTE_RE,
    _OPENAI_VIDEO_CONTENT_ROUTE_RE,
    _OPENAI_VIDEO_REMIX_ROUTE_RE,
    _OPENAI_VIDEO_TASK_ROUTE_RE,
    GatewayAuthContext,
    GatewayExecuteRequest,
    GatewayExecutionDecisionResponse,
    GatewayExecutionPlanResponse,
    GatewayResolveRequest,
    GatewayRouteDecision,
    classify_gateway_route,
)


def _gateway_module() -> Any:
    from . import gateway as gateway_module

    return gateway_module


def _parse_query_string(query_string: str | None) -> dict[str, str]:
    return {
        str(key): str(value) for key, value in parse_qsl(query_string or "", keep_blank_values=True)
    }


async def _resolve_auth_context(
    payload: GatewayResolveRequest,
    decision: GatewayRouteDecision,
) -> dict[str, Any] | None:
    gateway_module = _gateway_module()
    return await gateway_module._resolve_auth_context_signature(
        headers=payload.headers,
        query_string=payload.query_string,
        auth_endpoint_signature=str(decision.auth_endpoint_signature or ""),
    )


async def _resolve_auth_context_signature(
    *,
    headers: dict[str, str] | None,
    query_string: str | None,
    auth_endpoint_signature: str,
) -> dict[str, Any] | None:
    normalized_signature = str(auth_endpoint_signature or "").strip().lower()
    if not normalized_signature:
        return None

    client_api_key = extract_client_api_key_for_endpoint_with_query(
        headers or {},
        _parse_query_string(query_string),
        normalized_signature,
    )
    if not client_api_key:
        return None

    auth_result = await AuthService.authenticate_api_key_threadsafe(client_api_key)
    if not auth_result or not auth_result.user or not auth_result.api_key:
        return None

    return GatewayAuthContext(
        user_id=str(auth_result.user.id),
        api_key_id=str(auth_result.api_key.id),
        balance_remaining=auth_result.balance_remaining,
        access_allowed=bool(auth_result.access_allowed),
    ).model_dump(exclude_none=True)


async def _resolve_gateway_execute_auth_context(
    *,
    payload: GatewayExecuteRequest,
    decision: GatewayRouteDecision,
) -> GatewayAuthContext | None:
    gateway_module = _gateway_module()
    if payload.auth_context is not None:
        return payload.auth_context

    resolved = await gateway_module._resolve_auth_context_signature(
        headers=payload.headers,
        query_string=payload.query_string,
        auth_endpoint_signature=str(decision.auth_endpoint_signature or ""),
    )
    if not resolved:
        return None
    return GatewayAuthContext.model_validate(resolved)


def _resolve_gateway_sync_adapter(
    decision: GatewayRouteDecision,
    path: str,
) -> tuple[Any | None, dict[str, Any]]:
    gateway_module = _gateway_module()
    if decision.route_class != "ai_public":
        return None, {}

    family = str(decision.route_family or "").strip().lower()
    kind = str(decision.route_kind or "").strip().lower()
    if family == "openai" and kind == "chat":
        from src.api.handlers.openai import OpenAIChatAdapter

        return OpenAIChatAdapter(), {}
    if family == "openai" and kind == "cli":
        from src.api.handlers.openai_cli import OpenAICliAdapter

        return OpenAICliAdapter(), {}
    if family == "openai" and kind == "compact":
        from src.api.handlers.openai_cli import OpenAICompactAdapter

        return OpenAICompactAdapter(), {}
    if family == "claude" and kind == "chat":
        from src.api.handlers.claude.adapter import ClaudeChatAdapter

        return ClaudeChatAdapter(), {}
    if family == "claude" and kind == "cli":
        from src.api.handlers.claude_cli import ClaudeCliAdapter

        return ClaudeCliAdapter(), {}
    if family == "gemini" and kind == "chat":
        from src.api.handlers.gemini.adapter import GeminiChatAdapter

        return GeminiChatAdapter(), gateway_module._extract_gemini_path_params(path)
    if family == "gemini" and kind == "cli":
        from src.api.handlers.gemini_cli import GeminiCliAdapter

        return GeminiCliAdapter(), gateway_module._extract_gemini_path_params(path)
    if family == "openai" and kind == "video":
        from src.api.handlers.openai.video_adapter import OpenAIVideoAdapter

        return OpenAIVideoAdapter(), gateway_module._extract_openai_video_path_params(path)
    if family == "gemini" and kind == "video":
        from src.api.handlers.gemini.video_adapter import GeminiVeoAdapter

        return GeminiVeoAdapter(), gateway_module._extract_gemini_video_path_params(path)
    return None, {}


async def _build_gateway_stream_plan_response(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
) -> GatewayExecutionPlanResponse | None:
    gateway_module = _gateway_module()
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    auth_context = await gateway_module._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )
    if auth_context is None or not auth_context.access_allowed:
        return None

    if decision.route_family == "openai" and decision.route_kind == "chat":
        planned = await gateway_module._build_openai_chat_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "claude" and decision.route_kind == "chat":
        planned = await gateway_module._build_claude_chat_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "gemini" and decision.route_kind == "chat":
        planned = await gateway_module._build_gemini_chat_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "openai" and decision.route_kind in {"cli", "compact"}:
        planned = await gateway_module._build_openai_cli_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "claude" and decision.route_kind == "cli":
        planned = await gateway_module._build_claude_cli_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "gemini" and decision.route_kind == "cli":
        planned = await gateway_module._build_gemini_cli_stream_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "gemini" and decision.route_kind == "files":
        download_match = _GEMINI_FILES_DOWNLOAD_ROUTE_RE.match(str(payload.path or "").strip())
        if not download_match or str(payload.method or "").strip().upper() != "GET":
            return None

        gateway_request = gateway_module._build_gateway_forward_request(
            request=request, payload=payload
        )
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
        plan = await gateway_module._build_gemini_files_download_stream_plan(
            request=gateway_request,
            db=db,
            user=user,
            user_api_key=api_key,
            file_id=download_match.group("file_id"),
        )
        return GatewayExecutionPlanResponse(
            action="executor_stream",
            plan_kind="gemini_files_download",
            plan=plan,
            auth_context=auth_context,
        )

    if decision.route_family == "openai" and decision.route_kind == "video":
        content_match = _OPENAI_VIDEO_CONTENT_ROUTE_RE.match(str(payload.path or "").strip())
        if not content_match or str(payload.method or "").strip().upper() != "GET":
            return None

        gateway_request = gateway_module._build_gateway_forward_request(
            request=request, payload=payload
        )
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
        plan = await gateway_module._build_openai_video_content_stream_plan(
            request=gateway_request,
            db=db,
            user=user,
            user_api_key=api_key,
            task_id=content_match.group("task_id"),
        )
        return GatewayExecutionPlanResponse(
            action="executor_stream",
            plan_kind="openai_video_content",
            plan=plan,
            auth_context=auth_context,
        )

    return None


async def _build_gateway_stream_decision_response(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    gateway_module = _gateway_module()
    if decision.route_family == "openai" and decision.route_kind == "chat":
        return await gateway_module._build_openai_chat_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "claude" and decision.route_kind == "chat":
        return await gateway_module._build_claude_chat_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "gemini" and decision.route_kind == "chat":
        return await gateway_module._build_gemini_chat_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "openai" and decision.route_kind in {"cli", "compact"}:
        return await gateway_module._build_openai_cli_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "claude" and decision.route_kind == "cli":
        return await gateway_module._build_claude_cli_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "gemini" and decision.route_kind == "cli":
        return await gateway_module._build_gemini_cli_stream_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "gemini" and decision.route_kind == "files":
        normalized_path = str(payload.path or "").strip()
        download_match = _GEMINI_FILES_DOWNLOAD_ROUTE_RE.match(normalized_path)
        if str(payload.method or "").strip().upper() == "GET" and download_match:
            gateway_request = gateway_module._build_gateway_forward_request(
                request=request, payload=payload
            )
            user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
            return await gateway_module._build_gemini_files_download_stream_decision(
                request=gateway_request,
                db=db,
                user=user,
                user_api_key=api_key,
                file_id=str(download_match.group("file_id") or "").strip(),
            )

    if decision.route_family == "openai" and decision.route_kind == "video":
        normalized_path = str(payload.path or "").strip()
        content_match = _OPENAI_VIDEO_CONTENT_ROUTE_RE.match(normalized_path)
        if str(payload.method or "").strip().upper() == "GET" and content_match:
            gateway_request = gateway_module._build_gateway_forward_request(
                request=request, payload=payload
            )
            user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
            return await gateway_module._build_openai_video_content_stream_decision(
                request=gateway_request,
                db=db,
                user=user,
                user_api_key=api_key,
                task_id=str(content_match.group("task_id") or "").strip(),
            )

    return None


async def _build_gateway_sync_plan_response(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
) -> GatewayExecutionPlanResponse | None:
    gateway_module = _gateway_module()
    decision = classify_gateway_route(payload.method, payload.path, payload.headers)
    auth_context = await gateway_module._resolve_gateway_execute_auth_context(
        payload=payload,
        decision=decision,
    )
    if auth_context is None or not auth_context.access_allowed:
        return None

    if decision.route_family == "openai" and decision.route_kind == "chat":
        planned = await gateway_module._build_openai_chat_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "claude" and decision.route_kind == "chat":
        planned = await gateway_module._build_claude_chat_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "gemini" and decision.route_kind == "chat":
        planned = await gateway_module._build_gemini_chat_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "openai" and decision.route_kind in {"cli", "compact"}:
        planned = await gateway_module._build_openai_cli_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "claude" and decision.route_kind == "cli":
        planned = await gateway_module._build_claude_cli_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "gemini" and decision.route_kind == "cli":
        planned = await gateway_module._build_gemini_cli_sync_plan(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )
        if planned is not None:
            planned.auth_context = auth_context
            return planned

    if decision.route_family == "openai" and decision.route_kind == "video":
        normalized_path = str(payload.path or "").strip()
        normalized_method = str(payload.method or "").strip().upper()
        if normalized_method == "POST" and normalized_path == "/v1/videos":
            planned = await gateway_module._build_openai_video_create_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                decision=decision,
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned
        remix_match = _OPENAI_VIDEO_REMIX_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and remix_match:
            planned = await gateway_module._build_openai_video_remix_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(remix_match.group("task_id") or "").strip(),
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned
        cancel_match = _OPENAI_VIDEO_CANCEL_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and cancel_match:
            planned = await gateway_module._build_openai_video_cancel_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(cancel_match.group("task_id") or "").strip(),
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned
        task_match = _OPENAI_VIDEO_TASK_ROUTE_RE.match(normalized_path)
        if normalized_method == "DELETE" and task_match:
            planned = await gateway_module._build_openai_video_delete_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(task_match.group("task_id") or "").strip(),
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned

    if decision.route_family == "gemini" and decision.route_kind == "video":
        normalized_path = str(payload.path or "").strip()
        normalized_method = str(payload.method or "").strip().upper()
        create_match = _GEMINI_VIDEO_CREATE_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and create_match:
            planned = await gateway_module._build_gemini_video_create_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                model=str(create_match.group("model") or "").strip(),
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned
        if normalized_method == "POST" and (
            _GEMINI_MODEL_OPERATION_CANCEL_RE.match(normalized_path)
            or _GEMINI_OPERATION_CANCEL_RE.match(normalized_path)
        ):
            task_id = str(
                gateway_module._extract_gemini_video_path_params(normalized_path).get("task_id")
                or ""
            )
            planned = await gateway_module._build_gemini_video_cancel_sync_plan(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=task_id,
            )
            if planned is not None:
                planned.auth_context = auth_context
                return planned

    if decision.route_family == "gemini" and decision.route_kind == "files":
        resource_match = _GEMINI_FILES_RESOURCE_ROUTE_RE.match(str(payload.path or "").strip())
        download_match = _GEMINI_FILES_DOWNLOAD_ROUTE_RE.match(str(payload.path or "").strip())
        method = str(payload.method or "").strip().upper()

        if method == "POST" and str(payload.path or "").strip() == "/upload/v1beta/files":
            gateway_request = gateway_module._build_gateway_forward_request(
                request=request, payload=payload
            )
            user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
            pipeline = gateway_module.get_pipeline()
            await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
            plan, report_context = await gateway_module._build_gemini_files_proxy_sync_plan(
                request=gateway_request,
                db=db,
                method="POST",
                upstream_path="/v1beta/files",
                is_upload=True,
            )
            return GatewayExecutionPlanResponse(
                action="executor_sync",
                plan_kind="gemini_files_upload",
                plan=plan,
                report_kind="gemini_files_store_mapping",
                report_context=report_context,
                auth_context=auth_context,
            )

        if method == "GET" and str(payload.path or "").strip() == "/v1beta/files":
            gateway_request = gateway_module._build_gateway_forward_request(
                request=request, payload=payload
            )
            user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
            pipeline = gateway_module.get_pipeline()
            await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
            plan, report_context = await gateway_module._build_gemini_files_proxy_sync_plan(
                request=gateway_request,
                db=db,
                method="GET",
                upstream_path="/v1beta/files",
            )
            return GatewayExecutionPlanResponse(
                action="executor_sync",
                plan_kind="gemini_files_list",
                plan=plan,
                report_kind="gemini_files_store_mapping",
                report_context=report_context,
                auth_context=auth_context,
            )

        if (
            not resource_match
            or download_match
            or str(payload.path or "").strip() == "/v1beta/files"
        ):
            return None

        gateway_request = gateway_module._build_gateway_forward_request(
            request=request, payload=payload
        )
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)
        file_name = resource_match.group("file_name")
        if method == "GET":
            plan, report_context = await gateway_module._build_gemini_files_get_sync_plan(
                request=gateway_request,
                db=db,
                file_name=file_name,
            )
            return GatewayExecutionPlanResponse(
                action="executor_sync",
                plan_kind="gemini_files_get",
                plan=plan,
                report_kind="gemini_files_store_mapping",
                report_context=report_context,
                auth_context=auth_context,
            )
        if method == "DELETE":
            normalized_file_name = (
                file_name if str(file_name or "").startswith("files/") else f"files/{file_name}"
            )
            plan, _report_context = await gateway_module._build_gemini_files_proxy_sync_plan(
                request=gateway_request,
                db=db,
                method="DELETE",
                upstream_path=f"/v1beta/{normalized_file_name}",
            )
            return GatewayExecutionPlanResponse(
                action="executor_sync",
                plan_kind="gemini_files_delete",
                plan=plan,
                report_kind="gemini_files_delete_mapping",
                report_context={"file_name": normalized_file_name},
                auth_context=auth_context,
            )

    return None


async def _build_gateway_sync_decision_response(
    *,
    request: Request,
    payload: GatewayExecuteRequest,
    db: Session,
    auth_context: GatewayAuthContext,
    decision: GatewayRouteDecision,
) -> GatewayExecutionDecisionResponse | None:
    gateway_module = _gateway_module()
    if decision.route_family == "openai" and decision.route_kind == "chat":
        return await gateway_module._build_openai_chat_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "openai" and decision.route_kind in {"cli", "compact"}:
        return await gateway_module._build_openai_cli_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "claude" and decision.route_kind == "chat":
        return await gateway_module._build_claude_chat_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "claude" and decision.route_kind == "cli":
        return await gateway_module._build_claude_cli_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "gemini" and decision.route_kind == "chat":
        return await gateway_module._build_gemini_chat_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "gemini" and decision.route_kind == "cli":
        return await gateway_module._build_gemini_cli_sync_decision(
            request=request,
            payload=payload,
            db=db,
            auth_context=auth_context,
            decision=decision,
        )

    if decision.route_family == "openai" and decision.route_kind == "video":
        normalized_path = str(payload.path or "").strip()
        normalized_method = str(payload.method or "").strip().upper()
        if normalized_method == "POST" and normalized_path == "/v1/videos":
            return await gateway_module._build_openai_video_create_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                decision=decision,
            )
        remix_match = _OPENAI_VIDEO_REMIX_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and remix_match:
            return await gateway_module._build_openai_video_remix_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(remix_match.group("task_id") or "").strip(),
            )
        cancel_match = _OPENAI_VIDEO_CANCEL_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and cancel_match:
            return await gateway_module._build_openai_video_cancel_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(cancel_match.group("task_id") or "").strip(),
            )
        task_match = _OPENAI_VIDEO_TASK_ROUTE_RE.match(normalized_path)
        if normalized_method == "DELETE" and task_match:
            return await gateway_module._build_openai_video_delete_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=str(task_match.group("task_id") or "").strip(),
            )

    if decision.route_family == "gemini" and decision.route_kind == "video":
        normalized_path = str(payload.path or "").strip()
        normalized_method = str(payload.method or "").strip().upper()
        create_match = _GEMINI_VIDEO_CREATE_ROUTE_RE.match(normalized_path)
        if normalized_method == "POST" and create_match:
            return await gateway_module._build_gemini_video_create_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                model=str(create_match.group("model") or "").strip(),
            )
        if normalized_method == "POST" and (
            _GEMINI_MODEL_OPERATION_CANCEL_RE.match(normalized_path)
            or _GEMINI_OPERATION_CANCEL_RE.match(normalized_path)
        ):
            task_id = str(
                gateway_module._extract_gemini_video_path_params(normalized_path).get("task_id")
                or ""
            )
            return await gateway_module._build_gemini_video_cancel_sync_decision(
                request=request,
                payload=payload,
                db=db,
                auth_context=auth_context,
                task_id=task_id,
            )

    if decision.route_family == "gemini" and decision.route_kind == "files":
        normalized_path = str(payload.path or "").strip()
        method = str(payload.method or "").strip().upper()
        resource_match = _GEMINI_FILES_RESOURCE_ROUTE_RE.match(normalized_path)
        download_match = _GEMINI_FILES_DOWNLOAD_ROUTE_RE.match(normalized_path)

        gateway_request = gateway_module._build_gateway_forward_request(
            request=request, payload=payload
        )
        user, api_key = gateway_module._load_gateway_auth_models(db, auth_context)
        pipeline = gateway_module.get_pipeline()
        await pipeline._check_user_rate_limit(gateway_request, db, user, api_key)

        if method == "GET" and normalized_path == "/v1beta/files":
            return await gateway_module._build_gemini_files_proxy_sync_decision(
                request=gateway_request,
                db=db,
                method="GET",
                upstream_path="/v1beta/files",
                decision_kind="gemini_files_list",
                report_kind="gemini_files_store_mapping",
            )

        if (
            method == "GET"
            and resource_match
            and not download_match
            and normalized_path != "/v1beta/files"
        ):
            file_name = str(resource_match.group("file_name") or "").strip()
            return await gateway_module._build_gemini_files_proxy_sync_decision(
                request=gateway_request,
                db=db,
                method="GET",
                upstream_path=f"/v1beta/{file_name if file_name.startswith('files/') else f'files/{file_name}'}",
                decision_kind="gemini_files_get",
                report_kind="gemini_files_store_mapping",
            )

        if (
            method == "DELETE"
            and resource_match
            and not download_match
            and normalized_path != "/v1beta/files"
        ):
            file_name = str(resource_match.group("file_name") or "").strip()
            normalized_file_name = (
                file_name if file_name.startswith("files/") else f"files/{file_name}"
            )
            return await gateway_module._build_gemini_files_proxy_sync_decision(
                request=gateway_request,
                db=db,
                method="DELETE",
                upstream_path=f"/v1beta/{normalized_file_name}",
                decision_kind="gemini_files_delete",
                report_kind="gemini_files_delete_mapping",
                report_context={"file_name": normalized_file_name},
            )

    return None
