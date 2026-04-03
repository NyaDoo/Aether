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
from src.models.database import ApiKey, RequestCandidate, User, VideoTask
from src.services.auth.service import AuthService
from src.utils.async_utils import safe_create_task

from .common import ensure_loopback
from .gateway_chat import (
    _build_chat_stream_decision,
    _build_chat_stream_plan,
    _build_chat_sync_decision,
    _build_chat_sync_plan,
    _build_claude_chat_stream_decision,
    _build_claude_chat_stream_plan,
    _build_claude_chat_sync_decision,
    _build_claude_chat_sync_plan,
    _build_gemini_chat_stream_decision,
    _build_gemini_chat_stream_plan,
    _build_gemini_chat_sync_decision,
    _build_gemini_chat_sync_plan,
    _build_openai_chat_stream_decision,
    _build_openai_chat_stream_plan,
    _build_openai_chat_sync_decision,
    _build_openai_chat_sync_plan,
)
from .gateway_cli import (
    _build_claude_cli_stream_decision,
    _build_claude_cli_stream_plan,
    _build_claude_cli_sync_decision,
    _build_claude_cli_sync_plan,
    _build_cli_stream_decision,
    _build_cli_stream_plan,
    _build_cli_sync_decision,
    _build_cli_sync_plan,
    _build_gemini_cli_stream_decision,
    _build_gemini_cli_stream_plan,
    _build_gemini_cli_sync_decision,
    _build_gemini_cli_sync_plan,
    _build_openai_cli_stream_decision,
    _build_openai_cli_stream_plan,
    _build_openai_cli_sync_decision,
    _build_openai_cli_sync_plan,
)
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
from .gateway_decision_plan import (
    _build_gateway_stream_decision_response,
    _build_gateway_stream_plan_response,
    _build_gateway_sync_decision_response,
    _build_gateway_sync_plan_response,
    _parse_query_string,
    _resolve_auth_context,
    _resolve_auth_context_signature,
    _resolve_gateway_execute_auth_context,
    _resolve_gateway_sync_adapter,
)
from .gateway_files import (
    _build_gemini_files_download_stream_decision,
    _build_gemini_files_download_stream_plan,
    _build_gemini_files_get_sync_plan,
    _build_gemini_files_proxy_sync_decision,
    _build_gemini_files_proxy_sync_plan,
)
from .gateway_finalize import (
    _build_gateway_embedded_error_payload,
    _build_gateway_sync_error_payload,
    _extract_gateway_report_body_bytes,
    _extract_gateway_sync_error_message,
    _finalize_gateway_chat_sync,
    _finalize_gateway_cli_sync,
    _finalize_gateway_sync_response,
    _resolve_gateway_finalize_db,
    _resolve_gateway_sync_error_status_code,
    _run_gateway_chat_sync_finalize_background,
    _run_gateway_chat_sync_finalize_background_with_session,
    _run_gateway_cli_sync_finalize_background,
    _run_gateway_cli_sync_finalize_background_with_session,
)
from .gateway_reporting import (
    _apply_gateway_stream_report,
    _apply_gateway_sync_report,
    _build_gateway_sync_telemetry_writer,
    _build_gateway_usage_metadata,
    _close_gateway_session,
    _dispatch_gateway_sync_telemetry,
    _ensure_gateway_request_candidate,
    _gateway_sync_report_requires_inline,
    _mark_gateway_sync_candidate_terminal_state,
    _postprocess_gateway_report_provider_response,
    _record_gateway_chat_sync_failure,
    _record_gateway_cli_sync_failure,
    _record_gateway_direct_candidate_graph,
    _record_gateway_gemini_video_cancel_sync_success,
    _record_gateway_gemini_video_create_sync_success,
    _record_gateway_openai_chat_stream_success,
    _record_gateway_openai_chat_sync_success,
    _record_gateway_openai_video_cancel_sync_success,
    _record_gateway_openai_video_create_sync_success,
    _record_gateway_openai_video_delete_sync_success,
    _record_gateway_openai_video_remix_sync_success,
    _record_gateway_passthrough_chat_stream_success,
    _record_gateway_passthrough_chat_sync_success,
    _record_gateway_passthrough_cli_stream_success,
    _record_gateway_passthrough_cli_sync_success,
    _record_gateway_video_sync_failure,
    _resolve_gateway_background_db,
    _resolve_gateway_failure_adapter,
    _run_gateway_stream_report_background,
    _run_gateway_stream_report_background_with_session,
    _run_gateway_sync_report_background,
    _run_gateway_sync_report_background_with_session,
    _run_gateway_video_finalize_submitted_background,
    _schedule_gateway_sync_telemetry,
)
from .gateway_shared import (
    LEGACY_CHAT_CLI_INTERNAL_GATEWAY_HEADER,
    _allows_legacy_chat_cli_internal_route,
    _allows_legacy_chat_cli_report_route,
    _build_retired_internal_gateway_response,
    _build_gateway_forward_request,
    _build_gateway_request_context,
    _build_proxy_public_fallback_response,
    _dispatch_gateway_files_handler,
    _execute_gateway_control_request,
    _execute_gateway_files_control_request,
    _extract_gateway_upstream_auth,
    _extract_gemini_path_params,
    _extract_gemini_video_path_params,
    _extract_openai_video_path_params,
    _is_gateway_control_executed_response,
    _is_gemini_files_route,
    _is_stream_request_payload,
    _is_video_route,
    _request_allows_legacy_chat_cli_internal_gateway,
    _load_gateway_auth_models,
    _serialize_gateway_sync_proxy,
    _serialize_gateway_sync_timeouts,
    _stream_executor_requires_python_rewrite,
)
from .gateway_video import (
    _build_gemini_video_cancel_sync_decision,
    _build_gemini_video_cancel_sync_plan,
    _build_gemini_video_create_sync_decision,
    _build_gemini_video_create_sync_plan,
    _build_openai_video_cancel_sync_decision,
    _build_openai_video_cancel_sync_plan,
    _build_openai_video_content_stream_decision,
    _build_openai_video_content_stream_plan,
    _build_openai_video_create_sync_decision,
    _build_openai_video_create_sync_plan,
    _build_openai_video_delete_sync_decision,
    _build_openai_video_delete_sync_plan,
    _build_openai_video_remix_sync_decision,
    _build_openai_video_remix_sync_plan,
)


def _is_streaming_sync_payload(
    body_json: dict[str, Any],
    path_params: dict[str, Any] | None = None,
) -> bool:
    return _is_stream_request_payload(body_json, path_params)


def _decode_gateway_body(payload: GatewayExecuteRequest) -> bytes:
    if not payload.body_base64:
        return b""
    try:
        return base64.b64decode(payload.body_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid gateway body payload") from exc


def _extract_gateway_raw_body(payload: GatewayExecuteRequest) -> bytes:
    if payload.body_base64:
        return _decode_gateway_body(payload)
    if payload.body_json:
        return JSONResponse(content=payload.body_json).body
    return b""


async def _select_gateway_direct_candidate(
    *,
    db: Session,
    redis_client: Any | None,
    api_format: str,
    model_name: str,
    user_api_key: ApiKey,
    request_id: str,
    is_stream: bool,
    capability_requirements: dict[str, bool] | None,
    preferred_key_ids: list[str] | None,
    request_body: dict[str, Any] | None,
) -> Any | None:
    from src.services.candidate.resolver import CandidateResolver
    from src.services.scheduling.aware_scheduler import (
        CacheAwareScheduler,
        get_cache_aware_scheduler,
    )
    from src.services.system.config import SystemConfigService
    from src.services.task.execute.pool import TaskPoolOperationsService

    priority_mode = SystemConfigService.get_config(
        db,
        "provider_priority_mode",
        CacheAwareScheduler.PRIORITY_MODE_PROVIDER,
    )
    scheduling_mode = SystemConfigService.get_config(
        db,
        "scheduling_mode",
        CacheAwareScheduler.SCHEDULING_MODE_CACHE_AFFINITY,
    )
    cache_scheduler = await get_cache_aware_scheduler(
        redis_client,
        priority_mode=priority_mode,
        scheduling_mode=scheduling_mode,
    )
    await cache_scheduler._ensure_initialized()

    candidate_resolver = CandidateResolver(
        db=db,
        cache_scheduler=cache_scheduler,
    )
    candidates, _global_model_id = await candidate_resolver.fetch_candidates(
        api_format=api_format,
        model_name=model_name,
        affinity_key=str(user_api_key.id),
        user_api_key=user_api_key,
        request_id=request_id,
        is_stream=is_stream,
        capability_requirements=capability_requirements,
        preferred_key_ids=preferred_key_ids,
        request_body=request_body,
    )
    candidates, _pool_traces = await TaskPoolOperationsService().apply_pool_reorder(
        candidates,
        request_body=request_body,
    )
    if candidates:
        _record_gateway_direct_candidate_graph(
            db=db,
            candidate_resolver=candidate_resolver,
            candidates=candidates,
            request_id=request_id,
            user_api_key=user_api_key,
            required_capabilities=capability_requirements,
            selected_candidate_index=0,
        )
    return candidates[0] if candidates else None


def _gateway_sync_report_context_flag(context: dict[str, Any], key: str) -> bool | None:
    value = context.get(key)
    if isinstance(value, bool):
        return value
    return None


def _gateway_report_context_envelope_name(envelope: Any) -> str | None:
    name = getattr(envelope, "name", None)
    if isinstance(name, str):
        normalized = name.strip()
        if normalized:
            return normalized
    return None


async def _aggregate_gateway_sync_response_bytes(
    *,
    body_bytes: bytes,
    provider_api_format: str,
    client_api_format: str,
    provider_name: str,
    model: str,
    request_id: str,
) -> dict[str, Any]:
    from src.api.handlers.base.parsers import get_parser_for_format
    from src.api.handlers.base.upstream_stream_bridge import (
        aggregate_upstream_stream_to_internal_response,
    )
    from src.api.handlers.base.utils import get_format_converter_registry

    provider_parser = get_parser_for_format(provider_api_format) if provider_api_format else None

    async def _byte_iter() -> Any:
        yield body_bytes

    internal_resp = await aggregate_upstream_stream_to_internal_response(
        _byte_iter(),
        provider_api_format=provider_api_format,
        provider_name=provider_name,
        model=model,
        request_id=request_id,
        envelope=None,
        provider_parser=provider_parser,
    )

    registry = get_format_converter_registry()
    tgt_norm = registry.get_normalizer(client_api_format) if client_api_format else None
    if tgt_norm is None:
        raise RuntimeError(f"未注册 Normalizer: {client_api_format}")

    response_json = tgt_norm.response_from_internal(
        internal_resp,
        requested_model=model,
    )
    return response_json if isinstance(response_json, dict) else {}


async def _maybe_build_gateway_core_sync_fast_success_response(
    payload: GatewaySyncReportRequest,
) -> Response | None:
    from src.api.handlers.base.parsers import get_parser_for_format
    from src.api.handlers.base.utils import (
        build_json_response_for_client,
        filter_proxy_response_headers,
        get_format_converter_registry,
        resolve_client_accept_encoding,
    )
    from src.core.exceptions import EmbeddedErrorException

    if payload.status_code >= 400:
        return None

    context = dict(payload.report_context or {})
    if _gateway_sync_report_context_flag(context, "has_envelope") is not False:
        return None

    provider_api_format = str(context.get("provider_api_format") or "").strip().lower()
    client_api_format = str(context.get("client_api_format") or "").strip().lower()
    if not provider_api_format or not client_api_format:
        return None

    provider_parser = get_parser_for_format(provider_api_format or client_api_format or "")
    if isinstance(payload.body_json, dict):
        try:
            if provider_parser.is_error_response(dict(payload.body_json)):
                return None
        except Exception:
            if payload.body_json.get("error") is not None:
                return None

    try:
        needs_conversion = _gateway_sync_report_context_flag(context, "needs_conversion")
        if needs_conversion is None:
            needs_conversion = provider_api_format != client_api_format

        if payload.body_base64:
            response_json = await _aggregate_gateway_sync_response_bytes(
                body_bytes=_extract_gateway_report_body_bytes(payload),
                provider_api_format=provider_api_format,
                client_api_format=client_api_format,
                provider_name=str(context.get("provider_name") or "unknown"),
                model=str(context.get("model") or "unknown"),
                request_id=str(
                    context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]
                ),
            )
        elif isinstance(payload.body_json, dict):
            response_json = dict(payload.body_json)
            if needs_conversion and provider_api_format and client_api_format:
                registry = get_format_converter_registry()
                response_json = registry.convert_response(
                    response_json,
                    provider_api_format,
                    client_api_format,
                    requested_model=str(context.get("model") or "unknown"),
                )
        else:
            return None
    except EmbeddedErrorException:
        return None
    except Exception:
        return None

    if not isinstance(response_json, dict):
        return None

    original_headers = dict(context.get("original_headers") or {})
    client_response_headers = filter_proxy_response_headers(dict(payload.headers or {}))
    client_response_headers["content-type"] = "application/json"
    return build_json_response_for_client(
        status_code=payload.status_code,
        content=response_json,
        headers=client_response_headers,
        client_accept_encoding=resolve_client_accept_encoding(original_headers, None),
    )


def _coerce_gateway_video_local_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _resolve_gateway_video_response_time_ms(payload: GatewaySyncReportRequest) -> int:
    telemetry = payload.telemetry if isinstance(payload.telemetry, dict) else {}
    raw_elapsed_ms = telemetry.get("elapsed_ms")
    try:
        return max(int(raw_elapsed_ms or 0), 0)
    except (TypeError, ValueError):
        return 0


def _gateway_report_context_uses_rust_video_task_owner(context: dict[str, Any]) -> bool:
    return bool(context.get("rust_video_task_persisted"))


def _find_gateway_existing_video_task(
    *,
    db: Session,
    request_id: str | None = None,
    local_task_id: str | None = None,
    local_short_id: str | None = None,
) -> VideoTask | None:
    if local_task_id:
        task = db.query(VideoTask).filter(VideoTask.id == local_task_id).first()
        if task is not None:
            return task
    if local_short_id:
        task = db.query(VideoTask).filter(VideoTask.short_id == local_short_id).first()
        if task is not None:
            return task
    if request_id:
        return db.query(VideoTask).filter(VideoTask.request_id == request_id).first()
    return None


async def _finalize_gateway_openai_video_create_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
    background_tasks: BackgroundTasks | None = None,
) -> Response:
    from sqlalchemy.exc import IntegrityError

    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.models.database import Provider, ProviderAPIKey, ProviderEndpoint
    from src.services.billing.rule_service import BillingRuleService
    from src.services.scheduling.schemas import ProviderCandidate
    from src.services.usage.service import UsageService

    if payload.status_code >= 400:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )
    if not isinstance(payload.body_json, dict):
        raise HTTPException(status_code=502, detail="Invalid upstream response from video provider")

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    provider_id = str(context.get("provider_id") or "").strip()
    endpoint_id = str(context.get("endpoint_id") or "").strip()
    key_id = str(context.get("key_id") or "").strip()
    request_id = str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]).strip()
    local_task_id = str(context.get("local_task_id") or "").strip()
    local_created_at = _coerce_gateway_video_local_timestamp(context.get("local_created_at"))
    rust_owned_task = _gateway_report_context_uses_rust_video_task_owner(context)
    if not all([user_id, api_key_id]) or (
        not rust_owned_task and not all([provider_id, endpoint_id, key_id])
    ):
        raise HTTPException(status_code=400, detail="Missing gateway video finalize context")

    external_task_id = str(payload.body_json.get("id") or "").strip()
    if not external_task_id:
        raise HTTPException(status_code=502, detail="Upstream video response missing task id")

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video finalize context")

    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=request_id,
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
    )

    original_request_body = dict(context.get("original_request_body") or {})
    original_headers = dict(context.get("original_headers") or {})
    response_time_ms = _resolve_gateway_video_response_time_ms(payload)

    if rust_owned_task:
        task = _find_gateway_existing_video_task(
            db=db,
            request_id=request_id,
            local_task_id=local_task_id or None,
        )
        if task is not None:
            logger.info(
                "gateway reusing rust-owned openai video task: request_id={}, task_id={}",
                request_id,
                task.id,
            )
            response_body = handler._normalizer.video_task_from_internal(handler._task_to_internal(task))
            if background_tasks is not None:
                background_tasks.add_task(
                    _run_gateway_video_finalize_submitted_background,
                    db=db,
                    request_id=request_id,
                    provider_name=str(context.get("provider_name") or "openai"),
                    provider_id=provider_id or None,
                    provider_endpoint_id=endpoint_id or None,
                    provider_api_key_id=key_id or None,
                    response_time_ms=response_time_ms,
                    status_code=payload.status_code,
                    endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
                    or None,
                    provider_request_headers=dict(context.get("provider_request_headers") or {}),
                    response_headers=dict(payload.headers or {}),
                    response_body=response_body,
                )
            else:
                await _run_gateway_video_finalize_submitted_background(
                    db=db,
                    request_id=request_id,
                    provider_name=str(context.get("provider_name") or "openai"),
                    provider_id=provider_id or None,
                    provider_endpoint_id=endpoint_id or None,
                    provider_api_key_id=key_id or None,
                    response_time_ms=response_time_ms,
                    status_code=payload.status_code,
                    endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
                    or None,
                    provider_request_headers=dict(context.get("provider_request_headers") or {}),
                    response_headers=dict(payload.headers or {}),
                    response_body=response_body,
                )
            return JSONResponse(response_body)

        logger.warning(
            "gateway rust-owned openai video task missing, falling back to legacy create: request_id={}, local_task_id={}",
            request_id,
            local_task_id,
        )

    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    endpoint = db.query(ProviderEndpoint).filter(ProviderEndpoint.id == endpoint_id).first()
    key = db.query(ProviderAPIKey).filter(ProviderAPIKey.id == key_id).first()
    if not provider or not endpoint or not key:
        raise HTTPException(status_code=400, detail="Invalid gateway video finalize context")

    internal_request = handler._normalizer.video_request_to_internal(original_request_body)
    candidate = ProviderCandidate(provider=provider, endpoint=endpoint, key=key)

    rule_lookup = BillingRuleService.find_rule(
        db,
        provider_id=provider.id,
        model_name=internal_request.model,
        task_type="video",
    )
    billing_rule_snapshot = handler._build_billing_rule_snapshot(rule_lookup)

    task = handler._create_task_record(
        external_task_id=external_task_id,
        candidate=candidate,
        original_request_body=original_request_body,
        internal_request=internal_request,
        original_headers=original_headers,
        billing_rule_snapshot=billing_rule_snapshot,
    )
    if local_task_id:
        task.id = local_task_id
    if local_created_at is not None:
        task.created_at = local_created_at
        task.submitted_at = local_created_at
        task.updated_at = local_created_at

    try:
        db.add(task)
        UsageService.begin_pending_usage(
            db,
            request_id=request_id,
            user=user,
            api_key=api_key,
            model=internal_request.model,
            is_stream=False,
            request_type="video",
            api_format=handler.FORMAT_ID,
            request_headers=original_headers,
            request_body=original_request_body,
        )

        response_body = handler._normalizer.video_task_from_internal(handler._task_to_internal(task))

        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Task already exists") from exc
    except Exception:
        db.rollback()
        raise

    if background_tasks is not None:
        background_tasks.add_task(
            _run_gateway_video_finalize_submitted_background,
            db=db,
            request_id=request_id,
            provider_name=str(context.get("provider_name") or "openai"),
            provider_id=provider_id or None,
            provider_endpoint_id=endpoint_id or None,
            provider_api_key_id=key_id or None,
            response_time_ms=response_time_ms,
            status_code=payload.status_code,
            endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
            or None,
            provider_request_headers=dict(context.get("provider_request_headers") or {}),
            response_headers=dict(payload.headers or {}),
            response_body=response_body,
        )
    else:
        await _run_gateway_video_finalize_submitted_background(
            db=db,
            request_id=request_id,
            provider_name=str(context.get("provider_name") or "openai"),
            provider_id=provider_id or None,
            provider_endpoint_id=endpoint_id or None,
            provider_api_key_id=key_id or None,
            response_time_ms=response_time_ms,
            status_code=payload.status_code,
            endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
            or None,
            provider_request_headers=dict(context.get("provider_request_headers") or {}),
            response_headers=dict(payload.headers or {}),
            response_body=response_body,
        )

    return JSONResponse(response_body)


async def _finalize_gateway_openai_video_remix_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> Response:
    from sqlalchemy.exc import IntegrityError

    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format.conversion.internal_video import InternalVideoTask, VideoStatus
    from src.models.database import Provider
    from src.services.scheduling.schemas import ProviderCandidate

    if payload.status_code >= 400:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )
    if not isinstance(payload.body_json, dict):
        raise HTTPException(status_code=502, detail="Invalid upstream response from video provider")

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    task_id = str(context.get("task_id") or "").strip()
    request_id = str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]).strip()
    local_task_id = str(context.get("local_task_id") or "").strip()
    local_created_at = _coerce_gateway_video_local_timestamp(context.get("local_created_at"))
    if not all([user_id, api_key_id, task_id]):
        raise HTTPException(status_code=400, detail="Missing gateway video remix finalize context")

    external_task_id = str(payload.body_json.get("id") or "").strip()
    if not external_task_id:
        raise HTTPException(status_code=502, detail="Upstream video response missing task id")

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video remix finalize context")

    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=request_id,
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
    )

    if _gateway_report_context_uses_rust_video_task_owner(context):
        task = _find_gateway_existing_video_task(
            db=db,
            request_id=request_id,
            local_task_id=local_task_id or None,
        )
        if task is not None:
            logger.info(
                "gateway reusing rust-owned openai remix task: request_id={}, task_id={}",
                request_id,
                task.id,
            )
            response_body = handler._normalizer.video_task_from_internal(handler._task_to_internal(task))
            return JSONResponse(response_body)

        logger.warning(
            "gateway rust-owned openai remix task missing, falling back to legacy create: request_id={}, local_task_id={}",
            request_id,
            local_task_id,
        )

    original_task = handler._get_task(task_id)
    if original_task.status != VideoStatus.COMPLETED.value:
        raise HTTPException(
            status_code=409,
            detail=f"Can only remix completed videos (current status: {original_task.status})",
        )
    if not original_task.external_task_id:
        raise HTTPException(status_code=500, detail="Original task missing external_task_id")

    endpoint, key = handler._get_endpoint_and_key(original_task)
    provider = db.query(Provider).filter(Provider.id == original_task.provider_id).first()
    if provider is None:
        raise HTTPException(status_code=500, detail="Provider not found")

    original_request_body = dict(context.get("original_request_body") or {})
    original_headers = dict(context.get("original_headers") or {})
    try:
        internal_request = handler._normalizer.video_request_to_internal(
            {
                "prompt": original_request_body.get("prompt", ""),
                "model": original_task.model,
                "size": original_task.size,
                "seconds": original_task.duration_seconds,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    billing_rule_snapshot = None
    if isinstance(original_task.request_metadata, dict):
        billing_rule_snapshot = original_task.request_metadata.get("billing_rule_snapshot")

    candidate = ProviderCandidate(provider=provider, endpoint=endpoint, key=key)
    task = handler._create_task_record(
        external_task_id=external_task_id,
        candidate=candidate,
        original_request_body={
            **original_request_body,
            "remix_video_id": task_id,
        },
        internal_request=internal_request,
        original_headers=original_headers,
        billing_rule_snapshot=billing_rule_snapshot,
    )
    if local_task_id:
        task.id = local_task_id
    if local_created_at is not None:
        task.created_at = local_created_at
        task.submitted_at = local_created_at
        task.updated_at = local_created_at

    try:
        db.add(task)
        db.flush()
        db.commit()
        db.refresh(task)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Task already exists") from exc
    except Exception:
        db.rollback()
        raise

    internal_task = InternalVideoTask(
        id=task.id,
        external_id=external_task_id,
        status=VideoStatus.SUBMITTED,
        created_at=task.created_at,
        original_request=internal_request,
    )
    response_body = handler._normalizer.video_task_from_internal(internal_task)
    return JSONResponse(response_body)


async def _finalize_gateway_gemini_video_create_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
    background_tasks: BackgroundTasks | None = None,
) -> Response:
    from sqlalchemy.exc import IntegrityError

    from src.api.handlers.base.video_handler_base import normalize_gemini_operation_id
    from src.api.handlers.gemini.video_handler import GeminiVeoHandler
    from src.core.api_format.conversion.internal_video import InternalVideoTask, VideoStatus
    from src.models.database import Provider, ProviderAPIKey, ProviderEndpoint
    from src.services.billing.rule_service import BillingRuleService
    from src.services.scheduling.schemas import ProviderCandidate
    from src.services.usage.service import UsageService

    if payload.status_code >= 400:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )
    if not isinstance(payload.body_json, dict):
        raise HTTPException(status_code=502, detail="Invalid upstream response from video provider")

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    provider_id = str(context.get("provider_id") or "").strip()
    endpoint_id = str(context.get("endpoint_id") or "").strip()
    key_id = str(context.get("key_id") or "").strip()
    request_id = str(context.get("request_id") or payload.trace_id or uuid.uuid4().hex[:8]).strip()
    model = str(context.get("model") or "").strip()
    local_short_id = str(context.get("local_short_id") or "").strip()
    local_created_at = _coerce_gateway_video_local_timestamp(context.get("local_created_at"))
    rust_owned_task = _gateway_report_context_uses_rust_video_task_owner(context)
    if not all([user_id, api_key_id, model]) or (
        not rust_owned_task and not all([provider_id, endpoint_id, key_id])
    ):
        raise HTTPException(status_code=400, detail="Missing gateway video finalize context")

    operation_name = str(payload.body_json.get("name") or payload.body_json.get("id") or "").strip()
    if not operation_name:
        raise HTTPException(
            status_code=502, detail="Upstream video response missing operation name"
        )
    external_task_id = normalize_gemini_operation_id(operation_name)

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video finalize context")

    handler = GeminiVeoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=request_id,
        client_ip="127.0.0.1",
        user_agent=str(
            (context.get("original_headers") or {}).get("user-agent") or "aether-gateway"
        ),
        start_time=time.time(),
    )

    original_request_body = dict(context.get("original_request_body") or {})
    original_headers = dict(context.get("original_headers") or {})
    response_time_ms = _resolve_gateway_video_response_time_ms(payload)

    if rust_owned_task:
        task = _find_gateway_existing_video_task(
            db=db,
            request_id=request_id,
            local_short_id=local_short_id or None,
        )
        if task is not None:
            logger.info(
                "gateway reusing rust-owned gemini video task: request_id={}, short_id={}",
                request_id,
                task.short_id,
            )
            response_body = handler._normalizer.video_task_from_internal(handler._task_to_internal(task))
            if background_tasks is not None:
                background_tasks.add_task(
                    _run_gateway_video_finalize_submitted_background,
                    db=db,
                    request_id=request_id,
                    provider_name=str(context.get("provider_name") or "gemini"),
                    provider_id=provider_id or None,
                    provider_endpoint_id=endpoint_id or None,
                    provider_api_key_id=key_id or None,
                    response_time_ms=response_time_ms,
                    status_code=payload.status_code,
                    endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
                    or None,
                    provider_request_headers=dict(context.get("provider_request_headers") or {}),
                    response_headers=dict(payload.headers or {}),
                    response_body=response_body,
                )
            else:
                await _run_gateway_video_finalize_submitted_background(
                    db=db,
                    request_id=request_id,
                    provider_name=str(context.get("provider_name") or "gemini"),
                    provider_id=provider_id or None,
                    provider_endpoint_id=endpoint_id or None,
                    provider_api_key_id=key_id or None,
                    response_time_ms=response_time_ms,
                    status_code=payload.status_code,
                    endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
                    or None,
                    provider_request_headers=dict(context.get("provider_request_headers") or {}),
                    response_headers=dict(payload.headers or {}),
                    response_body=response_body,
                )
            return JSONResponse(response_body)

        logger.warning(
            "gateway rust-owned gemini video task missing, falling back to legacy create: request_id={}, local_short_id={}",
            request_id,
            local_short_id,
        )

    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    endpoint = db.query(ProviderEndpoint).filter(ProviderEndpoint.id == endpoint_id).first()
    key = db.query(ProviderAPIKey).filter(ProviderAPIKey.id == key_id).first()
    if not provider or not endpoint or not key:
        raise HTTPException(status_code=400, detail="Invalid gateway video finalize context")

    request_with_model = {**original_request_body, "model": model}
    try:
        internal_request = handler._normalizer.video_request_to_internal(request_with_model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidate = ProviderCandidate(provider=provider, endpoint=endpoint, key=key)
    rule_lookup = BillingRuleService.find_rule(
        db,
        provider_id=provider.id,
        model_name=internal_request.model,
        task_type="video",
    )
    billing_rule_snapshot = handler._build_billing_rule_snapshot(rule_lookup)

    task = handler._create_task_record(
        external_task_id=external_task_id,
        candidate=candidate,
        original_request_body=original_request_body,
        converted_request_body=original_request_body,
        internal_request=internal_request,
        original_headers=original_headers,
        billing_rule_snapshot=billing_rule_snapshot,
        format_converted=False,
    )
    if local_short_id:
        task.short_id = local_short_id
    if local_created_at is not None:
        task.created_at = local_created_at
        task.submitted_at = local_created_at
        task.updated_at = local_created_at

    try:
        db.add(task)
        UsageService.begin_pending_usage(
            db,
            request_id=request_id,
            user=user,
            api_key=api_key,
            model=internal_request.model,
            is_stream=False,
            request_type="video",
            api_format=handler.FORMAT_ID,
            request_headers=original_headers,
            request_body=original_request_body,
        )
        db.flush()
        db.commit()
        db.refresh(task)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Task already exists") from exc
    except Exception:
        db.rollback()
        raise

    internal_task = InternalVideoTask(
        id=task.short_id,
        external_id=external_task_id,
        status=VideoStatus.SUBMITTED,
        created_at=task.created_at,
        original_request=internal_request,
    )
    response_body = handler._normalizer.video_task_from_internal(internal_task)

    if background_tasks is not None:
        background_tasks.add_task(
            _run_gateway_video_finalize_submitted_background,
            db=db,
            request_id=request_id,
            provider_name=str(context.get("provider_name") or "gemini"),
            provider_id=provider_id or None,
            provider_endpoint_id=endpoint_id or None,
            provider_api_key_id=key_id or None,
            response_time_ms=response_time_ms,
            status_code=payload.status_code,
            endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
            or None,
            provider_request_headers=dict(context.get("provider_request_headers") or {}),
            response_headers=dict(payload.headers or {}),
            response_body=response_body,
        )
    else:
        await _run_gateway_video_finalize_submitted_background(
            db=db,
            request_id=request_id,
            provider_name=str(context.get("provider_name") or "gemini"),
            provider_id=provider_id or None,
            provider_endpoint_id=endpoint_id or None,
            provider_api_key_id=key_id or None,
            response_time_ms=response_time_ms,
            status_code=payload.status_code,
            endpoint_api_format=str(context.get("provider_api_format") or handler.FORMAT_ID)
            or None,
            provider_request_headers=dict(context.get("provider_request_headers") or {}),
            response_headers=dict(payload.headers or {}),
            response_body=response_body,
        )

    return JSONResponse(response_body)


async def _finalize_gateway_openai_video_delete_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> Response:
    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format.conversion.internal_video import VideoStatus

    if payload.status_code >= 400 and payload.status_code != 404:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    task_id = str(context.get("task_id") or "").strip()
    if not all([user_id, api_key_id, task_id]):
        raise HTTPException(status_code=400, detail="Missing gateway video delete finalize context")

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video delete finalize context")

    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent="aether-gateway",
        start_time=time.time(),
    )
    task = handler._get_task(task_id)
    if task.status not in (VideoStatus.COMPLETED.value, VideoStatus.FAILED.value):
        raise HTTPException(
            status_code=400,
            detail=f"Can only delete completed or failed videos (current status: {task.status})",
        )

    db.delete(task)
    db.commit()
    return JSONResponse({"id": task_id, "object": "video", "deleted": True})


async def _finalize_gateway_openai_video_cancel_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> Response:
    from datetime import datetime, timezone

    from src.api.handlers.openai.video_handler import OpenAIVideoHandler
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.services.usage.service import UsageService

    if payload.status_code >= 400:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    task_id = str(context.get("task_id") or "").strip()
    if not all([user_id, api_key_id, task_id]):
        raise HTTPException(status_code=400, detail="Missing gateway video cancel finalize context")

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video cancel finalize context")

    handler = OpenAIVideoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent="aether-gateway",
        start_time=time.time(),
    )
    task = handler._get_task(task_id)
    if task.status in (
        VideoStatus.COMPLETED.value,
        VideoStatus.FAILED.value,
        VideoStatus.CANCELLED.value,
        VideoStatus.EXPIRED.value,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Task cannot be cancelled in status: {task.status}",
        )

    now = datetime.now(timezone.utc)
    task.status = VideoStatus.CANCELLED.value
    task.completed_at = getattr(task, "completed_at", None) or now
    task.updated_at = now

    try:
        UsageService.finalize_void(
            db,
            request_id=task.request_id,
            reason="cancelled_by_user",
            finalized_at=task.completed_at,
        )
    except Exception:
        pass

    db.commit()
    return JSONResponse({})


async def _finalize_gateway_gemini_video_cancel_sync(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
) -> Response:
    from datetime import datetime, timezone

    from src.api.handlers.gemini.video_handler import GeminiVeoHandler
    from src.core.api_format.conversion.internal_video import VideoStatus
    from src.services.usage.service import UsageService

    if payload.status_code >= 400:
        return JSONResponse(
            status_code=payload.status_code,
            content=payload.body_json if isinstance(payload.body_json, dict) else {},
            headers=dict(payload.headers or {}),
        )

    context = dict(payload.report_context or {})
    user_id = str(context.get("user_id") or "").strip()
    api_key_id = str(context.get("api_key_id") or "").strip()
    task_id = str(context.get("task_id") or "").strip()
    if not all([user_id, api_key_id, task_id]):
        raise HTTPException(status_code=400, detail="Missing gateway video cancel finalize context")

    user = db.query(User).filter(User.id == user_id).first()
    api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first()
    if not user or not api_key:
        raise HTTPException(status_code=400, detail="Invalid gateway video cancel finalize context")

    handler = GeminiVeoHandler(
        db=db,
        user=user,
        api_key=api_key,
        request_id=str(payload.trace_id or uuid.uuid4().hex[:8]),
        client_ip="127.0.0.1",
        user_agent="aether-gateway",
        start_time=time.time(),
    )
    task = handler._get_task_by_external_id(task_id)
    if task.status in (
        VideoStatus.COMPLETED.value,
        VideoStatus.FAILED.value,
        VideoStatus.CANCELLED.value,
        VideoStatus.EXPIRED.value,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Task cannot be cancelled in status: {task.status}",
        )

    now = datetime.now(timezone.utc)
    task.status = VideoStatus.CANCELLED.value
    task.completed_at = getattr(task, "completed_at", None) or now
    task.updated_at = now

    try:
        UsageService.finalize_void(
            db,
            request_id=task.request_id,
            reason="cancelled_by_user",
            finalized_at=task.completed_at,
        )
    except Exception:
        pass

    db.commit()
    return JSONResponse({})


from .gateway_routes import router
