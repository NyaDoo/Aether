from __future__ import annotations

import inspect
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import BackgroundTasks
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.models.database import ApiKey, RequestCandidate, User

from .gateway_contract import GatewayStreamReportRequest, GatewaySyncReportRequest
from .gateway_reporting_common import _gateway_module


def _load_existing_gateway_direct_candidate_record_map(
    *,
    db: Session,
    request_id: str,
) -> dict[tuple[int, int], str]:
    if not request_id or not hasattr(db, "query"):
        return {}

    rows = (
        db.query(RequestCandidate)
        .filter(RequestCandidate.request_id == request_id)
        .order_by(RequestCandidate.candidate_index, RequestCandidate.retry_index)
        .all()
    )
    return {
        (int(row.candidate_index or 0), int(row.retry_index or 0)): str(row.id)
        for row in rows
    }


def _record_gateway_direct_candidate_graph(
    *,
    db: Session,
    candidate_resolver: Any,
    candidates: list[Any],
    request_id: str,
    user_api_key: ApiKey,
    required_capabilities: dict[str, bool] | None,
    selected_candidate_index: int,
) -> None:
    from src.services.request.candidate import RequestCandidateService
    from src.services.scheduling.schemas import PoolCandidate

    if not candidates or not request_id or not hasattr(db, "query"):
        return

    selected_candidate_index = max(0, min(selected_candidate_index, len(candidates) - 1))
    selected_candidate = candidates[selected_candidate_index]

    try:
        user = getattr(user_api_key, "user", None)
    except Exception:
        user = None
    user_id = str(getattr(user_api_key, "user_id", "") or getattr(user, "id", "") or "") or None

    try:
        candidate_record_map = _load_existing_gateway_direct_candidate_record_map(
            db=db,
            request_id=request_id,
        )
        if not candidate_record_map:
            candidate_record_map = candidate_resolver.create_candidate_records(
                candidates,
                request_id,
                user_id,
                user_api_key,
                required_capabilities,
                expand_retries=False,
            )
    except IntegrityError as exc:
        db.rollback()
        candidate_record_map = _load_existing_gateway_direct_candidate_record_map(
            db=db,
            request_id=request_id,
        )
        if not candidate_record_map:
            logger.warning(
                "[Gateway] failed to create direct candidate graph for request {}: {}",
                request_id,
                exc,
            )
            return
        logger.debug(
            "[Gateway] reused existing direct candidate graph for request {} after duplicate insert",
            request_id,
        )
    except Exception as exc:
        db.rollback()
        logger.warning(
            "[Gateway] failed to create direct candidate graph for request {}: {}",
            request_id,
            exc,
        )
        return

    selected_retry_index = 0
    if isinstance(selected_candidate, PoolCandidate):
        selected_retry_index = int(getattr(selected_candidate, "_pool_key_index", 0) or 0)

    selected_record_id = candidate_record_map.get((selected_candidate_index, selected_retry_index))
    if not selected_record_id:
        selected_record_id = candidate_record_map.get((selected_candidate_index, 0))
        selected_retry_index = 0
    if not selected_record_id:
        return

    setattr(selected_candidate, "request_candidate_id", selected_record_id)
    RequestCandidateService.mark_candidate_started(db, selected_record_id)

    unused_record_ids = [
        record_id
        for (candidate_index, retry_index), record_id in candidate_record_map.items()
        if (candidate_index, retry_index) != (selected_candidate_index, selected_retry_index)
    ]
    if not unused_record_ids:
        return

    now = datetime.now(timezone.utc)
    unused_candidates = (
        db.query(RequestCandidate)
        .filter(
            RequestCandidate.id.in_(unused_record_ids),
            RequestCandidate.status == "available",
        )
        .all()
    )
    for candidate in unused_candidates:
        candidate.status = "unused"
        candidate.finished_at = now
    if unused_candidates:
        db.flush()


def _ensure_gateway_request_candidate(
    *,
    db: Session,
    report_context: dict[str, Any],
    trace_id: str,
    initial_status: str,
) -> RequestCandidate | None:
    from src.services.request.candidate import RequestCandidateService

    if not hasattr(db, "query"):
        return None

    request_id = str(report_context.get("request_id") or trace_id or "").strip()
    if not request_id:
        return None

    candidate_id = str(report_context.get("candidate_id") or "").strip() or None
    provider_id = str(report_context.get("provider_id") or "").strip() or None
    endpoint_id = str(report_context.get("endpoint_id") or "").strip() or None
    key_id = str(report_context.get("key_id") or "").strip() or None
    user_id = str(report_context.get("user_id") or "").strip() or None
    api_key_id = str(report_context.get("api_key_id") or "").strip() or None
    client_api_format = str(report_context.get("client_api_format") or "").strip() or None

    if not any([candidate_id, provider_id, endpoint_id, key_id, client_api_format]):
        return None

    candidate: RequestCandidate | None = None
    if candidate_id:
        candidate = db.query(RequestCandidate).filter(RequestCandidate.id == candidate_id).first()

    if candidate is None:
        lookup = db.query(RequestCandidate).filter(RequestCandidate.request_id == request_id)
        if provider_id:
            lookup = lookup.filter(RequestCandidate.provider_id == provider_id)
        if endpoint_id:
            lookup = lookup.filter(RequestCandidate.endpoint_id == endpoint_id)
        if key_id:
            lookup = lookup.filter(RequestCandidate.key_id == key_id)
        candidate = lookup.order_by(
            RequestCandidate.retry_index.desc(),
            RequestCandidate.candidate_index.desc(),
            RequestCandidate.created_at.desc(),
        ).first()

    if candidate is None:
        user = db.query(User).filter(User.id == user_id).first() if user_id else None
        api_key = db.query(ApiKey).filter(ApiKey.id == api_key_id).first() if api_key_id else None
        candidate_index = (
            db.query(RequestCandidate).filter(RequestCandidate.request_id == request_id).count()
        )
        candidate = RequestCandidateService.create_candidate(
            db,
            request_id=request_id,
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            user_id=user_id,
            api_key_id=api_key_id,
            username=str(getattr(user, "username", "") or "") or None,
            api_key_name=str(getattr(api_key, "name", "") or "") or None,
            provider_id=provider_id,
            endpoint_id=endpoint_id,
            key_id=key_id,
            status=initial_status,
            extra_data={
                "gateway_direct_executor": True,
                "phase": "3c_trial",
                "client_api_format": client_api_format,
                "provider_api_format": str(report_context.get("provider_api_format") or "") or None,
            },
        )

    current_status = str(candidate.status or "").strip().lower()
    if candidate.started_at is None:
        candidate.started_at = datetime.now(timezone.utc)
    if initial_status == "streaming" and current_status in {
        "",
        "available",
        "unused",
        "skipped",
        "pending",
        "streaming",
    }:
        candidate.status = "streaming"
    elif current_status in {"", "available", "unused", "skipped"}:
        candidate.status = "pending"
    db.commit()
    return candidate


def _mark_gateway_sync_candidate_terminal_state(
    *,
    db: Session,
    candidate: RequestCandidate | None,
    payload: GatewaySyncReportRequest,
) -> None:
    from src.services.request.candidate import RequestCandidateService

    if candidate is None:
        return

    gateway_module = _gateway_module()
    elapsed_ms = 0
    if isinstance(payload.telemetry, dict):
        raw_elapsed_ms = payload.telemetry.get("elapsed_ms")
        try:
            elapsed_ms = max(int(raw_elapsed_ms or 0), 0)
        except (TypeError, ValueError):
            elapsed_ms = 0

    has_error_payload = (
        isinstance(payload.body_json, dict) and payload.body_json.get("error") is not None
    )
    if payload.status_code >= 400 or has_error_payload:
        RequestCandidateService.mark_candidate_failed(
            db=db,
            candidate_id=candidate.id,
            error_type="gateway_error",
            error_message=gateway_module._extract_gateway_sync_error_message(payload),
            status_code=payload.status_code,
            latency_ms=elapsed_ms or None,
        )
        return

    RequestCandidateService.mark_candidate_success(
        db=db,
        candidate_id=candidate.id,
        status_code=payload.status_code,
        latency_ms=elapsed_ms,
        extra_data={"gateway_direct_executor": True, "phase": "3c_trial"},
    )
