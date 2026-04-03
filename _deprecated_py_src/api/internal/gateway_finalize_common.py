from __future__ import annotations

import base64
import inspect
import time
import uuid
from typing import Any

from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session

from src.core.logger import logger
from src.database import create_session, get_db
from src.models.database import ApiKey, User

from .gateway_contract import GatewaySyncReportRequest


def _gateway_module() -> Any:
    from . import gateway as gateway_module

    return gateway_module


async def _finalize_gateway_sync_response(
    payload: GatewaySyncReportRequest,
    *,
    db: Session,
    background_tasks: BackgroundTasks | None = None,
) -> Response:
    gateway_module = _gateway_module()
    if payload.report_kind == "openai_chat_sync_finalize":
        return await gateway_module._finalize_gateway_chat_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "claude_chat_sync_finalize":
        return await gateway_module._finalize_gateway_chat_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "gemini_chat_sync_finalize":
        return await gateway_module._finalize_gateway_chat_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "openai_cli_sync_finalize":
        return await gateway_module._finalize_gateway_cli_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "openai_compact_sync_finalize":
        return await gateway_module._finalize_gateway_cli_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "claude_cli_sync_finalize":
        return await gateway_module._finalize_gateway_cli_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "gemini_cli_sync_finalize":
        return await gateway_module._finalize_gateway_cli_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "openai_video_create_sync_finalize":
        return await gateway_module._finalize_gateway_openai_video_create_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "openai_video_remix_sync_finalize":
        return await gateway_module._finalize_gateway_openai_video_remix_sync(payload, db=db)
    if payload.report_kind == "gemini_video_create_sync_finalize":
        return await gateway_module._finalize_gateway_gemini_video_create_sync(
            payload,
            db=db,
            background_tasks=background_tasks,
        )
    if payload.report_kind == "openai_video_cancel_sync_finalize":
        return await gateway_module._finalize_gateway_openai_video_cancel_sync(payload, db=db)
    if payload.report_kind == "openai_video_delete_sync_finalize":
        return await gateway_module._finalize_gateway_openai_video_delete_sync(payload, db=db)
    if payload.report_kind == "gemini_video_cancel_sync_finalize":
        return await gateway_module._finalize_gateway_gemini_video_cancel_sync(payload, db=db)
    raise HTTPException(status_code=400, detail="Unsupported gateway sync finalize kind")


def _resolve_gateway_finalize_db(
    request: Request,
) -> tuple[Any, Any | None]:
    gateway_module = _gateway_module()
    overrides = getattr(getattr(request, "app", None), "dependency_overrides", None)
    if isinstance(overrides, dict):
        override = overrides.get(get_db)
        if callable(override):
            override_value = override()
            if inspect.isgenerator(override_value):
                generator = override_value
                db = next(generator)

                def _cleanup_override_generator() -> None:
                    try:
                        next(generator)
                    except StopIteration:
                        return
                    except Exception as exc:
                        logger.warning(
                            "gateway finalize override generator cleanup failed: {}",
                            exc,
                        )

                return db, _cleanup_override_generator
            return override_value, None

    db = create_session()
    return db, lambda: gateway_module._close_gateway_session(db)


def _extract_gateway_report_body_bytes(payload: Any) -> bytes:
    if payload.body_base64:
        try:
            return base64.b64decode(payload.body_base64, validate=True)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail="Invalid gateway report body payload"
            ) from exc
    if hasattr(payload, "body_json") and payload.body_json is not None:
        return JSONResponse(content=payload.body_json).body
    return b""


def _build_gateway_embedded_error_payload(exc: Exception) -> dict[str, Any]:
    from src.core.error_utils import extract_client_error_message
    from src.core.exceptions import EmbeddedErrorException

    message = extract_client_error_message(exc)
    payload: dict[str, Any] = {
        "error": {
            "message": message,
        }
    }
    if isinstance(exc, EmbeddedErrorException):
        if exc.error_message and str(exc.error_message).strip():
            payload["error"]["message"] = str(exc.error_message).strip()
        if exc.error_code is not None:
            payload["error"]["code"] = int(exc.error_code)
        if exc.error_status:
            payload["error"]["status"] = str(exc.error_status)
    return payload


def _extract_gateway_sync_error_message(payload: GatewaySyncReportRequest) -> str:
    if isinstance(payload.body_json, dict):
        error_obj = payload.body_json.get("error")
        if isinstance(error_obj, dict):
            for key in ("message", "detail", "status", "type", "code"):
                value = error_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(error_obj, str) and error_obj.strip():
            return error_obj.strip()

        for key in ("message", "detail", "status", "type"):
            value = payload.body_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if payload.body_base64:
        try:
            body_text = base64.b64decode(payload.body_base64, validate=True).decode(
                "utf-8", errors="replace"
            )
            if body_text.strip():
                return body_text[:4000]
        except Exception:
            pass

    return f"HTTP {payload.status_code}"


def _resolve_gateway_sync_error_status_code(
    payload: GatewaySyncReportRequest,
    *,
    provider_parser: Any | None = None,
) -> int:
    status_code = int(getattr(payload, "status_code", 0) or 0)
    if 400 <= status_code < 600:
        return status_code

    if isinstance(payload.body_json, dict):
        if provider_parser is not None:
            try:
                parsed = provider_parser.parse_response(dict(payload.body_json), status_code or 200)
                embedded_status = getattr(parsed, "embedded_status_code", None)
                if isinstance(embedded_status, int) and 100 <= embedded_status < 600:
                    return embedded_status
            except Exception:
                pass

        error_obj = payload.body_json.get("error")
        if isinstance(error_obj, dict):
            for key in ("code", "status"):
                value = error_obj.get(key)
                if isinstance(value, int) and 100 <= value < 600:
                    return value
                if isinstance(value, str) and value.isdigit():
                    parsed_value = int(value)
                    if 100 <= parsed_value < 600:
                        return parsed_value

    return status_code if 400 <= status_code < 600 else 400


def _build_gateway_sync_error_payload(
    payload: GatewaySyncReportRequest,
    *,
    client_api_format: str,
    provider_api_format: str,
    needs_conversion: bool,
) -> dict[str, Any]:
    from src.api.handlers.base.chat_error_utils import (
        _build_client_error_response_best_effort,
        _convert_error_response_best_effort,
    )

    if isinstance(payload.body_json, dict):
        if needs_conversion and provider_api_format and client_api_format:
            return _convert_error_response_best_effort(
                dict(payload.body_json),
                provider_api_format,
                client_api_format,
            )
        return dict(payload.body_json)

    return _build_client_error_response_best_effort(
        _extract_gateway_sync_error_message(payload),
        client_api_format or provider_api_format or "openai:chat",
    )
