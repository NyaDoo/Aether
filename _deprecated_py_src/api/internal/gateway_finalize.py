"""Compatibility re-export layer for gateway finalize helpers."""

from __future__ import annotations

from .gateway_finalize_chat import (
    _finalize_gateway_chat_sync,
    _run_gateway_chat_sync_finalize_background,
    _run_gateway_chat_sync_finalize_background_with_session,
)
from .gateway_finalize_cli import (
    _finalize_gateway_cli_sync,
    _run_gateway_cli_sync_finalize_background,
    _run_gateway_cli_sync_finalize_background_with_session,
)
from .gateway_finalize_common import (
    _build_gateway_embedded_error_payload,
    _build_gateway_sync_error_payload,
    _extract_gateway_report_body_bytes,
    _extract_gateway_sync_error_message,
    _finalize_gateway_sync_response,
    _gateway_module,
    _resolve_gateway_finalize_db,
    _resolve_gateway_sync_error_status_code,
)

__all__ = [
    "_build_gateway_embedded_error_payload",
    "_build_gateway_sync_error_payload",
    "_extract_gateway_report_body_bytes",
    "_extract_gateway_sync_error_message",
    "_finalize_gateway_chat_sync",
    "_finalize_gateway_cli_sync",
    "_finalize_gateway_sync_response",
    "_gateway_module",
    "_resolve_gateway_finalize_db",
    "_resolve_gateway_sync_error_status_code",
    "_run_gateway_chat_sync_finalize_background",
    "_run_gateway_chat_sync_finalize_background_with_session",
    "_run_gateway_cli_sync_finalize_background",
    "_run_gateway_cli_sync_finalize_background_with_session",
]
