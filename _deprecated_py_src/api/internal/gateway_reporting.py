"""Compatibility re-export layer for gateway reporting helpers."""

from __future__ import annotations

from .gateway_reporting_background import (
    _apply_gateway_stream_report,
    _apply_gateway_sync_report,
    _close_gateway_session,
    _gateway_sync_report_requires_inline,
    _resolve_gateway_background_db,
    _run_gateway_stream_report_background,
    _run_gateway_stream_report_background_with_session,
    _run_gateway_sync_report_background,
    _run_gateway_sync_report_background_with_session,
    _run_gateway_video_finalize_submitted_background,
)
from .gateway_reporting_candidates import (
    _ensure_gateway_request_candidate,
    _mark_gateway_sync_candidate_terminal_state,
    _record_gateway_direct_candidate_graph,
)
from .gateway_reporting_common import _gateway_module
from .gateway_reporting_failures import (
    _record_gateway_chat_sync_failure,
    _record_gateway_cli_sync_failure,
    _record_gateway_sync_failure,
    _record_gateway_video_sync_failure,
    _resolve_gateway_failure_adapter,
)
from .gateway_reporting_success_stream import (
    _GatewayReportStreamContext,
    _iter_gateway_report_body_chunks,
    _record_gateway_openai_chat_stream_success,
    _record_gateway_passthrough_chat_stream_success,
    _record_gateway_passthrough_cli_stream_success,
)
from .gateway_reporting_success_sync import (
    _postprocess_gateway_report_provider_response,
    _record_gateway_gemini_video_cancel_sync_success,
    _record_gateway_gemini_video_create_sync_success,
    _record_gateway_openai_chat_sync_success,
    _record_gateway_openai_video_cancel_sync_success,
    _record_gateway_openai_video_create_sync_success,
    _record_gateway_openai_video_delete_sync_success,
    _record_gateway_openai_video_remix_sync_success,
    _record_gateway_passthrough_chat_sync_success,
    _record_gateway_passthrough_cli_sync_success,
)
from .gateway_reporting_telemetry import (
    _build_gateway_sync_telemetry_writer,
    _build_gateway_usage_metadata,
    _dispatch_gateway_sync_telemetry,
    _schedule_gateway_sync_telemetry,
)

__all__ = [
    "_GatewayReportStreamContext",
    "_apply_gateway_stream_report",
    "_apply_gateway_sync_report",
    "_build_gateway_sync_telemetry_writer",
    "_build_gateway_usage_metadata",
    "_close_gateway_session",
    "_dispatch_gateway_sync_telemetry",
    "_ensure_gateway_request_candidate",
    "_gateway_module",
    "_gateway_sync_report_requires_inline",
    "_iter_gateway_report_body_chunks",
    "_record_gateway_gemini_video_create_sync_success",
    "_mark_gateway_sync_candidate_terminal_state",
    "_postprocess_gateway_report_provider_response",
    "_record_gateway_gemini_video_cancel_sync_success",
    "_record_gateway_chat_sync_failure",
    "_record_gateway_cli_sync_failure",
    "_record_gateway_direct_candidate_graph",
    "_record_gateway_openai_chat_stream_success",
    "_record_gateway_openai_chat_sync_success",
    "_record_gateway_openai_video_create_sync_success",
    "_record_gateway_openai_video_cancel_sync_success",
    "_record_gateway_openai_video_delete_sync_success",
    "_record_gateway_openai_video_remix_sync_success",
    "_record_gateway_passthrough_chat_stream_success",
    "_record_gateway_passthrough_chat_sync_success",
    "_record_gateway_passthrough_cli_stream_success",
    "_record_gateway_passthrough_cli_sync_success",
    "_record_gateway_sync_failure",
    "_record_gateway_video_sync_failure",
    "_resolve_gateway_background_db",
    "_resolve_gateway_failure_adapter",
    "_run_gateway_stream_report_background",
    "_run_gateway_stream_report_background_with_session",
    "_run_gateway_sync_report_background",
    "_run_gateway_sync_report_background_with_session",
    "_run_gateway_video_finalize_submitted_background",
    "_schedule_gateway_sync_telemetry",
]
