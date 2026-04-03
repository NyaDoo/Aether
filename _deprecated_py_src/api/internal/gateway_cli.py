"""Compatibility re-export layer for gateway CLI builders."""

from __future__ import annotations

from .gateway_cli_decision import (
    _build_claude_cli_stream_decision,
    _build_claude_cli_sync_decision,
    _build_cli_stream_decision,
    _build_cli_sync_decision,
    _build_gemini_cli_stream_decision,
    _build_gemini_cli_sync_decision,
    _build_openai_cli_stream_decision,
    _build_openai_cli_sync_decision,
)
from .gateway_cli_plan import (
    _build_claude_cli_stream_plan,
    _build_claude_cli_sync_plan,
    _build_cli_stream_plan,
    _build_cli_sync_plan,
    _build_gemini_cli_stream_plan,
    _build_gemini_cli_sync_plan,
    _build_openai_cli_stream_plan,
    _build_openai_cli_sync_plan,
)

__all__ = [
    "_build_cli_sync_decision",
    "_build_cli_stream_decision",
    "_build_openai_cli_sync_decision",
    "_build_openai_cli_stream_decision",
    "_build_claude_cli_sync_decision",
    "_build_claude_cli_stream_decision",
    "_build_gemini_cli_sync_decision",
    "_build_gemini_cli_stream_decision",
    "_build_cli_stream_plan",
    "_build_cli_sync_plan",
    "_build_openai_cli_stream_plan",
    "_build_claude_cli_stream_plan",
    "_build_gemini_cli_stream_plan",
    "_build_openai_cli_sync_plan",
    "_build_claude_cli_sync_plan",
    "_build_gemini_cli_sync_plan",
]
