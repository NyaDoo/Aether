"""Claude Code CLI client restriction.

When cli_only_enabled is True, only requests from genuine Claude Code CLI
clients are allowed. Non-CLI traffic receives a 403 response.
"""

from __future__ import annotations

import contextvars
from typing import Any

from src.core.logger import logger

# Contextvar to carry the original request headers into the envelope layer.
_original_request_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "claude_code_original_request_headers",
    default=None,
)


def set_original_request_headers(headers: dict[str, str] | None) -> None:
    _original_request_headers.set(headers)


def get_original_request_headers() -> dict[str, str] | None:
    return _original_request_headers.get()


# Known Claude Code CLI User-Agent patterns.
_CLI_USER_AGENT_PATTERNS = (
    "claude-code",
    "claudecode",
    "claude_code",
)

# Known originator / x-app values indicating CLI usage.
_CLI_APP_VALUES = {"cli"}


def is_claude_code_client(headers: dict[str, Any]) -> bool:
    """Detect whether the request originates from a Claude Code CLI client.

    Detection signals (any match is sufficient):
    1. User-Agent contains a known Claude Code CLI pattern
    2. x-app header equals "cli"
    """
    # Normalize header keys to lowercase for case-insensitive matching.
    lower_headers = {k.lower(): v for k, v in headers.items()}

    # Check User-Agent
    ua = str(lower_headers.get("user-agent", "")).lower()
    for pattern in _CLI_USER_AGENT_PATTERNS:
        if pattern in ua:
            return True

    # Check x-app header
    x_app = str(lower_headers.get("x-app", "")).strip().lower()
    if x_app in _CLI_APP_VALUES:
        return True

    return False


def enforce_cli_only(cli_only_enabled: bool) -> None:
    """Enforce CLI-only restriction if enabled.

    Reads original request headers from contextvar, checks whether the
    client is a Claude Code CLI, and raises HTTPException(403) if not.
    """
    if not cli_only_enabled:
        return

    headers = get_original_request_headers()
    if headers is None:
        # No headers available; skip enforcement (should not happen in normal flow).
        logger.debug("CLI-only check skipped: no request headers in context")
        return

    if is_claude_code_client(headers):
        return

    from fastapi import HTTPException

    logger.info("CLI-only restriction: rejected non-CLI client")
    raise HTTPException(
        status_code=403,
        detail="This endpoint only accepts requests from Claude Code CLI clients.",
    )


__all__ = [
    "enforce_cli_only",
    "get_original_request_headers",
    "is_claude_code_client",
    "set_original_request_headers",
]
