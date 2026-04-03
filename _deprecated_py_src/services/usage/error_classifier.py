"""Error classification helpers for Usage records."""

from __future__ import annotations

from src.core.enums import ErrorCategory

_STATUS_CODE_MAP: dict[int, ErrorCategory] = {
    400: ErrorCategory.INVALID_REQUEST,
    401: ErrorCategory.AUTH,
    403: ErrorCategory.AUTH,
    404: ErrorCategory.NOT_FOUND,
    408: ErrorCategory.TIMEOUT,
    429: ErrorCategory.RATE_LIMIT,
    500: ErrorCategory.SERVER_ERROR,
    502: ErrorCategory.SERVER_ERROR,
    503: ErrorCategory.SERVER_ERROR,
    504: ErrorCategory.TIMEOUT,
}

_CONTEXT_LENGTH_PATTERNS = (
    "context_length_exceeded",
    "maximum context length",
    "too many tokens",
    "input is too long",
)

_CONTENT_FILTER_PATTERNS = (
    "content_filter",
    "content_policy",
    "safety_block",
    "blocked by content",
)

_NETWORK_PATTERNS = ("connection", "network", "dns", "socket")


def classify_error(
    status_code: int | None,
    error_message: str | None,
    status: str | None = None,
) -> ErrorCategory:
    """Map provider errors to ErrorCategory."""
    if status and status.lower() == "cancelled":
        return ErrorCategory.CANCELLED

    if status_code is not None:
        mapped = _STATUS_CODE_MAP.get(status_code)
        if mapped:
            return mapped

    if error_message:
        msg_lower = error_message.lower()
        if any(p in msg_lower for p in _CONTEXT_LENGTH_PATTERNS):
            return ErrorCategory.CONTEXT_LENGTH
        if any(p in msg_lower for p in _CONTENT_FILTER_PATTERNS):
            return ErrorCategory.CONTENT_FILTER
        if "rate limit" in msg_lower or "rate_limit" in msg_lower:
            return ErrorCategory.RATE_LIMIT
        if "timeout" in msg_lower or "timed out" in msg_lower:
            return ErrorCategory.TIMEOUT
        if any(p in msg_lower for p in _NETWORK_PATTERNS):
            return ErrorCategory.NETWORK

    if status_code is not None:
        if status_code >= 500:
            return ErrorCategory.SERVER_ERROR
        if status_code >= 400:
            return ErrorCategory.INVALID_REQUEST

    return ErrorCategory.UNKNOWN
