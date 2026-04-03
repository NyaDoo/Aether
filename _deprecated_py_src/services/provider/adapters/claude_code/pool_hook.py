"""Claude Code pool scheduling hook."""

from __future__ import annotations

from typing import Any


class ClaudeCodePoolHook:
    """Pool scheduling hook for Claude Code providers.

    Extracts the session UUID from ``metadata.user_id`` which follows the
    pattern ``<user>_session_<uuid>``.
    """

    name = "claude_code"

    def extract_session_uuid(self, request_body: dict[str, Any]) -> str | None:
        metadata = request_body.get("metadata")
        if isinstance(metadata, dict):
            user_id = metadata.get("user_id")
            if isinstance(user_id, str) and "_session_" in user_id:
                idx = user_id.rfind("_session_")
                return user_id[idx + len("_session_") :].strip() or None
        return None


claude_code_pool_hook = ClaudeCodePoolHook()
