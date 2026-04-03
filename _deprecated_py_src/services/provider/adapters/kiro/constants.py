"""Kiro adapter constants.

Kiro upstream uses AWS Event Stream (binary frames) for streaming responses.
"""

from __future__ import annotations

import platform

AWS_EVENTSTREAM_CONTENT_TYPE = "application/vnd.amazon.eventstream"

# Kiro API endpoints
KIRO_GENERATE_ASSISTANT_PATH = "/generateAssistantResponse"
KIRO_USAGE_LIMITS_PATH = "/getUsageLimits"

# Default AWS region when not specified in credentials
DEFAULT_REGION = "us-east-1"

# Default client fingerprints used in headers (best-effort)
DEFAULT_KIRO_VERSION = "0.8.0"
DEFAULT_NODE_VERSION = "22.21.1"


def _detect_system_version() -> str:
    system = platform.system().lower() or "other"
    release = platform.release() or "unknown"
    # Match KiroIDE style: darwin#24.6.0, windows#10.0.22631, linux#6.8.0-...
    return f"{system}#{release}"


DEFAULT_SYSTEM_VERSION = _detect_system_version()

# Header constants
KIRO_AGENT_MODE = "vibe"
CODEWHISPERER_OPTOUT = "true"

# aws-sdk-js versions observed in kiro.rs
AWS_SDK_JS_MAIN_VERSION = "1.0.27"
AWS_SDK_JS_USAGE_VERSION = "1.0.0"

# Claude model context window used by kiro.rs to convert contextUsage percentage -> tokens
CONTEXT_WINDOW_TOKENS = 200_000

# ---------------------------------------------------------------------------
# Chunked-write policy injected into tool descriptions and system prompt
# ---------------------------------------------------------------------------
# Kiro upstream has lower per-message size limits than standard Claude.
# We inject instructions for Write/Edit tools and a system-level policy
# so the model splits large writes into smaller chunks automatically.

WRITE_TOOL_DESCRIPTION_SUFFIX = (
    "- IMPORTANT: If the content to write exceeds 150 lines, you MUST only write "
    "the first 50 lines using this tool, then use `Edit` tool to append the "
    "remaining content in chunks of no more than 50 lines each. If needed, leave "
    "a unique placeholder to help append content. Do NOT attempt to write all "
    "content at once."
)

EDIT_TOOL_DESCRIPTION_SUFFIX = (
    "- IMPORTANT: If the `new_string` content exceeds 50 lines, you MUST split "
    "it into multiple Edit calls, each replacing no more than 50 lines at a time. "
    "If used to append content, leave a unique placeholder to help append content. "
    "On the final chunk, do NOT include the placeholder."
)

TOOL_DESCRIPTION_SUFFIXES: dict[str, str] = {
    "Write": WRITE_TOOL_DESCRIPTION_SUFFIX,
    "Edit": EDIT_TOOL_DESCRIPTION_SUFFIX,
}

SYSTEM_CHUNKED_POLICY = (
    "When the Write or Edit tool has content size limits, always comply silently. "
    "Never suggest bypassing these limits via alternative tools. "
    "Never ask the user whether to switch approaches. "
    "Complete all chunked operations without commentary."
)

__all__ = [
    "AWS_EVENTSTREAM_CONTENT_TYPE",
    "AWS_SDK_JS_MAIN_VERSION",
    "AWS_SDK_JS_USAGE_VERSION",
    "CODEWHISPERER_OPTOUT",
    "CONTEXT_WINDOW_TOKENS",
    "DEFAULT_KIRO_VERSION",
    "DEFAULT_NODE_VERSION",
    "DEFAULT_REGION",
    "DEFAULT_SYSTEM_VERSION",
    "EDIT_TOOL_DESCRIPTION_SUFFIX",
    "KIRO_AGENT_MODE",
    "KIRO_GENERATE_ASSISTANT_PATH",
    "KIRO_USAGE_LIMITS_PATH",
    "SYSTEM_CHUNKED_POLICY",
    "TOOL_DESCRIPTION_SUFFIXES",
    "WRITE_TOOL_DESCRIPTION_SUFFIX",
]
