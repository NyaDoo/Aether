"""
Endpoint signature 辅助函数。

调度/编排链路使用 endpoint signature key：`family:kind`（如 "claude:chat", "openai:cli"）。
"""

from __future__ import annotations

from src.core.api_format.enums import ApiFamily, EndpointKind
from src.core.api_format.signature import (
    EndpointSignature,
    make_signature_key,
    normalize_signature_key,
)

DEFAULT_ENDPOINT_SIGNATURE: str = make_signature_key(ApiFamily.CLAUDE, EndpointKind.CHAT)


def normalize_endpoint_signature(
    value: str | EndpointSignature | tuple[ApiFamily, EndpointKind] | None,
    *,
    default: str = DEFAULT_ENDPOINT_SIGNATURE,
) -> str:
    """
    将任意输入归一化为 canonical signature key（`family:kind`，小写）。

    不支持旧格式（如 "CLAUDE_CLI"），仅接受 `family:kind` 格式。
    解析失败时返回默认值。
    """
    if value is None:
        return default
    if isinstance(value, EndpointSignature):
        return value.key
    if isinstance(value, tuple) and len(value) == 2:
        fam, kind = value
        if isinstance(fam, ApiFamily) and isinstance(kind, EndpointKind):
            return make_signature_key(fam, kind)
        return default
    if isinstance(value, str):
        try:
            return normalize_signature_key(value)
        except ValueError:
            # 如果解析失败，返回默认值
            return default
    return default
