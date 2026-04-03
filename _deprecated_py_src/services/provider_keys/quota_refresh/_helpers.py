"""Shared helpers for quota refresh strategies."""

from __future__ import annotations

from typing import Any

from src.models.database import ProviderAPIKey
from src.services.provider.pool.account_state import OAUTH_REFRESH_FAILED_PREFIX


def build_success_state_update(key: ProviderAPIKey) -> dict[str, Any]:
    """配额刷新成功时的 state_updates 构建。

    如果当前 key 携带 [REFRESH_FAILED] 标记，保留该标记（配额刷新不等于 token 刷新成功）。
    """
    current_reason = str(getattr(key, "oauth_invalid_reason", None) or "").strip()
    if current_reason.startswith(OAUTH_REFRESH_FAILED_PREFIX):
        return {
            "oauth_invalid_at": getattr(key, "oauth_invalid_at", None),
            "oauth_invalid_reason": current_reason,
        }
    return {
        "oauth_invalid_at": None,
        "oauth_invalid_reason": None,
    }
