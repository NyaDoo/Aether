"""Pool account state helpers.

Provides a shared way to classify account-level hard-block states
from upstream metadata and OAuth invalid reasons.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from src.services.provider_keys.quota_reader import get_quota_reader

OAUTH_ACCOUNT_BLOCK_PREFIX = "[ACCOUNT_BLOCK] "
OAUTH_REFRESH_FAILED_PREFIX = "[REFRESH_FAILED] "
OAUTH_EXPIRED_PREFIX = "[OAUTH_EXPIRED] "
OAUTH_REQUEST_FAILED_PREFIX = "[REQUEST_FAILED] "

# -- 按原因细分的关键词组 --
# 封禁类 (suspended / banned)
_KEYWORDS_SUSPENDED: tuple[str, ...] = (
    "suspended",
    "account_block",
    "account blocked",
    "封禁",
    "封号",
    "被封",
    "账户已封禁",
    "账号异常",
)

# 停用类 (disabled / deactivated)
_KEYWORDS_DISABLED: tuple[str, ...] = (
    "account has been disabled",
    "account disabled",
    "account has been deactivated",
    "account_deactivated",
    "account deactivated",
    "organization has been disabled",
    "organization_disabled",
    "deactivated_workspace",
    "deactivated",
    "访问被禁止",
    "账户访问被禁止",
)

_TOKEN_INVALID_KEYWORDS: tuple[str, ...] = (
    "authentication token has been invalidated",
    "token has been invalidated",
    "codex token 无效或已过期",
)

# 需要验证类
_KEYWORDS_VERIFICATION: tuple[str, ...] = (
    "validation_required",
    "verify your account",
    "需要验证",
    "验证账号",
    "验证身份",
)

# 合并的完整列表（用于 is_account_level_block_reason 快速判断）
ACCOUNT_BLOCK_REASON_KEYWORDS: tuple[str, ...] = (
    *_KEYWORDS_SUSPENDED,
    *_KEYWORDS_DISABLED,
    *_TOKEN_INVALID_KEYWORDS,
    *_KEYWORDS_VERIFICATION,
)

AUTO_REMOVABLE_ACCOUNT_STATE_CODES: frozenset[str] = frozenset(
    {
        "account_banned",
        "account_suspended",
        "account_disabled",
        "workspace_deactivated",
        "account_forbidden",
    }
)


def _classify_block_reason(text: str) -> tuple[str, str]:
    """Return (code, label) based on the oauth_invalid_reason text."""
    lowered = text.lower()
    if any(kw in lowered for kw in _TOKEN_INVALID_KEYWORDS):
        return "oauth_expired", "Token 失效"
    if any(kw in lowered for kw in _KEYWORDS_VERIFICATION):
        return "account_verification", "需要验证"
    if "deactivated_workspace" in lowered:
        return "workspace_deactivated", "工作区停用"
    if any(kw in lowered for kw in _KEYWORDS_DISABLED):
        return "account_disabled", "账号停用"
    if any(kw in lowered for kw in _KEYWORDS_SUSPENDED):
        return "account_suspended", "账号封禁"
    return "account_blocked", "账号异常"


@dataclass(frozen=True, slots=True)
class PoolAccountState:
    """Resolved account-level state for one key."""

    blocked: bool
    code: str | None = None  # account_banned / account_forbidden / account_blocked
    label: str | None = None
    reason: str | None = None
    source: str | None = None  # metadata / oauth_invalid / oauth_refresh / oauth_request
    recoverable: bool = False


@dataclass(frozen=True, slots=True)
class OAuthStatusSnapshot:
    code: str = "none"  # none / valid / expiring / expired / invalid / check_failed
    label: str | None = None
    reason: str | None = None
    expires_at: int | None = None
    invalid_at: int | None = None
    source: str | None = None
    requires_reauth: bool = False
    expiring_soon: bool = False


@dataclass(frozen=True, slots=True)
class AccountStatusSnapshot:
    code: str = "ok"
    label: str | None = None
    reason: str | None = None
    blocked: bool = False
    source: str | None = None
    recoverable: bool = False


@dataclass(frozen=True, slots=True)
class QuotaStatusSnapshot:
    code: str = "unknown"  # unknown / ok / exhausted
    label: str | None = None
    reason: str | None = None
    exhausted: bool = False
    usage_ratio: float | None = None
    updated_at: int | None = None
    reset_seconds: float | None = None
    plan_type: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderKeyStatusSnapshot:
    oauth: OAuthStatusSnapshot
    account: AccountStatusSnapshot
    quota: QuotaStatusSnapshot


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y"}
    return False


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _extract_reason(source: dict[str, Any] | None, *fields: str) -> str | None:
    if not isinstance(source, dict):
        return None
    for field in fields:
        text = _clean_text(source.get(field))
        if text:
            return text
    return None


def _is_workspace_deactivated_reason(reason: str | None) -> bool:
    text = _clean_text(reason)
    return bool(text and "deactivated_workspace" in text.lower())


_TAGGED_REASON_PATTERN = re.compile(
    r"(?:^|\n)\[(?P<tag>[A-Z_]+)\]\s*(?P<detail>.*?)(?=\n\[[A-Z_]+\]|\Z)",
    re.S,
)


def _extract_tagged_reason_sections(reason: str | None) -> dict[str, str]:
    text = _clean_text(reason)
    if not text:
        return {}
    sections: dict[str, str] = {}
    for match in _TAGGED_REASON_PATTERN.finditer(text):
        tag = str(match.group("tag") or "").strip().upper()
        if not tag or tag in sections:
            continue
        detail = str(match.group("detail") or "").strip()
        sections[tag] = detail
    return sections


def _resolve_from_metadata(
    provider_type: str | None,
    upstream_metadata: Any,
) -> PoolAccountState | None:
    if not isinstance(upstream_metadata, dict):
        return None

    normalized_provider = str(provider_type or "").strip().lower()
    provider_bucket: dict[str, Any] | None = None
    if normalized_provider:
        maybe_bucket = upstream_metadata.get(normalized_provider)
        if isinstance(maybe_bucket, dict):
            provider_bucket = maybe_bucket

    quota_block = get_quota_reader(normalized_provider, upstream_metadata).account_block()
    if quota_block.blocked:
        return PoolAccountState(
            blocked=True,
            code=quota_block.code,
            label=quota_block.label,
            reason=quota_block.reason,
            source="metadata",
        )

    for source in (provider_bucket, upstream_metadata):
        if not isinstance(source, dict):
            continue
        if _is_truthy_flag(source.get("is_banned")):
            reason = _extract_reason(source, "ban_reason", "forbidden_reason", "reason", "message")
            return PoolAccountState(
                blocked=True,
                code="account_banned",
                label="账号封禁",
                reason=reason or "账号已封禁",
                source="metadata",
            )
        if _is_truthy_flag(source.get("is_forbidden")) or _is_truthy_flag(
            source.get("account_disabled")
        ):
            reason = _extract_reason(source, "forbidden_reason", "ban_reason", "reason", "message")
            if _is_workspace_deactivated_reason(reason):
                return PoolAccountState(
                    blocked=True,
                    code="workspace_deactivated",
                    label="工作区停用",
                    reason=reason or "工作区已停用",
                    source="metadata",
                )
            return PoolAccountState(
                blocked=True,
                code="account_forbidden",
                label="访问受限",
                reason=reason or "账号访问受限",
                source="metadata",
            )

    return None


def _resolve_from_oauth_invalid_reason(reason: str | None) -> PoolAccountState | None:
    text = _clean_text(reason)
    if not text:
        return None

    if text.startswith(OAUTH_ACCOUNT_BLOCK_PREFIX):
        cleaned = text[len(OAUTH_ACCOUNT_BLOCK_PREFIX) :].strip()
        code, label = (
            _classify_block_reason(cleaned) if cleaned else ("account_blocked", "账号异常")
        )
        return PoolAccountState(
            blocked=True,
            code=code,
            label=label,
            reason=cleaned or "账号异常",
            source="oauth_invalid",
        )

    if text.startswith(OAUTH_EXPIRED_PREFIX):
        cleaned = text[len(OAUTH_EXPIRED_PREFIX) :].strip()
        return PoolAccountState(
            blocked=True,
            code="oauth_expired",
            label="Token 失效",
            reason=cleaned or "OAuth Token 已过期且无法续期",
            source="oauth_invalid",
            recoverable=True,
        )

    if text.startswith(OAUTH_REFRESH_FAILED_PREFIX):
        cleaned = text[len(OAUTH_REFRESH_FAILED_PREFIX) :].strip()
        return PoolAccountState(
            blocked=False,
            code="oauth_refresh_failed",
            label="续期失败",
            reason=cleaned or "OAuth Token 续期失败",
            source="oauth_refresh",
            recoverable=True,
        )

    if text.startswith(OAUTH_REQUEST_FAILED_PREFIX):
        cleaned = text[len(OAUTH_REQUEST_FAILED_PREFIX) :].strip()
        return PoolAccountState(
            blocked=False,
            code="oauth_request_failed",
            label="请求失败",
            reason=cleaned or "账号状态检查失败",
            source="oauth_request",
            recoverable=True,
        )

    if text.startswith("["):
        return None

    lowered = text.lower()
    if any(keyword in lowered for keyword in ACCOUNT_BLOCK_REASON_KEYWORDS):
        code, label = _classify_block_reason(text)
        return PoolAccountState(
            blocked=True,
            code=code,
            label=label,
            reason=text,
            source="oauth_invalid",
        )

    return None


def resolve_account_status_snapshot(
    *,
    provider_type: str | None,
    upstream_metadata: Any,
    oauth_invalid_reason: str | None,
) -> AccountStatusSnapshot:
    from_metadata = _resolve_from_metadata(provider_type, upstream_metadata)
    if from_metadata is not None:
        return AccountStatusSnapshot(
            code=from_metadata.code or "ok",
            label=from_metadata.label,
            reason=from_metadata.reason,
            blocked=from_metadata.blocked,
            source=from_metadata.source,
            recoverable=from_metadata.recoverable,
        )

    text = _clean_text(oauth_invalid_reason)
    if not text:
        return AccountStatusSnapshot()

    tagged_sections = _extract_tagged_reason_sections(text)
    if "ACCOUNT_BLOCK" in tagged_sections:
        cleaned = tagged_sections["ACCOUNT_BLOCK"]
        code, label = (
            _classify_block_reason(cleaned) if cleaned else ("account_blocked", "账号异常")
        )
        return AccountStatusSnapshot(
            code=code,
            label=label,
            reason=cleaned or "账号异常",
            blocked=True,
            source="oauth_invalid",
        )

    if text.startswith("["):
        return AccountStatusSnapshot()

    lowered = text.lower()
    if any(keyword in lowered for keyword in ACCOUNT_BLOCK_REASON_KEYWORDS):
        code, label = _classify_block_reason(text)
        return AccountStatusSnapshot(
            code=code,
            label=label,
            reason=text,
            blocked=True,
            source="oauth_invalid",
        )

    return AccountStatusSnapshot()


def resolve_oauth_status_snapshot(
    *,
    auth_type: str | None,
    oauth_expires_at: int | None,
    oauth_invalid_at: int | None,
    oauth_invalid_reason: str | None,
    now_ts: int | None = None,
) -> OAuthStatusSnapshot:
    if str(auth_type or "").strip().lower() != "oauth":
        return OAuthStatusSnapshot()

    now = int(now_ts if now_ts is not None else time.time())
    invalid_at = int(oauth_invalid_at) if isinstance(oauth_invalid_at, int) else None
    tagged_sections = _extract_tagged_reason_sections(oauth_invalid_reason)
    raw_reason = _clean_text(oauth_invalid_reason)

    expired_reason = tagged_sections.get("OAUTH_EXPIRED")
    if expired_reason:
        return OAuthStatusSnapshot(
            code="invalid",
            label="已失效",
            reason=expired_reason,
            invalid_at=invalid_at,
            expires_at=oauth_expires_at,
            source="oauth_invalid",
            requires_reauth=True,
        )

    refresh_failed_reason = tagged_sections.get("REFRESH_FAILED")
    if refresh_failed_reason:
        return OAuthStatusSnapshot(
            code="invalid",
            label="已失效",
            reason=refresh_failed_reason,
            invalid_at=invalid_at,
            expires_at=oauth_expires_at,
            source="oauth_refresh",
            requires_reauth=True,
        )

    request_failed_reason = tagged_sections.get("REQUEST_FAILED")
    if request_failed_reason:
        return OAuthStatusSnapshot(
            code="check_failed",
            label="检查失败",
            reason=request_failed_reason,
            expires_at=oauth_expires_at,
            source="oauth_request",
        )

    account_snapshot = resolve_account_status_snapshot(
        provider_type=None,
        upstream_metadata=None,
        oauth_invalid_reason=raw_reason,
    )
    if account_snapshot.blocked:
        if oauth_expires_at is None:
            return OAuthStatusSnapshot()
    elif raw_reason or invalid_at is not None:
        return OAuthStatusSnapshot(
            code="invalid",
            label="已失效",
            reason=raw_reason,
            invalid_at=invalid_at,
            expires_at=oauth_expires_at,
            source="oauth_invalid",
            requires_reauth=True,
        )

    expires_at = int(oauth_expires_at) if isinstance(oauth_expires_at, int) else None
    if expires_at is None:
        return OAuthStatusSnapshot()
    if expires_at <= now:
        return OAuthStatusSnapshot(
            code="expired",
            label="已过期",
            reason="Token 已过期，请重新授权",
            expires_at=expires_at,
            source="expires_at",
            requires_reauth=True,
        )
    expiring_soon = (expires_at - now) < 24 * 3600
    return OAuthStatusSnapshot(
        code="expiring" if expiring_soon else "valid",
        label="即将过期" if expiring_soon else "有效",
        expires_at=expires_at,
        source="expires_at",
        expiring_soon=expiring_soon,
    )


def resolve_quota_status_snapshot(
    *,
    provider_type: str | None,
    upstream_metadata: Any,
) -> QuotaStatusSnapshot:
    normalized_provider = str(provider_type or "").strip().lower()
    reader = get_quota_reader(normalized_provider, upstream_metadata)
    quota_state = reader.is_exhausted()
    usage_ratio = reader.usage_ratio()
    updated_at = reader.updated_at()
    reset_seconds = reader.reset_seconds()
    plan_type = reader.plan_type()

    if quota_state.exhausted:
        return QuotaStatusSnapshot(
            code="exhausted",
            label="额度耗尽",
            reason=quota_state.reason,
            exhausted=True,
            usage_ratio=usage_ratio,
            updated_at=updated_at,
            reset_seconds=reset_seconds,
            plan_type=plan_type,
        )

    if any(value is not None for value in (usage_ratio, updated_at, reset_seconds, plan_type)):
        return QuotaStatusSnapshot(
            code="ok",
            exhausted=False,
            usage_ratio=usage_ratio,
            updated_at=updated_at,
            reset_seconds=reset_seconds,
            plan_type=plan_type,
        )

    return QuotaStatusSnapshot()


def build_provider_key_status_snapshot(
    *,
    auth_type: str | None,
    oauth_expires_at: int | None,
    oauth_invalid_at: int | None,
    oauth_invalid_reason: str | None,
    provider_type: str | None,
    upstream_metadata: Any,
    now_ts: int | None = None,
) -> ProviderKeyStatusSnapshot:
    account = resolve_account_status_snapshot(
        provider_type=provider_type,
        upstream_metadata=upstream_metadata,
        oauth_invalid_reason=oauth_invalid_reason,
    )
    oauth = resolve_oauth_status_snapshot(
        auth_type=auth_type,
        oauth_expires_at=oauth_expires_at,
        oauth_invalid_at=oauth_invalid_at,
        oauth_invalid_reason=oauth_invalid_reason,
        now_ts=now_ts,
    )
    quota = resolve_quota_status_snapshot(
        provider_type=provider_type,
        upstream_metadata=upstream_metadata,
    )
    return ProviderKeyStatusSnapshot(oauth=oauth, account=account, quota=quota)


def resolve_pool_account_state(
    *,
    provider_type: str | None,
    upstream_metadata: Any,
    oauth_invalid_reason: str | None,
) -> PoolAccountState:
    """Resolve account-level hard-block state for pool scheduling."""

    from_metadata = _resolve_from_metadata(provider_type, upstream_metadata)
    if from_metadata is not None:
        return from_metadata

    from_oauth = _resolve_from_oauth_invalid_reason(oauth_invalid_reason)
    if from_oauth is not None:
        return from_oauth

    return PoolAccountState(blocked=False)


def should_auto_remove_account_state(state: PoolAccountState) -> bool:
    """Whether a resolved account state is safe to auto-remove.

    Auto-removal is limited to hard, non-recoverable account abnormalities.
    Pure token failures (`oauth_expired`, `oauth_refresh_failed`) and
    softer/manual-recoverable states like `account_verification` are excluded.
    """

    return bool(
        state.blocked
        and not state.recoverable
        and str(state.code or "").strip().lower() in AUTO_REMOVABLE_ACCOUNT_STATE_CODES
    )


__all__ = [
    "ACCOUNT_BLOCK_REASON_KEYWORDS",
    "AUTO_REMOVABLE_ACCOUNT_STATE_CODES",
    "AccountStatusSnapshot",
    "OAUTH_ACCOUNT_BLOCK_PREFIX",
    "OAUTH_EXPIRED_PREFIX",
    "OAUTH_REFRESH_FAILED_PREFIX",
    "OAUTH_REQUEST_FAILED_PREFIX",
    "OAuthStatusSnapshot",
    "PoolAccountState",
    "ProviderKeyStatusSnapshot",
    "QuotaStatusSnapshot",
    "build_provider_key_status_snapshot",
    "resolve_account_status_snapshot",
    "resolve_oauth_status_snapshot",
    "resolve_pool_account_state",
    "resolve_quota_status_snapshot",
    "should_auto_remove_account_state",
]
