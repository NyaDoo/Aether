from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Connection

from src.core.crypto import crypto_service
from src.models.database import Provider, ProviderAPIKey
from src.services.provider.pool.account_state import (
    AccountStatusSnapshot,
    OAuthStatusSnapshot,
    ProviderKeyStatusSnapshot,
    QuotaStatusSnapshot,
    build_provider_key_status_snapshot,
    resolve_oauth_status_snapshot,
)


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None
    return None


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def extract_oauth_auth_config(key: ProviderAPIKey) -> dict[str, Any] | None:
    if str(getattr(key, "auth_type", "") or "").strip().lower() != "oauth":
        return None

    auth_config_raw = getattr(key, "auth_config", None)
    if not auth_config_raw:
        return None

    try:
        decrypted = crypto_service.decrypt(auth_config_raw)
        if isinstance(decrypted, str) and decrypted.strip():
            parsed = json.loads(decrypted)
            if isinstance(parsed, dict):
                return parsed
    except Exception:
        pass

    return None


def normalize_oauth_expires_at(raw: Any) -> int | None:
    value = _coerce_float(raw)
    if value is None or value <= 0:
        return None
    if value > 1_000_000_000_000:
        value /= 1000
    return int(value)


def hydrate_provider_key_status_snapshot(raw: Any) -> ProviderKeyStatusSnapshot | None:
    if not isinstance(raw, dict):
        return None

    oauth_raw = raw.get("oauth") if isinstance(raw.get("oauth"), dict) else {}
    account_raw = raw.get("account") if isinstance(raw.get("account"), dict) else {}
    quota_raw = raw.get("quota") if isinstance(raw.get("quota"), dict) else {}

    return ProviderKeyStatusSnapshot(
        oauth=OAuthStatusSnapshot(
            code=_clean_text(oauth_raw.get("code")) or "none",
            label=_clean_text(oauth_raw.get("label")),
            reason=_clean_text(oauth_raw.get("reason")),
            expires_at=_coerce_int(oauth_raw.get("expires_at")),
            invalid_at=_coerce_int(oauth_raw.get("invalid_at")),
            source=_clean_text(oauth_raw.get("source")),
            requires_reauth=_coerce_bool(oauth_raw.get("requires_reauth")),
            expiring_soon=_coerce_bool(oauth_raw.get("expiring_soon")),
        ),
        account=AccountStatusSnapshot(
            code=_clean_text(account_raw.get("code")) or "ok",
            label=_clean_text(account_raw.get("label")),
            reason=_clean_text(account_raw.get("reason")),
            blocked=_coerce_bool(account_raw.get("blocked")),
            source=_clean_text(account_raw.get("source")),
            recoverable=_coerce_bool(account_raw.get("recoverable")),
        ),
        quota=QuotaStatusSnapshot(
            code=_clean_text(quota_raw.get("code")) or "unknown",
            label=_clean_text(quota_raw.get("label")),
            reason=_clean_text(quota_raw.get("reason")),
            exhausted=_coerce_bool(quota_raw.get("exhausted")),
            usage_ratio=_coerce_float(quota_raw.get("usage_ratio")),
            updated_at=_coerce_int(quota_raw.get("updated_at")),
            reset_seconds=_coerce_float(quota_raw.get("reset_seconds")),
            plan_type=_clean_text(quota_raw.get("plan_type")),
        ),
    )


def resolve_provider_type_for_key(
    key: ProviderAPIKey,
    *,
    provider_type: str | None = None,
    connection: Connection | None = None,
) -> str | None:
    normalized = _clean_text(provider_type)
    if normalized:
        return normalized

    provider_rel = getattr(key, "__dict__", {}).get("provider")
    rel_type = _clean_text(getattr(provider_rel, "provider_type", None)) or _clean_text(
        getattr(provider_rel, "type", None)
    )
    if rel_type:
        return rel_type

    provider_id = _clean_text(getattr(key, "provider_id", None))
    if provider_id and connection is not None:
        result = connection.execute(
            select(Provider.provider_type).where(Provider.id == provider_id)
        ).scalar_one_or_none()
        return _clean_text(result)

    return None


def derive_oauth_expires_at(
    key: ProviderAPIKey,
    *,
    auth_config: dict[str, Any] | None = None,
) -> int | None:
    if str(getattr(key, "auth_type", "") or "").strip().lower() != "oauth":
        return None

    cfg = auth_config if isinstance(auth_config, dict) else extract_oauth_auth_config(key)
    if cfg:
        for field in ("expires_at", "expiresAt", "expiry", "exp"):
            expires_at = normalize_oauth_expires_at(cfg.get(field))
            if expires_at is not None:
                return expires_at

    expires_dt = getattr(key, "expires_at", None)
    if isinstance(expires_dt, datetime):
        return int(expires_dt.timestamp())
    return None


def resolve_provider_key_status_snapshot(
    key: ProviderAPIKey,
    *,
    provider_type: str | None = None,
    connection: Connection | None = None,
    auth_config: dict[str, Any] | None = None,
    oauth_expires_at: int | None = None,
    now_ts: int | None = None,
) -> ProviderKeyStatusSnapshot:
    persisted_snapshot = hydrate_provider_key_status_snapshot(getattr(key, "status_snapshot", None))
    current_snapshot = _build_snapshot_from_current_fields(
        key,
        provider_type=provider_type,
        connection=connection,
        auth_config=auth_config,
        oauth_expires_at=oauth_expires_at,
        now_ts=now_ts,
    )
    if persisted_snapshot is None:
        return current_snapshot

    resolved_oauth_expires_at = current_snapshot.oauth.expires_at
    if resolved_oauth_expires_at is None and persisted_snapshot.oauth.expires_at is not None:
        resolved_oauth_expires_at = int(persisted_snapshot.oauth.expires_at)
    resolved_oauth_invalid_at = current_snapshot.oauth.invalid_at
    if resolved_oauth_invalid_at is None and persisted_snapshot.oauth.invalid_at is not None:
        resolved_oauth_invalid_at = int(persisted_snapshot.oauth.invalid_at)
    oauth_invalid_reason = _clean_text(getattr(key, "oauth_invalid_reason", None)) or (
        persisted_snapshot.oauth.reason if persisted_snapshot is not None else None
    )

    return ProviderKeyStatusSnapshot(
        oauth=resolve_oauth_status_snapshot(
            auth_type=str(getattr(key, "auth_type", "api_key") or "api_key"),
            oauth_expires_at=resolved_oauth_expires_at,
            oauth_invalid_at=resolved_oauth_invalid_at,
            oauth_invalid_reason=oauth_invalid_reason,
            now_ts=now_ts,
        ),
        account=persisted_snapshot.account,
        quota=persisted_snapshot.quota,
    )


def _build_snapshot_from_current_fields(
    key: ProviderAPIKey,
    *,
    provider_type: str | None = None,
    connection: Connection | None = None,
    auth_config: dict[str, Any] | None = None,
    oauth_expires_at: int | None = None,
    now_ts: int | None = None,
) -> ProviderKeyStatusSnapshot:
    resolved_provider_type = resolve_provider_type_for_key(
        key, provider_type=provider_type, connection=connection
    )
    oauth_auth_config = (
        auth_config if isinstance(auth_config, dict) else extract_oauth_auth_config(key)
    )
    normalized_oauth_expires_at = normalize_oauth_expires_at(oauth_expires_at)
    resolved_oauth_expires_at = (
        normalized_oauth_expires_at
        if normalized_oauth_expires_at is not None
        else derive_oauth_expires_at(
            key,
            auth_config=oauth_auth_config,
        )
    )
    raw_invalid_at = getattr(key, "oauth_invalid_at", None)
    oauth_invalid_at = (
        int(raw_invalid_at.timestamp()) if isinstance(raw_invalid_at, datetime) else None
    )
    oauth_invalid_reason = _clean_text(getattr(key, "oauth_invalid_reason", None))

    return build_provider_key_status_snapshot(
        auth_type=str(getattr(key, "auth_type", "api_key") or "api_key"),
        oauth_expires_at=resolved_oauth_expires_at,
        oauth_invalid_at=oauth_invalid_at,
        oauth_invalid_reason=oauth_invalid_reason,
        provider_type=resolved_provider_type,
        upstream_metadata=getattr(key, "upstream_metadata", None),
        now_ts=now_ts,
    )


def sync_provider_key_status_snapshot(
    key: ProviderAPIKey,
    *,
    provider_type: str | None = None,
    connection: Connection | None = None,
    auth_config: dict[str, Any] | None = None,
    oauth_expires_at: int | None = None,
) -> dict[str, Any]:
    snapshot = _build_snapshot_from_current_fields(
        key,
        provider_type=provider_type,
        connection=connection,
        auth_config=auth_config,
        oauth_expires_at=oauth_expires_at,
    )
    snapshot_dict = asdict(snapshot)
    key.status_snapshot = snapshot_dict
    return snapshot_dict
