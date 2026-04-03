"""Helpers for resolving provider metadata without touching detached ORM relations."""

from __future__ import annotations

import time
from typing import Any


def _safe_getattr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    try:
        return getattr(obj, attr)
    except Exception:
        # Intentionally broad: ORM objects may raise DetachedInstanceError,
        # MissingGreenlet, or other SQLAlchemy errors when accessing
        # lazy-loaded attributes on expired/detached objects.
        return None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _extract_provider_id(*, endpoint: Any | None = None, key: Any | None = None) -> str | None:
    for source in (key, endpoint):
        provider_id = _normalize_text(_safe_getattr(source, "provider_id"))
        if provider_id:
            return provider_id

    for provider_obj in (_safe_getattr(endpoint, "provider"), _safe_getattr(key, "provider")):
        provider_id = _normalize_text(_safe_getattr(provider_obj, "id"))
        if provider_id:
            return provider_id

    return None


_SNAPSHOT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SNAPSHOT_TTL: float = 30.0  # seconds


def _load_provider_snapshot(provider_id: str | None) -> dict[str, Any] | None:
    normalized_id = _normalize_text(provider_id)
    if not normalized_id:
        return None

    now = time.monotonic()
    cached = _SNAPSHOT_CACHE.get(normalized_id)
    if cached is not None and (now - cached[0]) < _SNAPSHOT_TTL:
        return cached[1]

    from src.database import create_session
    from src.models.database import Provider

    with create_session() as db:
        row = db.query(Provider).filter(Provider.id == normalized_id).first()
        if row is None:
            return None
        snapshot = {
            "provider_type": _safe_getattr(row, "provider_type"),
            "proxy": _safe_getattr(row, "proxy"),
        }
        _SNAPSHOT_CACHE[normalized_id] = (now, snapshot)
        return snapshot


def resolve_provider_type(
    *,
    endpoint: Any | None = None,
    key: Any | None = None,
    explicit_provider_type: str | None = None,
    decrypted_auth_config: dict[str, Any] | None = None,
) -> str | None:
    provider_type = _normalize_text(explicit_provider_type).lower()
    if provider_type:
        return provider_type

    for source in (endpoint, key):
        provider_type = _normalize_text(_safe_getattr(source, "provider_type")).lower()
        if provider_type:
            return provider_type

    for provider_obj in (_safe_getattr(endpoint, "provider"), _safe_getattr(key, "provider")):
        provider_type = _normalize_text(_safe_getattr(provider_obj, "provider_type")).lower()
        if provider_type:
            return provider_type

    if isinstance(decrypted_auth_config, dict):
        provider_type = _normalize_text(decrypted_auth_config.get("provider_type")).lower()
        if provider_type:
            return provider_type

    snapshot = _load_provider_snapshot(_extract_provider_id(endpoint=endpoint, key=key))
    provider_type = _normalize_text((snapshot or {}).get("provider_type")).lower()
    return provider_type or None


def resolve_provider_proxy(
    *,
    endpoint: Any | None = None,
    key: Any | None = None,
) -> dict[str, Any] | None:
    for source in (endpoint, key):
        provider_proxy = _safe_getattr(source, "provider_proxy")
        if isinstance(provider_proxy, dict):
            return provider_proxy

    for provider_obj in (_safe_getattr(endpoint, "provider"), _safe_getattr(key, "provider")):
        provider_proxy = _safe_getattr(provider_obj, "proxy")
        if isinstance(provider_proxy, dict):
            return provider_proxy

    snapshot = _load_provider_snapshot(_extract_provider_id(endpoint=endpoint, key=key))
    provider_proxy = (snapshot or {}).get("proxy")
    return provider_proxy if isinstance(provider_proxy, dict) else None
