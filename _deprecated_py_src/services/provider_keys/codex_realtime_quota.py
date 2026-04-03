"""
Codex 配额实时同步（基于响应头）。
"""

from __future__ import annotations

import json
import time
from threading import Lock
from typing import Any

from sqlalchemy.orm import Session, joinedload

from src.core.logger import logger
from src.core.provider_types import ProviderType, normalize_provider_type
from src.models.database import ProviderAPIKey
from src.services.model.upstream_fetcher import merge_upstream_metadata
from src.services.provider_keys.codex_usage_parser import (
    CodexUsageParseError,
    parse_codex_usage_headers,
)

_COMPARE_IGNORE_FIELDS = frozenset(
    {
        "updated_at",
        "primary_reset_seconds",
        "secondary_reset_seconds",
    }
)
_CACHE_TTL_SECONDS = 30.0
_CACHE_MAX_ENTRIES = 4096
_header_fingerprint_cache: dict[str, tuple[str, float]] = {}
_cache_lock = Lock()


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _build_compare_payload(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k not in _COMPARE_IGNORE_FIELDS}


def _get_cached_fingerprint(key_id: str, now_ts: float) -> str | None:
    with _cache_lock:
        cached = _header_fingerprint_cache.get(key_id)
        if not cached:
            return None
        fp, expires_at = cached
        if expires_at <= now_ts:
            _header_fingerprint_cache.pop(key_id, None)
            return None
        return fp


def _set_cached_fingerprint(key_id: str, fingerprint: str, now_ts: float) -> None:
    with _cache_lock:
        _header_fingerprint_cache[key_id] = (fingerprint, now_ts + _CACHE_TTL_SECONDS)
        _prune_cache_locked(now_ts)


def _prune_cache_locked(now_ts: float) -> None:
    expired_keys = [
        key for key, (_, expires_at) in _header_fingerprint_cache.items() if expires_at <= now_ts
    ]
    for key in expired_keys:
        _header_fingerprint_cache.pop(key, None)

    overflow = len(_header_fingerprint_cache) - _CACHE_MAX_ENTRIES
    if overflow <= 0:
        return

    keys_by_expiry = sorted(_header_fingerprint_cache.items(), key=lambda item: item[1][1])
    for key, _ in keys_by_expiry[:overflow]:
        _header_fingerprint_cache.pop(key, None)


def sync_codex_quota_from_response_headers(
    *,
    db: Session,
    provider_api_key_id: str | None,
    response_headers: dict[str, Any] | None,
) -> bool:
    """
    从响应头同步 Codex 配额到 ProviderAPIKey.upstream_metadata。

    返回值:
    - True: 已产生数据库更新（由调用方统一 commit）
    - False: 无更新（无配额头/命中缓存/内容未变化/非 codex key）
    """
    if not provider_api_key_id or not isinstance(response_headers, dict):
        return False

    try:
        parsed = parse_codex_usage_headers(response_headers)
    except CodexUsageParseError as exc:
        logger.warning(
            "实时同步 Codex 配额头解析失败，已跳过: provider_api_key_id={}, error={}",
            provider_api_key_id,
            exc,
        )
        return False
    if not parsed:
        return False

    now_ts = time.time()
    incoming_compare = _build_compare_payload(parsed)
    incoming_fingerprint = _fingerprint_payload(incoming_compare)

    cached_fp = _get_cached_fingerprint(provider_api_key_id, now_ts)
    if cached_fp == incoming_fingerprint:
        return False

    key = (
        db.query(ProviderAPIKey)
        .options(joinedload(ProviderAPIKey.provider))
        .filter(ProviderAPIKey.id == provider_api_key_id)
        .first()
    )
    if key is None:
        _set_cached_fingerprint(provider_api_key_id, incoming_fingerprint, now_ts)
        return False

    provider_type = normalize_provider_type(
        getattr(getattr(key, "provider", None), "provider_type", None)
    )
    if provider_type != ProviderType.CODEX:
        _set_cached_fingerprint(provider_api_key_id, incoming_fingerprint, now_ts)
        return False

    current_metadata = key.upstream_metadata if isinstance(key.upstream_metadata, dict) else {}
    current_codex = current_metadata.get("codex")
    if not isinstance(current_codex, dict):
        current_codex = {}

    merged_codex = dict(current_codex)
    merged_codex.update(parsed)

    current_fingerprint = _fingerprint_payload(_build_compare_payload(current_codex))
    merged_fingerprint = _fingerprint_payload(_build_compare_payload(merged_codex))
    if current_fingerprint == merged_fingerprint:
        _set_cached_fingerprint(provider_api_key_id, merged_fingerprint, now_ts)
        return False

    key.upstream_metadata = merge_upstream_metadata(
        current_metadata,
        {
            "codex": merged_codex,
        },
    )
    db.add(key)
    _set_cached_fingerprint(provider_api_key_id, merged_fingerprint, now_ts)
    return True
