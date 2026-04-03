"""Backward compat shim -- canonical definitions moved to src.services.provider.pool.oauth_cache."""

from src.services.provider.pool.oauth_cache import (  # noqa: F401
    cache_token,
    get_cached_token,
    invalidate_token,
)

__all__ = ["get_cached_token", "cache_token", "invalidate_token"]
