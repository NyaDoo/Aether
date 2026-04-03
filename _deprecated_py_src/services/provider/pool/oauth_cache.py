"""OAuth token Redis cache for the Account Pool.

Additions over the base ``auth.py`` refresh flow:

- **Redis token cache**: Avoids repeated DB decryption for hot keys.
  Cache key: ``provider_oauth_token_cache:{key_id}``
- **Configurable proactive refresh skew**: Default 180 s (3 min) instead
  of the base 120 s, configurable via ``PoolConfig.proactive_refresh_seconds``.
- **401 immediate invalidation**: Clears the Redis cache so the next request
  triggers a fresh refresh.

This module does NOT replace ``auth.py``; it adds a caching layer that
``auth.py`` can consult before decrypting from DB.
"""

from __future__ import annotations

from src.core.logger import logger
from src.services.provider.pool import redis_ops


async def get_cached_token(key_id: str) -> str | None:
    """Return cached access token from Redis, or None."""
    return await redis_ops.get_cached_oauth_token(key_id)


async def cache_token(key_id: str, token: str, expires_in_seconds: int) -> None:
    """Cache an access token in Redis.

    *expires_in_seconds* is the remaining lifetime of the token.  We shave
    off 60 s so the cache expires slightly before the token itself, giving
    the refresh flow time to act.
    """
    ttl = max(1, expires_in_seconds - 60)
    await redis_ops.cache_oauth_token(key_id, token, ttl)
    logger.debug("Pool OAuth: cached token for key {} (TTL={}s)", key_id[:8], ttl)


async def invalidate_token(key_id: str) -> None:
    """Invalidate the cached token (e.g. after a 401)."""
    await redis_ops.invalidate_oauth_token_cache(key_id)
    logger.debug("Pool OAuth: invalidated token cache for key {}", key_id[:8])
