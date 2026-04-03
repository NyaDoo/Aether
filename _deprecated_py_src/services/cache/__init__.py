"""通用缓存模块。

包含缓存后端、缓存失效与缓存同步等能力（backend/sync/*_cache）。

调度/候选/缓存亲和性相关逻辑已迁移到 `src.services.scheduling`。
"""

from src.services.cache.backend import BaseCacheBackend, LocalCache, RedisCache, get_cache_backend

__all__ = [
    "BaseCacheBackend",
    "LocalCache",
    "RedisCache",
    "get_cache_backend",
]
