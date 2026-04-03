"""
/v1/models 列表缓存管理。

从 api/base/models_service.py 迁移到 services 层，
消除 services→api 的反向依赖。
"""

from __future__ import annotations

from src.core.cache_service import CacheService
from src.core.logger import logger

# 缓存 key 前缀（models_service.py 也使用此常量）
MODELS_LIST_CACHE_PREFIX = "models:list"


async def invalidate_models_list_cache() -> None:
    """
    清除所有 /v1/models 列表缓存

    在模型创建、更新、删除时调用，确保模型列表实时更新
    """
    try:
        # 使用通配符删除所有 models:list:* 缓存（包括多格式组合的 key）
        deleted = await CacheService.delete_pattern(f"{MODELS_LIST_CACHE_PREFIX}:*")
        if deleted > 0:
            logger.info("[ModelsService] 已清除 {} 个 {} 缓存", deleted, MODELS_LIST_CACHE_PREFIX)
        else:
            logger.debug("[ModelsService] 无 {} 缓存需要清除", MODELS_LIST_CACHE_PREFIX)
    except Exception as e:
        logger.warning("[ModelsService] 清除缓存失败: {}", e)
