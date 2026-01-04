"""缓存装饰器工具"""

import functools
import json
from typing import Any, Callable, Optional

from src.core.logger import logger

from src.clients.redis_client import get_redis_client_sync


def cache_result(key_prefix: str, ttl: int = 60, user_specific: bool = True) -> Callable:
    """
    缓存函数结果的装饰器

    Args:
        key_prefix: 缓存键前缀
        ttl: 缓存过期时间（秒）
        user_specific: 是否针对用户缓存（从 context.user.id 获取）
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            redis_client = get_redis_client_sync()

            # 如果 Redis 不可用，直接执行原函数
            if redis_client is None:
                return await func(*args, **kwargs)

            # 构建缓存键
            try:
                # 从 args 中获取 context（通常是第一个参数）
                context = args[0] if args else None

                if user_specific and context and hasattr(context, "user") and context.user:
                    cache_key = f"{key_prefix}:user:{context.user.id}"
                else:
                    cache_key = f"{key_prefix}:global"

                # 如果有额外的参数（如 days），添加到键中
                if hasattr(args[0], "__dict__"):
                    # 如果是 dataclass 或对象，获取其属性
                    for attr_name in ["days", "limit"]:
                        if hasattr(args[0], attr_name):
                            attr_value = getattr(args[0], attr_name)
                            cache_key += f":{attr_name}:{attr_value}"

                # 尝试从缓存获取
                cached = await redis_client.get(cache_key)
                if cached:
                    try:
                        result = json.loads(cached)
                        logger.debug(f"缓存命中: {cache_key}")
                        return result
                    except json.JSONDecodeError as e:
                        logger.warning(f"缓存解析失败，删除损坏缓存: {cache_key}, 错误: {e}")
                        try:
                            await redis_client.delete(cache_key)
                        except Exception:
                            pass

                # 执行原函数
                result = await func(*args, **kwargs)

                # 保存到缓存
                try:
                    await redis_client.setex(
                        cache_key, ttl, json.dumps(result, ensure_ascii=False, default=str)
                    )
                    logger.debug(f"缓存已保存: {cache_key}, TTL: {ttl}s")
                except Exception as e:
                    logger.warning(f"保存缓存失败: {e}")

                return result

            except Exception as e:
                logger.warning(f"缓存处理出错: {e}, 直接执行原函数")
                return await func(*args, **kwargs)

        return wrapper

    return decorator
