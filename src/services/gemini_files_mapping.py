"""
Gemini Files API - 文件与 Key 绑定映射服务

用于在上传文件后记录 file_id -> provider_key_id，
并在后续 generateContent 请求中优先使用同一 Key。

存储策略：
- 数据库（持久化）：主存储，支持服务重启后恢复
- Redis（缓存）：加速读取，TTL=48小时

读取策略：
1. 先查 Redis 缓存
2. 缓存未命中时回查数据库
3. 从数据库读取后回填缓存

清理策略：
- 数据库中 expires_at 过期的记录由定时任务清理
- Redis 缓存由 TTL 自动过期
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete
from sqlalchemy.orm import Session

from src.core.cache_service import CacheService
from src.core.logger import logger

FILE_MAPPING_TTL_SECONDS = 60 * 60 * 48  # 48小时
FILE_MAPPING_CACHE_PREFIX = "gemini_files:key"


def _normalize_file_name(file_name: str) -> str:
    """规范化文件名，确保以 files/ 开头"""
    name = (file_name or "").strip()
    if not name:
        return ""
    return name if name.startswith("files/") else f"files/{name}"


def build_file_mapping_key(file_name: str) -> str:
    """构建 Redis 缓存键"""
    normalized = _normalize_file_name(file_name)
    return f"{FILE_MAPPING_CACHE_PREFIX}:{normalized}" if normalized else ""


# =============================================================================
# 异步接口（用于请求处理流程）
# =============================================================================


async def store_file_key_mapping(
    file_name: str,
    key_id: str,
    user_id: str | None = None,
    display_name: str | None = None,
    mime_type: str | None = None,
    source_hash: str | None = None,
) -> None:
    """
    存储文件→Key 映射（同时写入 Redis 和数据库）

    Args:
        file_name: 文件名（如 files/abc123）
        key_id: Provider Key ID
        user_id: 用户 ID（可选，用于权限验证）
        display_name: 文件显示名（可选）
        mime_type: 文件 MIME 类型（可选）
        source_hash: 源文件哈希（可选，用于关联相同源文件的不同上传）
    """
    normalized_name = _normalize_file_name(file_name)
    if not normalized_name or not key_id:
        return

    # 1. 写入 Redis 缓存
    cache_key = build_file_mapping_key(normalized_name)
    await CacheService.set(cache_key, str(key_id), ttl_seconds=FILE_MAPPING_TTL_SECONDS)

    # 2. 写入数据库（异步执行，不阻塞主流程）
    try:
        await _store_to_database(
            file_name=normalized_name,
            key_id=key_id,
            user_id=user_id,
            display_name=display_name,
            mime_type=mime_type,
            source_hash=source_hash,
        )
    except Exception as e:
        # 数据库写入失败只记录警告，不影响主流程
        logger.warning(f"Failed to persist Gemini file mapping to database: {e}")


async def _store_to_database(
    file_name: str,
    key_id: str,
    user_id: str | None = None,
    display_name: str | None = None,
    mime_type: str | None = None,
    source_hash: str | None = None,
) -> None:
    """将映射写入数据库"""
    from src.database import get_db_context
    from src.models.database import GeminiFileMapping

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=48)

    with get_db_context() as db:
        # 使用 upsert 逻辑：存在则更新，不存在则插入
        existing = (
            db.query(GeminiFileMapping).filter(GeminiFileMapping.file_name == file_name).first()
        )

        if existing:
            # 更新现有记录
            existing.key_id = key_id
            existing.user_id = user_id
            existing.display_name = display_name
            existing.mime_type = mime_type
            existing.source_hash = source_hash
            existing.expires_at = expires_at
        else:
            # 插入新记录
            mapping = GeminiFileMapping(
                id=str(uuid.uuid4()),
                file_name=file_name,
                key_id=key_id,
                user_id=user_id,
                display_name=display_name,
                mime_type=mime_type,
                source_hash=source_hash,
                created_at=now,
                expires_at=expires_at,
            )
            db.add(mapping)

        db.commit()


async def get_file_key_mapping(file_name: str) -> str | None:
    """
    获取文件→Key 映射

    读取策略：
    1. 先查 Redis 缓存
    2. 缓存未命中时回查数据库
    3. 从数据库读取后回填缓存

    Args:
        file_name: 文件名（如 files/abc123）

    Returns:
        Provider Key ID，如果不存在或已过期则返回 None
    """
    normalized_name = _normalize_file_name(file_name)
    if not normalized_name:
        return None

    cache_key = build_file_mapping_key(normalized_name)

    # 1. 先查 Redis 缓存
    cached_value = await CacheService.get(cache_key)
    if cached_value:
        return str(cached_value)

    # 2. 缓存未命中，回查数据库
    key_id = await _get_from_database(normalized_name)

    if key_id:
        # 3. 回填缓存（使用剩余有效期或默认 TTL）
        await CacheService.set(cache_key, key_id, ttl_seconds=FILE_MAPPING_TTL_SECONDS)
        logger.debug(f"Gemini file mapping cache refilled from database: {normalized_name}")

    return key_id


async def _get_from_database(file_name: str) -> str | None:
    """从数据库查询映射"""
    from src.database import get_db_context
    from src.models.database import GeminiFileMapping

    now = datetime.now(timezone.utc)

    try:
        with get_db_context() as db:
            mapping = (
                db.query(GeminiFileMapping)
                .filter(
                    GeminiFileMapping.file_name == file_name,
                    GeminiFileMapping.expires_at > now,  # 只返回未过期的
                )
                .first()
            )

            if mapping:
                return str(mapping.key_id)
    except Exception as e:
        logger.warning(f"Failed to query Gemini file mapping from database: {e}")

    return None


async def get_all_key_ids_for_file(file_name: str) -> list[str]:
    """
    获取支持指定文件的所有 Key ID 列表

    当同一个源文件被上传到多个 Key 时，返回所有可用的 Key ID。
    这允许系统在首选 Key 不可用时选择其他 Key。

    Args:
        file_name: 文件名（如 files/abc123）

    Returns:
        所有支持该文件的 Key ID 列表（包括原始映射和具有相同 source_hash 的映射）
    """
    from src.database import get_db_context
    from src.models.database import GeminiFileMapping

    normalized_name = _normalize_file_name(file_name)
    if not normalized_name:
        return []

    now = datetime.now(timezone.utc)

    try:
        with get_db_context() as db:
            # 首先获取原始映射
            original_mapping = (
                db.query(GeminiFileMapping)
                .filter(
                    GeminiFileMapping.file_name == normalized_name,
                    GeminiFileMapping.expires_at > now,
                )
                .first()
            )

            if not original_mapping:
                return []

            key_ids = [str(original_mapping.key_id)]

            # 如果有 source_hash，查找所有具有相同 source_hash 的映射
            if original_mapping.source_hash:
                related_mappings = (
                    db.query(GeminiFileMapping)
                    .filter(
                        GeminiFileMapping.source_hash == original_mapping.source_hash,
                        GeminiFileMapping.expires_at > now,
                        GeminiFileMapping.file_name != normalized_name,  # 排除原始映射
                    )
                    .all()
                )

                for mapping in related_mappings:
                    kid = str(mapping.key_id)
                    if kid not in key_ids:
                        key_ids.append(kid)

            return key_ids
    except Exception as e:
        logger.warning(f"Failed to query related Gemini file mappings: {e}")
        return []


async def delete_file_key_mapping(file_name: str) -> None:
    """
    删除文件→Key 映射（同时从 Redis 和数据库删除）

    Args:
        file_name: 文件名（如 files/abc123）
    """
    normalized_name = _normalize_file_name(file_name)
    if not normalized_name:
        return

    # 1. 从 Redis 删除
    cache_key = build_file_mapping_key(normalized_name)
    await CacheService.delete(cache_key)

    # 2. 从数据库删除
    try:
        await _delete_from_database(normalized_name)
    except Exception as e:
        logger.warning(f"Failed to delete Gemini file mapping from database: {e}")


async def _delete_from_database(file_name: str) -> None:
    """从数据库删除映射"""
    from src.database import get_db_context
    from src.models.database import GeminiFileMapping

    with get_db_context() as db:
        db.execute(delete(GeminiFileMapping).where(GeminiFileMapping.file_name == file_name))
        db.commit()


# =============================================================================
# 同步接口（用于定时任务等场景）
# =============================================================================


def cleanup_expired_mappings(db: Session) -> int:
    """
    清理过期的文件映射记录（同步方法，供定时任务调用）

    Args:
        db: 数据库会话

    Returns:
        删除的记录数
    """
    from src.models.database import GeminiFileMapping

    now = datetime.now(timezone.utc)

    result = db.execute(delete(GeminiFileMapping).where(GeminiFileMapping.expires_at <= now))
    db.commit()

    deleted_count = result.rowcount
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} expired Gemini file mappings")

    return deleted_count


# =============================================================================
# 请求解析工具函数
# =============================================================================


def _extract_file_name_from_uri(file_uri: str) -> str | None:
    """
    从 fileUri 提取 files/xxx 名称。

    支持两种格式：
    - 完整 URL: https://generativelanguage.googleapis.com/v1beta/files/abc123
    - 短格式: files/abc123
    """
    if not file_uri:
        return None
    # 完整 URL 格式
    if "/files/" in file_uri:
        idx = file_uri.rfind("/files/")
        return file_uri[idx + 1 :]  # 提取 files/xxx 部分
    # 短格式
    if file_uri.startswith("files/"):
        return file_uri
    return None


def extract_file_names_from_request(payload: dict[str, Any] | None) -> set[str]:
    """
    从 Gemini 请求体中提取 fileUri 使用到的 files/xxx 名称集合。
    """
    results: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            file_data = node.get("fileData") or node.get("file_data")
            if isinstance(file_data, dict):
                file_uri = file_data.get("fileUri") or file_data.get("file_uri")
                if isinstance(file_uri, str):
                    file_name = _extract_file_name_from_uri(file_uri)
                    if file_name:
                        results.add(file_name)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    if payload:
        walk(payload)

    return results
