"""
Gemini Files 管理 API

提供文件映射的管理功能：
- 列出所有文件映射
- 删除文件映射
- 查看文件映射统计
- 上传文件到 Gemini

优化：HTTP 上传期间不持有数据库连接，避免阻塞其他请求。
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, func
from sqlalchemy.orm import Session

from src.clients.http_client import HTTPClientPool
from src.core.crypto import crypto_service
from src.core.logger import logger
from src.database import create_session, get_db
from src.models.database import GeminiFileMapping, ProviderAPIKey, User
from src.services.gemini_files_mapping import delete_file_key_mapping, store_file_key_mapping


@dataclass
class KeyInfo:
    """Key 信息（用于 HTTP 上传，不依赖数据库会话）"""

    id: str
    name: str | None
    decrypted_api_key: str


router = APIRouter(prefix="/api/admin/gemini-files", tags=["Gemini Files Management"])

# Gemini Files API 基础 URL
GEMINI_FILES_BASE_URL = "https://generativelanguage.googleapis.com"


# ============ Schema ============


class FileMappingResponse(BaseModel):
    """文件映射响应"""

    id: str
    file_name: str
    key_id: str
    key_name: str | None = None
    user_id: str | None = None
    username: str | None = None
    display_name: str | None = None
    mime_type: str | None = None
    created_at: datetime
    expires_at: datetime
    is_expired: bool


class FileMappingListResponse(BaseModel):
    """文件映射列表响应"""

    items: list[FileMappingResponse]
    total: int
    page: int
    page_size: int


class FileMappingStatsResponse(BaseModel):
    """文件映射统计响应"""

    total_mappings: int
    active_mappings: int
    expired_mappings: int
    by_mime_type: dict[str, int]
    capable_keys_count: int


# ============ Routes ============


@router.get("/mappings", response_model=FileMappingListResponse)
async def list_file_mappings(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_expired: bool = Query(False),
    search: str | None = Query(None),
) -> Any:
    """
    列出所有文件映射

    - **page**: 页码
    - **page_size**: 每页数量
    - **include_expired**: 是否包含已过期的映射
    - **search**: 搜索文件名或显示名
    """
    now = datetime.now(timezone.utc)

    query = db.query(GeminiFileMapping)

    # 过滤过期
    if not include_expired:
        query = query.filter(GeminiFileMapping.expires_at > now)

    # 搜索
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (GeminiFileMapping.file_name.ilike(search_pattern))
            | (GeminiFileMapping.display_name.ilike(search_pattern))
        )

    # 总数
    total = query.count()

    # 分页
    offset = (page - 1) * page_size
    mappings = (
        query.order_by(GeminiFileMapping.created_at.desc()).offset(offset).limit(page_size).all()
    )

    # 获取关联的 Key 和 User 信息
    key_ids = {m.key_id for m in mappings}
    user_ids = {m.user_id for m in mappings if m.user_id}

    keys_map = {}
    if key_ids:
        keys = db.query(ProviderAPIKey).filter(ProviderAPIKey.id.in_(key_ids)).all()
        keys_map = {str(k.id): k.name for k in keys}

    users_map = {}
    if user_ids:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        users_map = {str(u.id): u.username for u in users}

    items = []
    for m in mappings:
        items.append(
            FileMappingResponse(
                id=str(m.id),
                file_name=m.file_name,
                key_id=str(m.key_id),
                key_name=keys_map.get(str(m.key_id)),
                user_id=str(m.user_id) if m.user_id else None,
                username=users_map.get(str(m.user_id)) if m.user_id else None,
                display_name=m.display_name,
                mime_type=m.mime_type,
                created_at=m.created_at,
                expires_at=m.expires_at,
                is_expired=m.expires_at <= now,
            )
        )

    return FileMappingListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/stats", response_model=FileMappingStatsResponse)
async def get_file_mapping_stats(
    db: Session = Depends(get_db),
) -> Any:
    """获取文件映射统计信息"""
    now = datetime.now(timezone.utc)

    # 总数
    total_mappings = db.query(func.count(GeminiFileMapping.id)).scalar() or 0

    # 活跃数（未过期）
    active_mappings = (
        db.query(func.count(GeminiFileMapping.id))
        .filter(GeminiFileMapping.expires_at > now)
        .scalar()
        or 0
    )

    # 过期数
    expired_mappings = total_mappings - active_mappings

    # 按 MIME 类型统计
    mime_stats = (
        db.query(GeminiFileMapping.mime_type, func.count(GeminiFileMapping.id))
        .filter(GeminiFileMapping.expires_at > now)
        .group_by(GeminiFileMapping.mime_type)
        .all()
    )
    by_mime_type = {(mt or "unknown"): count for mt, count in mime_stats}

    # 有 gemini_files 能力的 Key 数量
    keys = db.query(ProviderAPIKey).filter(ProviderAPIKey.is_active.is_(True)).all()
    capable_keys_count = sum(
        1 for key in keys if key.capabilities and key.capabilities.get("gemini_files", False)
    )

    return FileMappingStatsResponse(
        total_mappings=total_mappings,
        active_mappings=active_mappings,
        expired_mappings=expired_mappings,
        by_mime_type=by_mime_type,
        capable_keys_count=capable_keys_count,
    )


@router.delete("/mappings/{mapping_id}")
async def delete_mapping(
    mapping_id: str,
    db: Session = Depends(get_db),
) -> Any:
    """
    删除指定的文件映射

    注意：这只删除映射记录，不会删除 Gemini 上的实际文件
    """
    mapping = db.query(GeminiFileMapping).filter(GeminiFileMapping.id == mapping_id).first()

    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    file_name = mapping.file_name

    # 从数据库删除
    db.delete(mapping)
    db.commit()

    # 同时从 Redis 删除
    await delete_file_key_mapping(file_name)

    return {"message": "Mapping deleted successfully", "file_name": file_name}


@router.delete("/mappings")
async def cleanup_expired_mappings(
    db: Session = Depends(get_db),
) -> Any:
    """清理所有过期的文件映射"""
    now = datetime.now(timezone.utc)

    result = db.execute(delete(GeminiFileMapping).where(GeminiFileMapping.expires_at <= now))
    db.commit()

    deleted_count = result.rowcount

    return {
        "message": f"Cleaned up {deleted_count} expired mappings",
        "deleted_count": deleted_count,
    }


class CapableKeyResponse(BaseModel):
    """可用 Key 响应"""

    id: str
    name: str
    provider_name: str | None = None


class UploadResultItem(BaseModel):
    """单个 Key 的上传结果"""

    key_id: str
    key_name: str | None = None
    success: bool
    file_name: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    """上传响应"""

    display_name: str
    mime_type: str
    size_bytes: int
    results: list[UploadResultItem]
    success_count: int
    fail_count: int


@router.get("/capable-keys", response_model=list[CapableKeyResponse])
async def list_capable_keys(
    db: Session = Depends(get_db),
) -> Any:
    """获取所有具有 gemini_files 能力的 Key 列表"""
    from src.models.database import Provider

    keys = db.query(ProviderAPIKey).filter(ProviderAPIKey.is_active.is_(True)).all()
    capable_keys = [
        key for key in keys if key.capabilities and key.capabilities.get("gemini_files", False)
    ]

    # 获取 Provider 名称
    provider_ids = {key.provider_id for key in capable_keys}
    providers = db.query(Provider).filter(Provider.id.in_(provider_ids)).all()
    provider_map = {str(p.id): p.name for p in providers}

    return [
        CapableKeyResponse(
            id=str(key.id),
            name=key.name,
            provider_name=provider_map.get(str(key.provider_id)),
        )
        for key in capable_keys
    ]


async def _upload_to_key(
    key_info: KeyInfo,
    content: bytes,
    file_size: int,
    mime_type: str,
    display_name: str,
    source_hash: str,
) -> UploadResultItem:
    """上传文件到指定的 Key（不依赖数据库会话）"""
    api_key = key_info.decrypted_api_key
    client = await HTTPClientPool.get_default_client_async()

    try:
        # 第一步：初始化可恢复上传
        init_url = f"{GEMINI_FILES_BASE_URL}/upload/v1beta/files?key={api_key}"
        init_headers = {
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        init_body = {"file": {"display_name": display_name}}

        init_response = await client.post(init_url, headers=init_headers, json=init_body)

        if init_response.status_code != 200:
            logger.error(
                f"Gemini upload init failed for key {key_info.id}: {init_response.status_code}"
            )
            return UploadResultItem(
                key_id=key_info.id,
                key_name=key_info.name,
                success=False,
                error=f"初始化失败: {init_response.status_code}",
            )

        # 获取上传 URL
        upload_url = init_response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            return UploadResultItem(
                key_id=key_info.id,
                key_name=key_info.name,
                success=False,
                error="未获取到上传 URL",
            )

        # 第二步：上传文件内容
        upload_headers = {
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Length": str(file_size),
            "Content-Type": mime_type,
        }

        upload_response = await client.post(upload_url, headers=upload_headers, content=content)

        if upload_response.status_code != 200:
            logger.error(
                f"Gemini upload failed for key {key_info.id}: {upload_response.status_code}"
            )
            return UploadResultItem(
                key_id=key_info.id,
                key_name=key_info.name,
                success=False,
                error=f"上传失败: {upload_response.status_code}",
            )

        # 解析响应
        result = upload_response.json()
        file_info = result.get("file", {})
        file_name = file_info.get("name", "")
        response_display_name = file_info.get("displayName", display_name)
        response_mime_type = file_info.get("mimeType", mime_type)

        # 存储文件映射（包含源文件哈希，用于关联相同源文件的不同上传）
        await store_file_key_mapping(
            file_name=file_name,
            key_id=key_info.id,
            user_id=None,
            display_name=response_display_name,
            mime_type=response_mime_type,
            source_hash=source_hash,
        )

        return UploadResultItem(
            key_id=key_info.id,
            key_name=key_info.name,
            success=True,
            file_name=file_name,
        )

    except Exception as exc:
        logger.error(f"Gemini upload error for key {key_info.id}: {exc}")
        return UploadResultItem(
            key_id=key_info.id,
            key_name=key_info.name,
            success=False,
            error=str(exc),
        )


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    key_ids: str = Query(..., description="逗号分隔的 Key ID 列表"),
) -> Any:
    """
    上传文件到 Gemini Files API

    - **file**: 要上传的文件
    - **key_ids**: 逗号分隔的 Key ID 列表，文件将上传到所有指定的 Key

    支持的文件类型：视频、图片、音频、文档等
    文件大小限制：2GB
    文件有效期：48小时

    优化：HTTP 上传期间不持有数据库连接
    """
    # 解析 Key IDs
    key_id_list = [kid.strip() for kid in key_ids.split(",") if kid.strip()]
    if not key_id_list:
        raise HTTPException(status_code=400, detail="请至少选择一个 Key")

    # ========== 阶段 1：读取文件内容并计算哈希 ==========
    content = await file.read()
    file_size = len(content)
    mime_type = file.content_type or "application/octet-stream"
    display_name = file.filename or "uploaded_file"

    # 计算源文件哈希（用于重复检测和关联相同源文件的不同上传）
    source_hash = hashlib.sha256(content).hexdigest()

    # ========== 阶段 2：查询数据库（短暂持有连接）==========
    key_infos: list[KeyInfo] = []
    existing_mappings: dict[str, str] = {}  # key_id -> existing file_name

    with create_session() as db:
        keys = (
            db.query(ProviderAPIKey)
            .filter(
                ProviderAPIKey.id.in_(key_id_list),
                ProviderAPIKey.is_active.is_(True),
            )
            .all()
        )

        # 过滤有 gemini_files 能力的 Key，并提取必要信息
        capable_key_ids = []
        for key in keys:
            if key.capabilities and key.capabilities.get("gemini_files", False):
                try:
                    decrypted_key = crypto_service.decrypt(key.api_key)
                    key_infos.append(
                        KeyInfo(
                            id=str(key.id),
                            name=key.name,
                            decrypted_api_key=decrypted_key,
                        )
                    )
                    capable_key_ids.append(str(key.id))
                except Exception as exc:
                    logger.error(f"Failed to decrypt provider key {key.id}: {exc}")

        # 检查是否已存在相同 source_hash 的映射（重复检测）
        if capable_key_ids:
            now = datetime.now(timezone.utc)
            existing = (
                db.query(GeminiFileMapping)
                .filter(
                    GeminiFileMapping.source_hash == source_hash,
                    GeminiFileMapping.key_id.in_(capable_key_ids),
                    GeminiFileMapping.expires_at > now,  # 只查未过期的
                )
                .all()
            )
            for mapping in existing:
                existing_mappings[str(mapping.key_id)] = mapping.file_name

    if not key_infos:
        raise HTTPException(
            status_code=400,
            detail="选中的 Key 都没有「Gemini 文件 API」能力或解密失败",
        )

    # ========== 阶段 3：并发上传（跳过已有相同文件的 Key）==========
    results: list[UploadResultItem] = []
    keys_to_upload: list[KeyInfo] = []

    for key_info in key_infos:
        if key_info.id in existing_mappings:
            # 该 Key 已有相同文件，跳过上传
            results.append(
                UploadResultItem(
                    key_id=key_info.id,
                    key_name=key_info.name,
                    success=True,
                    file_name=existing_mappings[key_info.id],
                    error=None,
                )
            )
            logger.info(
                f"跳过重复上传: Key {key_info.id} 已有文件 {existing_mappings[key_info.id]}"
            )
        else:
            keys_to_upload.append(key_info)

    # 只对需要上传的 Key 执行上传
    if keys_to_upload:
        tasks = [
            _upload_to_key(key_info, content, file_size, mime_type, display_name, source_hash)
            for key_info in keys_to_upload
        ]
        upload_results = await asyncio.gather(*tasks)
        results.extend(upload_results)

    success_count = sum(1 for r in results if r.success)
    fail_count = len(results) - success_count

    return UploadResponse(
        display_name=display_name,
        mime_type=mime_type,
        size_bytes=file_size,
        results=results,
        success_count=success_count,
        fail_count=fail_count,
    )
