"""
Gemini Files 管理 API

提供文件映射管理与能力查询；上传入口已收成 Rust-only 兼容壳。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, func
from sqlalchemy.orm import Session, load_only

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.database import get_db
from src.models.database import GeminiFileMapping, ProviderAPIKey, User
from src.services.gemini_files_mapping import delete_file_key_mapping

router = APIRouter(prefix="/api/admin/gemini-files", tags=["Gemini Files Management"])
pipeline = get_pipeline()

_RUST_UPLOADER_DETAIL = "Admin Gemini file upload requires Rust uploader"


class FileMappingResponse(BaseModel):
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
    items: list[FileMappingResponse]
    total: int
    page: int
    page_size: int


class FileMappingStatsResponse(BaseModel):
    total_mappings: int
    active_mappings: int
    expired_mappings: int
    by_mime_type: dict[str, int]
    capable_keys_count: int


class CapableKeyResponse(BaseModel):
    id: str
    name: str
    provider_name: str | None = None


class UploadResultItem(BaseModel):
    key_id: str
    key_name: str | None = None
    success: bool
    file_name: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    display_name: str
    mime_type: str
    size_bytes: int
    results: list[UploadResultItem]
    success_count: int
    fail_count: int


async def _list_file_mappings_response(
    *,
    db: Session,
    page: int,
    page_size: int,
    include_expired: bool,
    search: str | None,
) -> FileMappingListResponse:
    now = datetime.now(timezone.utc)

    query = db.query(GeminiFileMapping)
    count_query = db.query(func.count(GeminiFileMapping.id))

    if not include_expired:
        active_filter = GeminiFileMapping.expires_at > now
        query = query.filter(active_filter)
        count_query = count_query.filter(active_filter)

    if search:
        search_pattern = f"%{search}%"
        search_filter = (GeminiFileMapping.file_name.ilike(search_pattern)) | (
            GeminiFileMapping.display_name.ilike(search_pattern)
        )
        query = query.filter(search_filter)
        count_query = count_query.filter(search_filter)

    total = int(count_query.scalar() or 0)
    offset = (page - 1) * page_size
    mappings = (
        query.options(
            load_only(
                GeminiFileMapping.id,
                GeminiFileMapping.file_name,
                GeminiFileMapping.key_id,
                GeminiFileMapping.user_id,
                GeminiFileMapping.display_name,
                GeminiFileMapping.mime_type,
                GeminiFileMapping.created_at,
                GeminiFileMapping.expires_at,
            )
        )
        .order_by(GeminiFileMapping.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    key_ids = {m.key_id for m in mappings}
    user_ids = {m.user_id for m in mappings if m.user_id}

    keys_map: dict[str, str | None] = {}
    if key_ids:
        keys = (
            db.query(ProviderAPIKey)
            .options(load_only(ProviderAPIKey.id, ProviderAPIKey.name))
            .filter(ProviderAPIKey.id.in_(key_ids))
            .all()
        )
        keys_map = {str(k.id): k.name for k in keys}

    users_map: dict[str, str | None] = {}
    if user_ids:
        users = (
            db.query(User)
            .options(load_only(User.id, User.username))
            .filter(User.id.in_(user_ids))
            .all()
        )
        users_map = {str(u.id): u.username for u in users}

    return FileMappingListResponse(
        items=[
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
            for m in mappings
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def _get_file_mapping_stats_response(*, db: Session) -> FileMappingStatsResponse:
    now = datetime.now(timezone.utc)
    total_mappings = db.query(func.count(GeminiFileMapping.id)).scalar() or 0
    active_mappings = (
        db.query(func.count(GeminiFileMapping.id))
        .filter(GeminiFileMapping.expires_at > now)
        .scalar()
        or 0
    )
    expired_mappings = total_mappings - active_mappings
    mime_stats = (
        db.query(GeminiFileMapping.mime_type, func.count(GeminiFileMapping.id))
        .filter(GeminiFileMapping.expires_at > now)
        .group_by(GeminiFileMapping.mime_type)
        .all()
    )
    by_mime_type = {(mime_type or "unknown"): count for mime_type, count in mime_stats}
    keys = db.query(ProviderAPIKey.capabilities).filter(ProviderAPIKey.is_active.is_(True)).all()
    capable_keys_count = sum(
        1
        for (capabilities,) in keys
        if isinstance(capabilities, dict) and capabilities.get("gemini_files", False)
    )
    return FileMappingStatsResponse(
        total_mappings=total_mappings,
        active_mappings=active_mappings,
        expired_mappings=expired_mappings,
        by_mime_type=by_mime_type,
        capable_keys_count=capable_keys_count,
    )


async def _delete_mapping_response(*, db: Session, mapping_id: str) -> dict[str, Any]:
    mapping = db.query(GeminiFileMapping).filter(GeminiFileMapping.id == mapping_id).first()
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    file_name = mapping.file_name
    db.delete(mapping)
    db.commit()
    await delete_file_key_mapping(file_name)
    return {"message": "Mapping deleted successfully", "file_name": file_name}


async def _cleanup_expired_mappings_response(*, db: Session) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    result = db.execute(delete(GeminiFileMapping).where(GeminiFileMapping.expires_at <= now))
    db.commit()
    deleted_count = result.rowcount
    return {
        "message": f"Cleaned up {deleted_count} expired mappings",
        "deleted_count": deleted_count,
    }


async def _list_capable_keys_response(*, db: Session) -> list[CapableKeyResponse]:
    from src.models.database import Provider

    key_rows = (
        db.query(
            ProviderAPIKey.id,
            ProviderAPIKey.name,
            ProviderAPIKey.provider_id,
            ProviderAPIKey.capabilities,
        )
        .filter(ProviderAPIKey.is_active.is_(True))
        .all()
    )
    capable_keys = [
        key
        for key in key_rows
        if isinstance(key.capabilities, dict) and key.capabilities.get("gemini_files", False)
    ]

    provider_ids = {key.provider_id for key in capable_keys if key.provider_id}
    provider_map: dict[str, str] = {}
    if provider_ids:
        providers = db.query(Provider.id, Provider.name).filter(Provider.id.in_(provider_ids)).all()
        provider_map = {str(provider_id): provider_name for provider_id, provider_name in providers}

    return [
        CapableKeyResponse(
            id=str(key.id),
            name=key.name,
            provider_name=provider_map.get(str(key.provider_id)),
        )
        for key in capable_keys
    ]


async def _upload_file_response(*, file: UploadFile, key_ids: str) -> Any:
    del file, key_ids
    raise HTTPException(status_code=503, detail=_RUST_UPLOADER_DETAIL)


@dataclass
class AdminGeminiFilesListMappingsAdapter(AdminApiAdapter):
    page: int
    page_size: int
    include_expired: bool
    search: str | None

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _list_file_mappings_response(
            db=context.db,
            page=self.page,
            page_size=self.page_size,
            include_expired=self.include_expired,
            search=self.search,
        )


class AdminGeminiFilesStatsAdapter(AdminApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _get_file_mapping_stats_response(db=context.db)


@dataclass
class AdminGeminiFilesDeleteMappingAdapter(AdminApiAdapter):
    mapping_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _delete_mapping_response(db=context.db, mapping_id=self.mapping_id)


class AdminGeminiFilesCleanupMappingsAdapter(AdminApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _cleanup_expired_mappings_response(db=context.db)


class AdminGeminiFilesCapableKeysAdapter(AdminApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _list_capable_keys_response(db=context.db)


@dataclass
class AdminGeminiFilesUploadAdapter(AdminApiAdapter):
    file: UploadFile
    key_ids: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        del context
        return await _upload_file_response(file=self.file, key_ids=self.key_ids)


@router.get("/mappings", response_model=FileMappingListResponse)
async def list_file_mappings(
    request: Request,
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_expired: bool = Query(False),
    search: str | None = Query(None),
) -> Any:
    adapter = AdminGeminiFilesListMappingsAdapter(
        page=page,
        page_size=page_size,
        include_expired=include_expired,
        search=search,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/stats", response_model=FileMappingStatsResponse)
async def get_file_mapping_stats(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = AdminGeminiFilesStatsAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.delete("/mappings/{mapping_id}")
async def delete_mapping(
    mapping_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = AdminGeminiFilesDeleteMappingAdapter(mapping_id=mapping_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.delete("/mappings")
async def cleanup_expired_mappings(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = AdminGeminiFilesCleanupMappingsAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/capable-keys", response_model=list[CapableKeyResponse])
async def list_capable_keys(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = AdminGeminiFilesCapableKeysAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    key_ids: str = Query(..., description="逗号分隔的 Key ID 列表"),
    db: Session = Depends(get_db),
) -> Any:
    adapter = AdminGeminiFilesUploadAdapter(file=file, key_ids=key_ids)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
