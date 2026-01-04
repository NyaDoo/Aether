"""用户个人 API 端点。"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.authenticated_adapter import AuthenticatedApiAdapter
from src.api.base.pipeline import ApiRequestPipeline
from src.core.crypto import crypto_service
from src.core.exceptions import ForbiddenException, InvalidRequestException, NotFoundException, translate_pydantic_error
from src.core.logger import logger
from src.database import get_db
from src.models.api import (
    ChangePasswordRequest,
    CreateMyApiKeyRequest,
    UpdateApiKeyProvidersRequest,
    UpdatePreferencesRequest,
    UpdateProfileRequest,
)
from src.models.database import ApiKey, Provider, Usage, User
from src.services.usage.service import UsageService
from src.services.user.apikey import ApiKeyService
from src.services.user.preference import PreferenceService


router = APIRouter(prefix="/api/users/me", tags=["User Profile"])
pipeline = ApiRequestPipeline()


@router.get("")
async def get_my_profile(request: Request, db: Session = Depends(get_db)):
    """获取当前用户完整信息（包含偏好设置）"""
    adapter = MeProfileAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.put("")
async def update_my_profile(request: Request, db: Session = Depends(get_db)):
    adapter = UpdateProfileAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.patch("/password")
async def change_my_password(request: Request, db: Session = Depends(get_db)):
    """Change current user's password"""
    adapter = ChangePasswordAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ============== API密钥管理 ==============


@router.get("/api-keys")
async def list_my_api_keys(request: Request, db: Session = Depends(get_db)):
    adapter = ListMyApiKeysAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/api-keys")
async def create_my_api_key(request: Request, db: Session = Depends(get_db)):
    adapter = CreateMyApiKeyAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/api-keys/{key_id}")
async def get_my_api_key(
    key_id: str,
    request: Request,
    include_key: bool = Query(False, description="Include full decrypted key in response"),
    db: Session = Depends(get_db),
):
    """Get API key detail, optionally include full key"""
    if include_key:
        adapter = GetMyFullKeyAdapter(key_id=key_id)
    else:
        adapter = GetMyApiKeyDetailAdapter(key_id=key_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.delete("/api-keys/{key_id}")
async def delete_my_api_key(key_id: str, request: Request, db: Session = Depends(get_db)):
    adapter = DeleteMyApiKeyAdapter(key_id=key_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.patch("/api-keys/{key_id}")
async def toggle_my_api_key(key_id: str, request: Request, db: Session = Depends(get_db)):
    """Toggle API key active status"""
    adapter = ToggleMyApiKeyAdapter(key_id=key_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ============== 使用统计 ==============


@router.get("/usage")
async def get_my_usage(
    request: Request,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=200, description="每页记录数，默认100，最大200"),
    offset: int = Query(0, ge=0, le=2000, description="偏移量，用于分页，最大2000"),
    db: Session = Depends(get_db),
):
    adapter = GetUsageAdapter(start_date=start_date, end_date=end_date, limit=limit, offset=offset)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/usage/active")
async def get_my_active_requests(
    request: Request,
    ids: Optional[str] = Query(None, description="Comma-separated request IDs to query"),
    db: Session = Depends(get_db),
):
    """获取用户活跃请求状态（用于轮询更新）"""
    adapter = GetActiveRequestsAdapter(ids=ids)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/usage/interval-timeline")
async def get_my_interval_timeline(
    request: Request,
    hours: int = Query(24, ge=1, le=720, description="分析最近多少小时的数据"),
    limit: int = Query(5000, ge=100, le=20000, description="最大返回数据点数量"),
    db: Session = Depends(get_db),
):
    """获取当前用户的请求间隔时间线数据，用于散点图展示"""
    adapter = GetMyIntervalTimelineAdapter(hours=hours, limit=limit)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/usage/heatmap")
async def get_my_activity_heatmap(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Get user's activity heatmap data for the past 365 days.

    This endpoint is cached for 5 minutes to reduce database load.
    """
    adapter = GetMyActivityHeatmapAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/providers")
async def list_available_providers(request: Request, db: Session = Depends(get_db)):
    adapter = ListAvailableProvidersAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/endpoint-status")
async def get_endpoint_status(request: Request, db: Session = Depends(get_db)):
    """获取端点状态（简化版，不包含敏感信息）"""
    adapter = GetEndpointStatusAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ============== API密钥与提供商关联 ==============


# UpdateApiKeyProvidersRequest 已移至 src/models/api.py


@router.put("/api-keys/{api_key_id}/providers")
async def update_api_key_providers(
    api_key_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    adapter = UpdateApiKeyProvidersAdapter(api_key_id=api_key_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.put("/api-keys/{api_key_id}/capabilities")
async def update_api_key_capabilities(
    api_key_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """更新 API Key 的强制能力配置"""
    adapter = UpdateApiKeyCapabilitiesAdapter(api_key_id=api_key_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ============== 偏好设置 ==============


@router.get("/preferences")
async def get_my_preferences(request: Request, db: Session = Depends(get_db)):
    adapter = GetPreferencesAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.put("/preferences")
async def update_my_preferences(request: Request, db: Session = Depends(get_db)):
    adapter = UpdatePreferencesAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/model-capabilities")
async def get_model_capability_settings(request: Request, db: Session = Depends(get_db)):
    """获取用户的模型能力配置"""
    adapter = GetModelCapabilitySettingsAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.put("/model-capabilities")
async def update_model_capability_settings(request: Request, db: Session = Depends(get_db)):
    """更新用户的模型能力配置"""
    adapter = UpdateModelCapabilitySettingsAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ============== Pipeline 适配器 ==============


class MeProfileAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        return PreferenceService.get_user_with_preferences(context.db, context.user.id)


class UpdateProfileAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user
        payload = context.ensure_json_body()
        try:
            request = UpdateProfileRequest.model_validate(payload)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                raise InvalidRequestException(translate_pydantic_error(errors[0]))
            raise InvalidRequestException("请求数据验证失败")

        if request.email:
            existing = (
                db.query(User).filter(User.email == request.email, User.id != user.id).first()
            )
            if existing:
                raise InvalidRequestException("邮箱已被使用")
            user.email = request.email

        if request.username:
            existing = (
                db.query(User).filter(User.username == request.username, User.id != user.id).first()
            )
            if existing:
                raise InvalidRequestException("用户名已被使用")
            user.username = request.username

        user.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user)
        return {"message": "个人信息更新成功"}


class ChangePasswordAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user
        payload = context.ensure_json_body()
        try:
            request = ChangePasswordRequest.model_validate(payload)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                raise InvalidRequestException(translate_pydantic_error(errors[0]))
            raise InvalidRequestException("请求数据验证失败")

        if not user.verify_password(request.old_password):
            raise InvalidRequestException("旧密码错误")
        if len(request.new_password) < 6:
            raise InvalidRequestException("密码长度至少6位")

        user.set_password(request.new_password)
        user.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.info(f"用户修改密码: {user.email}")
        return {"message": "密码修改成功"}


class ListMyApiKeysAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user

        # 一次性查询所有 API keys
        api_keys = (
            db.query(ApiKey)
            .filter(ApiKey.user_id == user.id)
            .order_by(ApiKey.created_at.desc())
            .all()
        )

        if not api_keys:
            return []

        # 批量查询所有 API keys 的统计数据（单次查询）
        api_key_ids = [key.id for key in api_keys]
        stats_query = (
            db.query(
                Usage.api_key_id,
                func.count(Usage.id).label("requests"),
                func.sum(Usage.total_cost_usd).label("cost"),
                func.max(Usage.created_at).label("last_used"),
            )
            .filter(Usage.api_key_id.in_(api_key_ids))
            .group_by(Usage.api_key_id)
            .all()
        )

        # 构建统计数据映射
        stats_map = {
            row.api_key_id: {
                "total_requests": row.requests or 0,
                "total_cost_usd": float(row.cost or 0),
                "last_used_at": row.last_used,
            }
            for row in stats_query
        }

        result = []
        for key in api_keys:
            # 从映射中获取统计，没有则使用默认值
            real_stats = stats_map.get(
                key.id,
                {"total_requests": 0, "total_cost_usd": 0.0, "last_used_at": None},
            )

            result.append(
                {
                    "id": key.id,
                    "name": key.name,
                    "key_display": key.get_display_key(),
                    "is_active": key.is_active,
                    "last_used_at": (
                        real_stats["last_used_at"].isoformat()
                        if real_stats["last_used_at"]
                        else None
                    ),
                    "created_at": key.created_at.isoformat(),
                    "total_requests": real_stats["total_requests"],
                    "total_cost_usd": real_stats["total_cost_usd"],
                    "allowed_providers": key.allowed_providers,
                    "force_capabilities": key.force_capabilities,
                }
            )
        return result


class CreateMyApiKeyAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        payload = context.ensure_json_body()
        try:
            request = CreateMyApiKeyRequest.model_validate(payload)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                raise InvalidRequestException(translate_pydantic_error(errors[0]))
            raise InvalidRequestException("请求数据验证失败")
        try:
            api_key, plain_key = ApiKeyService.create_api_key(
                db=context.db,
                user_id=context.user.id,
                name=request.name,
            )
        except ValueError as exc:
            raise InvalidRequestException(str(exc))

        return {
            "id": api_key.id,
            "name": api_key.name,
            "key": plain_key,
            "key_display": api_key.get_display_key(),
            "message": "API密钥创建成功",
        }


@dataclass
class GetMyFullKeyAdapter(AuthenticatedApiAdapter):
    key_id: str

    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user

        # 查找API密钥，确保属于当前用户
        api_key = (
            db.query(ApiKey).filter(ApiKey.id == self.key_id, ApiKey.user_id == user.id).first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在", "api_key")

        # 解密完整密钥
        if not api_key.key_encrypted:
            raise HTTPException(status_code=400, detail="该密钥没有存储完整密钥信息")

        try:
            full_key = crypto_service.decrypt(api_key.key_encrypted)
        except Exception as e:
            logger.error(f"解密API密钥失败: Key ID {self.key_id}, 错误: {e}")
            raise HTTPException(status_code=500, detail="解密密钥失败")

        logger.info(f"用户 {user.email} 查看完整API密钥: Key ID {self.key_id}")

        return {
            "key": full_key,
        }


@dataclass
class GetMyApiKeyDetailAdapter(AuthenticatedApiAdapter):
    """Get API key detail without full key"""
    key_id: str

    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user

        api_key = (
            db.query(ApiKey).filter(ApiKey.id == self.key_id, ApiKey.user_id == user.id).first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在", "api_key")

        return {
            "id": api_key.id,
            "name": api_key.name,
            "key_display": api_key.get_display_key(),
            "is_active": api_key.is_active,
            "allowed_providers": api_key.allowed_providers,
            "force_capabilities": api_key.force_capabilities,
            "rate_limit": api_key.rate_limit,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "created_at": api_key.created_at.isoformat(),
        }


@dataclass
class DeleteMyApiKeyAdapter(AuthenticatedApiAdapter):
    key_id: str

    async def handle(self, context):  # type: ignore[override]
        api_key = (
            context.db.query(ApiKey)
            .filter(ApiKey.id == self.key_id, ApiKey.user_id == context.user.id)
            .first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在", "api_key")
        context.db.delete(api_key)
        context.db.commit()
        return {"message": "API密钥已删除"}


@dataclass
class ToggleMyApiKeyAdapter(AuthenticatedApiAdapter):
    key_id: str

    async def handle(self, context):  # type: ignore[override]
        api_key = (
            context.db.query(ApiKey)
            .filter(ApiKey.id == self.key_id, ApiKey.user_id == context.user.id)
            .first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在", "api_key")
        api_key.is_active = not api_key.is_active
        context.db.commit()
        context.db.refresh(api_key)
        return {
            "id": api_key.id,
            "is_active": api_key.is_active,
            "message": f"API密钥已{'启用' if api_key.is_active else '禁用'}",
        }


@dataclass
class GetUsageAdapter(AuthenticatedApiAdapter):
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    limit: int = 100
    offset: int = 0

    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user
        summary_list = UsageService.get_usage_summary(
            db=db,
            user_id=user.id,
            start_date=self.start_date,
            end_date=self.end_date,
        )

        # 过滤掉 unknown/pending provider 的记录（请求未到达任何提供商）
        filtered_summary = [
            item for item in summary_list
            if item.get("provider") not in ("unknown", "pending", None)
        ]

        total_requests = sum(item["requests"] for item in filtered_summary)
        total_input_tokens = (
            sum(item["input_tokens"] for item in filtered_summary) if filtered_summary else 0
        )
        total_output_tokens = (
            sum(item["output_tokens"] for item in filtered_summary) if filtered_summary else 0
        )
        total_tokens = sum(item["total_tokens"] for item in filtered_summary) if filtered_summary else 0
        total_cost = sum(item["total_cost_usd"] for item in filtered_summary) if filtered_summary else 0.0

        # 管理员可以看到真实成本
        total_actual_cost = 0.0
        if user.role == "admin":
            total_actual_cost = (
                sum(item.get("actual_total_cost_usd", 0.0) for item in filtered_summary)
                if filtered_summary
                else 0.0
            )

        model_summary = {}
        for item in filtered_summary:
            model_name = item["model"]
            base_stats = {
                "model": model_name,
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }
            # 管理员可以看到真实成本
            if user.role == "admin":
                base_stats["actual_total_cost_usd"] = 0.0

            stats = model_summary.setdefault(model_name, base_stats)
            stats["requests"] += item["requests"]
            stats["input_tokens"] += item["input_tokens"]
            stats["output_tokens"] += item["output_tokens"]
            stats["total_tokens"] += item["total_tokens"]
            stats["total_cost_usd"] += item["total_cost_usd"]
            # 管理员可以看到真实成本
            if user.role == "admin":
                stats["actual_total_cost_usd"] += item.get("actual_total_cost_usd", 0.0)

        summary_by_model = sorted(model_summary.values(), key=lambda x: x["requests"], reverse=True)

        # 按提供商汇总（用于 UsageProviderTable）
        provider_summary = {}
        for item in filtered_summary:
            provider_name = item["provider"]
            base_stats = {
                "provider": provider_name,
                "requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "success_count": 0,
                "total_response_time_ms": 0.0,
                "response_time_count": 0,
            }
            stats = provider_summary.setdefault(provider_name, base_stats)
            stats["requests"] += item["requests"]
            stats["total_tokens"] += item["total_tokens"]
            stats["total_cost_usd"] += item["total_cost_usd"]
            # 假设 summary 中的都是成功的请求
            stats["success_count"] += item["requests"]
            if item.get("avg_response_time_ms") is not None:
                stats["total_response_time_ms"] += item["avg_response_time_ms"] * item["requests"]
                stats["response_time_count"] += item["requests"]

        summary_by_provider = []
        for stats in provider_summary.values():
            avg_response_time_ms = (
                stats["total_response_time_ms"] / stats["response_time_count"]
                if stats["response_time_count"] > 0 else 0
            )
            success_rate = (
                (stats["success_count"] / stats["requests"] * 100)
                if stats["requests"] > 0 else 100
            )
            summary_by_provider.append({
                "provider": stats["provider"],
                "requests": stats["requests"],
                "total_tokens": stats["total_tokens"],
                "total_cost_usd": stats["total_cost_usd"],
                "success_rate": round(success_rate, 2),
                "avg_response_time_ms": round(avg_response_time_ms, 2),
            })
        summary_by_provider = sorted(summary_by_provider, key=lambda x: x["requests"], reverse=True)

        query = db.query(Usage).filter(Usage.user_id == user.id)
        if self.start_date:
            query = query.filter(Usage.created_at >= self.start_date)
        if self.end_date:
            query = query.filter(Usage.created_at <= self.end_date)

        # 计算总数用于分页
        total_records = query.count()
        usage_records = query.order_by(Usage.created_at.desc()).offset(self.offset).limit(self.limit).all()

        avg_resp_query = db.query(func.avg(Usage.response_time_ms)).filter(
            Usage.user_id == user.id,
            Usage.status_code == 200,
            Usage.response_time_ms.isnot(None),
        )
        if self.start_date:
            avg_resp_query = avg_resp_query.filter(Usage.created_at >= self.start_date)
        if self.end_date:
            avg_resp_query = avg_resp_query.filter(Usage.created_at <= self.end_date)
        avg_response_ms = avg_resp_query.scalar() or 0
        avg_response_time = float(avg_response_ms) / 1000.0 if avg_response_ms else 0

        # 构建响应数据
        response_data = {
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_response_time": avg_response_time,
            "quota_usd": user.quota_usd,
            "used_usd": user.used_usd,
            "summary_by_model": summary_by_model,
            "summary_by_provider": summary_by_provider,
            # 分页信息
            "pagination": {
                "total": total_records,
                "limit": self.limit,
                "offset": self.offset,
                "has_more": self.offset + self.limit < total_records,
            },
            "records": [
                {
                    "id": r.id,
                    "provider": r.provider,
                    "model": r.model,
                    "target_model": r.target_model,  # 映射后的目标模型名
                    "api_format": r.api_format,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "total_tokens": r.total_tokens,
                    "cost": r.total_cost_usd,
                    "response_time_ms": r.response_time_ms,
                    "is_stream": r.is_stream,
                    "status": r.status,  # 请求状态: pending, streaming, completed, failed
                    "created_at": r.created_at.isoformat(),
                    "cache_creation_input_tokens": r.cache_creation_input_tokens,
                    "cache_read_input_tokens": r.cache_read_input_tokens,
                    "status_code": r.status_code,
                    "error_message": r.error_message,
                    "input_price_per_1m": r.input_price_per_1m,
                    "output_price_per_1m": r.output_price_per_1m,
                    "cache_creation_price_per_1m": r.cache_creation_price_per_1m,
                    "cache_read_price_per_1m": r.cache_read_price_per_1m,
                }
                for r in usage_records
            ],
        }

        # 管理员可以看到真实成本
        if user.role == "admin":
            response_data["total_actual_cost"] = total_actual_cost
            # 为每条记录添加真实成本和倍率信息
            for i, r in enumerate(usage_records):
                # 确保字段有值，避免前端显示 -
                actual_cost = (
                    r.actual_total_cost_usd if r.actual_total_cost_usd is not None else 0.0
                )
                rate_mult = r.rate_multiplier if r.rate_multiplier is not None else 1.0
                response_data["records"][i]["actual_cost"] = actual_cost
                response_data["records"][i]["rate_multiplier"] = rate_mult

                # 调试日志：检查前几条记录
                if i < 3:
                    from src.core.logger import logger
                    logger.debug(
                        f"Usage record {i}: id={r.id}, actual_total_cost_usd={r.actual_total_cost_usd}, "
                        f"rate_multiplier={r.rate_multiplier}, returned: actual_cost={actual_cost}, rate_mult={rate_mult}"
                    )

        return response_data


@dataclass
class GetActiveRequestsAdapter(AuthenticatedApiAdapter):
    """轻量级活跃请求状态查询适配器（用于用户端轮询）"""

    ids: Optional[str] = None

    async def handle(self, context):  # type: ignore[override]
        from src.services.usage import UsageService

        db = context.db
        user = context.user
        id_list = None
        if self.ids:
            id_list = [id.strip() for id in self.ids.split(",") if id.strip()]
            if not id_list:
                return {"requests": []}

        requests = UsageService.get_active_requests_status(db=db, ids=id_list, user_id=user.id)
        return {"requests": requests}


@dataclass
class GetMyIntervalTimelineAdapter(AuthenticatedApiAdapter):
    """获取当前用户的请求间隔时间线适配器"""

    hours: int
    limit: int

    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user

        result = UsageService.get_interval_timeline(
            db=db,
            hours=self.hours,
            limit=self.limit,
            user_id=str(user.id),
        )

        return result


class GetMyActivityHeatmapAdapter(AuthenticatedApiAdapter):
    """Activity heatmap adapter with Redis caching for user."""

    async def handle(self, context):  # type: ignore[override]
        user = context.user
        result = await UsageService.get_cached_heatmap(
            db=context.db,
            user_id=user.id,
            include_actual_cost=user.role == "admin",
        )
        context.add_audit_metadata(action="activity_heatmap")
        return result


class ListAvailableProvidersAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        from sqlalchemy.orm import selectinload

        from src.models.database import Model, ProviderEndpoint

        db = context.db

        # 使用 selectinload 预加载所有关联数据，避免 N+1 查询
        providers = (
            db.query(Provider)
            .options(
                selectinload(Provider.endpoints),
                selectinload(Provider.models).selectinload(Model.global_model),
            )
            .filter(Provider.is_active.is_(True))
            .all()
        )

        result = []
        for provider in providers:
            # 直接使用预加载的 endpoints，无需额外查询
            endpoints_data = [
                {
                    "id": ep.id,
                    "api_format": ep.api_format if ep.api_format else None,
                    "base_url": ep.base_url,
                    "is_active": ep.is_active,
                }
                for ep in provider.endpoints
            ]

            models_data = []
            # 直接使用预加载的 models，无需额外查询
            direct_models = provider.models
            for model in direct_models:
                global_model = model.global_model
                display_name = (
                    global_model.display_name if global_model else model.provider_model_name
                )
                unified_name = global_model.name if global_model else model.provider_model_name
                models_data.append(
                    {
                        "id": model.id,
                        "name": unified_name,
                        "display_name": display_name,
                        "input_price_per_1m": model.input_price_per_1m,
                        "output_price_per_1m": model.output_price_per_1m,
                        "cache_creation_price_per_1m": model.cache_creation_price_per_1m,
                        "cache_read_price_per_1m": model.cache_read_price_per_1m,
                        "supports_vision": model.supports_vision,
                        "supports_function_calling": model.supports_function_calling,
                        "supports_streaming": model.supports_streaming,
                    }
                )

            result.append(
                {
                    "id": provider.id,
                    "name": provider.name,
                    "display_name": provider.display_name,
                    "description": provider.description,
                    "provider_priority": provider.provider_priority,
                    "endpoints": endpoints_data,
                    "models": models_data,
                }
            )
        return result


@dataclass
class UpdateApiKeyProvidersAdapter(AuthenticatedApiAdapter):
    api_key_id: str

    async def handle(self, context):  # type: ignore[override]
        db = context.db
        user = context.user
        payload = context.ensure_json_body()
        try:
            request = UpdateApiKeyProvidersRequest.model_validate(payload)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                raise InvalidRequestException(translate_pydantic_error(errors[0]))
            raise InvalidRequestException("请求数据验证失败")

        api_key = (
            db.query(ApiKey).filter(ApiKey.id == self.api_key_id, ApiKey.user_id == user.id).first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在")

        if request.allowed_providers is not None and len(request.allowed_providers) > 0:
            provider_ids = [cfg.provider_id for cfg in request.allowed_providers]
            valid = (
                db.query(Provider.id)
                .filter(Provider.id.in_(provider_ids), Provider.is_active.is_(True))
                .all()
            )
            valid_ids = {p.id for p in valid}
            invalid = set(provider_ids) - valid_ids
            if invalid:
                raise InvalidRequestException(f"无效的提供商ID: {', '.join(invalid)}")

        # 只存储 provider_id 列表，而不是完整的 ProviderConfig 字典
        # 因为 allowed_providers 字段设计为存储 provider ID 字符串列表
        api_key.allowed_providers = (
            [cfg.provider_id for cfg in request.allowed_providers]
            if request.allowed_providers
            else None
        )
        api_key.updated_at = datetime.now(timezone.utc)
        db.commit()
        logger.debug(f"用户 {user.id} 更新API密钥 {self.api_key_id} 的可用提供商")
        return {"message": "API密钥可用提供商已更新"}


@dataclass
class UpdateApiKeyCapabilitiesAdapter(AuthenticatedApiAdapter):
    """更新 API Key 的强制能力配置"""

    api_key_id: str

    async def handle(self, context):  # type: ignore[override]
        from src.core.key_capabilities import CAPABILITY_DEFINITIONS, CapabilityConfigMode
        from src.models.database import AuditEventType
        from src.services.system.audit import audit_service

        db = context.db
        user = context.user
        payload = context.ensure_json_body()

        api_key = (
            db.query(ApiKey).filter(ApiKey.id == self.api_key_id, ApiKey.user_id == user.id).first()
        )
        if not api_key:
            raise NotFoundException("API密钥不存在")

        # 保存旧值用于审计
        old_capabilities = api_key.force_capabilities

        # 验证 force_capabilities 字段
        force_capabilities = payload.get("force_capabilities")
        if force_capabilities is not None:
            if not isinstance(force_capabilities, dict):
                raise InvalidRequestException("force_capabilities 必须是对象类型")

            # 验证只允许用户可配置的能力
            for cap_name, cap_value in force_capabilities.items():
                cap_def = CAPABILITY_DEFINITIONS.get(cap_name)
                if not cap_def:
                    raise InvalidRequestException(f"未知的能力类型: {cap_name}")
                if cap_def.config_mode != CapabilityConfigMode.USER_CONFIGURABLE:
                    raise InvalidRequestException(f"能力 {cap_name} 不支持用户配置")
                if not isinstance(cap_value, bool):
                    raise InvalidRequestException(f"能力 {cap_name} 的值必须是布尔类型")

        api_key.force_capabilities = force_capabilities
        api_key.updated_at = datetime.now(timezone.utc)
        db.commit()

        # 记录审计日志
        audit_service.log_event(
            db=db,
            event_type=AuditEventType.CONFIG_CHANGED,
            description=f"用户更新 API Key 能力配置",
            user_id=user.id,
            api_key_id=api_key.id,
            metadata={
                "action": "update_api_key_capabilities",
                "old_capabilities": old_capabilities,
                "new_capabilities": force_capabilities,
            },
        )

        logger.debug(f"用户 {user.id} 更新API密钥 {self.api_key_id} 的强制能力配置: {force_capabilities}")
        return {
            "message": "API密钥能力配置已更新",
            "force_capabilities": api_key.force_capabilities,
        }


class GetPreferencesAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        preferences = PreferenceService.get_or_create_preferences(context.db, context.user.id)
        return {
            "avatar_url": preferences.avatar_url,
            "bio": preferences.bio,
            "default_provider_id": preferences.default_provider_id,
            "default_provider": (
                preferences.default_provider.name if preferences.default_provider else None
            ),
            "theme": preferences.theme,
            "language": preferences.language,
            "timezone": preferences.timezone,
            "notifications": {
                "email": preferences.email_notifications,
                "usage_alerts": preferences.usage_alerts,
                "announcements": preferences.announcement_notifications,
            },
        }


class UpdatePreferencesAdapter(AuthenticatedApiAdapter):
    async def handle(self, context):  # type: ignore[override]
        payload = context.ensure_json_body()
        try:
            request = UpdatePreferencesRequest.model_validate(payload)
        except ValidationError as e:
            errors = e.errors()
            if errors:
                raise InvalidRequestException(translate_pydantic_error(errors[0]))
            raise InvalidRequestException("请求数据验证失败")

        PreferenceService.update_preferences(
            db=context.db,
            user_id=context.user.id,
            avatar_url=request.avatar_url,
            bio=request.bio,
            default_provider_id=request.default_provider_id,
            theme=request.theme,
            language=request.language,
            timezone=request.timezone,
            email_notifications=request.email_notifications,
            usage_alerts=request.usage_alerts,
            announcement_notifications=request.announcement_notifications,
        )
        return {"message": "偏好设置更新成功"}


class GetModelCapabilitySettingsAdapter(AuthenticatedApiAdapter):
    """获取用户的模型能力配置"""

    async def handle(self, context):  # type: ignore[override]
        user = context.user
        return {
            "model_capability_settings": user.model_capability_settings or {},
        }


class UpdateModelCapabilitySettingsAdapter(AuthenticatedApiAdapter):
    """更新用户的模型能力配置"""

    async def handle(self, context):  # type: ignore[override]
        from src.core.key_capabilities import CAPABILITY_DEFINITIONS, CapabilityConfigMode
        from src.models.database import AuditEventType
        from src.services.cache.user_cache import UserCacheService
        from src.services.system.audit import audit_service

        db = context.db
        # 重新从数据库查询用户，确保在 session 中（context.user 可能来自缓存，是分离对象）
        user = db.query(User).filter(User.id == context.user.id).first()
        if not user:
            raise NotFoundException("用户不存在")
        payload = context.ensure_json_body()

        # 保存旧值用于审计
        old_settings = user.model_capability_settings

        # 验证 model_capability_settings 字段
        settings = payload.get("model_capability_settings")
        if settings is not None:
            if not isinstance(settings, dict):
                raise InvalidRequestException("model_capability_settings 必须是对象类型")

            # 验证每个模型的能力配置
            for model_name, capabilities in settings.items():
                if not isinstance(model_name, str):
                    raise InvalidRequestException("模型名称必须是字符串")
                if not isinstance(capabilities, dict):
                    raise InvalidRequestException(f"模型 {model_name} 的能力配置必须是对象类型")

                # 验证只允许用户可配置的能力
                for cap_name, cap_value in capabilities.items():
                    cap_def = CAPABILITY_DEFINITIONS.get(cap_name)
                    if not cap_def:
                        raise InvalidRequestException(f"未知的能力类型: {cap_name}")
                    if cap_def.config_mode != CapabilityConfigMode.USER_CONFIGURABLE:
                        raise InvalidRequestException(f"能力 {cap_name} 不支持用户配置")
                    if not isinstance(cap_value, bool):
                        raise InvalidRequestException(f"能力 {cap_name} 的值必须是布尔类型")

        user.model_capability_settings = settings
        user.updated_at = datetime.now(timezone.utc)
        db.commit()

        # 清除用户缓存，确保下次读取时获取最新数据
        await UserCacheService.invalidate_user_cache(user.id, user.email)

        # 记录审计日志
        audit_service.log_event(
            db=db,
            event_type=AuditEventType.CONFIG_CHANGED,
            description=f"用户更新模型能力配置",
            user_id=user.id,
            metadata={
                "action": "update_model_capability_settings",
                "old_settings": old_settings,
                "new_settings": settings,
            },
        )

        logger.debug(f"用户 {user.id} 更新模型能力配置: {settings}")
        return {
            "message": "模型能力配置已更新",
            "model_capability_settings": user.model_capability_settings,
        }


class GetEndpointStatusAdapter(AuthenticatedApiAdapter):
    """获取端点状态（简化版，不包含敏感信息）"""

    # 类级别缓存实例（延迟初始化）
    _cache_backend = None
    _cache_ttl = 60  # 缓存60秒

    @classmethod
    async def _get_cache(cls):
        """获取缓存后端实例（懒加载）"""
        if cls._cache_backend is None:
            from src.services.cache.backend import get_cache_backend

            cls._cache_backend = await get_cache_backend(
                name="endpoint_status",
                backend_type="auto",
                ttl=cls._cache_ttl,  # 使用 ttl 而不是 default_ttl
            )
        return cls._cache_backend

    async def handle(self, context):  # type: ignore[override]
        from src.services.health.endpoint import EndpointHealthService

        db = context.db

        # 尝试从缓存获取
        cache = await self._get_cache()
        cache_key = "endpoint_status:all"

        try:
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            pass  # 缓存失败不影响正常流程

        # 使用共享服务获取健康状态（普通用户视图）
        result = EndpointHealthService.get_endpoint_health_by_format(
            db=db,
            lookback_hours=6,
            include_admin_fields=False,  # 不包含敏感的管理员字段
            use_cache=True,
        )

        # 写入缓存
        try:
            await cache.set(cache_key, result, ttl=self._cache_ttl)
        except Exception:
            pass  # 缓存失败不影响正常流程

        return result
