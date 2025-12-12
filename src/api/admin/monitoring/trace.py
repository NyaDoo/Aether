"""
请求链路追踪 API 端点
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.pipeline import ApiRequestPipeline
from src.database import get_db
from src.models.database import Provider, ProviderEndpoint, ProviderAPIKey
from src.core.crypto import crypto_service
from src.services.request.candidate import RequestCandidateService

router = APIRouter(prefix="/api/admin/monitoring/trace", tags=["Admin - Monitoring: Trace"])
pipeline = ApiRequestPipeline()


class CandidateResponse(BaseModel):
    """候选记录响应"""

    id: str
    request_id: str
    candidate_index: int
    retry_index: int = 0  # 重试序号（从0开始）
    provider_id: Optional[str] = None
    provider_name: Optional[str] = None
    provider_website: Optional[str] = None  # Provider 官网
    endpoint_id: Optional[str] = None
    endpoint_name: Optional[str] = None  # 端点显示名称（api_format）
    key_id: Optional[str] = None
    key_name: Optional[str] = None  # 密钥名称
    key_preview: Optional[str] = None  # 密钥脱敏预览（如 sk-***abc）
    key_capabilities: Optional[dict] = None  # Key 支持的能力
    required_capabilities: Optional[dict] = None  # 请求实际需要的能力标签
    status: str  # 'pending', 'success', 'failed', 'skipped'
    skip_reason: Optional[str] = None
    is_cached: bool = False
    # 执行结果字段
    status_code: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    latency_ms: Optional[int] = None
    concurrent_requests: Optional[int] = None
    extra_data: Optional[dict] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class RequestTraceResponse(BaseModel):
    """请求追踪完整响应"""

    request_id: str
    total_candidates: int
    final_status: str  # 'success', 'failed', 'streaming', 'pending'
    total_latency_ms: int
    candidates: List[CandidateResponse]


@router.get("/{request_id}", response_model=RequestTraceResponse)
async def get_request_trace(
    request_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    """获取特定请求的完整追踪信息"""

    adapter = AdminGetRequestTraceAdapter(request_id=request_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/stats/provider/{provider_id}")
async def get_provider_failure_rate(
    provider_id: str,
    request: Request,
    limit: int = Query(100, ge=1, le=1000, description="统计最近的尝试数量"),
    db: Session = Depends(get_db),
):
    """
    获取某个 Provider 的失败率统计

    需要管理员权限
    """
    adapter = AdminProviderFailureRateAdapter(provider_id=provider_id, limit=limit)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# -------- 请求追踪适配器 --------


@dataclass
class AdminGetRequestTraceAdapter(AdminApiAdapter):
    request_id: str

    async def handle(self, context):  # type: ignore[override]
        db = context.db

        # 只查询 candidates
        candidates = RequestCandidateService.get_candidates_by_request_id(db, self.request_id)

        # 如果没有数据，返回 404
        if not candidates:
            raise HTTPException(status_code=404, detail="Request not found")

        # 计算总延迟（只统计已完成的候选：success 或 failed）
        # 使用显式的 is not None 检查，避免过滤掉 0ms 的快速响应
        total_latency = sum(
            c.latency_ms
            for c in candidates
            if c.status in ("success", "failed") and c.latency_ms is not None
        )

        # 判断最终状态：
        # 1. status="success" 即视为成功（无论 status_code 是什么）
        #    - 流式请求即使客户端断开（499），只要 Provider 成功返回数据，也算成功
        # 2. 同时检查 status_code 在 200-299 范围，作为额外的成功判断条件
        #    - 用于兼容非流式请求或未正确设置 status 的旧数据
        # 3. status="streaming" 表示流式请求正在进行中
        # 4. status="pending" 表示请求尚未开始执行
        has_success = any(
            c.status == "success"
            or (c.status_code is not None and 200 <= c.status_code < 300)
            for c in candidates
        )
        has_streaming = any(c.status == "streaming" for c in candidates)
        has_pending = any(c.status == "pending" for c in candidates)

        if has_success:
            final_status = "success"
        elif has_streaming:
            # 有候选正在流式传输中
            final_status = "streaming"
        elif has_pending:
            # 有候选正在等待执行
            final_status = "pending"
        else:
            final_status = "failed"

        # 批量加载 provider 信息，避免 N+1 查询
        provider_ids = {c.provider_id for c in candidates if c.provider_id}
        provider_map = {}
        provider_website_map = {}
        if provider_ids:
            providers = db.query(Provider).filter(Provider.id.in_(provider_ids)).all()
            for p in providers:
                provider_map[p.id] = p.name
                provider_website_map[p.id] = p.website

        # 批量加载 endpoint 信息
        endpoint_ids = {c.endpoint_id for c in candidates if c.endpoint_id}
        endpoint_map = {}
        if endpoint_ids:
            endpoints = db.query(ProviderEndpoint).filter(ProviderEndpoint.id.in_(endpoint_ids)).all()
            endpoint_map = {e.id: e.api_format for e in endpoints}

        # 批量加载 key 信息
        key_ids = {c.key_id for c in candidates if c.key_id}
        key_map = {}
        key_preview_map = {}
        key_capabilities_map = {}
        if key_ids:
            keys = db.query(ProviderAPIKey).filter(ProviderAPIKey.id.in_(key_ids)).all()
            for k in keys:
                key_map[k.id] = k.name
                key_capabilities_map[k.id] = k.capabilities
                # 生成脱敏预览：先解密再脱敏
                try:
                    decrypted_key = crypto_service.decrypt(k.api_key)
                    if len(decrypted_key) > 8:
                        # 检测常见前缀模式
                        prefix_end = 0
                        for prefix in ["sk-", "key-", "api-", "ak-"]:
                            if decrypted_key.lower().startswith(prefix):
                                prefix_end = len(prefix)
                                break
                        if prefix_end > 0:
                            key_preview_map[k.id] = f"{decrypted_key[:prefix_end]}***{decrypted_key[-4:]}"
                        else:
                            key_preview_map[k.id] = f"{decrypted_key[:4]}***{decrypted_key[-4:]}"
                    elif len(decrypted_key) > 4:
                        key_preview_map[k.id] = f"***{decrypted_key[-4:]}"
                    else:
                        key_preview_map[k.id] = "***"
                except Exception:
                    key_preview_map[k.id] = "***"

        # 构建 candidate 响应列表
        candidate_responses: List[CandidateResponse] = []
        for candidate in candidates:
            provider_name = (
                provider_map.get(candidate.provider_id) if candidate.provider_id else None
            )
            provider_website = (
                provider_website_map.get(candidate.provider_id) if candidate.provider_id else None
            )
            endpoint_name = (
                endpoint_map.get(candidate.endpoint_id) if candidate.endpoint_id else None
            )
            key_name = (
                key_map.get(candidate.key_id) if candidate.key_id else None
            )
            key_preview = (
                key_preview_map.get(candidate.key_id) if candidate.key_id else None
            )
            key_capabilities = (
                key_capabilities_map.get(candidate.key_id) if candidate.key_id else None
            )

            candidate_responses.append(
                CandidateResponse(
                    id=candidate.id,
                    request_id=candidate.request_id,
                    candidate_index=candidate.candidate_index,
                    retry_index=candidate.retry_index,
                    provider_id=candidate.provider_id,
                    provider_name=provider_name,
                    provider_website=provider_website,
                    endpoint_id=candidate.endpoint_id,
                    endpoint_name=endpoint_name,
                    key_id=candidate.key_id,
                    key_name=key_name,
                    key_preview=key_preview,
                    key_capabilities=key_capabilities,
                    required_capabilities=candidate.required_capabilities,
                    status=candidate.status,
                    skip_reason=candidate.skip_reason,
                    is_cached=candidate.is_cached,
                    status_code=candidate.status_code,
                    error_type=candidate.error_type,
                    error_message=candidate.error_message,
                    latency_ms=candidate.latency_ms,
                    concurrent_requests=candidate.concurrent_requests,
                    extra_data=candidate.extra_data,
                    created_at=candidate.created_at,
                    started_at=candidate.started_at,
                    finished_at=candidate.finished_at,
                )
            )

        response = RequestTraceResponse(
            request_id=self.request_id,
            total_candidates=len(candidates),
            final_status=final_status,
            total_latency_ms=total_latency,
            candidates=candidate_responses,
        )
        context.add_audit_metadata(
            action="trace_request_detail",
            request_id=self.request_id,
            total_candidates=len(candidates),
            final_status=final_status,
            total_latency_ms=total_latency,
        )
        return response


@dataclass
class AdminProviderFailureRateAdapter(AdminApiAdapter):
    provider_id: str
    limit: int

    async def handle(self, context):  # type: ignore[override]
        result = RequestCandidateService.get_candidate_stats_by_provider(
            db=context.db,
            provider_id=self.provider_id,
            limit=self.limit,
        )
        context.add_audit_metadata(
            action="trace_provider_failure_rate",
            provider_id=self.provider_id,
            limit=self.limit,
            total_attempts=result.get("total_attempts"),
            failure_rate=result.get("failure_rate"),
        )
        return result
