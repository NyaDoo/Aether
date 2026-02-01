from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from src.config.settings import config
from src.core.logger import logger
from src.models.database import ApiKey, Provider, Usage, User, VideoTask
from src.services.billing.dimension_collector_service import DimensionCollectorService
from src.services.billing.formula_engine import BillingIncompleteError, FormulaEngine
from src.services.billing.rule_service import BillingRuleService
from src.services.usage.service import UsageService


class TaskApplicationService:
    """
    TaskApplicationService (Phase2)

    当前仅先收敛"终态结算"入口，用于替代旧版 VideoTelemetry 直写 Usage 的流程。
    后续将扩展 submit/cancel 并迁移候选编排逻辑。
    """

    def __init__(self, db: Session, *, redis_client: Any | None = None) -> None:
        self.db = db
        self.redis = redis_client

    async def finalize_video_task(self, task: VideoTask) -> bool:
        """
        更新视频任务的计费信息（轮询完成后调用）。

        异步任务的计费流程：
        1. 提交成功时：Usage 已结算（billing_status='settled'，费用=0）
        2. 轮询完成时：更新实际费用（成功则计费，失败则保持0）

        返回 True 表示成功更新，False 表示无需更新（如已是最终状态）
        """
        request_id = getattr(task, "request_id", None) or task.id

        existing = self.db.query(Usage).filter(Usage.request_id == request_id).first()
        if not existing:
            # Usage 不存在，尝试创建并结算（兜底逻辑）
            logger.warning(
                "Usage not found for video task, creating fallback: task_id=%s request_id=%s",
                task.id,
                request_id,
            )
            return await self._create_fallback_usage(task, request_id)

        # 检查是否已有计费更新标记（避免重复计费）
        metadata = existing.request_metadata or {}
        if metadata.get("billing_updated_at"):
            logger.debug(
                "Video task billing already updated: task_id=%s request_id=%s",
                task.id,
                request_id,
            )
            return False

        # 计算异步任务总耗时（ms）
        response_time_ms: int | None = None
        if task.submitted_at and task.completed_at:
            delta = task.completed_at - task.submitted_at
            response_time_ms = int(delta.total_seconds() * 1000)

        # === 收集计费维度 ===
        base_dimensions: dict[str, Any] = {
            "duration_seconds": task.duration_seconds,
            "resolution": task.resolution,
            "aspect_ratio": task.aspect_ratio,
            "size": task.size or "",
            "retry_count": task.retry_count,
        }

        collector_metadata: dict[str, Any] = {
            "task": {
                "id": task.id,
                "external_task_id": task.external_task_id,
                "model": task.model,
                "duration_seconds": task.duration_seconds,
                "resolution": task.resolution,
                "aspect_ratio": task.aspect_ratio,
                "size": task.size,
                "retry_count": task.retry_count,
                "video_size_bytes": task.video_size_bytes,
            },
            "result": {
                "video_url": task.video_url,
                "video_urls": task.video_urls or [],
            },
        }

        dims = DimensionCollectorService(self.db).collect_dimensions(
            api_format=task.provider_api_format,
            task_type="video",
            request=task.original_request_body or {},
            response=(
                (task.request_metadata or {}).get("poll_raw_response")
                if isinstance(task.request_metadata, dict)
                else None
            ),
            metadata=collector_metadata,
            base_dimensions=base_dimensions,
        )

        # === 计算成本（优先使用冻结的 billing_rule_snapshot）===
        rule_snapshot = None
        if isinstance(task.request_metadata, dict):
            rule_snapshot = task.request_metadata.get("billing_rule_snapshot")

        expression = None
        variables: dict[str, Any] | None = None
        dimension_mappings: dict[str, dict[str, Any]] | None = None
        rule_id = None
        rule_name = None
        rule_scope = None

        if isinstance(rule_snapshot, dict) and rule_snapshot.get("status") == "ok":
            rule_id = rule_snapshot.get("rule_id")
            rule_name = rule_snapshot.get("rule_name")
            rule_scope = rule_snapshot.get("scope")
            expression = rule_snapshot.get("expression")
            variables = rule_snapshot.get("variables") or {}
            dimension_mappings = rule_snapshot.get("dimension_mappings") or {}
        else:
            lookup = BillingRuleService.find_rule(
                self.db,
                provider_id=task.provider_id,
                model_name=task.model,
                task_type="video",
            )
            if lookup:
                rule = lookup.rule
                rule_id = rule.id
                rule_name = rule.name
                rule_scope = getattr(lookup, "scope", None)
                expression = rule.expression
                variables = rule.variables or {}
                dimension_mappings = rule.dimension_mappings or {}

        billing_snapshot: dict[str, Any] = {
            "schema_version": "1.0",
            "rule_id": str(rule_id) if rule_id else None,
            "rule_name": str(rule_name) if rule_name else None,
            "scope": str(rule_scope) if rule_scope else None,
            "expression": str(expression) if expression else None,
            "dimensions_used": dims,
            "missing_required": [],
            "cost": 0.0,
            "status": "no_rule",
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

        cost = 0.0
        # 只有任务成功时才计费
        if task.status == "completed" and expression:
            engine = FormulaEngine()
            try:
                result = engine.evaluate(
                    expression=str(expression),
                    variables=variables,
                    dimensions=dims,
                    dimension_mappings=dimension_mappings,
                    strict_mode=config.billing_strict_mode,
                )
                billing_snapshot["status"] = result.status
                billing_snapshot["missing_required"] = result.missing_required
                if result.status == "complete":
                    cost = float(result.cost)
                billing_snapshot["cost"] = cost
            except BillingIncompleteError as exc:
                # strict_mode=true：标记任务失败并隐藏产物，避免"免费放行"
                task.status = "failed"
                task.error_code = "billing_incomplete"
                task.error_message = f"Missing required dimensions: {exc.missing_required}"
                task.video_url = None
                task.video_urls = None
                billing_snapshot["status"] = "incomplete"
                billing_snapshot["missing_required"] = exc.missing_required
                billing_snapshot["cost"] = 0.0
            except Exception as exc:
                billing_snapshot["status"] = "incomplete"
                billing_snapshot["error"] = str(exc)
                billing_snapshot["cost"] = 0.0

        # 回写到 task.request_metadata 便于审计/重算
        # 重新赋值整个字典，确保 SQLAlchemy 检测到变更
        metadata = dict(task.request_metadata) if task.request_metadata else {}
        metadata["billing_snapshot"] = billing_snapshot
        task.request_metadata = metadata

        # === 更新已结算的 Usage 计费信息 ===
        updated = UsageService.update_settled_billing(
            self.db,
            request_id=request_id,
            total_cost_usd=cost,
            request_cost_usd=cost,
            status="completed" if task.status == "completed" else "failed",
            status_code=200 if task.status == "completed" else 500,
            error_message=(
                None
                if task.status == "completed"
                else (task.error_message or task.error_code or "video_task_failed")
            ),
            response_time_ms=response_time_ms,
            billing_snapshot=billing_snapshot,
            extra_metadata={
                "dimensions": dims,
                "raw_response_ref": {
                    "video_task_id": task.id,
                    "field": "video_tasks.request_metadata.poll_raw_response",
                },
            },
        )

        if updated:
            logger.debug(
                "Updated video task billing: task_id=%s request_id=%s cost=%.6f",
                task.id,
                request_id,
                cost,
            )
        else:
            logger.warning(
                "Failed to update video task billing (may already be updated): "
                "task_id=%s request_id=%s",
                task.id,
                request_id,
            )

        return updated

    async def _create_fallback_usage(self, task: VideoTask, request_id: str) -> bool:
        """
        兜底逻辑：当 Usage 不存在时创建完整记录。
        这种情况理论上不应发生（submit 阶段已创建），但保留以防万一。
        """
        user_obj = self.db.query(User).filter(User.id == task.user_id).first()
        api_key_obj = (
            self.db.query(ApiKey).filter(ApiKey.id == task.api_key_id).first()
            if task.api_key_id
            else None
        )
        provider_obj = (
            self.db.query(Provider).filter(Provider.id == task.provider_id).first()
            if task.provider_id
            else None
        )
        provider_name = provider_obj.name if provider_obj else "unknown"

        # 计算响应时间
        response_time_ms: int | None = None
        if task.submitted_at and task.completed_at:
            delta = task.completed_at - task.submitted_at
            response_time_ms = int(delta.total_seconds() * 1000)

        try:
            await UsageService.record_usage_with_custom_cost(
                db=self.db,
                user=user_obj,
                api_key=api_key_obj,
                provider=provider_name,
                model=task.model,
                request_type="video",
                total_cost_usd=0.0,  # 兜底记录不计费
                request_cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                api_format=task.client_api_format,
                endpoint_api_format=task.provider_api_format,
                has_format_conversion=bool(task.format_converted),
                is_stream=False,
                response_time_ms=response_time_ms,
                first_byte_time_ms=None,
                status_code=200 if task.status == "completed" else 500,
                error_message=(
                    None
                    if task.status == "completed"
                    else (task.error_message or task.error_code or "video_task_failed")
                ),
                metadata={
                    "fallback_created": True,
                    "video_task_id": task.id,
                },
                request_headers=(
                    (task.request_metadata or {}).get("request_headers")
                    if isinstance(task.request_metadata, dict)
                    else None
                ),
                request_body=task.original_request_body,
                provider_request_headers=None,
                response_headers=None,
                client_response_headers=None,
                response_body=None,
                request_id=request_id,
                provider_id=task.provider_id,
                provider_endpoint_id=task.endpoint_id,
                provider_api_key_id=task.key_id,
                status="completed" if task.status == "completed" else "failed",
                target_model=None,
            )
            return True
        except Exception as exc:
            logger.exception(
                "Failed to create fallback usage for video task=%s: %s",
                task.id,
                str(exc),
            )
            return False
