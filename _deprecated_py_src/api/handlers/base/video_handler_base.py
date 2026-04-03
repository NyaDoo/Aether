"""
Video Handler 基类

定义视频生成相关操作的统一接口。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy.orm import Session

from src.config.settings import config
from src.core.api_format.conversion.internal_video import InternalVideoTask, VideoStatus
from src.core.exceptions import ProviderNotAvailableException
from src.core.logger import logger
from src.core.video_utils import (
    extract_short_id_from_operation,
    normalize_gemini_operation_id,
    sanitize_error_message,
)
from src.models.database import ApiKey, ProviderAPIKey, ProviderEndpoint, User, VideoTask
from src.services.billing.rule_service import BillingRuleLookupResult
from src.services.scheduling.aware_scheduler import ProviderCandidate

if TYPE_CHECKING:
    from src.services.candidate.submit import SubmitOutcome


class VideoHandlerBase(ABC):
    """视频处理器基类"""

    FORMAT_ID: str = ""

    def __init__(
        self,
        db: Session,
        user: User,
        api_key: ApiKey,
        request_id: str,
        client_ip: str,
        user_agent: str,
        start_time: float,
        allowed_api_formats: list[str] | None = None,
    ):
        self.db = db
        self.user = user
        self.api_key = api_key
        self.request_id = request_id
        self.client_ip = client_ip
        self.user_agent = user_agent
        self.start_time = start_time
        self.allowed_api_formats = allowed_api_formats or [self.FORMAT_ID]

    @abstractmethod
    async def handle_create_task(
        self,
        *,
        http_request: Request,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """创建视频任务"""

    @abstractmethod
    async def handle_get_task(
        self,
        *,
        task_id: str,
        http_request: Request,
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """获取视频任务状态"""

    @abstractmethod
    async def handle_list_tasks(
        self,
        *,
        http_request: Request,
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """列出任务"""

    @abstractmethod
    async def handle_cancel_task(
        self,
        *,
        task_id: str,
        http_request: Request,
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """取消任务"""

    async def handle_delete_task(
        self,
        *,
        task_id: str,
        http_request: Request,
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """删除已完成或失败的视频任务 - 可选实现"""
        raise HTTPException(status_code=501, detail="Delete not supported for this provider")

    async def handle_remix_task(
        self,
        *,
        task_id: str,
        http_request: Request,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> JSONResponse:
        """Remix 任务（基于已完成视频创建新视频）- 可选实现"""
        raise HTTPException(status_code=501, detail="Remix not supported for this provider")

    @abstractmethod
    async def handle_download_content(
        self,
        *,
        task_id: str,
        http_request: Request,
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
    ) -> Response | StreamingResponse:
        """下载视频内容"""

    def _build_error_response(self, response: "httpx.Response") -> JSONResponse:
        """
        构建脱敏后的错误响应

        子类可重写 _format_error_payload 自定义错误格式。
        """
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                error_data = response.json()
                if isinstance(error_data, dict) and "error" in error_data:
                    payload = self._format_error_payload(error_data["error"], response.status_code)
                    return JSONResponse(
                        status_code=response.status_code,
                        content={"error": payload},
                    )
            except (ValueError, KeyError, TypeError):
                pass
        message = sanitize_error_message(response.text or "Upstream error")
        fallback_payload = self._format_error_payload({"message": message}, response.status_code)
        return JSONResponse(
            status_code=response.status_code,
            content={"error": fallback_payload},
        )

    async def _try_rust_sync_http_response(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any = None,
        provider_name: str | None = None,
        provider_id: str | None = None,
        endpoint_id: str | None = None,
        key_id: str | None = None,
        provider_api_format: str | None = None,
        client_api_format: str | None = None,
        model_name: str | None = None,
        content_type: str | None = None,
        content_encoding: str | None = None,
        proxy: Any = None,
        tls_profile: str | None = None,
        request_timeout_ms: int = 300_000,
        connect_timeout_ms: int = 30_000,
        pool_timeout_ms: int = 30_000,
        log_label: str = "VideoRequest",
    ) -> httpx.Response:
        from src.services.request.execution_runtime_plan import (
            ExecutionPlan,
            ExecutionPlanTimeouts,
            build_execution_plan_body,
        )
        from src.services.request.execution_runtime_client import (
            ExecutionRuntimeClient,
            ExecutionRuntimeClientError,
        )

        resolved_provider_name = str(provider_name or "").strip() or None
        if config.execution_runtime_backend != "rust":
            raise ProviderNotAvailableException(
                "Video 请求仅支持 Rust executor",
                provider_name=resolved_provider_name,
                upstream_response=f"executor_backend={config.execution_runtime_backend}",
            )

        request_headers = dict(headers)
        if (
            body is not None
            and content_type
            and not any(str(key).lower() == "content-type" for key in request_headers)
        ):
            request_headers["content-type"] = content_type

        try:
            plan = ExecutionPlan(
                request_id=str(self.request_id or ""),
                candidate_id=None,
                provider_name=str(provider_name or ""),
                provider_id=str(provider_id or ""),
                endpoint_id=str(endpoint_id or ""),
                key_id=str(key_id or ""),
                method=str(method or "POST").upper(),
                url=url,
                headers=request_headers,
                body=build_execution_plan_body(body, content_type=content_type),
                stream=False,
                provider_api_format=str(provider_api_format or self.FORMAT_ID),
                client_api_format=str(client_api_format or self.FORMAT_ID),
                model_name=str(model_name or ""),
                content_type=content_type,
                content_encoding=content_encoding,
                proxy=proxy,
                tls_profile=tls_profile,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=connect_timeout_ms,
                    read_ms=request_timeout_ms,
                    write_ms=request_timeout_ms,
                    pool_ms=pool_timeout_ms,
                    total_ms=request_timeout_ms,
                ),
            )
        except Exception as exc:
            logger.warning(
                "[{}] Rust execution plan build failed request_id={} method={} url={}: {}",
                log_label,
                self.request_id,
                method,
                url,
                sanitize_error_message(str(exc)),
            )
            raise ProviderNotAvailableException(
                "Rust executor 请求计划构建失败",
                provider_name=resolved_provider_name,
                upstream_response=sanitize_error_message(str(exc)),
            ) from exc

        try:
            rust_result = await ExecutionRuntimeClient().execute_sync_json(plan)
        except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning(
                "[{}] Rust executor unavailable request_id={} method={} url={}: {}",
                log_label,
                self.request_id,
                method,
                url,
                sanitize_error_message(str(exc)),
            )
            raise ProviderNotAvailableException(
                "执行器暂时不可用，请稍后重试",
                provider_name=resolved_provider_name,
                upstream_response=sanitize_error_message(str(exc)),
            ) from exc

        response_headers = dict(rust_result.headers)
        if rust_result.response_json is not None:
            response_headers.setdefault("content-type", "application/json")
            response_body = json.dumps(rust_result.response_json, ensure_ascii=False).encode(
                "utf-8"
            )
        elif rust_result.response_body_bytes is not None:
            response_body = rust_result.response_body_bytes
        else:
            response_body = b""

        return httpx.Response(
            status_code=rust_result.status_code,
            request=httpx.Request(str(method or "POST").upper(), url, headers=request_headers),
            headers=response_headers,
            content=response_body,
        )

    def _format_error_payload(self, error: dict[str, Any], status_code: int) -> dict[str, Any]:
        """
        格式化错误负载，子类可重写以匹配特定 API 格式

        默认返回 OpenAI 风格格式。
        """
        return {
            "type": error.get("type", "upstream_error"),
            "message": sanitize_error_message(error.get("message", "Request failed")),
        }

    def _get_task(self, task_id: str) -> VideoTask:
        """通过 UUID 查找任务（OpenAI Sora 风格）"""
        task = (
            self.db.query(VideoTask)
            .filter(VideoTask.id == task_id, VideoTask.user_id == self.user.id)
            .first()
        )
        if not task:
            raise HTTPException(status_code=404, detail="Video task not found")
        return task

    def _get_endpoint_and_key(self, task: VideoTask) -> tuple[ProviderEndpoint, ProviderAPIKey]:
        endpoint = (
            self.db.query(ProviderEndpoint).filter(ProviderEndpoint.id == task.endpoint_id).first()
        )
        key = self.db.query(ProviderAPIKey).filter(ProviderAPIKey.id == task.key_id).first()
        if not endpoint or not key:
            raise HTTPException(status_code=500, detail="Provider endpoint or key not found")
        return endpoint, key

    def _task_to_internal(self, task: VideoTask) -> InternalVideoTask:
        try:
            status = VideoStatus(task.status)
        except ValueError:
            status = VideoStatus.PENDING
        return InternalVideoTask(
            id=task.id,  # OpenAI Sora 使用 UUID
            external_id=task.external_task_id,
            status=status,
            progress_percent=task.progress_percent or 0,
            progress_message=task.progress_message,
            video_url=task.video_url,
            video_urls=task.video_urls or [],
            created_at=task.created_at,
            completed_at=task.completed_at,
            error_code=task.error_code,
            error_message=task.error_message,
            extra={"model": task.model},
        )

    def _finalize_usage_on_submit_failure(
        self,
        candidate_keys: list[dict[str, Any]],
        status_code: int | None,
    ) -> None:
        """
        提交失败时结算 pending usage（避免遗留 pending 状态）。

        从 candidate_keys 中提取最后尝试的 provider 信息，更新 Usage 记录。
        """
        from src.services.usage.service import UsageService

        # 提取 provider 信息：优先取最后一个有 attempt 的候选
        provider_name = "unknown"
        provider_id = None
        endpoint_id = None
        key_id = None

        for ck in reversed(candidate_keys):
            if ck.get("attempt_status") or ck.get("selected"):
                provider_name = ck.get("provider_name") or "unknown"
                provider_id = ck.get("provider_id")
                endpoint_id = ck.get("endpoint_id")
                key_id = ck.get("key_id")
                break

        try:
            # 更新 usage 状态并设置 provider 信息
            UsageService.update_usage_status(
                self.db,
                request_id=self.request_id,
                status="failed",
                error_message=f"submit_failed (status_code={status_code or 'unknown'})",
                provider=provider_name,
                provider_id=provider_id,
                provider_endpoint_id=endpoint_id,
                provider_api_key_id=key_id,
                status_code=status_code,
            )
        except Exception as exc:
            logger.warning(
                "Failed to finalize usage on submit failure: request_id={}, error={}",
                self.request_id,
                sanitize_error_message(str(exc)),
            )

    def _build_billing_rule_snapshot(
        self, rule_lookup: BillingRuleLookupResult | None
    ) -> dict[str, Any]:
        """
        构建 billing_rule 快照，用于冻结到视频任务的 request_metadata 中。

        快照确保异步任务完成时使用创建时刻的计费规则，避免规则变更导致成本计算不一致。
        """
        if not rule_lookup:
            return {"status": "no_rule"}

        rule = rule_lookup.rule
        return {
            "status": "ok",
            "scope": rule_lookup.scope,
            "effective_task_type": rule_lookup.effective_task_type,
            "rule_id": rule.id,
            "rule_name": rule.name,
            "expression": rule.expression,
            "variables": rule.variables,
            "dimension_mappings": rule.dimension_mappings,
        }

    async def _submit_with_failover(
        self,
        *,
        api_format: str,
        model_name: str,
        task_type: str,
        submit_func: Callable[[ProviderCandidate], Awaitable["httpx.Response"]],
        extract_external_task_id: Callable[[dict[str, Any]], str | None],
        supported_auth_types: set[str] | None,
        allow_format_conversion: bool = False,
        capability_requirements: dict[str, bool] | None = None,
        max_candidates: int = 10,
    ) -> "SubmitOutcome | JSONResponse":
        """
        提交阶段故障转移（只负责拿到 external_task_id）。

        返回：
        - 成功：SubmitOutcome
        - 命中上游终止规则：直接返回脱敏后的 JSONResponse（保留 API 格式差异）

        失败时：
        - 无可用候选 / 全部失败：抛 HTTPException(503)
        """
        # 延迟导入，避免 handler 基类层引入过多依赖导致循环
        from src.services.candidate.submit import (
            AllCandidatesFailedError,
            SubmitOutcome,
            UpstreamClientRequestError,
        )

        # 统一入口：总是通过 TaskService（内部可继续委托 CandidateService，便于逐步内核统一）
        from src.services.task import TaskService

        submitter: Any = TaskService(self.db)
        submit_call = submitter.submit_with_failover

        try:
            return await submit_call(
                api_format=api_format,
                model_name=model_name,
                affinity_key=str(self.api_key.id),
                user_api_key=self.api_key,
                request_id=self.request_id,
                task_type=task_type,
                submit_func=submit_func,
                extract_external_task_id=extract_external_task_id,
                supported_auth_types=supported_auth_types,
                allow_format_conversion=allow_format_conversion,
                capability_requirements=capability_requirements,
                max_candidates=max_candidates,
            )
        except UpstreamClientRequestError as exc:
            # 将 pending usage 结算为 failed，并记录 provider 信息
            self._finalize_usage_on_submit_failure(exc.candidate_keys, exc.response.status_code)
            return self._build_error_response(exc.response)
        except AllCandidatesFailedError as exc:
            # 将 pending usage 结算为 failed
            self._finalize_usage_on_submit_failure(exc.candidate_keys, exc.last_status_code)
            detail = "No available provider for video generation"
            if config.billing_require_rule:
                detail = "No available provider with billing rule for video generation"
            # 记录候选信息到日志
            logger.warning(
                "[VideoHandler] All candidates failed: reason={}, candidate_keys={}",
                exc.reason,
                exc.candidate_keys,
            )
            # 创建带有 candidate_keys 的 HTTPException
            http_exc = HTTPException(status_code=503, detail=detail)
            http_exc.candidate_keys = exc.candidate_keys  # type: ignore[attr-defined]
            raise http_exc
        except ProviderNotAvailableException:
            detail = "No available provider for video generation"
            if config.billing_require_rule:
                detail = "No available provider with billing rule for video generation"
            raise HTTPException(status_code=503, detail=detail)


__all__ = ["VideoHandlerBase", "normalize_gemini_operation_id", "sanitize_error_message"]
