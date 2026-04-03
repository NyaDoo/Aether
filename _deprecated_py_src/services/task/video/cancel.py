from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from src.config.settings import config
from src.core.logger import logger
from src.models.database import ProviderAPIKey, ProviderEndpoint
from src.services.usage.service import UsageService


class VideoTaskCancelService:
    """视频任务取消服务（上游取消 + 本地状态与计费回写）。"""

    def __init__(self, db: Session) -> None:
        self.db = db

    async def cancel_task(
        self,
        *,
        task: Any,
        task_id: str,
        original_headers: dict[str, str] | None = None,
    ) -> Any:
        """
        Cancel a video task (best-effort) and void its Usage (no charge).

        Returns:
        - None on success
        - upstream httpx.Response when upstream returns an error (status >= 400)
        """
        import json
        from datetime import datetime, timezone

        import httpx
        from fastapi import HTTPException

        from src.core.api_format import (
            build_upstream_headers_for_endpoint,
            get_extra_headers_from_endpoint,
            make_signature_key,
        )
        from src.core.api_format.conversion.internal_video import VideoStatus
        from src.core.crypto import crypto_service
        from src.services.provider.auth import get_provider_auth
        from src.services.provider.transport import build_provider_url

        current_status = str(getattr(task, "status", "") or "")
        non_cancellable_statuses = {
            VideoStatus.COMPLETED.value,
            VideoStatus.FAILED.value,
            VideoStatus.CANCELLED.value,
            VideoStatus.EXPIRED.value,
        }
        if current_status in non_cancellable_statuses:
            raise HTTPException(
                status_code=409,
                detail=f"Task cannot be cancelled in status: {current_status}",
            )

        external_task_id = getattr(task, "external_task_id", None)
        if not external_task_id:
            raise HTTPException(status_code=500, detail="Task missing external_task_id")

        endpoint = (
            self.db.query(ProviderEndpoint).filter(ProviderEndpoint.id == task.endpoint_id).first()
        )
        key = self.db.query(ProviderAPIKey).filter(ProviderAPIKey.id == task.key_id).first()
        if not endpoint or not key:
            raise HTTPException(status_code=500, detail="Provider endpoint or key not found")
        if not getattr(key, "api_key", None):
            raise HTTPException(status_code=500, detail="Provider key not configured")

        upstream_key = crypto_service.decrypt(key.api_key)
        extra_headers = get_extra_headers_from_endpoint(endpoint)

        raw_family = str(getattr(endpoint, "api_family", "") or "").strip().lower()
        raw_kind = str(getattr(endpoint, "endpoint_kind", "") or "").strip().lower()
        provider_format = (
            make_signature_key(raw_family, raw_kind)
            if raw_family and raw_kind
            else str(
                getattr(endpoint, "api_format", "")
                or getattr(task, "provider_api_format", "")
                or ""
            )
        )
        provider_format_norm = provider_format.strip().lower()

        headers = build_upstream_headers_for_endpoint(
            original_headers or {},
            provider_format,
            upstream_key,
            endpoint_headers=extra_headers,
            header_rules=getattr(endpoint, "header_rules", None),
        )

        async def _try_rust_cancel_response(
            *,
            method: str,
            url: str,
            request_headers: dict[str, str],
            body: Any,
            content_type: str | None = None,
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

            if config.execution_runtime_backend != "rust":
                return httpx.Response(
                    status_code=503,
                    request=httpx.Request(method, url, headers=request_headers),
                    json={"error": {"message": "Video 取消仅支持 Rust executor"}},
                )

            final_headers = dict(request_headers)
            if (
                body is not None
                and content_type
                and not any(str(key).lower() == "content-type" for key in final_headers)
            ):
                final_headers["content-type"] = content_type

            try:
                result = await ExecutionRuntimeClient().execute_sync_json(
                    ExecutionPlan(
                        request_id=str(getattr(task, "request_id", "") or task_id),
                        candidate_id=None,
                        provider_name=provider_format_norm.split(":", 1)[0],
                        provider_id=str(getattr(endpoint, "provider_id", "") or ""),
                        endpoint_id=str(getattr(endpoint, "id", "") or ""),
                        key_id=str(getattr(key, "id", "") or ""),
                        method=method,
                        url=url,
                        headers=final_headers,
                        body=build_execution_plan_body(body, content_type=content_type),
                        stream=False,
                        provider_api_format=provider_format,
                        client_api_format=provider_format,
                        model_name=str(getattr(task, "model", "") or ""),
                        content_type=content_type,
                        timeouts=ExecutionPlanTimeouts(
                            connect_ms=30_000,
                            read_ms=300_000,
                            write_ms=300_000,
                            pool_ms=30_000,
                            total_ms=300_000,
                        ),
                    )
                )
            except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[VideoCancel] Rust executor unavailable task={} method={} url={}: {}",
                    getattr(task, "id", task_id),
                    method,
                    url,
                    str(exc),
                )
                return httpx.Response(
                    status_code=503,
                    request=httpx.Request(method, url, headers=final_headers),
                    json={"error": {"message": "执行器暂时不可用，请稍后重试"}},
                )

            response_headers = dict(result.headers)
            if result.response_json is not None:
                response_headers.setdefault("content-type", "application/json")
                response_body = json.dumps(result.response_json, ensure_ascii=False).encode("utf-8")
            elif result.response_body_bytes is not None:
                response_body = result.response_body_bytes
            else:
                response_body = b""

            return httpx.Response(
                status_code=result.status_code,
                request=httpx.Request(method, url, headers=final_headers),
                headers=response_headers,
                content=response_body,
            )

        if provider_format_norm.startswith("openai:"):
            upstream_url = build_provider_url(endpoint, is_stream=False, key=key)
            upstream_url = f"{upstream_url.rstrip('/')}/{str(external_task_id).lstrip('/')}"
            response = await _try_rust_cancel_response(
                method="DELETE",
                url=upstream_url,
                request_headers=headers,
                body=None,
            )
            if response.status_code >= 400:
                return response

        elif provider_format_norm.startswith("gemini:"):
            # Gemini cancel endpoint supports both:
            # - operations/{id}:cancel
            # - models/{model}/operations/{id}:cancel
            operation_name = str(external_task_id)
            if not (
                operation_name.startswith("operations/") or operation_name.startswith("models/")
            ):
                operation_name = f"operations/{operation_name}"

            base = (
                getattr(endpoint, "base_url", None) or "https://generativelanguage.googleapis.com"
            ).rstrip("/")
            if base.endswith("/v1beta"):
                base = base[: -len("/v1beta")]
            upstream_url = f"{base}/v1beta/{operation_name}:cancel"

            auth_info = await get_provider_auth(endpoint, key)
            if auth_info:
                headers.pop("x-goog-api-key", None)
                headers[auth_info.auth_header] = auth_info.auth_value

            response = await _try_rust_cancel_response(
                method="POST",
                url=upstream_url,
                request_headers=headers,
                body={},
                content_type="application/json",
            )
            if response.status_code >= 400:
                return response

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Cancel not supported for provider format: {provider_format}",
            )

        now = datetime.now(timezone.utc)
        task.status = VideoStatus.CANCELLED.value
        task.completed_at = getattr(task, "completed_at", None) or now
        task.updated_at = now

        # Void Usage (no charge)
        try:
            voided = UsageService.finalize_void(
                self.db,
                request_id=task.request_id,
                reason="cancelled_by_user",
                finalized_at=task.completed_at,
            )
            if not voided:
                logger.warning(
                    "Skip voiding video usage because billing is already terminal: task_id={} request_id={}",
                    getattr(task, "id", task_id),
                    getattr(task, "request_id", None),
                )
        except Exception as exc:
            logger.warning(
                "Failed to void usage for cancelled task={}: {}",
                getattr(task, "id", task_id),
                str(exc),
            )

        self.db.commit()
        return None
