"""
Image Handler 基类

两条路径共用同一个客户端入口 `openai:image`：

1. 透传路径（非 Codex 候选）— 请求原样转发到上游 `/v1/images/generations`，
   流式 SSE 原样透传，尾帧解 usage 落库。

2. Codex 反代路径 — 把 OpenAI images payload 转成 Responses API + image_generation
   tool 的 payload，上游强制 stream=true，读取 SSE 并聚合
   `response.output_item.done` 里的 base64 为标准 images JSON。
   - 客户端 stream=false: 返回聚合 JSON
   - 客户端 stream=true:  伪造一帧 `image_generation.completed` + `[DONE]` SSE

握手阶段（首字节前）遇到上游错误仍可切换候选，和 chat 流式一致。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.api.handlers.base.request_builder import evaluate_condition
from src.clients.http_client import HTTPClientPool
from src.core.api_format import (
    ApiFamily,
    EndpointKind,
    build_upstream_headers_for_endpoint,
    get_extra_headers_from_endpoint,
    make_signature_key,
)
from src.core.api_format.headers import HOP_BY_HOP_HEADERS
from src.core.crypto import crypto_service
from src.core.logger import logger
from src.core.provider_types import ProviderType
from src.database.database import create_session
from src.models.database import ApiKey, ProviderAPIKey, ProviderEndpoint, User
from src.services.provider.auth import get_provider_auth
from src.services.provider.body_transformer import (
    get_body_transformer,
    transform_request_body,
)
from src.services.provider.transport import build_provider_url
from src.services.provider.upstream_headers import build_upstream_extra_headers
from src.services.proxy_node.resolver import (
    get_proxy_label,
    resolve_delegate_config_async,
    resolve_effective_proxy,
    resolve_proxy_info_async,
)
from src.services.scheduling.aware_scheduler import ProviderCandidate
from src.services.usage.service import UsageService


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """脱敏请求头用于审计。"""
    drop = {"authorization", "x-api-key", "cookie", "x-goog-api-key"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


def _is_codex_candidate(candidate: ProviderCandidate) -> bool:
    pt = str(getattr(candidate.provider, "provider_type", "") or "").strip().lower()
    return pt == ProviderType.CODEX.value


class ImageHandlerBase(ABC):
    """图像生成处理器基类。

    子类必须提供：
    - FORMAT_ID / API_FAMILY
    - _build_upstream_url(base_url) → str (透传路径使用；Codex 走 transport hook)
    """

    FORMAT_ID: str = "UNKNOWN"
    API_FAMILY: ClassVar[ApiFamily | None] = None
    ENDPOINT_KIND: ClassVar[EndpointKind] = EndpointKind.IMAGE

    # 流式握手阶段最多等待的时间（秒）——用于 failover 决策窗口
    STREAM_HANDSHAKE_TIMEOUT: ClassVar[float] = 30.0

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

    # ------------------------------------------------------------------ #
    # 子类抽象接口
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _build_upstream_url(
        self, base_url: str | None, *, path_suffix: str | None = None
    ) -> str:
        """透传路径的 URL 构建。Codex 候选不走这个，由 transport hook 改写到 /responses。

        path_suffix: "generations" | "edits"，由 adapter 根据路由传入。
        """

    # ------------------------------------------------------------------ #
    # 公共编排入口
    # ------------------------------------------------------------------ #

    async def process_sync(
        self,
        *,
        http_request: Request,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        query_params: dict[str, str] | None = None,
        multipart_context: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        raw_content_type: str | None = None,
        upstream_path_suffix: str | None = None,
    ) -> JSONResponse:
        """同步图像生成。成功后 candidate 选中 Codex 会在闭包内聚合 SSE 再返回 JSON。

        raw_body + raw_content_type: 用于 multipart /v1/images/edits 透传路径。
        upstream_path_suffix: "generations" | "edits"，决定透传 URL 尾部。
        """
        _ = query_params
        model = str(original_request_body.get("model") or "unknown")

        self._create_pending_usage(
            model=model,
            is_stream=False,
            original_headers=original_headers,
            original_request_body=original_request_body,
        )

        async def _submit(candidate: ProviderCandidate) -> httpx.Response:
            return await self._perform_candidate_call(
                candidate=candidate,
                original_headers=original_headers,
                original_request_body=original_request_body,
                multipart_context=multipart_context,
                raw_body=raw_body,
                raw_content_type=raw_content_type,
                upstream_path_suffix=upstream_path_suffix,
                client_is_stream=False,
            )

        def _extract_task_id(payload: dict[str, Any]) -> str | None:
            _ = payload
            return self.request_id

        from src.services.candidate.submit import (
            AllCandidatesFailedError,
            UpstreamClientRequestError,
        )
        from src.services.task import TaskService

        try:
            outcome = await TaskService(self.db).submit_with_failover(
                api_format=self.FORMAT_ID,
                model_name=model,
                affinity_key=str(self.api_key.id),
                user_api_key=self.api_key,
                request_id=self.request_id,
                task_type="image",
                submit_func=_submit,
                extract_external_task_id=_extract_task_id,
                # Codex 反代走 oauth；其他上游是 api_key
                supported_auth_types={"api_key", "oauth"},
                allow_format_conversion=False,
                max_candidates=10,
                request_body=original_request_body,
            )
        except UpstreamClientRequestError as exc:
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=exc.response.status_code,
                error_message="upstream_client_error",
                is_stream=False,
            )
            return self._build_error_response_from_httpx(exc.response)
        except AllCandidatesFailedError as exc:
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=exc.last_status_code or 503,
                error_message=f"all_candidates_failed:{exc.reason}",
                is_stream=False,
            )
            raise HTTPException(status_code=503, detail="No available provider for image generation")
        except HTTPException as exc:
            # ProviderNotAvailableException / ProxyException 都是 HTTPException 子类，
            # 必须在 re-raise 前把 pending usage 结算成 failed。
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=exc.status_code,
                error_message=detail[:500],
                is_stream=False,
            )
            raise
        except Exception as exc:
            logger.warning(
                "[ImageHandler] sync submit failed request_id={}: {}",
                self.request_id,
                exc,
            )
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=500,
                error_message=str(exc)[:500],
                is_stream=False,
            )
            raise HTTPException(status_code=500, detail="Image generation failed")

        response_body = outcome.upstream_payload or {}
        response_headers = outcome.upstream_headers or {}
        status_code = outcome.upstream_status_code or 200

        await self._record_success(
            model=model,
            candidate=outcome.candidate,
            original_request_body=original_request_body,
            original_headers=original_headers,
            upstream_request_headers=self._build_upstream_headers_safe(
                original_headers, outcome.candidate.endpoint, original_request_body
            ),
            response_body=response_body,
            response_headers=response_headers,
            status_code=status_code,
            is_stream=False,
            has_format_conversion=_is_codex_candidate(outcome.candidate),
        )

        return JSONResponse(response_body, status_code=status_code)

    async def process_stream(
        self,
        *,
        http_request: Request,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        query_params: dict[str, str] | None = None,
        multipart_context: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        raw_content_type: str | None = None,
        upstream_path_suffix: str | None = None,
    ) -> StreamingResponse | JSONResponse:
        """流式图像生成。Codex 候选会把响应合成一帧 completed SSE。"""
        _ = query_params
        model = str(original_request_body.get("model") or "unknown")

        self._create_pending_usage(
            model=model,
            is_stream=True,
            original_headers=original_headers,
            original_request_body=original_request_body,
        )

        try:
            candidates = await self._prepare_stream_candidates(model, original_request_body)
        except HTTPException as exc:
            # ProviderNotAvailableException 等也是 HTTPException 子类，必须先结算 pending
            detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=exc.status_code,
                error_message=detail[:500],
                is_stream=True,
            )
            raise
        except Exception as exc:
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=500,
                error_message=str(exc)[:500],
                is_stream=True,
            )
            raise HTTPException(status_code=500, detail="Image generation failed")

        if not candidates:
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=503,
                error_message="no_candidates",
                is_stream=True,
            )
            raise HTTPException(status_code=503, detail="No available provider for image generation")

        # 依次尝试候选（与 chat 流式 failover 行为一致：连续失败后做退避+客户端轮换）
        last_error: str | None = None
        last_status: int | None = None
        last_exc: Exception | None = None
        chosen: ProviderCandidate | None = None
        opened_stream: httpx.Response | None = None
        codex_transformed_body: dict[str, Any] | None = None
        consecutive_failures = 0

        for candidate in candidates:
            # 在尝试本候选前按 chat 同款节奏做退避+客户端轮换
            await self._apply_stream_retry_pacing(
                candidate=candidate,
                consecutive_failures=consecutive_failures,
                error=last_exc,
            )

            # 调试日志：带代理标签，便于生产排障
            try:
                proxy_info = await resolve_proxy_info_async(
                    self._resolve_effective_proxy(candidate)
                )
                logger.debug(
                    "  [{}] 图像流式请求 provider={} 模型={} 代理={}",
                    self.request_id,
                    getattr(candidate.provider, "name", "?"),
                    str(original_request_body.get("model") or "?"),
                    get_proxy_label(proxy_info),
                )
            except Exception:
                pass

            try:
                opened = await self._open_stream_for_candidate(
                    candidate=candidate,
                    original_headers=original_headers,
                    original_request_body=original_request_body,
                    multipart_context=multipart_context,
                    raw_body=raw_body,
                    raw_content_type=raw_content_type,
                    upstream_path_suffix=upstream_path_suffix,
                )
            except Exception as exc:
                last_error = f"open_failed:{exc}"
                last_exc = exc
                consecutive_failures += 1
                continue

            response, transformed_body = opened
            if response.status_code >= 400:
                err_bytes = await response.aread()
                await response.aclose()
                last_status = response.status_code
                last_error = err_bytes.decode(errors="ignore")[:500]
                # 让 FailoverEngine 的 stream-capacity 检测能嗅到错误文本
                last_exc = RuntimeError(last_error)
                consecutive_failures += 1
                continue

            chosen = candidate
            opened_stream = response
            codex_transformed_body = transformed_body
            break

        if opened_stream is None or chosen is None:
            await self._record_failure(
                model=model,
                original_request_body=original_request_body,
                original_headers=original_headers,
                status_code=last_status or 502,
                error_message=last_error or "all_candidates_failed",
                is_stream=True,
            )
            raise HTTPException(
                status_code=last_status or 502,
                detail="Upstream image generation failed",
            )

        # Codex 流式：聚合完整 SSE → 合成单帧返回
        if _is_codex_candidate(chosen):
            aggregated = await self._consume_codex_stream_to_dict(opened_stream)
            await self._record_success(
                model=model,
                candidate=chosen,
                original_request_body=original_request_body,
                original_headers=original_headers,
                upstream_request_headers=self._build_upstream_headers_safe(
                    original_headers,
                    chosen.endpoint,
                    codex_transformed_body or original_request_body,
                ),
                response_body=aggregated,
                response_headers=dict(opened_stream.headers),
                status_code=opened_stream.status_code,
                is_stream=True,
                has_format_conversion=True,
            )
            return self._synthesize_stream_response(aggregated, opened_stream.status_code)

        # 透传路径：原样转发 SSE 字节
        return self._passthrough_stream_response(
            stream_response=opened_stream,
            candidate=chosen,
            model=model,
            original_request_body=original_request_body,
            original_headers=original_headers,
        )

    # ------------------------------------------------------------------ #
    # 候选执行：同步路径与流式握手阶段共用
    # ------------------------------------------------------------------ #

    async def _perform_candidate_call(
        self,
        *,
        candidate: ProviderCandidate,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        multipart_context: dict[str, Any] | None,
        raw_body: bytes | None,
        raw_content_type: str | None,
        upstream_path_suffix: str | None,
        client_is_stream: bool,
    ) -> httpx.Response:
        """发起一次上游调用。Codex 走 Responses + SSE 聚合；其他走透传。"""
        _ = client_is_stream  # 当前只影响 stream 路径的调用者

        if _is_codex_candidate(candidate):
            aggregated, synthetic = await self._perform_codex_sync_call(
                candidate=candidate,
                original_headers=original_headers,
                original_request_body=original_request_body,
                multipart_context=multipart_context,
            )
            _ = aggregated
            return synthetic

        # 透传路径：标准 OpenAI 生图端点
        upstream_key, endpoint, _provider_key, decrypted_auth_config = (
            await self._resolve_upstream_key(candidate)
        )
        provider_type = str(getattr(candidate.provider, "provider_type", "") or "").lower() or None
        upstream_url = self._build_upstream_url(
            endpoint.base_url, path_suffix=upstream_path_suffix
        )
        if raw_body is not None:
            # multipart 透传：保留原 content-type（含 boundary），按字节转发
            headers = self._build_upstream_headers_raw(
                original_headers,
                upstream_key,
                endpoint,
                content_type=raw_content_type,
                provider_type=provider_type,
                decrypted_auth_config=decrypted_auth_config,
            )
            client = await self._get_upstream_client(candidate)
            return await client.post(upstream_url, headers=headers, content=raw_body)

        headers = self._build_upstream_headers(
            original_headers,
            upstream_key,
            endpoint,
            body=original_request_body,
            original_body=original_request_body,
            provider_type=provider_type,
            decrypted_auth_config=decrypted_auth_config,
        )
        client = await self._get_upstream_client(candidate)
        return await client.post(upstream_url, headers=headers, json=original_request_body)

    async def _perform_codex_sync_call(
        self,
        *,
        candidate: ProviderCandidate,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        multipart_context: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], httpx.Response]:
        """Codex 同步路径：转换 → 流式上行 → 聚合 → 构造 synthetic JSON Response。"""
        upstream_key, endpoint, provider_key, decrypted_auth_config = (
            await self._resolve_upstream_key(candidate)
        )

        provider_body = await self._build_codex_request_body(
            candidate=candidate,
            original_request_body=original_request_body,
            multipart_context=multipart_context,
        )
        upstream_url = build_provider_url(
            endpoint,
            is_stream=True,
            key=provider_key,
            decrypted_auth_config=decrypted_auth_config,
        )
        headers = self._build_upstream_headers(
            original_headers,
            upstream_key,
            endpoint,
            body=provider_body,
            original_body=original_request_body,
            provider_type="codex",
            decrypted_auth_config=decrypted_auth_config,
        )

        client = await self._get_upstream_client(candidate)
        request = client.build_request(
            "POST",
            upstream_url,
            headers=headers,
            json=provider_body,
            timeout=httpx.Timeout(self.STREAM_HANDSHAKE_TIMEOUT, read=None),
        )
        stream_response = await client.send(request, stream=True)

        if stream_response.status_code >= 400:
            err_bytes = await stream_response.aread()
            await stream_response.aclose()
            # 返回错误响应供 submit_with_failover 抉择
            return ({}, httpx.Response(
                status_code=stream_response.status_code,
                headers=stream_response.headers,
                content=err_bytes,
            ))

        try:
            aggregated = await self._consume_codex_stream_to_dict(stream_response)
        finally:
            await stream_response.aclose()

        synthetic = httpx.Response(
            status_code=200,
            headers={
                k: v
                for k, v in stream_response.headers.items()
                if k.lower() not in HOP_BY_HOP_HEADERS
            },
            content=json.dumps(aggregated, ensure_ascii=False).encode("utf-8"),
        )
        return aggregated, synthetic

    async def _open_stream_for_candidate(
        self,
        *,
        candidate: ProviderCandidate,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
        multipart_context: dict[str, Any] | None,
        raw_body: bytes | None = None,
        raw_content_type: str | None = None,
        upstream_path_suffix: str | None = None,
    ) -> tuple[httpx.Response, dict[str, Any] | None]:
        """建立一次流式上行连接。Codex 会改写 URL/body；multipart 透传保留原字节。"""
        upstream_key, endpoint, provider_key, decrypted_auth_config = (
            await self._resolve_upstream_key(candidate)
        )
        provider_type = str(getattr(candidate.provider, "provider_type", "") or "").lower() or None

        transformed_body: dict[str, Any] | None = None
        is_codex = _is_codex_candidate(candidate)

        client = await self._get_upstream_client(candidate)

        if is_codex:
            transformed_body = await self._build_codex_request_body(
                candidate=candidate,
                original_request_body=original_request_body,
                multipart_context=multipart_context,
            )
            upstream_url = build_provider_url(
                endpoint,
                is_stream=True,
                key=provider_key,
                decrypted_auth_config=decrypted_auth_config,
            )
            headers = self._build_upstream_headers(
                original_headers,
                upstream_key,
                endpoint,
                body=transformed_body,
                original_body=original_request_body,
                provider_type="codex",
                decrypted_auth_config=decrypted_auth_config,
            )
            request = client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                json=transformed_body,
                timeout=httpx.Timeout(self.STREAM_HANDSHAKE_TIMEOUT, read=None),
            )
            response = await client.send(request, stream=True)
            return response, transformed_body

        # 透传路径
        upstream_url = self._build_upstream_url(
            endpoint.base_url, path_suffix=upstream_path_suffix
        )
        if raw_body is not None:
            headers = self._build_upstream_headers_raw(
                original_headers,
                upstream_key,
                endpoint,
                content_type=raw_content_type,
                provider_type=provider_type,
                decrypted_auth_config=decrypted_auth_config,
            )
            request = client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                content=raw_body,
                timeout=httpx.Timeout(self.STREAM_HANDSHAKE_TIMEOUT, read=None),
            )
        else:
            headers = self._build_upstream_headers(
                original_headers,
                upstream_key,
                endpoint,
                body=original_request_body,
                original_body=original_request_body,
                provider_type=provider_type,
                decrypted_auth_config=decrypted_auth_config,
            )
            request = client.build_request(
                "POST",
                upstream_url,
                headers=headers,
                json=original_request_body,
                timeout=httpx.Timeout(self.STREAM_HANDSHAKE_TIMEOUT, read=None),
            )
        response = await client.send(request, stream=True)
        return response, None

    async def _build_codex_request_body(
        self,
        *,
        candidate: ProviderCandidate,
        original_request_body: dict[str, Any],
        multipart_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """调用注册的 body transformer；使用候选的 mapped_model。"""
        transformer = get_body_transformer("codex", self.FORMAT_ID)
        if transformer is None:
            raise HTTPException(
                status_code=500,
                detail="Codex image body transformer not registered",
            )
        mapped_model = await self._resolve_mapped_model(candidate, original_request_body)
        operation = "edit" if multipart_context else "generate"
        context = {
            "mapped_model": mapped_model,
            "operation": operation,
            "multipart": multipart_context,
        }
        return transform_request_body(
            provider_type="codex",
            endpoint_sig=self.FORMAT_ID,
            request_body=original_request_body,
            context=context,
        )

    async def _resolve_mapped_model(
        self,
        candidate: ProviderCandidate,
        request_body: dict[str, Any],
    ) -> str:
        """解析候选对应的 mapped_model。完整 fallback 链与 chat 对齐：

        1. candidate.mapping_matched_model（通配符/pool 侧匹配）
        2. ModelMapperMiddleware.get_mapping(source, provider_id) — 读 Model.provider_model_mappings
        3. 原始 source 模型名（最终兜底）
        """
        mapped = getattr(candidate, "mapping_matched_model", None)
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()

        source = str(request_body.get("model") or "").strip()
        provider_id = str(getattr(candidate.provider, "id", "") or "").strip()
        if source and provider_id:
            try:
                from src.services.model.mapper import ModelMapperMiddleware

                mapper = ModelMapperMiddleware(self.db)
                mapping = await mapper.get_mapping(source, provider_id)
                if mapping and getattr(mapping, "model", None) is not None:
                    affinity_key = self.api_key.id if self.api_key else None
                    picked = mapping.model.select_provider_model_name(
                        affinity_key, api_format=self.FORMAT_ID
                    )
                    if isinstance(picked, str) and picked.strip():
                        logger.debug(
                            "[ImageHandler] model mapping {} -> {} (provider={})",
                            source,
                            picked,
                            provider_id[:8],
                        )
                        return picked.strip()
            except Exception as exc:
                logger.warning(
                    "[ImageHandler] model mapping lookup failed for {}: {}",
                    source,
                    exc,
                )

        return source or "unknown"

    async def _consume_codex_stream_to_dict(
        self,
        stream_response: httpx.Response,
    ) -> dict[str, Any]:
        """读取 Codex Responses SSE 字节流并聚合成标准 images JSON。"""
        from src.services.provider.adapters.codex.image_transform import (
            aggregate_codex_image_sse,
        )

        return await aggregate_codex_image_sse(stream_response.aiter_bytes())

    def _synthesize_stream_response(
        self,
        aggregated: dict[str, Any],
        upstream_status_code: int,
    ) -> StreamingResponse:
        from src.services.provider.adapters.codex.image_transform import (
            synthesize_stream_completed_frames,
        )

        payload = synthesize_stream_completed_frames(aggregated)

        async def _iter() -> AsyncIterator[bytes]:
            yield payload

        return StreamingResponse(
            _iter(),
            status_code=upstream_status_code if upstream_status_code < 400 else 200,
            media_type="text/event-stream",
            headers={"content-type": "text/event-stream"},
        )

    def _passthrough_stream_response(
        self,
        *,
        stream_response: httpx.Response,
        candidate: ProviderCandidate,
        model: str,
        original_request_body: dict[str, Any],
        original_headers: dict[str, str],
    ) -> StreamingResponse:
        captured_usage: dict[str, Any] = {}
        captured_completed_frame: dict[str, Any] = {}

        async def _iter_sse() -> AsyncIterator[bytes]:
            buffer = b""
            try:
                async for chunk in stream_response.aiter_bytes():
                    yield chunk
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event_bytes, buffer = buffer.split(b"\n\n", 1)
                        self._maybe_capture_completion(
                            event_bytes, captured_usage, captured_completed_frame
                        )
            except Exception as exc:
                logger.warning(
                    "[ImageHandler] stream interrupted request_id={}: {}",
                    self.request_id,
                    exc,
                )
            finally:
                await stream_response.aclose()
                try:
                    await self._record_success(
                        model=model,
                        candidate=candidate,
                        original_request_body=original_request_body,
                        original_headers=original_headers,
                        upstream_request_headers=self._build_upstream_headers_safe(
                            original_headers, candidate.endpoint, original_request_body
                        ),
                        response_body=captured_completed_frame
                        or ({"usage": captured_usage} if captured_usage else {}),
                        response_headers=dict(stream_response.headers),
                        status_code=stream_response.status_code,
                        is_stream=True,
                        has_format_conversion=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "[ImageHandler] stream usage record failed request_id={}: {}",
                        self.request_id,
                        exc,
                    )

        safe_headers = {
            k: v
            for k, v in stream_response.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        safe_headers.setdefault("content-type", "text/event-stream")

        return StreamingResponse(
            _iter_sse(),
            status_code=stream_response.status_code,
            headers=safe_headers,
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------ #
    # 内部辅助
    # ------------------------------------------------------------------ #

    async def _prepare_stream_candidates(
        self,
        model: str,
        request_body: dict[str, Any],
    ) -> list[ProviderCandidate]:
        from src.clients.redis_client import get_redis_client_sync
        from src.services.candidate.resolver import CandidateResolver
        from src.services.scheduling.aware_scheduler import (
            get_cache_aware_scheduler,
        )
        from src.services.user.group_service import UserGroupService

        scheduling_mode = UserGroupService.resolve_effective_scheduling_mode(
            self.db, self.api_key.user
        )
        redis = get_redis_client_sync()
        scheduler = await get_cache_aware_scheduler(redis, scheduling_mode=scheduling_mode)
        resolver = CandidateResolver(db=self.db, cache_scheduler=scheduler)
        candidates, _ = await resolver.fetch_candidates(
            api_format=self.FORMAT_ID,
            model_name=model,
            affinity_key=str(self.api_key.id),
            user_api_key=self.api_key,
            request_id=self.request_id,
            is_stream=True,
            capability_requirements=None,
            request_body=request_body,
        )
        return candidates or []

    def _maybe_capture_completion(
        self,
        event_bytes: bytes,
        captured_usage: dict[str, Any],
        captured_frame: dict[str, Any],
    ) -> None:
        try:
            text = event_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return
        data_payloads: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                data_payloads.append(line[5:].strip())
        for raw in data_payloads:
            if not raw or raw == "[DONE]":
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            evt_type = obj.get("type") or ""
            if str(evt_type).endswith("completed"):
                captured_frame.update(obj)
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    captured_usage.update(usage)

    def _resolve_effective_proxy(
        self, candidate: ProviderCandidate
    ) -> dict[str, Any] | None:
        """Key 级代理优先于 Provider 级，行为与 chat handler 一致。"""
        return resolve_effective_proxy(
            getattr(candidate.provider, "proxy", None),
            getattr(candidate.key, "proxy", None),
        )

    async def _get_upstream_client(
        self, candidate: ProviderCandidate
    ) -> httpx.AsyncClient:
        """解析候选的有效代理 → 返回代理感知的上游 HTTP 客户端。

        与 chat handler 同一机制（Key 级代理覆盖 Provider 级，回落系统默认代理节点）。
        直连场景等价于默认客户端，对本地开发无影响。

        tls_profile 目前传 None：Codex provider 未注册 envelope，chat 路径
        同样得到 None，两端保持一致。后续 Codex 若引入 envelope.prepare_context，
        此处需对齐。
        """
        effective_proxy = self._resolve_effective_proxy(candidate)
        delegate_cfg = await resolve_delegate_config_async(effective_proxy)
        return await HTTPClientPool.get_upstream_client(
            delegate_cfg,
            proxy_config=effective_proxy,
            tls_profile=None,
        )

    async def _reset_upstream_client(self, candidate: ProviderCandidate) -> bool:
        """重建上游客户端，释放失联的 HTTP/2 连接与流配额。

        与 src/services/candidate/failover.py 的 _rotate_upstream_client 同一实现，
        用于连续失败时轮换连接池。
        """
        effective_proxy = self._resolve_effective_proxy(candidate)
        delegate_cfg = await resolve_delegate_config_async(effective_proxy)
        return await HTTPClientPool.reset_upstream_client(
            delegate_cfg,
            proxy_config=effective_proxy,
            tls_profile=None,
        )

    async def _apply_stream_retry_pacing(
        self,
        *,
        candidate: ProviderCandidate,
        consecutive_failures: int,
        error: Exception | None,
    ) -> None:
        """复用 FailoverEngine 的节奏策略：连续失败触发退避+客户端轮换。

        策略完全等同 chat 流式 failover（RETRY_BACKOFF_EVERY_FAILURES /
        RETRY_ROTATE_CLIENT_EVERY_FAILURES / STREAM_CAPACITY_BACKOFF_SECONDS）。
        """
        if consecutive_failures <= 0:
            return

        import asyncio

        from src.services.candidate.failover import FailoverEngine

        if FailoverEngine._should_rotate_upstream_client(
            consecutive_failures=consecutive_failures,
            error=error,
        ):
            rotated = await self._reset_upstream_client(candidate)
            if rotated:
                logger.warning(
                    "  [{}] 图像流式连续失败 {} 次，已重建上游客户端",
                    self.request_id,
                    consecutive_failures,
                )

        backoff_seconds = FailoverEngine._compute_retry_backoff_seconds(
            consecutive_failures=consecutive_failures,
            error=error,
        )
        if backoff_seconds > 0:
            logger.warning(
                "  [{}] 图像流式连续失败 {} 次，退避 {:.0f}ms 后继续",
                self.request_id,
                consecutive_failures,
                backoff_seconds * 1000,
            )
            await asyncio.sleep(backoff_seconds)

    async def _resolve_upstream_key(
        self, candidate: ProviderCandidate
    ) -> tuple[str, ProviderEndpoint, ProviderAPIKey, dict[str, Any] | None]:
        """解密/刷新上游凭证。

        Returns:
            (upstream_key, endpoint, key, decrypted_auth_config)
            - upstream_key: 上游请求所用的 token（OAuth 是 access_token；api_key 是解密后的 key）
            - decrypted_auth_config: OAuth key 的 auth_config（Codex 需要读 account_id）；api_key None
        """
        auth_type = str(getattr(candidate.key, "auth_type", "api_key") or "api_key").lower()

        if auth_type == "oauth":
            try:
                auth_info = await get_provider_auth(candidate.endpoint, candidate.key)
            except Exception as exc:
                logger.error(
                    "[ImageHandler] OAuth token resolve failed key_id={}: {}",
                    candidate.key.id,
                    exc,
                )
                raise HTTPException(
                    status_code=500, detail="Failed to resolve OAuth access token"
                )
            if auth_info is None or not auth_info.auth_value:
                raise HTTPException(
                    status_code=500, detail="OAuth access token unavailable"
                )
            token = auth_info.auth_value
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            return (
                token,
                candidate.endpoint,
                candidate.key,
                auth_info.decrypted_auth_config,
            )

        # api_key 路径
        try:
            upstream_key = crypto_service.decrypt(candidate.key.api_key)
        except Exception as exc:
            logger.error(
                "[ImageHandler] Failed to decrypt provider key id={}: {}",
                candidate.key.id,
                exc,
            )
            raise HTTPException(status_code=500, detail="Failed to decrypt provider key")
        return upstream_key, candidate.endpoint, candidate.key, None

    def _build_upstream_headers(
        self,
        original_headers: dict[str, str],
        upstream_key: str,
        endpoint: ProviderEndpoint,
        *,
        body: dict[str, Any] | None = None,
        original_body: dict[str, Any] | None = None,
        provider_type: str | None = None,
        decrypted_auth_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        extra_headers = dict(get_extra_headers_from_endpoint(endpoint) or {})
        endpoint_sig = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )

        # 调用 provider-specific headers hook（如 Codex 的 chatgpt-account-id 注入）
        if provider_type:
            try:
                provider_extra = build_upstream_extra_headers(
                    provider_type=provider_type,
                    endpoint_sig=endpoint_sig,
                    request_body=body,
                    original_headers=original_headers,
                    decrypted_auth_config=decrypted_auth_config,
                )
                if provider_extra:
                    extra_headers.update(provider_extra)
            except Exception as exc:
                logger.warning(
                    "[ImageHandler] provider header hook failed pt={} sig={}: {}",
                    provider_type,
                    endpoint_sig,
                    exc,
                )

        headers = build_upstream_headers_for_endpoint(
            original_headers,
            endpoint_sig,
            upstream_key,
            endpoint_headers=extra_headers,
            header_rules=getattr(endpoint, "header_rules", None),
            body=body,
            original_body=original_body,
            condition_evaluator=evaluate_condition,
        )
        # 无论客户端怎么传，上游 body 我们已经序列化为 JSON
        headers["content-type"] = "application/json"
        return headers

    def _build_upstream_headers_safe(
        self,
        original_headers: dict[str, str],
        endpoint: ProviderEndpoint,
        body: dict[str, Any],
        *,
        provider_type: str | None = None,
        decrypted_auth_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """构建用于 usage 记录的脱敏版本 header。"""
        return self._build_upstream_headers(
            original_headers,
            "",
            endpoint,
            body=body,
            original_body=body,
            provider_type=provider_type,
            decrypted_auth_config=decrypted_auth_config,
        )

    def _build_upstream_headers_raw(
        self,
        original_headers: dict[str, str],
        upstream_key: str,
        endpoint: ProviderEndpoint,
        *,
        content_type: str | None,
        provider_type: str | None = None,
        decrypted_auth_config: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """multipart 透传专用：保留原 content-type（含 boundary），不覆盖为 JSON。"""
        extra_headers = dict(get_extra_headers_from_endpoint(endpoint) or {})
        endpoint_sig = make_signature_key(
            str(getattr(endpoint, "api_family", "")).strip().lower(),
            str(getattr(endpoint, "endpoint_kind", "")).strip().lower(),
        )

        if provider_type:
            try:
                provider_extra = build_upstream_extra_headers(
                    provider_type=provider_type,
                    endpoint_sig=endpoint_sig,
                    request_body=None,
                    original_headers=original_headers,
                    decrypted_auth_config=decrypted_auth_config,
                )
                if provider_extra:
                    extra_headers.update(provider_extra)
            except Exception as exc:
                logger.warning(
                    "[ImageHandler] provider header hook failed pt={} sig={}: {}",
                    provider_type,
                    endpoint_sig,
                    exc,
                )

        headers = build_upstream_headers_for_endpoint(
            original_headers,
            endpoint_sig,
            upstream_key,
            endpoint_headers=extra_headers,
            header_rules=getattr(endpoint, "header_rules", None),
            body=None,
            original_body=None,
            condition_evaluator=evaluate_condition,
        )
        if content_type:
            headers["content-type"] = content_type
        return headers

    def _build_error_response_from_httpx(self, response: httpx.Response) -> JSONResponse:
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return JSONResponse(status_code=response.status_code, content=response.json())
            except Exception:
                pass
        return JSONResponse(
            status_code=response.status_code,
            content={
                "error": {
                    "type": "upstream_error",
                    "message": (response.text or "Upstream error")[:500],
                }
            },
        )

    # ------------------------------------------------------------------ #
    # Usage 记录
    # ------------------------------------------------------------------ #

    def _create_pending_usage(
        self,
        *,
        model: str,
        is_stream: bool,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
    ) -> None:
        """请求进入时同步写一条 pending usage，便于前端实时看到"处理中"。

        失败容忍：pending 写入异常不阻塞请求，只记日志。后续 _record_success /
        _record_failure 若找到相同 request_id 的记录会原位更新。
        """
        try:
            UsageService.create_pending_usage(
                db=self.db,
                request_id=self.request_id,
                user=self.user,
                api_key=self.api_key,
                model=model,
                is_stream=is_stream,
                request_type="image",
                api_format=self.FORMAT_ID,
                request_headers=_sanitize_headers(original_headers),
                request_body=original_request_body,
            )
        except Exception as exc:
            logger.warning(
                "[ImageHandler] create pending usage failed request_id={}: {}",
                self.request_id,
                exc,
            )

    async def _record_success(
        self,
        *,
        model: str,
        candidate: ProviderCandidate,
        original_request_body: dict[str, Any],
        original_headers: dict[str, str],
        upstream_request_headers: dict[str, str],
        response_body: dict[str, Any],
        response_headers: dict[str, Any],
        status_code: int,
        is_stream: bool,
        has_format_conversion: bool = False,
        use_isolated_session: bool = False,
    ) -> None:
        response_time_ms = int((time.time() - self.start_time) * 1000)
        usage = response_body.get("usage") if isinstance(response_body, dict) else None
        input_tokens = 0
        output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

        endpoint_api_format = make_signature_key(
            str(getattr(candidate.endpoint, "api_family", "")).strip().lower(),
            str(getattr(candidate.endpoint, "endpoint_kind", "")).strip().lower(),
        )

        async def _do(db: Session) -> None:
            await UsageService.record_usage(
                db=db,
                user=self.user,
                api_key=self.api_key,
                provider=candidate.provider.name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                request_type="image",
                api_format=self.FORMAT_ID,
                api_family=self.API_FAMILY.value if self.API_FAMILY else None,
                endpoint_kind=self.ENDPOINT_KIND.value,
                endpoint_api_format=endpoint_api_format,
                has_format_conversion=has_format_conversion,
                is_stream=is_stream,
                response_time_ms=response_time_ms,
                status_code=status_code,
                request_headers=_sanitize_headers(original_headers),
                request_body=original_request_body,
                provider_request_headers=_sanitize_headers(upstream_request_headers),
                response_headers=response_headers,
                response_body=response_body,
                request_id=self.request_id,
                provider_id=candidate.provider.id,
                provider_endpoint_id=candidate.endpoint.id,
                provider_api_key_id=candidate.key.id,
                status="completed",
            )
            db.commit()

        await self._run_usage_write(_do, use_isolated_session)

    async def _record_failure(
        self,
        *,
        model: str,
        original_request_body: dict[str, Any],
        original_headers: dict[str, str],
        status_code: int,
        error_message: str,
        is_stream: bool,
        use_isolated_session: bool = False,
    ) -> None:
        response_time_ms = int((time.time() - self.start_time) * 1000)

        async def _do(db: Session) -> None:
            await UsageService.record_usage(
                db=db,
                user=self.user,
                api_key=self.api_key,
                provider="unknown",
                model=model,
                input_tokens=0,
                output_tokens=0,
                request_type="image",
                api_format=self.FORMAT_ID,
                api_family=self.API_FAMILY.value if self.API_FAMILY else None,
                endpoint_kind=self.ENDPOINT_KIND.value,
                has_format_conversion=False,
                is_stream=is_stream,
                response_time_ms=response_time_ms,
                status_code=status_code,
                error_message=error_message[:500],
                request_headers=_sanitize_headers(original_headers),
                request_body=original_request_body,
                request_id=self.request_id,
                status="failed",
            )
            db.commit()

        await self._run_usage_write(_do, use_isolated_session)

    async def _run_usage_write(
        self,
        write_fn,
        use_isolated_session: bool,
    ) -> None:
        if use_isolated_session:
            db = create_session()
            try:
                await write_fn(db)
            except Exception as exc:
                logger.warning(
                    "[ImageHandler] isolated usage write failed request_id={}: {}",
                    self.request_id,
                    exc,
                )
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()
            return

        try:
            await write_fn(self.db)
        except Exception as exc:
            logger.warning(
                "[ImageHandler] usage write failed request_id={}: {}",
                self.request_id,
                exc,
            )
            try:
                self.db.rollback()
            except Exception:
                pass


__all__ = ["ImageHandlerBase"]
