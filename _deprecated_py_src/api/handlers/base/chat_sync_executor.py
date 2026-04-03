"""
ChatSyncExecutor - 非流式请求执行器

从 ChatHandlerBase.process_sync() 提取的独立类，负责：
- 非流式请求的完整执行流程（请求构建、发送、响应解析）
- 通过 SyncRequestContext 管理可变状态（替代原来的 nonlocal 变量）
- 异常处理与 telemetry 记录
- 流式失败记录（_record_stream_failure）
- HTTP 错误文本提取（_extract_error_text）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx
from fastapi.responses import JSONResponse

from src.api.handlers.base.chat_error_utils import (
    _build_error_json_payload,
    _get_error_status_code,
)
from src.api.handlers.base.parsers import get_parser_for_format
from src.api.handlers.base.stream_context import (
    StreamContext,
    extract_proxy_timing,
    is_format_converted,
)
from src.api.handlers.base.utils import (
    build_json_response_for_client,
    filter_proxy_response_headers,
    get_format_converter_registry,
    resolve_client_accept_encoding,
    resolve_client_content_encoding,
)
from src.config.settings import config
from src.core.error_utils import extract_client_error_message
from src.core.exceptions import (
    EmbeddedErrorException,
    ProviderAuthException,
    ProviderNotAvailableException,
    ProviderRateLimitException,
    ProviderTimeoutException,
    ThinkingSignatureException,
    UpstreamClientException,
)
from src.core.logger import logger
from src.services.request.execution_runtime_plan import (
    ExecutionPlan,
    ExecutionPlanTimeouts,
    ExecutionProxySnapshot,
    PreparedExecutionPlan,
    build_execution_plan_body,
)
from src.services.request.execution_runtime_client import (
    ExecutionRuntimeClient,
    ExecutionRuntimeClientError,
)
from src.services.task.request_state import MutableRequestBodyState

if TYPE_CHECKING:
    from fastapi import Request

    from src.api.handlers.base.chat_handler_base import ChatHandlerBase
    from src.models.database import Provider, ProviderAPIKey, ProviderEndpoint
    from src.services.scheduling.aware_scheduler import ProviderCandidate


@dataclass
class SyncRequestContext:
    """同步请求的可变状态容器，替代原来的 nonlocal 变量"""

    provider_name: str | None = None
    response_json: dict[str, Any] | None = None
    status_code: int = 200
    response_headers: dict[str, str] = field(default_factory=dict)
    provider_request_headers: dict[str, str] = field(default_factory=dict)
    provider_request_body: dict[str, Any] | None = None
    provider_api_format_for_error: str | None = None
    client_api_format_for_error: str | None = None
    needs_conversion_for_error: bool = False
    provider_id: str | None = None
    endpoint_id: str | None = None
    key_id: str | None = None
    mapped_model_result: str | None = None
    sync_proxy_info: dict[str, Any] | None = None
    provider_response_json: dict[str, Any] | None = None  # 格式转换前的提供商原始响应
    pool_summary: dict[str, Any] | None = None


class ChatSyncExecutor:
    """非流式请求执行器，从 ChatHandlerBase 提取"""

    def __init__(self, handler: ChatHandlerBase) -> None:
        self._handler = handler
        self._ctx = SyncRequestContext()

    async def execute(
        self,
        request: Any,
        http_request: Request,
        original_headers: dict[str, Any],
        original_request_body: dict[str, Any],
        query_params: dict[str, str] | None = None,
        client_content_encoding: str | None = None,
        client_accept_encoding: str | None = None,
    ) -> JSONResponse:
        """处理非流式响应（原 process_sync 的完整逻辑）"""
        handler = self._handler
        logger.debug(f"开始非流式响应处理 ({handler.FORMAT_ID})")
        effective_client_content_encoding = resolve_client_content_encoding(
            original_headers,
            client_content_encoding,
        )
        effective_client_accept_encoding = resolve_client_accept_encoding(
            original_headers,
            client_accept_encoding,
        )

        # 转换请求格式
        converted_request = await handler._convert_request(request)
        model = getattr(converted_request, "model", original_request_body.get("model", "unknown"))
        api_format = handler.allowed_api_formats[0]

        # 提前创建 pending 记录，让前端可以立即看到"处理中"
        pending_usage_created = handler._create_pending_usage(
            model=model,
            is_stream=False,
            request_type="chat",
            api_format=handler.FORMAT_ID,
            request_headers=original_headers,
            request_body=original_request_body,
        )

        request_state = MutableRequestBodyState(original_request_body)

        # 捕获的上下文变量
        ctx = self._ctx

        async def sync_request_func(
            provider: Provider,
            endpoint: ProviderEndpoint,
            key: ProviderAPIKey,
            candidate: ProviderCandidate,
        ) -> dict[str, Any]:
            return await self._sync_request_func(
                provider,
                endpoint,
                key,
                candidate,
                model=model,
                api_format=api_format,
                original_headers=original_headers,
                request_state=request_state,
                query_params=query_params,
                client_content_encoding=effective_client_content_encoding,
            )

        try:
            # 解析能力需求
            capability_requirements = handler._resolve_capability_requirements(
                model_name=model,
                request_headers=original_headers,
                request_body=original_request_body,
            )
            preferred_key_ids = await handler._resolve_preferred_key_ids(
                model_name=model,
                request_body=original_request_body,
            )

            # 统一入口：总是通过 TaskService
            from src.services.task import TaskService
            from src.services.task.core.context import TaskMode

            exec_result = await TaskService(handler.db, handler.redis).execute(
                task_type="chat",
                task_mode=TaskMode.SYNC,
                api_format=api_format,
                model_name=model,
                user_api_key=handler.api_key,
                request_func=sync_request_func,
                request_id=handler.request_id,
                is_stream=False,
                capability_requirements=capability_requirements or None,
                preferred_key_ids=preferred_key_ids or None,
                request_body_state=request_state,
                request_headers=original_headers,
                request_body=original_request_body,
                # 预创建失败时，回退到 TaskService 侧创建，避免丢失 pending 状态。
                create_pending_usage=not pending_usage_created,
            )
            actual_provider_name = exec_result.provider_name or "unknown"
            ctx.provider_id = exec_result.provider_id
            ctx.endpoint_id = exec_result.endpoint_id
            ctx.key_id = exec_result.key_id

            ctx.provider_name = actual_provider_name
            response_time_ms = handler.elapsed_ms()

            # 确保 response_json 不为 None
            if ctx.response_json is None:
                ctx.response_json = {}

            # 规范化响应
            ctx.response_json = handler._normalize_response(ctx.response_json)

            # 提取 usage
            usage_info = handler._extract_usage(ctx.response_json)
            input_tokens = usage_info.get("input_tokens", 0)
            output_tokens = usage_info.get("output_tokens", 0)
            cache_creation_tokens = usage_info.get("cache_creation_input_tokens", 0)
            cached_tokens = usage_info.get("cache_read_input_tokens", 0)
            cache_creation_tokens_5m = usage_info.get("cache_creation_input_tokens_5m", 0)
            cache_creation_tokens_1h = usage_info.get("cache_creation_input_tokens_1h", 0)

            # 非流式成功时，返回给客户端的是提供商响应头（透传）
            # JSONResponse 会自动设置 content-type，但我们记录实际返回的完整头
            client_response_headers = filter_proxy_response_headers(ctx.response_headers)
            client_response_headers["content-type"] = "application/json"
            client_response = build_json_response_for_client(
                status_code=ctx.status_code,
                content=ctx.response_json,
                headers=client_response_headers,
                client_accept_encoding=effective_client_accept_encoding,
            )
            actual_client_response_headers = dict(client_response.headers)

            request_metadata = handler._build_request_metadata() or {}
            if ctx.sync_proxy_info:
                request_metadata["proxy"] = ctx.sync_proxy_info
            request_metadata = handler._merge_scheduling_metadata(
                request_metadata,
                exec_result=exec_result,
                selected_key_id=ctx.key_id,
            )
            total_cost = await handler.telemetry.record_success(  # noqa: F841
                provider=ctx.provider_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time_ms=response_time_ms,
                status_code=ctx.status_code,
                request_headers=original_headers,
                request_body=original_request_body,
                response_headers=ctx.response_headers,
                client_response_headers=actual_client_response_headers,
                response_body=ctx.provider_response_json or ctx.response_json,
                client_response_body=ctx.response_json if ctx.provider_response_json else None,
                provider_request_body=ctx.provider_request_body,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cached_tokens,
                cache_creation_tokens_5m=cache_creation_tokens_5m,
                cache_creation_tokens_1h=cache_creation_tokens_1h,
                is_stream=False,
                provider_request_headers=ctx.provider_request_headers,
                api_format=api_format,
                api_family=handler.api_family,
                endpoint_kind=handler.endpoint_kind,
                # 格式转换追踪
                endpoint_api_format=ctx.provider_api_format_for_error or None,
                has_format_conversion=is_format_converted(
                    ctx.provider_api_format_for_error, ctx.client_api_format_for_error
                ),
                provider_id=ctx.provider_id,
                provider_endpoint_id=ctx.endpoint_id,
                provider_api_key_id=ctx.key_id,
                # 模型映射信息
                target_model=ctx.mapped_model_result,
                request_metadata=request_metadata,
            )

            logger.debug(f"{handler.FORMAT_ID} 非流式响应完成")

            # 简洁的请求完成摘要
            logger.info(
                f"[OK] {handler.request_id[:8]} | {model} | "
                f"{ctx.provider_name or 'unknown'} | {response_time_ms}ms | "
                f"in:{input_tokens or 0} out:{output_tokens or 0}"
            )

            # 透传提供商的响应头
            return client_response

        except ThinkingSignatureException as e:
            # Thinking 签名错误：TaskService 层已处理整流重试但仍失败
            # 记录实际发送给 Provider 的请求体，便于排查问题根因
            response_time_ms = handler.elapsed_ms()
            request_metadata = handler._build_request_metadata() or {}
            if ctx.sync_proxy_info:
                request_metadata["proxy"] = ctx.sync_proxy_info
            request_metadata = handler._merge_scheduling_metadata(
                request_metadata,
                selected_key_id=ctx.key_id,
                pool_summary=ctx.pool_summary,
                fallback_from_request=True,
            )
            await handler.telemetry.record_failure(
                provider=ctx.provider_name or "unknown",
                model=model,
                response_time_ms=response_time_ms,
                status_code=e.status_code or 400,
                request_headers=original_headers,
                request_body=original_request_body,
                provider_request_body=ctx.provider_request_body,
                error_message=str(e),
                is_stream=False,
                provider_id=ctx.provider_id,
                provider_endpoint_id=ctx.endpoint_id,
                provider_api_key_id=ctx.key_id,
                request_metadata=request_metadata,
            )
            client_format = (ctx.client_api_format_for_error or "").upper()
            provider_format = (ctx.provider_api_format_for_error or client_format).upper()
            payload = _build_error_json_payload(
                e,
                client_format,
                provider_format,
                needs_conversion=ctx.needs_conversion_for_error,
            )
            return build_json_response_for_client(
                status_code=_get_error_status_code(e),
                content=payload,
                headers={"content-type": "application/json"},
                client_accept_encoding=effective_client_accept_encoding,
            )

        except UpstreamClientException as e:
            response_time_ms = handler.elapsed_ms()
            request_metadata = handler._build_request_metadata() or {}
            if ctx.sync_proxy_info:
                request_metadata["proxy"] = ctx.sync_proxy_info
            request_metadata = handler._merge_scheduling_metadata(
                request_metadata,
                selected_key_id=ctx.key_id,
                pool_summary=ctx.pool_summary,
                fallback_from_request=True,
            )
            client_format = (ctx.client_api_format_for_error or "").upper()
            provider_format = (ctx.provider_api_format_for_error or client_format).upper()
            payload = _build_error_json_payload(
                e,
                client_format,
                provider_format,
                needs_conversion=ctx.needs_conversion_for_error,
            )
            error_response = build_json_response_for_client(
                status_code=_get_error_status_code(e),
                content=payload,
                headers={"content-type": "application/json"},
                client_accept_encoding=effective_client_accept_encoding,
            )
            await handler.telemetry.record_failure(
                provider=ctx.provider_name or "unknown",
                model=model,
                response_time_ms=response_time_ms,
                status_code=_get_error_status_code(e),
                request_headers=original_headers,
                request_body=original_request_body,
                provider_request_body=ctx.provider_request_body,
                error_message=str(e),
                is_stream=False,
                api_format=api_format,
                api_family=handler.api_family,
                endpoint_kind=handler.endpoint_kind,
                provider_request_headers=ctx.provider_request_headers,
                response_headers=ctx.response_headers,
                client_response_headers=dict(error_response.headers),
                provider_id=ctx.provider_id,
                provider_endpoint_id=ctx.endpoint_id,
                provider_api_key_id=ctx.key_id,
                # 格式转换追踪
                endpoint_api_format=ctx.provider_api_format_for_error or None,
                has_format_conversion=is_format_converted(
                    ctx.provider_api_format_for_error, ctx.client_api_format_for_error
                ),
                target_model=ctx.mapped_model_result,
                request_metadata=request_metadata,
            )
            return error_response

        except Exception as e:
            response_time_ms = handler.elapsed_ms()

            status_code = 503
            if isinstance(e, ProviderAuthException):
                status_code = 503
            elif isinstance(e, ProviderRateLimitException):
                status_code = 429
            elif isinstance(e, ProviderTimeoutException):
                status_code = 504

            # 尝试从异常中提取响应头
            error_response_headers: dict[str, str] = {}
            if isinstance(e, ProviderRateLimitException) and e.response_headers:
                error_response_headers = e.response_headers
            elif isinstance(e, httpx.HTTPStatusError) and hasattr(e, "response"):
                error_response_headers = dict(e.response.headers)

            request_metadata = handler._build_request_metadata() or {}
            if ctx.sync_proxy_info:
                request_metadata["proxy"] = ctx.sync_proxy_info
            request_metadata = handler._merge_scheduling_metadata(
                request_metadata,
                selected_key_id=ctx.key_id,
                pool_summary=ctx.pool_summary,
                fallback_from_request=True,
            )
            await handler.telemetry.record_failure(
                provider=ctx.provider_name or "unknown",
                model=model,
                response_time_ms=response_time_ms,
                status_code=status_code,
                error_message=extract_client_error_message(e),
                request_headers=original_headers,
                request_body=original_request_body,
                provider_request_body=ctx.provider_request_body,
                is_stream=False,
                api_format=api_format,
                api_family=handler.api_family,
                endpoint_kind=handler.endpoint_kind,
                provider_request_headers=ctx.provider_request_headers,
                response_headers=error_response_headers,
                # 非流式失败返回给客户端的是 JSON 错误响应
                client_response_headers={"content-type": "application/json"},
                provider_id=ctx.provider_id,
                provider_endpoint_id=ctx.endpoint_id,
                provider_api_key_id=ctx.key_id,
                # 格式转换追踪
                endpoint_api_format=ctx.provider_api_format_for_error or None,
                has_format_conversion=is_format_converted(
                    ctx.provider_api_format_for_error, ctx.client_api_format_for_error
                ),
                # 模型映射信息
                target_model=ctx.mapped_model_result,
                request_metadata=request_metadata,
            )

            raise

    async def _sync_request_func(
        self,
        provider: Provider,
        endpoint: ProviderEndpoint,
        key: ProviderAPIKey,
        candidate: ProviderCandidate,
        *,
        model: str,
        api_format: Any,
        original_headers: dict[str, Any],
        request_state: MutableRequestBodyState,
        query_params: dict[str, str] | None = None,
        client_content_encoding: str | None = None,
    ) -> dict[str, Any]:
        """单次同步请求（原 sync_request_func 内嵌函数）"""
        prepared_plan = await self._build_sync_execution_plan(
            provider,
            endpoint,
            key,
            candidate,
            model=model,
            api_format=api_format,
            original_headers=original_headers,
            request_state=request_state,
            query_params=query_params,
            client_content_encoding=client_content_encoding,
        )
        return await self._execute_sync_plan(
            prepared_plan=prepared_plan,
            provider=provider,
            model=model,
        )

    async def _build_sync_execution_plan(
        self,
        provider: Provider,
        endpoint: ProviderEndpoint,
        key: ProviderAPIKey,
        candidate: ProviderCandidate,
        *,
        model: str,
        api_format: Any,
        original_headers: dict[str, Any],
        request_state: MutableRequestBodyState,
        query_params: dict[str, str] | None = None,
        client_content_encoding: str | None = None,
    ) -> PreparedExecutionPlan:
        """构建可序列化的执行计划，并保留本地执行所需的运行时上下文。"""
        handler = self._handler
        ctx = self._ctx

        ctx.provider_name = str(provider.name)
        ctx.provider_id = str(provider.id)
        ctx.endpoint_id = str(endpoint.id)
        ctx.key_id = str(key.id)
        provider_api_format = str(endpoint.api_format or api_format)
        client_api_format = api_format.value if hasattr(api_format, "value") else str(api_format)

        # 构建 Provider 请求（模型映射、格式转换、envelope 包装）
        prep = await handler._prepare_provider_request(
            model=model,
            provider=provider,
            endpoint=endpoint,
            key=key,
            working_request_body=request_state.build_attempt_body(),
            original_headers=original_headers,
            client_api_format=client_api_format,
            provider_api_format=provider_api_format,
            candidate=candidate,
            client_is_stream=False,
        )
        provider_api_format = prep.provider_api_format
        needs_conversion = prep.needs_conversion
        ctx.provider_api_format_for_error = provider_api_format
        ctx.client_api_format_for_error = client_api_format
        ctx.needs_conversion_for_error = needs_conversion
        mapped_model = prep.mapped_model
        if mapped_model:
            ctx.mapped_model_result = mapped_model
        request_body = prep.request_body
        url_model = prep.url_model
        envelope = prep.envelope
        upstream_is_stream = prep.upstream_is_stream
        auth_info = prep.auth_info
        tls_profile = prep.tls_profile

        # 构建请求（上游始终使用 header 认证，不跟随客户端的 query 方式）
        provider_payload, provider_hdrs = handler._request_builder.build(
            request_body,
            original_headers,
            endpoint,
            key,
            is_stream=upstream_is_stream,
            extra_headers=prep.extra_headers if prep.extra_headers else None,
            pre_computed_auth=auth_info.as_tuple() if auth_info else None,
            envelope=envelope,
            provider_api_format=prep.provider_api_format,
        )
        if upstream_is_stream:
            from src.core.api_format.headers import set_accept_if_absent

            set_accept_if_absent(provider_hdrs)

        ctx.provider_request_headers = provider_hdrs
        ctx.provider_request_body = provider_payload

        from src.services.provider.transport import (
            build_provider_url,
            redact_url_for_log,
        )

        url = build_provider_url(
            endpoint,
            query_params=query_params,
            path_params={"model": url_model},
            is_stream=upstream_is_stream,  # sync handler may still force upstream streaming
            key=key,
            decrypted_auth_config=auth_info.decrypted_auth_config if auth_info else None,
        )
        # 非流式：必须在 build_provider_url 调用后立即缓存（避免 contextvar 被后续调用覆盖）
        selected_base_url_cached = envelope.capture_selected_base_url() if envelope else None

        # 解析有效代理（Key 级别优先于 Provider 级别）
        from src.services.proxy_node.resolver import (
            build_proxy_url_async,
            get_proxy_label,
            get_system_proxy_config_async,
            resolve_effective_proxy,
            resolve_proxy_info_async,
        )

        _effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
        ctx.sync_proxy_info = await resolve_proxy_info_async(_effective_proxy)
        _proxy_label = get_proxy_label(ctx.sync_proxy_info)
        provider_type = str(getattr(provider, "provider_type", "") or "").lower()

        logger.info(
            f"  [{handler.request_id}] "
            f"发送{'上游流式(聚合)' if upstream_is_stream else '非流式'}请求: "
            f"Provider={provider.name}, 模型={model} -> {mapped_model or '无映射'}, "
            f"代理={_proxy_label}"
        )
        logger.debug(f"  [{handler.request_id}] 请求URL: {redact_url_for_log(url)}")

        # 解析 delegate 配置，用于后续本地执行或交给 Rust executor。
        from src.services.proxy_node.resolver import (
            resolve_delegate_config_async,
        )

        # 非流式请求使用 http_request_timeout 作为整体超时
        # 优先使用 Provider 配置，否则使用全局配置
        request_timeout = provider.request_timeout or config.http_request_timeout

        delegate_cfg = await resolve_delegate_config_async(_effective_proxy)
        is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
        effective_proxy_for_contract = _effective_proxy
        if not effective_proxy_for_contract or not effective_proxy_for_contract.get(
            "enabled", True
        ):
            effective_proxy_for_contract = await get_system_proxy_config_async()
        proxy_url: str | None = None
        if effective_proxy_for_contract and not is_tunnel_delegate:
            proxy_url = await build_proxy_url_async(effective_proxy_for_contract)

        return PreparedExecutionPlan(
            contract=ExecutionPlan(
                request_id=str(handler.request_id or ""),
                candidate_id=str(
                    getattr(candidate, "request_candidate_id", "")
                    or getattr(candidate, "id", "")
                    or ""
                )
                or None,
                provider_name=str(provider.name),
                provider_id=str(provider.id),
                endpoint_id=str(endpoint.id),
                key_id=str(key.id),
                method="POST",
                url=url,
                headers=dict(provider_hdrs),
                body=build_execution_plan_body(
                    provider_payload,
                    content_type=str(provider_hdrs.get("content-type") or "").strip() or None,
                ),
                stream=upstream_is_stream,
                provider_api_format=provider_api_format,
                client_api_format=client_api_format,
                model_name=str(model or ""),
                content_type=str(provider_hdrs.get("content-type") or "").strip() or None,
                content_encoding=client_content_encoding,
                proxy=ExecutionProxySnapshot.from_proxy_info(
                    ctx.sync_proxy_info,
                    proxy_url=proxy_url,
                    mode_override="tunnel" if is_tunnel_delegate else None,
                    node_id_override=(
                        str(delegate_cfg.get("node_id") or "").strip() or None
                        if is_tunnel_delegate
                        else None
                    ),
                ),
                tls_profile=tls_profile,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=int(config.http_connect_timeout * 1000),
                    read_ms=int(config.http_read_timeout * 1000),
                    write_ms=int(config.http_write_timeout * 1000),
                    pool_ms=int(config.http_pool_timeout * 1000),
                    total_ms=int(request_timeout * 1000),
                ),
            ),
            payload=provider_payload,
            headers=dict(provider_hdrs),
            upstream_is_stream=upstream_is_stream,
            needs_conversion=needs_conversion,
            provider_type=provider_type,
            request_timeout=request_timeout,
            delegate_config=delegate_cfg,
            proxy_config=_effective_proxy,
            envelope=envelope,
            selected_base_url=selected_base_url_cached,
            client_content_encoding=client_content_encoding,
            proxy_info=ctx.sync_proxy_info,
        )

    async def _execute_sync_plan(
        self,
        *,
        prepared_plan: PreparedExecutionPlan,
        provider: Provider,
        model: str,
    ) -> dict[str, Any]:
        handler = self._handler

        if not prepared_plan.remote_eligible:
            raise ProviderNotAvailableException(
                "执行器暂时不可用，请稍后重试",
                provider_name=str(provider.name),
                upstream_response="execution contract is not eligible for rust executor",
            )

        try:
            rust_result = await ExecutionRuntimeClient().execute_sync_json(prepared_plan.contract)
        except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
            logger.warning(
                "[{}] Rust executor unavailable: {}",
                handler.request_id,
                exc,
            )
            raise ProviderNotAvailableException(
                "执行器暂时不可用，请稍后重试",
                provider_name=str(provider.name),
                upstream_response=str(exc),
            ) from exc
        else:
            result = await self._finalize_rust_sync_result(
                prepared_plan=prepared_plan,
                provider=provider,
                model=model,
                status_code=rust_result.status_code,
                response_headers=rust_result.headers,
                response_json=rust_result.response_json,
                provider_response_json=rust_result.provider_response_json,
                response_body_bytes=rust_result.response_body_bytes,
            )
            logger.debug(
                "[{}] sync chat 请求由 Rust executor 执行完成",
                handler.request_id,
            )
            return result

    async def _finalize_rust_sync_result(
        self,
        *,
        prepared_plan: PreparedExecutionPlan,
        provider: Provider,
        model: str,
        status_code: int,
        response_headers: dict[str, str],
        response_json: dict[str, Any] | None,
        provider_response_json: dict[str, Any] | None = None,
        response_body_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        ctx = self._ctx

        synthetic_content = response_body_bytes
        if synthetic_content is None:
            synthetic_content = json.dumps(
                response_json or {},
                ensure_ascii=False,
            ).encode("utf-8")

        request = httpx.Request(
            prepared_plan.contract.method,
            prepared_plan.contract.url,
            headers=prepared_plan.headers,
        )
        synthetic_response = httpx.Response(
            status_code,
            request=request,
            headers=response_headers,
            content=synthetic_content,
        )

        ctx.status_code = status_code
        ctx.response_headers = dict(synthetic_response.headers)
        extract_proxy_timing(ctx.sync_proxy_info, ctx.response_headers)

        if prepared_plan.envelope:
            prepared_plan.envelope.on_http_status(
                base_url=prepared_plan.selected_base_url,
                status_code=ctx.status_code,
            )

        if status_code >= 400:
            error = httpx.HTTPStatusError(
                f"Upstream status error: {status_code}",
                request=request,
                response=synthetic_response,
            )
            error_body = ""
            try:
                if prepared_plan.envelope and hasattr(prepared_plan.envelope, "extract_error_text"):
                    error_body = await prepared_plan.envelope.extract_error_text(synthetic_response)
                else:
                    error_body = synthetic_response.text[:4000] if synthetic_response.text else ""
            except Exception:
                error_body = synthetic_response.text[:4000] if synthetic_response.text else ""
            error.upstream_response = error_body[:4000]  # type: ignore[attr-defined]
            raise error

        if prepared_plan.upstream_is_stream:
            if response_body_bytes is None:
                raise ExecutionRuntimeClientError("Rust executor stream result must contain body bytes")
            return await self._finalize_rust_stream_sync_result(
                prepared_plan=prepared_plan,
                provider=provider,
                model=model,
                response_body_bytes=response_body_bytes,
            )

        if response_json is None:
            raise ExecutionRuntimeClientError("Rust executor sync result must contain response_json")

        ctx.response_json = dict(response_json)
        if provider_response_json is not None:
            ctx.provider_response_json = dict(provider_response_json)

        if prepared_plan.envelope:
            ctx.response_json = prepared_plan.envelope.unwrap_response(ctx.response_json)
            prepared_plan.envelope.postprocess_unwrapped_response(
                model=model,
                data=ctx.response_json,
            )

        if isinstance(ctx.response_json, dict):
            parser = get_parser_for_format(ctx.provider_api_format_for_error or "")
            if parser.is_error_response(ctx.response_json):
                parsed = parser.parse_response(ctx.response_json, status_code)
                raise EmbeddedErrorException(
                    provider_name=str(provider.name),
                    error_code=parsed.embedded_status_code,
                    error_message=parsed.error_message,
                    error_status=parsed.error_type,
                )

        if ctx.needs_conversion_for_error and isinstance(ctx.response_json, dict):
            ctx.provider_response_json = ctx.response_json.copy()
            registry = get_format_converter_registry()
            ctx.response_json = registry.convert_response(
                ctx.response_json,
                ctx.provider_api_format_for_error or "",
                ctx.client_api_format_for_error or "",
                requested_model=model,
            )

        return ctx.response_json if isinstance(ctx.response_json, dict) else {}

    async def _finalize_rust_stream_sync_result(
        self,
        *,
        prepared_plan: PreparedExecutionPlan,
        provider: Provider,
        model: str,
        response_body_bytes: bytes,
    ) -> dict[str, Any]:
        async def _iter_body() -> Any:
            yield response_body_bytes

        return await self._aggregate_upstream_stream_response(
            byte_iter=_iter_body(),
            prepared_plan=prepared_plan,
            provider=provider,
            model=model,
        )

    async def _aggregate_upstream_stream_response(
        self,
        *,
        byte_iter: Any,
        prepared_plan: PreparedExecutionPlan,
        provider: Provider,
        model: str,
    ) -> dict[str, Any]:
        ctx = self._ctx

        provider_parser = (
            get_parser_for_format(ctx.provider_api_format_for_error)
            if ctx.provider_api_format_for_error
            else None
        )

        if (
            prepared_plan.provider_type == "kiro"
            and prepared_plan.envelope
            and prepared_plan.envelope.force_stream_rewrite()
        ):
            from src.services.provider.adapters.kiro.eventstream_rewriter import (
                apply_kiro_stream_rewrite,
            )

            byte_iter = apply_kiro_stream_rewrite(byte_iter, model=str(model or ""))

        from src.api.handlers.base.upstream_stream_bridge import (
            aggregate_upstream_stream_to_internal_response,
        )

        internal_resp = await aggregate_upstream_stream_to_internal_response(
            byte_iter,
            provider_api_format=ctx.provider_api_format_for_error or "",
            provider_name=str(provider.name),
            model=str(model or ""),
            request_id=str(self._handler.request_id or ""),
            envelope=prepared_plan.envelope,
            provider_parser=provider_parser,
        )

        registry = get_format_converter_registry()
        tgt_norm = (
            registry.get_normalizer(ctx.client_api_format_for_error)
            if ctx.client_api_format_for_error
            else None
        )
        if tgt_norm is None:
            raise RuntimeError(f"未注册 Normalizer: {ctx.client_api_format_for_error}")

        ctx.response_json = tgt_norm.response_from_internal(
            internal_resp,
            requested_model=model,
        )
        ctx.response_json = ctx.response_json if isinstance(ctx.response_json, dict) else {}
        return ctx.response_json

    async def _record_stream_failure(
        self,
        ctx: StreamContext,
        error: Exception,
        original_headers: dict[str, str],
        original_request_body: dict[str, Any],
    ) -> None:
        """记录流式请求失败"""
        handler = self._handler
        response_time_ms = handler.elapsed_ms()

        status_code = 503
        if isinstance(error, ThinkingSignatureException):
            status_code = 400
        elif isinstance(error, UpstreamClientException):
            status_code = _get_error_status_code(error)
        elif isinstance(error, ProviderAuthException):
            status_code = 503
        elif isinstance(error, ProviderRateLimitException):
            status_code = 429
        elif isinstance(error, ProviderTimeoutException):
            status_code = 504

        # 失败时返回给客户端的是 JSON 错误响应
        client_response_headers = {"content-type": "application/json"}

        stream_fail_metadata: dict[str, Any] | None = None
        if ctx.proxy_info:
            stream_fail_metadata = {"proxy": ctx.proxy_info}
        stream_fail_metadata = handler._merge_scheduling_metadata(
            stream_fail_metadata,
            selected_key_id=ctx.key_id,
            pool_summary=ctx.pool_summary,
            fallback_from_request=True,
        )

        await handler.telemetry.record_failure(
            provider=ctx.provider_name or "unknown",
            model=ctx.model,
            response_time_ms=response_time_ms,
            status_code=status_code,
            error_message=extract_client_error_message(error),
            request_headers=original_headers,
            request_body=original_request_body,
            provider_request_body=ctx.provider_request_body,
            is_stream=True,
            api_format=ctx.api_format,
            api_family=handler.api_family,
            endpoint_kind=handler.endpoint_kind,
            provider_request_headers=ctx.provider_request_headers,
            response_headers=ctx.response_headers,
            client_response_headers=client_response_headers,
            provider_id=ctx.provider_id,
            provider_endpoint_id=ctx.endpoint_id,
            provider_api_key_id=ctx.key_id,
            # 格式转换追踪
            endpoint_api_format=ctx.provider_api_format or None,
            has_format_conversion=ctx.has_format_conversion,
            target_model=ctx.mapped_model,
            request_metadata=stream_fail_metadata,
        )

    async def _extract_error_text(
        self,
        e: httpx.HTTPStatusError,
        *,
        envelope: Any = None,
    ) -> str:
        """从 HTTP 错误中提取错误文本"""
        if envelope and hasattr(envelope, "extract_error_text"):
            return await envelope.extract_error_text(e)
        try:
            if hasattr(e.response, "is_stream_consumed") and not e.response.is_stream_consumed:
                error_bytes = await e.response.aread()
                return error_bytes.decode("utf-8", errors="replace")
            else:
                return e.response.text if hasattr(e.response, "_content") else "Unable to read"
        except Exception as decode_error:
            return f"Unable to read error: {decode_error}"
