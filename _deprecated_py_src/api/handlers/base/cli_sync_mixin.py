"""CLI Handler - 同步处理 Mixin"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import httpx
from fastapi.responses import JSONResponse

from src.api.handlers.base.parsers import get_parser_for_format
from src.api.handlers.base.stream_context import extract_proxy_timing, is_format_converted
from src.api.handlers.base.upstream_stream_bridge import (
    aggregate_upstream_stream_to_internal_response,
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
    ProviderAuthException,
    ProviderNotAvailableException,
    ProviderRateLimitException,
    ProviderTimeoutException,
    ThinkingSignatureException,
)
from src.core.logger import logger
from src.services.request.execution_runtime_plan import (
    ExecutionPlan,
    ExecutionPlanTimeouts,
    ExecutionProxySnapshot,
    build_execution_plan_body,
    is_remote_execution_runtime_contract_eligible,
)
from src.services.request.execution_runtime_client import (
    ExecutionRuntimeClient,
    ExecutionRuntimeClientError,
)
from src.services.scheduling.aware_scheduler import ProviderCandidate
from src.services.task.request_state import MutableRequestBodyState

if TYPE_CHECKING:
    from src.api.handlers.base.cli_protocol import CliHandlerProtocol
    from src.models.database import Provider, ProviderAPIKey, ProviderEndpoint


class CliSyncMixin:
    """同步处理相关方法的 Mixin"""

    async def _aggregate_upstream_stream_sync_response(
        self: CliHandlerProtocol,
        *,
        body_bytes: bytes,
        provider_api_format: str,
        client_api_format: str,
        provider_name: str,
        provider_type: str,
        model: str,
        request_id: str,
        envelope: Any,
    ) -> dict[str, Any]:
        registry = get_format_converter_registry()
        provider_parser = (
            get_parser_for_format(provider_api_format) if provider_api_format else None
        )

        async def _byte_iter() -> Any:
            yield body_bytes

        byte_iter = _byte_iter()
        if provider_type == "kiro" and envelope and envelope.force_stream_rewrite():
            from src.services.provider.adapters.kiro.eventstream_rewriter import (
                apply_kiro_stream_rewrite,
            )

            byte_iter = apply_kiro_stream_rewrite(byte_iter, model=str(model or ""))

        internal_resp = await aggregate_upstream_stream_to_internal_response(
            byte_iter,
            provider_api_format=provider_api_format,
            provider_name=provider_name,
            model=model,
            request_id=request_id,
            envelope=envelope,
            provider_parser=provider_parser,
        )

        tgt_norm = registry.get_normalizer(client_api_format) if client_api_format else None
        if tgt_norm is None:
            raise RuntimeError(f"未注册 Normalizer: {client_api_format}")

        response_json = tgt_norm.response_from_internal(
            internal_resp,
            requested_model=model,
        )
        return response_json if isinstance(response_json, dict) else {}

    async def process_sync(
        self: CliHandlerProtocol,
        original_request_body: dict[str, Any],
        original_headers: dict[str, str],
        query_params: dict[str, str] | None = None,
        path_params: dict[str, Any] | None = None,
        client_content_encoding: str | None = None,
        client_accept_encoding: str | None = None,
    ) -> JSONResponse:
        """
        处理非流式请求

        通用流程：
        1. 构建请求
        2. 通过 TaskService/FailoverEngine 执行
        3. 解析响应并记录统计
        """
        logger.debug("开始非流式响应处理 ({})", self.FORMAT_ID)
        effective_client_content_encoding = resolve_client_content_encoding(
            original_headers,
            client_content_encoding,
        )
        effective_client_accept_encoding = resolve_client_accept_encoding(
            original_headers,
            client_accept_encoding,
        )

        # 使用子类实现的方法提取 model（不同 API 格式的 model 位置不同）
        model = self.extract_model_from_request(original_request_body, path_params)
        api_format = self.primary_api_format
        sync_start_time = time.time()

        # 提前创建 pending 记录，让前端可以立即看到"处理中"
        pending_usage_created = self._create_pending_usage(
            model=model,
            is_stream=False,
            request_type="chat",
            api_format=api_format,
            request_headers=original_headers,
            request_body=original_request_body,
        )

        provider_name = None
        response_json = None
        status_code = 200
        response_headers = {}
        provider_api_format = ""  # 用于追踪 Provider 的 API 格式
        provider_request_headers = {}  # 发送给 Provider 的请求头
        provider_request_body = None  # 实际发送给 Provider 的请求体
        provider_id = None  # Provider ID（用于失败记录）
        endpoint_id = None  # Endpoint ID（用于失败记录）
        key_id = None  # Key ID（用于失败记录）
        exec_result = None
        mapped_model_result = None  # 映射后的目标模型名（用于 Usage 记录）
        response_metadata_result: dict[str, Any] = {}  # Provider 响应元数据
        needs_conversion = False  # 是否需要格式转换（由 candidate 决定）
        sync_proxy_info: dict[str, Any] | None = None  # 代理信息

        request_state = MutableRequestBodyState(original_request_body)

        async def sync_request_func(
            provider: "Provider",
            endpoint: "ProviderEndpoint",
            key: "ProviderAPIKey",
            candidate: ProviderCandidate,
        ) -> dict[str, Any]:
            nonlocal provider_name, response_json, status_code, response_headers, provider_api_format, provider_request_headers, provider_request_body, mapped_model_result, response_metadata_result, needs_conversion, sync_proxy_info
            provider_name = str(provider.name)
            provider_api_format = str(endpoint.api_format) if endpoint.api_format else ""

            # 获取模型映射（优先使用映射匹配到的模型，其次是 Provider 级别的映射）
            mapped_model = candidate.mapping_matched_model if candidate else None
            if not mapped_model:
                mapped_model = await self._get_mapped_model(
                    source_model=model,
                    provider_id=str(provider.id),
                )

            request_body = request_state.build_attempt_body()
            if mapped_model:
                mapped_model_result = mapped_model  # 保存映射后的模型名，用于 Usage 记录
                request_body = self.apply_mapped_model(request_body, mapped_model)

            client_api_format = (
                api_format.value if hasattr(api_format, "value") else str(api_format)
            )
            needs_conversion = bool(getattr(candidate, "needs_conversion", False))

            upstream_request = await self._build_upstream_request(
                provider=provider,
                endpoint=endpoint,
                key=key,
                request_body=request_body,
                original_headers=original_headers,
                query_params=query_params,
                client_api_format=client_api_format,
                provider_api_format=provider_api_format,
                fallback_model=model,
                mapped_model=mapped_model,
                client_is_stream=False,
                needs_conversion=needs_conversion,
                output_limit=candidate.output_limit if candidate else None,
            )
            provider_headers = upstream_request.headers
            provider_payload = upstream_request.payload
            provider_request_headers = provider_headers
            provider_request_body = provider_payload
            url = upstream_request.url
            envelope = upstream_request.envelope
            upstream_is_stream = upstream_request.upstream_is_stream
            envelope_tls_profile = upstream_request.tls_profile
            selected_base_url_cached = upstream_request.selected_base_url

            # 解析有效代理（Key 级别优先于 Provider 级别）
            from src.services.proxy_node.resolver import (
                get_proxy_label,
                resolve_effective_proxy,
                resolve_proxy_info_async,
            )

            _effective_proxy = resolve_effective_proxy(provider.proxy, getattr(key, "proxy", None))
            sync_proxy_info = await resolve_proxy_info_async(_effective_proxy)
            _proxy_label = get_proxy_label(sync_proxy_info)

            logger.info(
                f"  └─ [{self.request_id}] 发送{'上游流式(聚合)' if upstream_is_stream else '非流式'}请求: "
                f"Provider={provider.name}, Endpoint={endpoint.id[:8] if endpoint.id else 'N/A'}..., "
                f"Key=***{key.api_key[-4:] if key.api_key else 'N/A'}, "
                f"原始模型={model}, 映射后={mapped_model or '无映射'}, URL模型={upstream_request.url_model}, "
                f"代理={_proxy_label}"
            )

            from src.services.proxy_node.resolver import (
                build_proxy_url_async,
                resolve_delegate_config_async,
            )

            # 非流式请求使用 http_request_timeout 作为整体超时
            # 优先使用 Provider 配置，否则使用全局配置
            request_timeout = provider.request_timeout or config.http_request_timeout

            delegate_cfg = await resolve_delegate_config_async(_effective_proxy)
            is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))
            proxy_url: str | None = None
            if _effective_proxy and not is_tunnel_delegate:
                proxy_url = await build_proxy_url_async(_effective_proxy)

            rust_plan = ExecutionPlan(
                request_id=str(self.request_id or ""),
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
                headers=dict(provider_headers),
                body=build_execution_plan_body(
                    provider_payload,
                    content_type=str(provider_headers.get("content-type") or "").strip() or None,
                ),
                stream=upstream_is_stream,
                provider_api_format=provider_api_format,
                client_api_format=client_api_format,
                model_name=str(model or ""),
                content_type=str(provider_headers.get("content-type") or "").strip() or None,
                content_encoding=effective_client_content_encoding,
                proxy=ExecutionProxySnapshot.from_proxy_info(
                    sync_proxy_info,
                    proxy_url=proxy_url,
                    mode_override="tunnel" if is_tunnel_delegate else None,
                    node_id_override=(
                        str(delegate_cfg.get("node_id") or "").strip() or None
                        if is_tunnel_delegate
                        else None
                    ),
                ),
                tls_profile=envelope_tls_profile,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=int(config.http_connect_timeout * 1000),
                    read_ms=int(config.http_read_timeout * 1000),
                    write_ms=int(config.http_write_timeout * 1000),
                    pool_ms=int(config.http_pool_timeout * 1000),
                    total_ms=int(request_timeout * 1000),
                ),
            )

            if not is_remote_execution_runtime_contract_eligible(rust_plan):
                raise ProviderNotAvailableException(
                    "CLI 请求暂不支持当前 Rust executor 契约",
                    provider_name=str(provider.name),
                    upstream_response="remote_contract_ineligible",
                )

            try:
                rust_result = await ExecutionRuntimeClient().execute_sync_json(rust_plan)
            except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
                logger.warning(
                    "[{}] CLI Rust executor unavailable: {}",
                    self.request_id,
                    exc,
                )
                raise ProviderNotAvailableException(
                    "执行器暂时不可用，请稍后重试",
                    provider_name=str(provider.name),
                    upstream_response=str(exc),
                ) from exc

            status_code = rust_result.status_code
            response_headers = dict(rust_result.headers)
            extract_proxy_timing(sync_proxy_info, response_headers)

            if envelope:
                envelope.on_http_status(
                    base_url=selected_base_url_cached,
                    status_code=status_code,
                )

            request = httpx.Request("POST", url, headers=provider_headers)
            synthetic_content = rust_result.response_body_bytes
            if synthetic_content is None:
                synthetic_content = json.dumps(
                    rust_result.response_json or {},
                    ensure_ascii=False,
                ).encode("utf-8")
            synthetic_response = httpx.Response(
                status_code,
                request=request,
                headers=response_headers,
                content=synthetic_content,
            )

            if status_code >= 400:
                error = httpx.HTTPStatusError(
                    f"Upstream status error: {status_code}",
                    request=request,
                    response=synthetic_response,
                )
                error_body = ""
                try:
                    if envelope and hasattr(envelope, "extract_error_text"):
                        error_body = await envelope.extract_error_text(synthetic_response)
                    else:
                        error_body = synthetic_response.text[:4000] if synthetic_response.text else ""
                except Exception:
                    error_body = synthetic_response.text[:4000] if synthetic_response.text else ""
                error.upstream_response = error_body[:4000]  # type: ignore[attr-defined]
                raise error

            if upstream_is_stream:
                if rust_result.response_body_bytes is None:
                    raise ExecutionRuntimeClientError(
                        "Rust executor stream sync result must contain body bytes"
                    )
                response_json = await self._aggregate_upstream_stream_sync_response(
                    body_bytes=rust_result.response_body_bytes,
                    provider_api_format=provider_api_format,
                    client_api_format=client_api_format,
                    provider_name=str(provider.name),
                    provider_type=str(getattr(provider, "provider_type", "") or "").lower(),
                    model=str(model or ""),
                    request_id=str(self.request_id or ""),
                    envelope=envelope,
                )
                response_metadata_result = self._extract_response_metadata(response_json or {})
                return response_json if isinstance(response_json, dict) else {}

            response_json = rust_result.response_json or {}
            if envelope:
                response_json = envelope.unwrap_response(response_json)
                envelope.postprocess_unwrapped_response(model=model, data=response_json)

            response_metadata_result = self._extract_response_metadata(response_json)
            return response_json if isinstance(response_json, dict) else {}

        try:
            # 解析能力需求
            capability_requirements = self._resolve_capability_requirements(
                model_name=model,
                request_headers=original_headers,
                request_body=original_request_body,
            )
            preferred_key_ids = await self._resolve_preferred_key_ids(
                model_name=model,
                request_body=original_request_body,
            )

            # 统一入口：总是通过 TaskService
            from src.services.task import TaskService
            from src.services.task.core.context import TaskMode

            exec_result = await TaskService(self.db, self.redis).execute(
                task_type="cli",
                task_mode=TaskMode.SYNC,
                api_format=api_format,
                model_name=model,
                user_api_key=self.api_key,
                request_func=sync_request_func,
                request_id=self.request_id,
                is_stream=False,
                capability_requirements=capability_requirements or None,
                preferred_key_ids=preferred_key_ids or None,
                request_body_state=request_state,
                request_headers=original_headers,
                request_body=original_request_body,
                # 预创建失败时，回退到 TaskService 侧创建，避免丢失 pending 状态。
                create_pending_usage=not pending_usage_created,
            )
            result = exec_result.response
            actual_provider_name = exec_result.provider_name or "unknown"
            attempt_id = exec_result.request_candidate_id
            provider_id = exec_result.provider_id
            endpoint_id = exec_result.endpoint_id
            key_id = exec_result.key_id

            provider_name = actual_provider_name
            response_time_ms = int((time.time() - sync_start_time) * 1000)

            # 确保 response_json 不为 None
            if response_json is None:
                response_json = {}

            # 跨格式：响应转换回 client_format（失败不触发 failover，保守回退为原始响应）
            provider_response_json: dict[str, Any] | None = None
            if (
                needs_conversion
                and provider_api_format
                and api_format
                and isinstance(response_json, dict)
            ):
                try:
                    provider_response_json = response_json.copy()
                    registry = get_format_converter_registry()
                    response_json = registry.convert_response(
                        response_json,
                        provider_api_format,
                        api_format,
                        requested_model=model,  # 使用用户请求的原始模型名
                    )
                    logger.debug(
                        "非流式响应格式转换完成: {} -> {}", provider_api_format, api_format
                    )
                except Exception as conv_err:
                    logger.warning("非流式响应格式转换失败，使用原始响应: {}", conv_err)
                    provider_response_json = None

            # 使用解析器提取 usage
            usage = self.parser.extract_usage_from_response(response_json)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("cache_read_tokens", 0)
            cache_creation_tokens = usage.get("cache_creation_tokens", 0)

            output_text = self.parser.extract_text_content(response_json)[:200]

            # 非流式成功时，返回给客户端的是提供商响应头（透传）
            client_response_headers = filter_proxy_response_headers(response_headers)
            client_response_headers["content-type"] = "application/json"
            client_response = build_json_response_for_client(
                status_code=status_code,
                content=response_json,
                headers=client_response_headers,
                client_accept_encoding=effective_client_accept_encoding,
            )
            actual_client_response_headers = dict(client_response.headers)

            request_metadata = self._build_request_metadata() or {}
            if sync_proxy_info:
                request_metadata["proxy"] = sync_proxy_info
            request_metadata = self._merge_scheduling_metadata(
                request_metadata,
                exec_result=exec_result,
                selected_key_id=key_id,
            )
            total_cost = await self.telemetry.record_success(
                provider=provider_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                response_time_ms=response_time_ms,
                status_code=status_code,
                request_headers=original_headers,
                request_body=original_request_body,
                response_headers=response_headers,
                client_response_headers=actual_client_response_headers,
                response_body=provider_response_json or response_json,
                client_response_body=response_json if provider_response_json else None,
                provider_request_body=provider_request_body,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cached_tokens,
                is_stream=False,
                provider_request_headers=provider_request_headers,
                api_format=api_format,
                api_family=self.api_family,
                endpoint_kind=self.endpoint_kind,
                # 格式转换追踪
                endpoint_api_format=provider_api_format or None,
                has_format_conversion=is_format_converted(provider_api_format, str(api_format)),
                # Provider 侧追踪信息（用于记录真实成本）
                provider_id=provider_id,
                provider_endpoint_id=endpoint_id,
                provider_api_key_id=key_id,
                # 模型映射信息
                target_model=mapped_model_result,
                # Provider 响应元数据（如 Gemini 的 modelVersion）
                response_metadata=response_metadata_result if response_metadata_result else None,
                request_metadata=request_metadata,
            )

            logger.info("{} 非流式响应处理完成", self.FORMAT_ID)

            # 透传提供商的响应头
            return client_response

        except ThinkingSignatureException as e:
            # Thinking 签名错误：TaskService 层已处理整流重试但仍失败
            # 记录实际发送给 Provider 的请求体，便于排查问题根因
            response_time_ms = int((time.time() - sync_start_time) * 1000)
            request_metadata = self._build_request_metadata() or {}
            if sync_proxy_info:
                request_metadata["proxy"] = sync_proxy_info
            request_metadata = self._merge_scheduling_metadata(
                request_metadata,
                selected_key_id=key_id,
                pool_summary=getattr(exec_result, "pool_summary", None),
                fallback_from_request=True,
            )
            await self.telemetry.record_failure(
                provider=provider_name or "unknown",
                model=model,
                response_time_ms=response_time_ms,
                status_code=e.status_code or 400,
                request_headers=original_headers,
                request_body=original_request_body,
                provider_request_body=provider_request_body,
                error_message=str(e),
                is_stream=False,
                api_format=api_format,
                api_family=self.api_family,
                endpoint_kind=self.endpoint_kind,
                request_metadata=request_metadata,
            )
            raise

        except Exception as e:
            response_time_ms = int((time.time() - sync_start_time) * 1000)

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

            request_metadata = self._build_request_metadata() or {}
            if sync_proxy_info:
                request_metadata["proxy"] = sync_proxy_info
            request_metadata = self._merge_scheduling_metadata(
                request_metadata,
                selected_key_id=key_id,
                pool_summary=getattr(exec_result, "pool_summary", None),
                fallback_from_request=True,
            )
            await self.telemetry.record_failure(
                provider=provider_name or "unknown",
                model=model,
                response_time_ms=response_time_ms,
                status_code=status_code,
                error_message=extract_client_error_message(e),
                request_headers=original_headers,
                request_body=original_request_body,
                provider_request_body=provider_request_body,
                is_stream=False,
                api_format=api_format,
                api_family=self.api_family,
                endpoint_kind=self.endpoint_kind,
                provider_request_headers=provider_request_headers,
                response_headers=error_response_headers,
                # 非流式失败返回给客户端的是 JSON 错误响应
                client_response_headers={"content-type": "application/json"},
                # 格式转换追踪
                endpoint_api_format=provider_api_format or None,
                has_format_conversion=is_format_converted(provider_api_format, str(api_format)),
                # 模型映射信息
                target_model=mapped_model_result,
                request_metadata=request_metadata,
            )

            raise

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

                for encoding in ["utf-8", "gbk", "latin1"]:
                    try:
                        return error_bytes.decode(encoding)
                    except (UnicodeDecodeError, LookupError):
                        continue

                return error_bytes.decode("utf-8", errors="replace")
            else:
                return (
                    e.response.text
                    if hasattr(e.response, "_content")
                    else "Unable to read response"
                )
        except Exception as decode_error:
            return f"Unable to read error response: {decode_error}"
