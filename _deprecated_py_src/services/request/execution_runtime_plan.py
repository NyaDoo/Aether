"""
Execution runtime 计划契约

用于在 Python 控制面和 Rust execution runtime 之间传递稳定的请求执行信息。
当前阶段先服务于非流式 chat 路径的计划构建与本地执行拆分。
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Any


def _drop_none(value: Any) -> Any:
    """递归移除 None 字段，便于序列化为紧凑 payload。"""
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


@dataclass(slots=True)
class ExecutionPlanTimeouts:
    connect_ms: int | None = None
    read_ms: int | None = None
    write_ms: int | None = None
    pool_ms: int | None = None
    total_ms: int | None = None


@dataclass(slots=True)
class ExecutionPlanBody:
    json_body: Any = None
    body_bytes_b64: str | None = None
    body_ref: str | None = None


@dataclass(slots=True)
class ExecutionProxySnapshot:
    enabled: bool
    mode: str | None = None
    node_id: str | None = None
    label: str | None = None
    url: str | None = None
    extra: dict[str, Any] | None = None

    @classmethod
    def from_proxy_info(
        cls,
        proxy_info: dict[str, Any] | None,
        *,
        proxy_url: str | None = None,
        mode_override: str | None = None,
        node_id_override: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> ExecutionProxySnapshot | None:
        if not proxy_info and not proxy_url:
            return None
        mode = (
            mode_override
            or str(proxy_info.get("type") or proxy_info.get("mode") or "").strip()
            or None
        )
        if not mode and proxy_url:
            mode = proxy_url.split("://", 1)[0].strip().lower() or None
        return cls(
            enabled=True,
            mode=mode,
            node_id=node_id_override
            or str((proxy_info or {}).get("node_id") or "").strip()
            or None,
            label=str((proxy_info or {}).get("label") or "").strip() or None,
            url=str(proxy_url or (proxy_info or {}).get("url") or "").strip() or None,
            extra=extra or None,
        )


@dataclass(slots=True)
class ExecutionPlan:
    request_id: str
    candidate_id: str | None
    provider_name: str
    provider_id: str
    endpoint_id: str
    key_id: str
    method: str
    url: str
    headers: dict[str, str]
    body: ExecutionPlanBody
    stream: bool
    provider_api_format: str
    client_api_format: str
    model_name: str
    content_type: str | None = None
    content_encoding: str | None = None
    proxy: ExecutionProxySnapshot | None = None
    tls_profile: str | None = None
    timeouts: ExecutionPlanTimeouts | None = None

    def to_payload(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


_REMOTE_EXECUTION_RUNTIME_BYPASS_FORMATS = {"openai:cli", "openai:compact"}


def should_bypass_remote_execution_runtime_url(
    url: str | None,
    *,
    provider_api_format: str | None = None,
    client_api_format: str | None = None,
) -> bool:
    normalized_url = str(url or "").strip().lower()
    if not normalized_url:
        return False

    normalized_provider_api_format = str(provider_api_format or "").strip().lower()
    normalized_client_api_format = str(client_api_format or "").strip().lower()
    if (
        normalized_provider_api_format not in _REMOTE_EXECUTION_RUNTIME_BYPASS_FORMATS
        and normalized_client_api_format not in _REMOTE_EXECUTION_RUNTIME_BYPASS_FORMATS
    ):
        return False

    return "/backend-api/codex" in normalized_url or "/backendapi/codex" in normalized_url


def should_bypass_remote_execution_runtime(contract: ExecutionPlan) -> bool:
    return should_bypass_remote_execution_runtime_url(
        contract.url,
        provider_api_format=contract.provider_api_format,
        client_api_format=contract.client_api_format,
    )


def is_remote_execution_runtime_proxy_supported(proxy: ExecutionProxySnapshot | None) -> bool:
    proxy_mode = str((proxy.mode if proxy else "") or "").strip().lower()
    return (
        proxy is None
        or (str(proxy.url or "").strip() != "" and proxy_mode not in {"tunnel"})
        or (proxy_mode == "tunnel" and str(proxy.node_id or "").strip() != "")
    )


def is_remote_execution_runtime_contract_eligible(contract: ExecutionPlan) -> bool:
    content_encoding = str(contract.content_encoding or "").strip().lower()
    has_json_body = contract.body.json_body is not None
    has_raw_body = bool(str(contract.body.body_bytes_b64 or "").strip())
    has_body = has_json_body or has_raw_body
    return (
        ((not has_body and content_encoding == "") or has_body)
        and (not has_json_body or content_encoding in {"", "gzip"} or has_raw_body)
        and is_remote_execution_runtime_proxy_supported(contract.proxy)
        and not should_bypass_remote_execution_runtime(contract)
    )


def build_execution_plan_body(
    payload: Any,
    *,
    content_type: str | None = None,
) -> ExecutionPlanBody:
    normalized_content_type = str(content_type or "").strip().lower()

    if isinstance(payload, dict):
        return ExecutionPlanBody(json_body=payload)

    if isinstance(payload, list) and "json" in normalized_content_type:
        return ExecutionPlanBody(json_body=payload)

    if isinstance(payload, (bytes, bytearray, memoryview)):
        return ExecutionPlanBody(body_bytes_b64=base64.b64encode(bytes(payload)).decode("ascii"))

    if isinstance(payload, str):
        return ExecutionPlanBody(
            body_bytes_b64=base64.b64encode(payload.encode("utf-8")).decode("ascii")
        )

    if payload is None:
        return ExecutionPlanBody()

    return ExecutionPlanBody(
        body_bytes_b64=base64.b64encode(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
    )


@dataclass(slots=True)
class PreparedExecutionPlan:
    """本地执行所需的运行时上下文；`contract` 是可序列化的稳定边界。"""

    contract: ExecutionPlan
    payload: dict[str, Any]
    headers: dict[str, str]
    upstream_is_stream: bool
    needs_conversion: bool
    provider_type: str
    request_timeout: float
    delegate_config: dict[str, Any] | None = None
    proxy_config: dict[str, Any] | None = None
    envelope: Any = None
    selected_base_url: str | None = None
    client_content_encoding: str | None = None
    proxy_info: dict[str, Any] | None = None

    @property
    def remote_eligible(self) -> bool:
        return is_remote_execution_runtime_contract_eligible(self.contract)


async def build_proxy_snapshot(
    proxy_config: dict[str, Any] | None,
    *,
    label: str = "adapter",
) -> ExecutionProxySnapshot | None:
    if not proxy_config:
        return None

    try:
        from src.services.proxy_node.resolver import (
            build_proxy_url_async,
            resolve_delegate_config_async,
            resolve_proxy_info_async,
        )

        delegate_cfg = await resolve_delegate_config_async(proxy_config)
        proxy_url: str | None = None
        if proxy_config and not (delegate_cfg and delegate_cfg.get("tunnel")):
            proxy_url = await build_proxy_url_async(proxy_config)
        proxy_info = await resolve_proxy_info_async(proxy_config)
        return ExecutionProxySnapshot.from_proxy_info(
            proxy_info,
            proxy_url=proxy_url,
            mode_override="tunnel" if delegate_cfg and delegate_cfg.get("tunnel") else None,
            node_id_override=(
                str(delegate_cfg.get("node_id") or "").strip() or None
                if delegate_cfg and delegate_cfg.get("tunnel")
                else None
            ),
        )
    except Exception as exc:
        from src.core.logger import logger

        logger.warning("Build {} proxy snapshot failed: {}", label, exc)
        return None


# Compatibility aliases for older call sites still using executor terminology.
should_bypass_remote_executor_url = should_bypass_remote_execution_runtime_url
should_bypass_remote_executor = should_bypass_remote_execution_runtime
is_remote_proxy_supported = is_remote_execution_runtime_proxy_supported
is_remote_contract_eligible = is_remote_execution_runtime_contract_eligible

__all__ = [
    "ExecutionPlan",
    "ExecutionPlanBody",
    "ExecutionPlanTimeouts",
    "ExecutionProxySnapshot",
    "PreparedExecutionPlan",
    "build_execution_plan_body",
    "build_proxy_snapshot",
    "is_remote_contract_eligible",
    "is_remote_execution_runtime_contract_eligible",
    "is_remote_execution_runtime_proxy_supported",
    "is_remote_proxy_supported",
    "should_bypass_remote_execution_runtime",
    "should_bypass_remote_execution_runtime_url",
    "should_bypass_remote_executor",
    "should_bypass_remote_executor_url",
]
