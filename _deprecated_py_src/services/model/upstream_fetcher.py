"""
上游模型获取公共模块

提供从上游 API 获取模型列表的公共函数；通用 api_format 抓取能力统一来自 core.api_format.capabilities，供以下场景使用：
- 定时任务自动获取（fetch_scheduler.py）
- 管理后台手动查询（provider_query.py）
"""

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from src.config.settings import config
from src.core.api_format import get_extra_headers_from_endpoint
from src.core.api_format.capabilities import BROWSER_FINGERPRINT_HEADERS
from src.core.api_format.headers import build_adapter_headers_for_endpoint
from src.core.logger import logger

# 并发请求限制
MAX_CONCURRENT_REQUESTS = 5

# 模型获取格式优先级：同族内优先使用 chat 端点，若无则回退到 cli 端点
MODEL_FETCH_FORMAT_PRIORITY: list[tuple[str, ...]] = [
    ("openai:chat", "openai:cli", "openai:compact"),
    ("claude:chat", "claude:cli"),
    ("gemini:chat", "gemini:cli"),
]

# Return tuple signature:
# (models, errors, has_success, upstream_metadata)
_ModelsFetcher = Callable[
    ["UpstreamModelsFetchContext", float],
    Awaitable[tuple[list[dict], list[str], bool, dict[str, Any] | None]],
]


@dataclass(frozen=True, slots=True)
class EndpointFetchConfig:
    """端点获取配置（纯数据，不依赖 DB session）。

    从 ProviderEndpoint ORM 对象提取必要字段，确保在 DB session 关闭后
    仍可安全使用（避免 DetachedInstanceError）。
    """

    base_url: str
    extra_headers: dict[str, str] | None = None


def build_format_to_config(endpoints: Iterable[Any]) -> dict[str, EndpointFetchConfig]:
    """将活跃的 ProviderEndpoint 转换为 api_format -> EndpointFetchConfig 映射。

    应在 DB session 活跃时调用，提取 ORM 对象上的 base_url 和 header_rules，
    转换为 session 无关的纯数据结构。
    """
    result: dict[str, EndpointFetchConfig] = {}
    for ep in endpoints:
        if not getattr(ep, "is_active", False):
            continue
        result[ep.api_format] = EndpointFetchConfig(
            base_url=ep.base_url,
            extra_headers=get_extra_headers_from_endpoint(ep),
        )
    return result


@dataclass(frozen=True)
class UpstreamModelsFetchContext:
    """上游模型获取上下文（Key 级别）。"""

    provider_type: str
    api_key_value: str
    format_to_endpoint: dict[str, EndpointFetchConfig]
    proxy_config: dict[str, Any] | None = None
    auth_config: dict[str, Any] | None = None


class UpstreamModelsFetcherRegistry:
    """按 provider_type 注册上游模型获取策略，避免到处写特判。"""

    _fetchers: dict[str, _ModelsFetcher] = {}

    @classmethod
    def register(cls, *, provider_types: list[str], fetcher: _ModelsFetcher) -> None:
        for pt in provider_types:
            if not pt:
                continue
            cls._fetchers[pt.lower()] = fetcher

    @classmethod
    def get(cls, provider_type: str) -> _ModelsFetcher | None:
        if not provider_type:
            return None
        return cls._fetchers.get(provider_type.lower())


async def _fetch_models_default(
    ctx: UpstreamModelsFetchContext,
    timeout_seconds: float,
) -> tuple[list[dict], list[str], bool, dict[str, Any] | None]:
    endpoint_configs = build_all_format_configs(ctx.api_key_value, ctx.format_to_endpoint)
    models, errors, has_success = await fetch_models_from_endpoints(
        endpoint_configs, timeout=timeout_seconds, proxy_config=ctx.proxy_config
    )
    return models, errors, has_success, None


async def fetch_models_for_key(
    ctx: UpstreamModelsFetchContext,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[list[dict], list[str], bool, dict[str, Any] | None]:
    """统一入口：按 provider_type 选择策略获取模型列表（可附带 upstream_metadata）。"""
    # Ensure provider plugins (including custom model fetchers) are registered.
    from src.services.provider.envelope import ensure_providers_bootstrapped

    ensure_providers_bootstrapped(provider_types=[ctx.provider_type] if ctx.provider_type else None)

    fetcher = UpstreamModelsFetcherRegistry.get(ctx.provider_type) or _fetch_models_default
    return await fetcher(ctx, timeout_seconds)


def merge_upstream_metadata(
    current: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """合并上游元数据，对 quota_by_model 做模型级深度合并。

    以上游返回为准（全量替换），不再保留上游已下架的模型。
    仅对上游返回的模型补充旧数据中的 reset_time（当新数据缺少时）。
    """
    merged: dict[str, Any] = dict(current) if isinstance(current, dict) else {}
    for ns_key, ns_val in incoming.items():
        old_ns = merged.get(ns_key)
        if (
            isinstance(ns_val, dict)
            and isinstance(old_ns, dict)
            and "quota_by_model" in ns_val
            and "quota_by_model" in old_ns
        ):
            old_qbm = old_ns["quota_by_model"]
            new_qbm = ns_val["quota_by_model"]
            if isinstance(old_qbm, dict) and isinstance(new_qbm, dict):
                # 保留新数据中已有模型的旧 reset_time
                for model_id, new_info in new_qbm.items():
                    if not isinstance(new_info, dict):
                        continue
                    old_info = old_qbm.get(model_id)
                    if (
                        isinstance(old_info, dict)
                        and "reset_time" in old_info
                        and "reset_time" not in new_info
                    ):
                        new_info["reset_time"] = old_info["reset_time"]
        merged[ns_key] = ns_val
    return merged


def build_all_format_configs(
    api_key_value: str,
    format_to_endpoint: dict[str, EndpointFetchConfig],
) -> list[dict]:
    """
    构建所有 API 格式的端点配置

    只对实际配置了端点的格式构建请求配置，不同端点的 base_url 可能不同，
    不应使用某个端点的 base_url 去尝试其他格式。

    Args:
        api_key_value: 解密后的 API Key
        format_to_endpoint: API 格式到 EndpointFetchConfig 的映射

    Returns:
        端点配置列表，每个配置包含 api_key, base_url, api_format, extra_headers
    """
    if not format_to_endpoint:
        return []

    # 同族内优先使用 chat 端点，若无则回退到 cli 端点
    configs: list[dict] = []
    for candidates in MODEL_FETCH_FORMAT_PRIORITY:
        fmt = next((f for f in candidates if f in format_to_endpoint), None)
        if fmt is not None:
            cfg = format_to_endpoint[fmt]

            base_url = str(getattr(cfg, "base_url", "") or "")
            extra_headers: dict[str, str] | None

            if isinstance(cfg, EndpointFetchConfig):
                extra_headers = cfg.extra_headers
            else:
                # 允许直接传递类似 ProviderEndpoint 的对象（测试/独立使用场景）。
                candidate_extra = getattr(cfg, "extra_headers", None)
                if isinstance(candidate_extra, dict):
                    extra_headers = {str(k): str(v) for k, v in candidate_extra.items() if k}
                else:
                    extra_headers = get_extra_headers_from_endpoint(cfg)

            configs.append(
                {
                    "api_key": api_key_value,
                    "base_url": base_url,
                    "api_format": fmt,
                    "extra_headers": extra_headers,
                }
            )
    return configs


async def _build_models_proxy_snapshot(
    proxy_config: dict[str, Any] | None,
) -> Any:
    from src.services.request.execution_runtime_plan import build_proxy_snapshot

    return await build_proxy_snapshot(proxy_config, label="upstream model fetch")


def _build_model_fetch_url(
    api_format: str,
    base_url: str,
    api_key: str,
    after_id: str | None = None,
) -> str:
    normalized_api_format = str(api_format or "").strip().lower()
    base_url = str(base_url or "").rstrip("/")

    if normalized_api_format.startswith("gemini:"):
        if base_url.endswith("/v1beta"):
            return f"{base_url}/models?key={api_key}"
        return f"{base_url}/v1beta/models?key={api_key}"

    if base_url.endswith("/v1"):
        url = f"{base_url}/models"
    else:
        url = f"{base_url}/v1/models"

    if normalized_api_format.startswith("claude:"):
        if after_id:
            return f"{url}?limit=100&after_id={after_id}"
        return f"{url}?limit=100"

    return url


def _build_model_fetch_headers(
    api_format: str,
    api_key: str,
    extra_headers: dict[str, str] | None,
) -> dict[str, str]:
    normalized_api_format = str(api_format or "").strip().lower()

    if normalized_api_format == "openai:cli":
        headers = {"User-Agent": config.internal_user_agent_openai_cli}
        if extra_headers:
            headers.update(extra_headers)
        return build_adapter_headers_for_endpoint("openai:chat", api_key, headers)

    if normalized_api_format == "openai:compact":
        headers = {"User-Agent": config.internal_user_agent_openai_cli}
        if extra_headers:
            headers.update(extra_headers)
        return build_adapter_headers_for_endpoint("openai:chat", api_key, headers)

    if normalized_api_format == "claude:cli":
        headers = {"User-Agent": config.internal_user_agent_claude_cli}
        if extra_headers:
            headers.update(extra_headers)
        return build_adapter_headers_for_endpoint(normalized_api_format, api_key, headers)

    if normalized_api_format == "claude:chat":
        headers = build_adapter_headers_for_endpoint(normalized_api_format, api_key, extra_headers)
        if "authorization" not in {str(k).lower() for k in headers}:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    if normalized_api_format == "gemini:cli":
        headers = {
            **BROWSER_FINGERPRINT_HEADERS,
            "User-Agent": config.internal_user_agent_gemini_cli,
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    if normalized_api_format == "gemini:chat":
        headers = {**BROWSER_FINGERPRINT_HEADERS}
        if extra_headers:
            headers.update(extra_headers)
        return headers

    return build_adapter_headers_for_endpoint(normalized_api_format, api_key, extra_headers)


def _parse_model_fetch_response(
    api_format: str,
    payload: Any,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    normalized_api_format = str(api_format or "").strip().lower()

    if normalized_api_format.startswith("gemini:"):
        if isinstance(payload, dict) and isinstance(payload.get("models"), list):
            out: list[dict[str, Any]] = []
            for model in payload["models"]:
                if not isinstance(model, dict):
                    continue
                out.append(
                    {
                        "id": str(model.get("name", "")).replace("models/", ""),
                        "owned_by": "google",
                        "display_name": model.get("displayName", ""),
                        "api_format": normalized_api_format,
                    }
                )
            return out, False, None
        return [], False, None

    page_models: list[dict[str, Any]] = []
    has_more = False
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        page_models = [m for m in payload["data"] if isinstance(m, dict)]
        has_more = bool(payload.get("has_more"))
    elif isinstance(payload, list):
        page_models = [m for m in payload if isinstance(m, dict)]

    for model in page_models:
        model.setdefault("api_format", normalized_api_format)

    next_cursor = None
    if normalized_api_format.startswith("claude:") and isinstance(payload, dict):
        raw_last_id = payload.get("last_id")
        if has_more and isinstance(raw_last_id, str) and raw_last_id.strip():
            next_cursor = raw_last_id.strip()

    return page_models, has_more, next_cursor


def _extract_rust_error_message(result: Any) -> str:
    payload = getattr(result, "response_json", None)
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            message = str(err.get("message") or "").strip()
            if message:
                return message[:500]
        message = str(payload.get("message") or "").strip()
        if message:
            return message[:500]
        try:
            return json.dumps(payload, ensure_ascii=False)[:500]
        except Exception:
            pass

    body_bytes = getattr(result, "response_body_bytes", None)
    if body_bytes:
        try:
            return body_bytes.decode("utf-8", errors="replace")[:500]
        except Exception:
            return "(binary body)"
    return "(empty)"


async def _try_rust_fetch_models_for_api_format(
    *,
    api_format: str,
    base_url: str,
    api_key: str,
    extra_headers: dict[str, str] | None,
    timeout: float,
    proxy_config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | None, bool] | None:
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
    )
    from src.services.request.execution_runtime_client import (
        ExecutionRuntimeClient,
        ExecutionRuntimeClientError,
    )

    if config.execution_runtime_backend != "rust":
        return None

    try:
        proxy_snapshot = await _build_models_proxy_snapshot(proxy_config)
        headers = _build_model_fetch_headers(api_format, api_key, extra_headers)
        models: list[dict[str, Any]] = []
        next_cursor: str | None = None
        seen_ids: set[str] = set()

        for _ in range(20):
            url = _build_model_fetch_url(api_format, base_url, api_key, next_cursor)
            result = await ExecutionRuntimeClient().execute_sync_json(
                ExecutionPlan(
                    request_id=f"model-fetch:{api_format}:{next_cursor or 'root'}",
                    candidate_id=None,
                    provider_name=str(api_format.split(":", 1)[0] or "unknown"),
                    provider_id="",
                    endpoint_id="",
                    key_id="",
                    method="GET",
                    url=url,
                    headers=dict(headers),
                    body=ExecutionPlanBody(),
                    stream=False,
                    provider_api_format=api_format,
                    client_api_format=api_format,
                    model_name="models",
                    proxy=proxy_snapshot,
                    timeouts=ExecutionPlanTimeouts(
                        connect_ms=min(int(timeout * 1000), 30_000),
                        read_ms=int(timeout * 1000),
                        write_ms=int(timeout * 1000),
                        pool_ms=min(int(timeout * 1000), 30_000),
                        total_ms=int(timeout * 1000),
                    ),
                )
            )

            if result.status_code != 200:
                error_body = _extract_rust_error_message(result)
                return [], f"HTTP {result.status_code}: {error_body}", False

            payload = result.response_json
            page_models, has_more, next_after_id = _parse_model_fetch_response(api_format, payload)
            for model in page_models:
                model_id = model.get("id")
                if isinstance(model_id, str) and model_id and model_id in seen_ids:
                    continue
                if isinstance(model_id, str) and model_id:
                    seen_ids.add(model_id)
                models.append(model)

            if not has_more or not next_after_id or next_after_id == next_cursor:
                return models, None, True
            next_cursor = next_after_id

        return models, None, True
    except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Rust model fetch fallback for {}: {}", api_format, exc)
        return None
    except Exception as exc:
        logger.warning("Rust model fetch unexpected fallback for {}: {}", api_format, exc)
        return None


async def fetch_models_from_endpoints(
    endpoint_configs: list[dict],
    timeout: float = 30.0,
    proxy_config: dict[str, Any] | None = None,
) -> tuple[list[dict], list[str], bool]:
    """
    从多个端点并发获取模型

    Args:
        endpoint_configs: 端点配置列表，每个配置包含 api_key, base_url, api_format, extra_headers
        timeout: 请求超时时间（秒）
        proxy_config: 代理配置（可选），支持系统默认回退

    Returns:
        (模型列表, 错误列表, 是否有成功)
    """
    all_models: list[dict] = []
    errors: list[str] = []
    has_success = False
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def fetch_one(config: dict) -> tuple[list, str | None, bool]:
        base_url = config["base_url"]
        if not base_url:
            return [], None, False
        base_url = base_url.rstrip("/")
        api_format = config["api_format"]
        api_key_value = config["api_key"]
        extra_headers = config.get("extra_headers")

        try:
            async with semaphore:
                rust_result = await _try_rust_fetch_models_for_api_format(
                    api_format=api_format,
                    base_url=base_url,
                    api_key=api_key_value,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    proxy_config=proxy_config,
                )
                if rust_result is None:
                    return [], f"{api_format}: rust executor unavailable", False
                models, error, success = rust_result

            for m in models:
                if "api_format" not in m:
                    m["api_format"] = api_format

            # 即使返回空列表，只要没有错误也算成功
            return models, error, success
        except httpx.TimeoutException:
            logger.warning("获取 {} 模型超时", api_format)
            return [], f"{api_format}: timeout", False
        except Exception:
            logger.exception("获取 {} 模型出错", api_format)
            return [], f"{api_format}: error", False

    results = await asyncio.gather(*[fetch_one(c) for c in endpoint_configs])
    for models, error, success in results:
        all_models.extend(models)
        if error:
            errors.append(error)
        if success:
            has_success = True

    return all_models, errors, has_success
