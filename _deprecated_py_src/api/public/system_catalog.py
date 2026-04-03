"""
System Catalog / 健康检查相关端点

这些是系统工具端点，不需要复杂的 Adapter 抽象。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session, load_only, selectinload

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.api.handlers.base.request_builder import (
    PassthroughRequestBuilder,
    build_test_request_body,
    get_provider_auth,
)
from src.clients.redis_client import get_redis_client
from src.config.settings import config
from src.core.logger import logger
from src.database import get_db
from src.database.database import get_pool_status
from src.models.database import GlobalModel, Model, Provider, ProviderAPIKey, ProviderEndpoint
from src.services.provider.provider_context import resolve_provider_proxy
from src.services.provider.transport import build_provider_url

router = APIRouter(tags=["System Catalog"])
pipeline = get_pipeline()


class PublicSystemCatalogApiAdapter(ApiAdapter):
    mode = ApiMode.PUBLIC

    def authorize(self, context: ApiRequestContext) -> None:  # type: ignore[override]
        return None


# ============== 辅助函数 ==============


def _as_bool(value: str | None, default: bool) -> bool:
    """将字符串转换为布尔值"""
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _serialize_provider(
    provider: Provider,
    include_models: bool,
    include_endpoints: bool,
) -> dict[str, Any]:
    """序列化 Provider 对象"""
    provider_data: dict[str, Any] = {
        "id": provider.id,
        "name": provider.name,
        "is_active": provider.is_active,
        "provider_priority": provider.provider_priority,
    }

    if include_endpoints:
        provider_data["endpoints"] = [
            {
                "id": endpoint.id,
                "base_url": endpoint.base_url,
                "api_format": endpoint.api_format if endpoint.api_format else None,
                "is_active": endpoint.is_active,
            }
            for endpoint in provider.endpoints or []
        ]

    if include_models:
        provider_data["models"] = [
            {
                "id": model.id,
                "name": (
                    model.global_model.name if model.global_model else model.provider_model_name
                ),
                "display_name": (
                    model.global_model.display_name
                    if model.global_model
                    else model.provider_model_name
                ),
                "is_active": model.is_active,
                "supports_streaming": model.supports_streaming,
            }
            for model in provider.models or []
            if model.is_active
        ]

    return provider_data


def _select_provider(db: Session, provider_name: str | None) -> Provider | None:
    """选择 Provider（按 provider_priority 优先级选择）"""
    query = db.query(Provider).filter(Provider.is_active.is_(True))
    if provider_name:
        provider = query.filter(Provider.name == provider_name).first()
        if provider:
            return provider

    # 按优先级选择（provider_priority 最小的优先）
    return query.order_by(Provider.provider_priority.asc()).first()


async def _build_test_connection_transport_context(
    endpoint: ProviderEndpoint,
    key: ProviderAPIKey,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Any]:
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import ExecutionProxySnapshot

    try:
        effective_proxy = resolve_effective_proxy(
            resolve_provider_proxy(endpoint=endpoint, key=key),
            getattr(key, "proxy", None),
        )
        if not effective_proxy or not effective_proxy.get("enabled", True):
            effective_proxy = await get_system_proxy_config_async()

        delegate_cfg = await resolve_delegate_config_async(effective_proxy)
        proxy_url: str | None = None
        if effective_proxy and not (delegate_cfg and delegate_cfg.get("tunnel")):
            proxy_url = await build_proxy_url_async(effective_proxy)

        proxy_info = await resolve_proxy_info_async(effective_proxy)
        proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
            proxy_info,
            proxy_url=proxy_url,
            mode_override="tunnel" if delegate_cfg and delegate_cfg.get("tunnel") else None,
            node_id_override=(
                str(delegate_cfg.get("node_id") or "").strip() or None
                if delegate_cfg and delegate_cfg.get("tunnel")
                else None
            ),
        )
        return effective_proxy, delegate_cfg, proxy_snapshot
    except Exception as exc:
        logger.warning(
            "Failed to build test-connection transport context endpoint={} key={}: {}",
            getattr(endpoint, "id", None),
            getattr(key, "id", None),
            exc,
        )
        return None, None, None


async def _try_rust_test_connection_response(
    *,
    request_id: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    provider_name: str,
    provider_id: str | None,
    endpoint_id: str | None,
    key_id: str | None,
    api_format: str,
    model_name: str,
    proxy_snapshot: Any,
) -> httpx.Response:
    import json

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
        raise HTTPException(
            status_code=503,
            detail="System catalog test-connection requires Rust executor",
        )

    try:
        result = await ExecutionRuntimeClient().execute_sync_json(
            ExecutionPlan(
                request_id=request_id,
                candidate_id=None,
                provider_name=provider_name,
                provider_id=str(provider_id or ""),
                endpoint_id=str(endpoint_id or ""),
                key_id=str(key_id or ""),
                method="POST",
                url=url,
                headers=dict(headers),
                body=build_execution_plan_body(body, content_type="application/json"),
                stream=False,
                provider_api_format=api_format,
                client_api_format=api_format,
                model_name=model_name,
                content_type="application/json",
                proxy=proxy_snapshot,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=30_000,
                    read_ms=30_000,
                    write_ms=30_000,
                    pool_ms=30_000,
                    total_ms=30_000,
                ),
            )
        )
    except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Rust test-connection unavailable url={}: {}", url, exc)
        raise HTTPException(
            status_code=503,
            detail="System catalog test-connection requires Rust executor",
        ) from exc

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
        request=httpx.Request("POST", url, headers=headers),
        headers=response_headers,
        content=response_body,
    )


async def _service_health_response(db: Session) -> dict[str, Any]:
    active_providers = (
        db.query(func.count(Provider.id)).filter(Provider.is_active.is_(True)).scalar() or 0
    )
    active_models = db.query(func.count(Model.id)).filter(Model.is_active.is_(True)).scalar() or 0

    redis_info: dict[str, Any] = {"status": "unknown"}
    try:
        redis = await get_redis_client()
        if redis:
            await redis.ping()
            redis_info = {"status": "ok"}
        else:
            redis_info = {"status": "degraded", "message": "Redis client not initialized"}
    except Exception as exc:
        redis_info = {"status": "error", "message": str(exc)}

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "active_providers": active_providers,
            "active_models": active_models,
        },
        "dependencies": {
            "database": {"status": "ok"},
            "redis": redis_info,
        },
    }


def _health_check_response() -> dict[str, Any]:
    try:
        pool_status = get_pool_status()
        pool_health = {
            "checked_out": pool_status["checked_out"],
            "pool_size": pool_status["pool_size"],
            "overflow": pool_status["overflow"],
            "max_capacity": pool_status["max_capacity"],
            "usage_rate": (
                f"{(pool_status['checked_out'] / pool_status['max_capacity'] * 100):.1f}%"
                if pool_status["max_capacity"] > 0
                else "0.0%"
            ),
        }
    except Exception as e:
        pool_health = {"error": str(e)}

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database_pool": pool_health,
    }


def _root_response(db: Session) -> dict[str, Any]:
    top_provider = (
        db.query(Provider)
        .options(load_only(Provider.id, Provider.name, Provider.provider_priority))
        .filter(Provider.is_active.is_(True))
        .order_by(Provider.provider_priority.asc())
        .first()
    )
    active_providers = (
        db.query(func.count(Provider.id)).filter(Provider.is_active.is_(True)).scalar() or 0
    )

    return {
        "message": "AI Proxy with Modular Architecture v4.0.0",
        "status": "running",
        "current_provider": top_provider.name if top_provider else "None",
        "available_providers": active_providers,
        "config": {},
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/v1/health",
            "providers": "/v1/providers",
            "test_connection": "/v1/test-connection",
        },
    }


def _list_providers_response(
    db: Session,
    *,
    include_models: bool,
    include_endpoints: bool,
    active_only: bool,
) -> dict[str, Any]:
    load_options = [
        load_only(Provider.id, Provider.name, Provider.is_active, Provider.provider_priority)
    ]
    if include_models:
        load_options.append(
            selectinload(Provider.models)
            .load_only(
                Model.id,
                Model.provider_model_name,
                Model.is_active,
                Model.supports_streaming,
                Model.global_model_id,
            )
            .selectinload(Model.global_model)
            .load_only(GlobalModel.id, GlobalModel.name, GlobalModel.display_name)
        )
    if include_endpoints:
        load_options.append(
            selectinload(Provider.endpoints).load_only(
                ProviderEndpoint.id,
                ProviderEndpoint.base_url,
                ProviderEndpoint.api_format,
                ProviderEndpoint.is_active,
            )
        )

    base_query = db.query(Provider)
    if load_options:
        base_query = base_query.options(*load_options)
    if active_only:
        base_query = base_query.filter(Provider.is_active.is_(True))
    base_query = base_query.order_by(Provider.provider_priority.asc(), Provider.name.asc())

    providers = base_query.all()
    return {
        "providers": [
            _serialize_provider(provider, include_models, include_endpoints)
            for provider in providers
        ]
    }


def _provider_detail_response(
    db: Session,
    *,
    provider_identifier: str,
    include_models: bool,
    include_endpoints: bool,
) -> dict[str, Any]:
    load_options = [
        load_only(Provider.id, Provider.name, Provider.is_active, Provider.provider_priority)
    ]
    if include_models:
        load_options.append(
            selectinload(Provider.models)
            .load_only(
                Model.id,
                Model.provider_model_name,
                Model.is_active,
                Model.supports_streaming,
                Model.global_model_id,
            )
            .selectinload(Model.global_model)
            .load_only(GlobalModel.id, GlobalModel.name, GlobalModel.display_name)
        )
    if include_endpoints:
        load_options.append(
            selectinload(Provider.endpoints).load_only(
                ProviderEndpoint.id,
                ProviderEndpoint.base_url,
                ProviderEndpoint.api_format,
                ProviderEndpoint.is_active,
            )
        )

    base_query = db.query(Provider)
    if load_options:
        base_query = base_query.options(*load_options)

    provider = base_query.filter(
        (Provider.id == provider_identifier) | (Provider.name == provider_identifier)
    ).first()
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    return _serialize_provider(provider, include_models, include_endpoints)


async def _test_connection_response(
    *,
    request: Request,
    db: Session,
    provider: str | None,
    model: str,
    api_format: str | None,
) -> dict[str, Any]:
    selected_provider = _select_provider(db, provider)
    if not selected_provider:
        raise HTTPException(status_code=503, detail="No active provider available")

    active_endpoints: list[ProviderEndpoint] = [
        ep for ep in (selected_provider.endpoints or []) if getattr(ep, "is_active", False)
    ]
    if not active_endpoints:
        raise HTTPException(status_code=503, detail="Provider has no active endpoints")

    if api_format:
        endpoint = next(
            (ep for ep in active_endpoints if (ep.api_format or "") == api_format),
            None,
        )
        if not endpoint:
            raise HTTPException(
                status_code=400,
                detail=f"Provider has no active endpoint for api_format={api_format}",
            )
        format_value = api_format
    else:
        endpoint = active_endpoints[0]
        format_value = endpoint.api_format or "claude:chat"

    active_keys: list[ProviderAPIKey] = [
        k for k in (selected_provider.api_keys or []) if getattr(k, "is_active", False)
    ]
    if not active_keys:
        raise HTTPException(status_code=503, detail="Provider has no active api keys")

    def _key_supports_format(k: ProviderAPIKey) -> bool:
        formats = getattr(k, "api_formats", None)
        if formats is None:
            return True
        if isinstance(formats, list):
            return str(format_value) in {str(x) for x in formats}
        return True

    key = next((k for k in active_keys if _key_supports_format(k)), active_keys[0])

    payload = build_test_request_body(
        format_value,
        request_data={
            "model": model,
            "messages": [{"role": "user", "content": "Health check"}],
            "max_tokens": 5,
        },
    )

    try:
        auth_info = await get_provider_auth(endpoint, key)

        request_builder = PassthroughRequestBuilder()
        provider_payload, provider_headers = request_builder.build(
            payload,
            {},
            endpoint,
            key,
            is_stream=False,
            pre_computed_auth=auth_info.as_tuple() if auth_info else None,
        )

        url = build_provider_url(
            endpoint,
            query_params=dict(request.query_params),
            path_params={"model": model},
            is_stream=False,
            key=key,
            decrypted_auth_config=auth_info.decrypted_auth_config if auth_info else None,
        )
        proxy_config, delegate_cfg, proxy_snapshot = await _build_test_connection_transport_context(
            endpoint,
            key,
        )

        resp = await _try_rust_test_connection_response(
            request_id=f"test-connection:{selected_provider.id}:{model}",
            url=url,
            headers=provider_headers,
            body=provider_payload,
            provider_name=selected_provider.name,
            provider_id=getattr(selected_provider, "id", None),
            endpoint_id=getattr(endpoint, "id", None),
            key_id=getattr(key, "id", None),
            api_format=format_value,
            model_name=model,
            proxy_snapshot=proxy_snapshot,
        )
        if resp is None:
            raise HTTPException(
                status_code=503,
                detail="System catalog test-connection requires Rust executor",
            )
        resp.raise_for_status()
        response = resp.json()

        return {
            "status": "success",
            "provider": selected_provider.name,
            "endpoint_id": getattr(endpoint, "id", None),
            "api_format": format_value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_id": response.get("id", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"API connectivity test failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))


class PublicServiceHealthAdapter(PublicSystemCatalogApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _service_health_response(context.db)


class PublicSimpleHealthCheckAdapter(PublicSystemCatalogApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        del context
        return _health_check_response()


class PublicRootCatalogAdapter(PublicSystemCatalogApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _root_response(context.db)


@dataclass
class PublicProvidersListAdapter(PublicSystemCatalogApiAdapter):
    include_models: bool
    include_endpoints: bool
    active_only: bool

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _list_providers_response(
            context.db,
            include_models=self.include_models,
            include_endpoints=self.include_endpoints,
            active_only=self.active_only,
        )


@dataclass
class PublicProviderDetailAdapter(PublicSystemCatalogApiAdapter):
    provider_identifier: str
    include_models: bool
    include_endpoints: bool

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _provider_detail_response(
            context.db,
            provider_identifier=self.provider_identifier,
            include_models=self.include_models,
            include_endpoints=self.include_endpoints,
        )


@dataclass
class PublicTestConnectionAdapter(PublicSystemCatalogApiAdapter):
    provider: str | None
    model: str
    api_format: str | None

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _test_connection_response(
            request=context.request,
            db=context.db,
            provider=self.provider,
            model=self.model,
            api_format=self.api_format,
        )


# ============== 端点 ==============


@router.get("/v1/health")
async def service_health(request: Request, db: Session = Depends(get_db)) -> Any:
    """返回服务健康状态与依赖信息"""
    adapter = PublicServiceHealthAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/health")
async def health_check(request: Request, db: Session = Depends(get_db)) -> Any:
    """简单健康检查端点（无需认证）"""
    adapter = PublicSimpleHealthCheckAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/")
async def root(request: Request, db: Session = Depends(get_db)) -> Any:
    """Root endpoint - 服务信息概览"""
    adapter = PublicRootCatalogAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/v1/providers")
async def list_providers(
    request: Request,
    db: Session = Depends(get_db),
    include_models: bool = Query(False),
    include_endpoints: bool = Query(False),
    active_only: bool = Query(True),
) -> Any:
    """列出所有 Provider"""
    adapter = PublicProvidersListAdapter(
        include_models=include_models,
        include_endpoints=include_endpoints,
        active_only=active_only,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/v1/providers/{provider_identifier}")
async def provider_detail(
    provider_identifier: str,
    request: Request,
    db: Session = Depends(get_db),
    include_models: bool = Query(False),
    include_endpoints: bool = Query(False),
) -> Any:
    """获取单个 Provider 详情"""
    adapter = PublicProviderDetailAdapter(
        provider_identifier=provider_identifier,
        include_models=include_models,
        include_endpoints=include_endpoints,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/v1/test-connection")
async def test_connection(
    request: Request,
    db: Session = Depends(get_db),
    provider: str | None = Query(None),
    model: str = Query("claude-3-haiku-20240307"),
    api_format: str | None = Query(None),
) -> Any:
    """测试 Provider 连接"""
    adapter = PublicTestConnectionAdapter(
        provider=provider,
        model=model,
        api_format=api_format,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/test-connection")
async def test_connection_legacy(
    request: Request,
    db: Session = Depends(get_db),
    provider: str | None = Query(None),
    model: str = Query("claude-3-haiku-20240307"),
    api_format: str | None = Query(None),
) -> Any:
    """测试 Provider 连接（legacy alias，已弃用）"""
    del request, db, provider, model, api_format
    raise HTTPException(
        status_code=410,
        detail="Deprecated endpoint. Please use /v1/test-connection.",
    )
