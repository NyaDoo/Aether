"""
Provider 操作 API 路由

提供 Provider 操作相关的 API 端点：
- 架构列表
- 连接管理
- 操作执行（余额查询、签到等）
- 配置管理
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.database import get_db
from src.models.database import Provider
from src.services.provider_ops import (
    ConnectorAuthType,
    ProviderActionType,
    ProviderOpsConfig,
    ProviderOpsService,
    get_registry,
)

router = APIRouter(prefix="/api/admin/provider-ops", tags=["Provider Operations"])
pipeline = get_pipeline()


# ==================== Request/Response Models ====================


class ArchitectureInfo(BaseModel):
    """架构信息"""

    architecture_id: str
    display_name: str
    description: str
    credentials_schema: dict[str, Any]
    supported_auth_types: list[dict[str, Any]]
    supported_actions: list[dict[str, Any]]
    default_connector: str | None


class ConnectorConfigRequest(BaseModel):
    """连接器配置请求"""

    auth_type: str = Field(..., description="认证类型")
    config: dict[str, Any] = Field(default_factory=dict, description="连接器配置")
    credentials: dict[str, Any] = Field(default_factory=dict, description="凭据信息")


class ActionConfigRequest(BaseModel):
    """操作配置请求"""

    enabled: bool = Field(True, description="是否启用")
    config: dict[str, Any] = Field(default_factory=dict, description="操作配置")


class SaveConfigRequest(BaseModel):
    """保存配置请求"""

    architecture_id: str = Field("generic_api", description="架构 ID")
    base_url: str | None = Field(None, description="API 基础地址")
    connector: ConnectorConfigRequest
    actions: dict[str, ActionConfigRequest] = Field(default_factory=dict)
    schedule: dict[str, str] = Field(default_factory=dict, description="定时任务配置")


class ConnectRequest(BaseModel):
    """连接请求"""

    credentials: dict[str, Any] | None = Field(None, description="凭据（可选，使用已保存的）")


class ExecuteActionRequest(BaseModel):
    """执行操作请求"""

    config: dict[str, Any] | None = Field(None, description="操作配置（覆盖默认）")


class ConnectionStatusResponse(BaseModel):
    """连接状态响应"""

    status: str
    auth_type: str
    connected_at: str | None
    expires_at: str | None
    last_error: str | None


class ActionResultResponse(BaseModel):
    """操作结果响应"""

    status: str
    action_type: str
    data: Any | None
    message: str | None
    executed_at: str
    response_time_ms: int | None
    cache_ttl_seconds: int


class ProviderOpsStatusResponse(BaseModel):
    """Provider 操作状态响应"""

    provider_id: str
    is_configured: bool
    architecture_id: str | None
    connection_status: ConnectionStatusResponse
    enabled_actions: list[str]


class ProviderOpsConfigResponse(BaseModel):
    """Provider 操作配置响应（脱敏）"""

    provider_id: str
    is_configured: bool
    architecture_id: str | None = None
    base_url: str | None = None
    connector: dict[str, Any] | None = None


class VerifyAuthResponse(BaseModel):
    """验证认证响应"""

    success: bool
    message: str | None = None
    data: dict[str, Any] | None = None
    updated_credentials: dict[str, Any] | None = None


# ==================== Helper Functions ====================


def _serialize_data(data: Any) -> Any:
    """序列化 dataclass 为字典，用于 JSON 响应"""
    if data is None:
        return None
    if is_dataclass(data) and not isinstance(data, type):
        return asdict(data)
    return data


def _build_action_result_response(result: Any) -> ActionResultResponse:
    return ActionResultResponse(
        status=result.status.value,
        action_type=result.action_type.value,
        data=_serialize_data(result.data),
        message=result.message,
        executed_at=result.executed_at.isoformat(),
        response_time_ms=result.response_time_ms,
        cache_ttl_seconds=result.cache_ttl_seconds,
    )


def _resolve_provider_base_url(
    provider_id: str,
    db: Session,
    *,
    saved_config: ProviderOpsConfig | None = None,
) -> str | None:
    base_url = saved_config.base_url if saved_config else None
    if base_url:
        return base_url

    provider = db.query(Provider).filter(Provider.id == provider_id).first()
    if not provider:
        return None

    if provider.endpoints:
        for endpoint in provider.endpoints:
            if endpoint.base_url:
                return endpoint.base_url

    provider_config = provider.config or {}
    return provider_config.get("base_url") or provider.website


def _list_architectures_response() -> list[dict[str, Any]]:
    registry = get_registry()
    return registry.to_dict_list()


def _get_architecture_response(architecture_id: str) -> dict[str, Any]:
    registry = get_registry()
    arch = registry.get(architecture_id)
    if not arch:
        raise HTTPException(status_code=404, detail=f"架构 {architecture_id} 不存在")
    return arch.to_dict()


def _get_provider_ops_status_response(provider_id: str, db: Session) -> ProviderOpsStatusResponse:
    service = ProviderOpsService(db)
    config = service.get_config(provider_id)
    conn_state = service.get_connection_status(provider_id)

    enabled_actions = []
    if config:
        for action_type, action_config in config.actions.items():
            if action_config.get("enabled", True):
                enabled_actions.append(action_type)

    return ProviderOpsStatusResponse(
        provider_id=provider_id,
        is_configured=config is not None,
        architecture_id=config.architecture_id if config else None,
        connection_status=ConnectionStatusResponse(
            status=conn_state.status.value,
            auth_type=conn_state.auth_type.value,
            connected_at=conn_state.connected_at.isoformat() if conn_state.connected_at else None,
            expires_at=conn_state.expires_at.isoformat() if conn_state.expires_at else None,
            last_error=conn_state.last_error,
        ),
        enabled_actions=enabled_actions,
    )


def _get_provider_ops_config_response(
    provider_id: str,
    db: Session,
) -> ProviderOpsConfigResponse:
    service = ProviderOpsService(db)
    config = service.get_config(provider_id)

    if not config:
        return ProviderOpsConfigResponse(
            provider_id=provider_id,
            is_configured=False,
        )

    masked_credentials = service.get_masked_credentials(config.connector_credentials)
    base_url = _resolve_provider_base_url(provider_id, db, saved_config=config)

    return ProviderOpsConfigResponse(
        provider_id=provider_id,
        is_configured=True,
        architecture_id=config.architecture_id,
        base_url=base_url,
        connector={
            "auth_type": config.connector_auth_type.value,
            "config": config.connector_config,
            "credentials": masked_credentials,
        },
    )


def _save_provider_ops_config_response(
    provider_id: str,
    payload: SaveConfigRequest,
    db: Session,
) -> dict[str, Any]:
    service = ProviderOpsService(db)
    credentials = service.merge_credentials_with_saved(
        provider_id, dict(payload.connector.credentials)
    )

    config = ProviderOpsConfig(
        architecture_id=payload.architecture_id,
        base_url=payload.base_url,
        connector_auth_type=ConnectorAuthType(payload.connector.auth_type),
        connector_config=payload.connector.config,
        connector_credentials=credentials,
        actions={
            action_type: {"enabled": action_config.enabled, "config": action_config.config}
            for action_type, action_config in payload.actions.items()
        },
        schedule=payload.schedule,
    )

    success = service.save_config(provider_id, config)
    if not success:
        raise HTTPException(status_code=404, detail="Provider 不存在")

    return {"success": True, "message": "配置保存成功"}


async def _verify_provider_auth_response(
    provider_id: str,
    payload: SaveConfigRequest,
    db: Session,
) -> VerifyAuthResponse:
    service = ProviderOpsService(db)
    base_url = payload.base_url or _resolve_provider_base_url(provider_id, db)
    if not base_url:
        return VerifyAuthResponse(
            success=False,
            message="请提供 API 地址",
        )

    credentials = service.merge_credentials_with_saved(
        provider_id, dict(payload.connector.credentials)
    )
    result = await service.verify_auth(
        base_url=base_url,
        architecture_id=payload.architecture_id,
        auth_type=ConnectorAuthType(payload.connector.auth_type),
        config=payload.connector.config,
        credentials=credentials,
        provider_id=provider_id,
    )

    return VerifyAuthResponse(
        success=result.get("success", False),
        message=result.get("message"),
        data=result.get("data"),
        updated_credentials=result.get("updated_credentials"),
    )


def _delete_provider_ops_config_response(provider_id: str, db: Session) -> dict[str, Any]:
    service = ProviderOpsService(db)
    success = service.delete_config(provider_id)
    if not success:
        raise HTTPException(status_code=404, detail="Provider 不存在")
    return {"success": True, "message": "配置已删除"}


async def _connect_provider_response(
    provider_id: str,
    payload: ConnectRequest,
    db: Session,
) -> dict[str, Any]:
    service = ProviderOpsService(db)
    success, message = await service.connect(provider_id, payload.credentials)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"success": True, "message": message}


async def _disconnect_provider_response(provider_id: str, db: Session) -> dict[str, Any]:
    service = ProviderOpsService(db)
    await service.disconnect(provider_id)
    return {"success": True, "message": "已断开连接"}


async def _execute_action_response(
    provider_id: str,
    action_type: str,
    payload: ExecuteActionRequest,
    db: Session,
) -> ActionResultResponse:
    service = ProviderOpsService(db)
    try:
        action_type_enum = ProviderActionType(action_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"无效的操作类型: {action_type}") from exc

    result = await service.execute_action(provider_id, action_type_enum, payload.config)
    return _build_action_result_response(result)


async def _get_balance_response(
    provider_id: str,
    refresh: bool,
    db: Session,
) -> ActionResultResponse:
    service = ProviderOpsService(db)
    result = await service.query_balance_with_cache(provider_id, trigger_refresh=refresh)
    return _build_action_result_response(result)


async def _refresh_balance_response(provider_id: str, db: Session) -> ActionResultResponse:
    service = ProviderOpsService(db)
    result = await service.query_balance(provider_id)
    return _build_action_result_response(result)


async def _checkin_response(provider_id: str, db: Session) -> ActionResultResponse:
    service = ProviderOpsService(db)
    result = await service.checkin(provider_id)
    return _build_action_result_response(result)


async def _batch_query_balance_response(
    provider_ids: list[str] | None,
    db: Session,
) -> dict[str, ActionResultResponse]:
    service = ProviderOpsService(db)
    results = await service.batch_query_balance(provider_ids)
    return {
        provider_id: _build_action_result_response(result)
        for provider_id, result in results.items()
    }


# ==================== Routes ====================


@router.get("/architectures", response_model=list[ArchitectureInfo])
async def list_architectures(request: Request, db: Session = Depends(get_db)) -> Any:
    """获取所有可用的架构"""
    adapter = AdminProviderOpsListArchitecturesAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/architectures/{architecture_id}", response_model=ArchitectureInfo)
async def get_architecture(
    architecture_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """获取指定架构的详情"""
    adapter = AdminProviderOpsGetArchitectureAdapter(architecture_id=architecture_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/providers/{provider_id}/status", response_model=ProviderOpsStatusResponse)
async def get_provider_ops_status(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """获取 Provider 的操作状态"""
    adapter = AdminProviderOpsStatusAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/providers/{provider_id}/config", response_model=ProviderOpsConfigResponse)
async def get_provider_ops_config(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """获取 Provider 的操作配置（脱敏）"""
    adapter = AdminProviderOpsConfigAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.put("/providers/{provider_id}/config")
async def save_provider_ops_config(
    provider_id: str,
    payload: SaveConfigRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """保存 Provider 的操作配置"""
    adapter = AdminProviderOpsSaveConfigAdapter(provider_id=provider_id, payload=payload)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/providers/{provider_id}/verify", response_model=VerifyAuthResponse)
async def verify_provider_auth(
    provider_id: str,
    payload: SaveConfigRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """验证 Provider 认证配置"""
    adapter = AdminProviderOpsVerifyAuthAdapter(provider_id=provider_id, payload=payload)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.delete("/providers/{provider_id}/config")
async def delete_provider_ops_config(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """删除 Provider 的操作配置"""
    adapter = AdminProviderOpsDeleteConfigAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/providers/{provider_id}/connect")
async def connect_provider(
    provider_id: str,
    payload: ConnectRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """建立与 Provider 的连接"""
    adapter = AdminProviderOpsConnectAdapter(provider_id=provider_id, payload=payload)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/providers/{provider_id}/disconnect")
async def disconnect_provider(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """断开与 Provider 的连接"""
    adapter = AdminProviderOpsDisconnectAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post(
    "/providers/{provider_id}/actions/{action_type}",
    response_model=ActionResultResponse,
)
async def execute_action(
    provider_id: str,
    action_type: str,
    payload: ExecuteActionRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """执行指定操作"""
    adapter = AdminProviderOpsExecuteActionAdapter(
        provider_id=provider_id,
        action_type=action_type,
        payload=payload,
    )
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.get("/providers/{provider_id}/balance", response_model=ActionResultResponse)
async def get_balance(
    provider_id: str,
    request: Request,
    refresh: bool = True,
    db: Session = Depends(get_db),
) -> Any:
    """获取余额（优先返回缓存，后台异步刷新）"""
    adapter = AdminProviderOpsGetBalanceAdapter(provider_id=provider_id, refresh=refresh)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/providers/{provider_id}/balance", response_model=ActionResultResponse)
async def refresh_balance(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """立即刷新余额（同步等待结果）"""
    adapter = AdminProviderOpsRefreshBalanceAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/providers/{provider_id}/checkin", response_model=ActionResultResponse)
async def checkin(
    provider_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """签到（快捷方法）"""
    adapter = AdminProviderOpsCheckinAdapter(provider_id=provider_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.post("/batch/balance")
async def batch_query_balance(
    request: Request,
    provider_ids: list[str] | None = Query(None),
    db: Session = Depends(get_db),
) -> Any:
    """批量查询余额"""
    adapter = AdminProviderOpsBatchBalanceAdapter(provider_ids=provider_ids)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


# ==================== Adapters ====================


class AdminProviderOpsListArchitecturesAdapter(AdminApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _list_architectures_response()


@dataclass
class AdminProviderOpsGetArchitectureAdapter(AdminApiAdapter):
    architecture_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _get_architecture_response(self.architecture_id)


@dataclass
class AdminProviderOpsStatusAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _get_provider_ops_status_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsConfigAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _get_provider_ops_config_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsSaveConfigAdapter(AdminApiAdapter):
    provider_id: str
    payload: SaveConfigRequest

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _save_provider_ops_config_response(self.provider_id, self.payload, context.db)


@dataclass
class AdminProviderOpsVerifyAuthAdapter(AdminApiAdapter):
    provider_id: str
    payload: SaveConfigRequest

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _verify_provider_auth_response(self.provider_id, self.payload, context.db)


@dataclass
class AdminProviderOpsDeleteConfigAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return _delete_provider_ops_config_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsConnectAdapter(AdminApiAdapter):
    provider_id: str
    payload: ConnectRequest

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _connect_provider_response(self.provider_id, self.payload, context.db)


@dataclass
class AdminProviderOpsDisconnectAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _disconnect_provider_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsExecuteActionAdapter(AdminApiAdapter):
    provider_id: str
    action_type: str
    payload: ExecuteActionRequest

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _execute_action_response(
            self.provider_id,
            self.action_type,
            self.payload,
            context.db,
        )


@dataclass
class AdminProviderOpsGetBalanceAdapter(AdminApiAdapter):
    provider_id: str
    refresh: bool

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _get_balance_response(self.provider_id, self.refresh, context.db)


@dataclass
class AdminProviderOpsRefreshBalanceAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _refresh_balance_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsCheckinAdapter(AdminApiAdapter):
    provider_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _checkin_response(self.provider_id, context.db)


@dataclass
class AdminProviderOpsBatchBalanceAdapter(AdminApiAdapter):
    provider_ids: list[str] | None

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _batch_query_balance_response(self.provider_ids, context.db)
