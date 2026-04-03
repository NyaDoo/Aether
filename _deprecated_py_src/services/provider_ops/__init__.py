"""
Provider 操作模块

提供对提供商的扩展操作支持：
- 多种鉴权方式（API Key、登录、Cookie）
- 可扩展的操作类型（余额查询、签到等）
"""

from src.services.provider_ops.registry import ArchitectureRegistry, get_registry
from src.services.provider_ops.service import ProviderOpsService
from src.services.provider_ops.types import (
    ActionResult,
    ActionStatus,
    BalanceInfo,
    CheckinInfo,
    ConnectorAuthType,
    ConnectorState,
    ConnectorStatus,
    ProviderActionType,
    ProviderOpsConfig,
)

__all__ = [
    # 服务
    "ProviderOpsService",
    # 注册表
    "ArchitectureRegistry",
    "get_registry",
    # 类型
    "ActionResult",
    "ActionStatus",
    "BalanceInfo",
    "CheckinInfo",
    "ConnectorAuthType",
    "ConnectorState",
    "ConnectorStatus",
    "ProviderActionType",
    "ProviderOpsConfig",
]
