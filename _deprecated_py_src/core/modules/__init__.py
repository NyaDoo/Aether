"""
模块化系统核心

提供可扩展的功能模块管理，支持：
- 声明式模块注册
- available/enabled 双层状态控制
- 延迟导入避免重依赖加载
- 前后端状态同步
"""

from src.core.modules.base import (
    ModuleCategory,
    ModuleDefinition,
    ModuleMetadata,
    ModuleStatus,
)
from src.core.modules.hooks import (
    AUTH_AUTHENTICATE,
    AUTH_CHECK_EXCLUSIVE_MODE,
    AUTH_CHECK_REGISTRATION,
    AUTH_GET_METHODS,
    AUTH_TOKEN_PREFIX_AUTHENTICATORS,
    HookDispatcher,
    HookSpec,
    HookStrategy,
    get_hook_dispatcher,
)
from src.core.modules.registry import ModuleRegistry, get_module_registry

__all__ = [
    "ModuleCategory",
    "ModuleMetadata",
    "ModuleDefinition",
    "ModuleStatus",
    "ModuleRegistry",
    "get_module_registry",
    # Hook system
    "HookDispatcher",
    "HookSpec",
    "HookStrategy",
    "get_hook_dispatcher",
    "AUTH_GET_METHODS",
    "AUTH_AUTHENTICATE",
    "AUTH_CHECK_REGISTRATION",
    "AUTH_CHECK_EXCLUSIVE_MODE",
    "AUTH_TOKEN_PREFIX_AUTHENTICATORS",
]
