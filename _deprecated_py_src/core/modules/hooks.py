"""
模块钩子系统

提供模块与核心代码之间的动态扩展点。
模块通过 ModuleDefinition.hooks 声明钩子实现，
核心代码通过 HookDispatcher 调用所有活跃模块的钩子。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

from src.core.logger import logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# 钩子处理器类型: 可以是同步或异步函数
HookHandler = Any  # Callable[..., Any]


class HookStrategy(str, Enum):
    """钩子执行策略"""

    FIRST_RESULT = "first_result"  # 返回第一个非 None 结果
    COLLECT_ALL = "collect_all"  # 收集所有结果到列表


@dataclass(frozen=True)
class HookSpec:
    """钩子规格定义"""

    name: str  # 如 "auth.authenticate"
    strategy: HookStrategy = HookStrategy.FIRST_RESULT
    requires_active_check: bool = True  # 是否过滤非活跃模块


# ==================== 预定义钩子规格 ====================

AUTH_GET_METHODS = HookSpec(
    name="auth.get_methods",
    strategy=HookStrategy.COLLECT_ALL,
)
"""查询所有可用认证方法。返回 list[dict]，每个 dict 包含认证方式信息。"""

AUTH_AUTHENTICATE = HookSpec(
    name="auth.authenticate",
    strategy=HookStrategy.FIRST_RESULT,
)
"""模块参与认证流程。kwargs: db, email, password, auth_type。返回 User 或 None。"""

AUTH_CHECK_REGISTRATION = HookSpec(
    name="auth.check_registration",
    strategy=HookStrategy.FIRST_RESULT,
)
"""模块检查是否允许本地注册。返回 {"blocked": True, "reason": "..."} 或 None。"""

AUTH_CHECK_EXCLUSIVE_MODE = HookSpec(
    name="auth.check_exclusive_mode",
    strategy=HookStrategy.FIRST_RESULT,
)
"""检查是否有模块开启了排他登录模式。返回 True 或 None。"""

AUTH_TOKEN_PREFIX_AUTHENTICATORS = HookSpec(
    name="auth.token_prefix_authenticators",
    strategy=HookStrategy.COLLECT_ALL,
    requires_active_check=False,  # token 前缀认证是核心鉴权路径，只要模块已注册即可
)
"""获取 token 前缀认证器列表。返回 list[{"prefix": "ae_", "module": "..."}]。"""


class HookDispatcher:
    """
    钩子分发器 -- 单例

    职责:
    - 注册模块的钩子实现
    - 在核心代码调用时，只执行活跃模块的钩子
    - 支持 FIRST_RESULT 和 COLLECT_ALL 两种执行策略
    """

    _instance: HookDispatcher | None = None

    def __init__(self) -> None:
        # {hook_name: [(module_name, handler), ...]}
        self._handlers: defaultdict[str, list[tuple[str, HookHandler]]] = defaultdict(list)

    @classmethod
    def get_instance(cls) -> HookDispatcher:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """重置单例（仅用于测试）"""
        cls._instance = None

    def register(self, hook_name: str, module_name: str, handler: HookHandler) -> None:
        """注册钩子处理器"""
        self._handlers[hook_name].append((module_name, handler))
        logger.debug("Hook [{}] registered handler from module [{}]", hook_name, module_name)

    def has_handlers(self, hook_name: str) -> bool:
        """检查是否有注册的处理器"""
        return bool(self._handlers.get(hook_name))

    def _get_active_handlers(
        self, spec: HookSpec, db: Session | None
    ) -> list[tuple[str, HookHandler]]:
        """获取活跃模块的处理器列表"""
        handlers = self._handlers.get(spec.name, [])
        if not handlers:
            return []

        if not spec.requires_active_check or db is None:
            return handlers

        from src.core.modules.registry import get_module_registry

        registry = get_module_registry()
        return [(name, handler) for name, handler in handlers if registry.is_active(name, db)]

    # ==================== 异步分发 ====================

    async def dispatch(
        self,
        spec: HookSpec,
        **kwargs: Any,
    ) -> Any:
        """
        异步分发钩子调用

        从 kwargs 中提取 db 参数用于活跃性检查，所有 kwargs 原样传递给处理器。

        Args:
            spec: 钩子规格
            **kwargs: 传递给处理器的参数（其中 db 同时用于活跃性检查）

        Returns:
            FIRST_RESULT: 第一个非 None 结果，或 None
            COLLECT_ALL: 结果列表
        """
        db = kwargs.get("db")
        active_handlers = self._get_active_handlers(spec, db)
        if not active_handlers:
            return [] if spec.strategy == HookStrategy.COLLECT_ALL else None

        if spec.strategy == HookStrategy.FIRST_RESULT:
            return await self._dispatch_first_result(spec.name, active_handlers, **kwargs)
        elif spec.strategy == HookStrategy.COLLECT_ALL:
            return await self._dispatch_collect_all(spec.name, active_handlers, **kwargs)
        return None

    async def _call_handler(self, handler: HookHandler, **kwargs: Any) -> Any:
        """调用处理器（支持同步和异步）"""
        result = handler(**kwargs)
        if isawaitable(result):
            return await result
        return result

    async def _dispatch_first_result(
        self, hook_name: str, handlers: list[tuple[str, HookHandler]], **kwargs: Any
    ) -> Any:
        for module_name, handler in handlers:
            try:
                result = await self._call_handler(handler, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.error("Hook [{}] handler from [{}] failed: {}", hook_name, module_name, e)
        return None

    async def _dispatch_collect_all(
        self, hook_name: str, handlers: list[tuple[str, HookHandler]], **kwargs: Any
    ) -> list[Any]:
        results: list[Any] = []
        for module_name, handler in handlers:
            try:
                result = await self._call_handler(handler, **kwargs)
                if result is not None:
                    if isinstance(result, list):
                        results.extend(result)
                    else:
                        results.append(result)
            except Exception as e:
                logger.error("Hook [{}] handler from [{}] failed: {}", hook_name, module_name, e)
        return results

    # ==================== 同步分发 ====================

    def dispatch_sync(
        self,
        spec: HookSpec,
        **kwargs: Any,
    ) -> Any:
        """
        同步版本的 dispatch（仅适用于同步钩子处理器）

        从 kwargs 中提取 db 参数用于活跃性检查，所有 kwargs 原样传递给处理器。
        用于无法使用 await 的同步上下文（如 OAuthService 的某些方法）。
        """
        db = kwargs.get("db")
        active_handlers = self._get_active_handlers(spec, db)
        if not active_handlers:
            return [] if spec.strategy == HookStrategy.COLLECT_ALL else None

        if spec.strategy == HookStrategy.FIRST_RESULT:
            for module_name, handler in active_handlers:
                try:
                    result = handler(**kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    logger.error(
                        "Hook [{}] sync handler from [{}] failed: {}", spec.name, module_name, e
                    )
            return None

        elif spec.strategy == HookStrategy.COLLECT_ALL:
            results: list[Any] = []
            for module_name, handler in active_handlers:
                try:
                    result = handler(**kwargs)
                    if result is not None:
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                except Exception as e:
                    logger.error(
                        "Hook [{}] sync handler from [{}] failed: {}", spec.name, module_name, e
                    )
            return results

        return None


def get_hook_dispatcher() -> HookDispatcher:
    """获取钩子分发器实例"""
    return HookDispatcher.get_instance()
