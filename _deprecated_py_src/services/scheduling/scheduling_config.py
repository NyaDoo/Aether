"""
调度配置 (SchedulingConfig)

从 CacheAwareScheduler 提取的常量定义和模式管理逻辑。
"""

from __future__ import annotations

from src.core.logger import logger


class SchedulingConfig:
    """调度配置：管理优先级模式和调度模式的常量、归一化和运行时更新。"""

    # 优先级模式常量
    PRIORITY_MODE_PROVIDER = "provider"  # 提供商优先模式
    PRIORITY_MODE_GLOBAL_KEY = "global_key"  # 全局 Key 优先模式
    ALLOWED_PRIORITY_MODES = {
        PRIORITY_MODE_PROVIDER,
        PRIORITY_MODE_GLOBAL_KEY,
    }

    # 调度模式常量
    SCHEDULING_MODE_FIXED_ORDER = "fixed_order"  # 固定顺序模式：严格按优先级，忽略缓存
    SCHEDULING_MODE_CACHE_AFFINITY = "cache_affinity"  # 缓存亲和模式：优先缓存，同优先级哈希分散
    SCHEDULING_MODE_LOAD_BALANCE = "load_balance"  # 负载均衡模式：忽略缓存，同优先级随机轮换
    ALLOWED_SCHEDULING_MODES = {
        SCHEDULING_MODE_FIXED_ORDER,
        SCHEDULING_MODE_CACHE_AFFINITY,
        SCHEDULING_MODE_LOAD_BALANCE,
    }

    def __init__(
        self,
        priority_mode: str | None = None,
        scheduling_mode: str | None = None,
    ) -> None:
        self.priority_mode = self._normalize_priority_mode(
            priority_mode or self.PRIORITY_MODE_PROVIDER
        )
        self.scheduling_mode = self._normalize_scheduling_mode(
            scheduling_mode or self.SCHEDULING_MODE_CACHE_AFFINITY
        )
        logger.debug(
            "[SchedulingConfig] 初始化优先级模式: {}, 调度模式: {}",
            self.priority_mode,
            self.scheduling_mode,
        )

    def _normalize_priority_mode(self, mode: str | None) -> str:
        normalized = (mode or "").strip().lower()
        if normalized not in self.ALLOWED_PRIORITY_MODES:
            if normalized:
                logger.warning("[SchedulingConfig] 无效的优先级模式 '{}'，回退为 provider", mode)
            return self.PRIORITY_MODE_PROVIDER
        return normalized

    def _normalize_scheduling_mode(self, mode: str | None) -> str:
        normalized = (mode or "").strip().lower()
        if normalized not in self.ALLOWED_SCHEDULING_MODES:
            if normalized:
                logger.warning(
                    "[SchedulingConfig] 无效的调度模式 '{}'，回退为 cache_affinity", mode
                )
            return self.SCHEDULING_MODE_CACHE_AFFINITY
        return normalized

    def set_priority_mode(self, mode: str | None) -> None:
        """运行时更新候选排序策略"""
        normalized = self._normalize_priority_mode(mode)
        if normalized == self.priority_mode:
            return
        self.priority_mode = normalized
        logger.debug("[SchedulingConfig] 切换优先级模式为: {}", self.priority_mode)

    def set_scheduling_mode(self, mode: str | None) -> None:
        """运行时更新调度模式"""
        normalized = self._normalize_scheduling_mode(mode)
        if normalized == self.scheduling_mode:
            return
        self.scheduling_mode = normalized
        logger.debug("[SchedulingConfig] 切换调度模式为: {}", self.scheduling_mode)
