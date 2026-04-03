"""
计费模块数据模型

定义计费相关的核心数据结构：
- BillingUnit: 计费单位枚举
- BillingDimension: 计费维度定义
- StandardizedUsage: 标准化的 usage 数据
- CostBreakdown: 计费明细结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BillingUnit(str, Enum):
    """计费单位"""

    PER_1M_TOKENS = "per_1m_tokens"  # 每百万 token
    PER_1M_TOKENS_HOUR = "per_1m_tokens_hour"  # 每百万 token 每小时（豆包缓存存储）
    PER_REQUEST = "per_request"  # 每次请求
    FIXED = "fixed"  # 固定费用


@dataclass
class BillingDimension:
    """
    计费维度定义

    每个维度描述一种计费方式，例如：
    - 输入 token 计费
    - 输出 token 计费
    - 缓存读取计费
    - 按次计费
    """

    name: str  # 维度名称，如 "input", "output", "cache_read"
    usage_field: str  # 从 usage 中取值的字段名
    price_field: str  # 价格配置中的字段名
    unit: BillingUnit = BillingUnit.PER_1M_TOKENS  # 计费单位
    default_price: float = 0.0  # 默认价格（当价格配置中没有时使用）

    def calculate(self, usage_value: float, price: float) -> float:
        """
        计算该维度的费用

        Args:
            usage_value: 使用量数值
            price: 单价

        Returns:
            计算后的费用
        """
        if usage_value <= 0 or price <= 0:
            return 0.0

        if self.unit == BillingUnit.PER_1M_TOKENS:
            return (usage_value / 1_000_000) * price
        elif self.unit == BillingUnit.PER_1M_TOKENS_HOUR:
            # 缓存存储按 token 数 * 小时数计费
            return (usage_value / 1_000_000) * price
        elif self.unit == BillingUnit.PER_REQUEST:
            return usage_value * price
        elif self.unit == BillingUnit.FIXED:
            return price

        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "name": self.name,
            "usage_field": self.usage_field,
            "price_field": self.price_field,
            "unit": self.unit.value,
            "default_price": self.default_price,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BillingDimension:
        """从字典创建实例"""
        return cls(
            name=data["name"],
            usage_field=data["usage_field"],
            price_field=data["price_field"],
            unit=BillingUnit(data.get("unit", "per_1m_tokens")),
            default_price=data.get("default_price", 0.0),
        )


@dataclass(init=False)
class StandardizedUsage:
    """
    标准化的 Usage 数据

    将不同 API 格式的 usage 统一为标准格式，便于计费计算。
    """

    # 基础 token 计数
    input_tokens: int = 0
    output_tokens: int = 0

    # 缓存相关
    cache_creation_tokens: int = 0  # Claude: 缓存创建
    cache_read_tokens: int = 0  # Claude/OpenAI/豆包: 缓存读取/命中

    # 特殊 token 类型
    reasoning_tokens: int = 0  # o1/豆包: 推理 token（通常包含在 output 中，单独记录用于分析）

    # 时间相关（用于按时计费）
    cache_storage_token_hours: float = 0.0  # 豆包: 缓存存储 token*小时

    # 请求计数（用于按次计费）
    request_count: int = 1

    # 任意维度存储（用于多维度计费；数值/字符串均可）
    # 兼容旧字段名：extra 作为 dimensions 的别名
    dimensions: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        reasoning_tokens: int = 0,
        cache_storage_token_hours: float = 0.0,
        request_count: int = 1,
        dimensions: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        # 基础字段
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.reasoning_tokens = reasoning_tokens
        self.cache_storage_token_hours = cache_storage_token_hours
        self.request_count = request_count

        # 兼容：支持 extra 与 dimensions 同时传入（dimensions 优先级更高）
        merged: dict[str, Any] = {}
        if isinstance(extra, dict):
            merged.update(extra)
        if isinstance(dimensions, dict):
            merged.update(dimensions)
        self.dimensions = merged

    @property
    def extra(self) -> dict[str, Any]:
        """向后兼容：旧代码使用 usage.extra 访问扩展维度。"""
        return self.dimensions

    @extra.setter
    def extra(self, value: dict[str, Any]) -> None:
        """向后兼容：允许旧代码写入 usage.extra。"""
        self.dimensions = value or {}

    def get(self, field_name: str, default: Any = 0) -> Any:
        """
        通用字段获取

        支持获取标准字段和扩展字段。

        Args:
            field_name: 字段名
            default: 默认值

        Returns:
            字段值
        """
        # 兼容旧字段名
        if field_name == "extra":
            return self.dimensions

        if hasattr(self, field_name) and field_name not in {"dimensions"}:
            return getattr(self, field_name)

        return self.dimensions.get(field_name, default)

    def set(self, field_name: str, value: Any) -> None:
        """
        通用字段设置

        Args:
            field_name: 字段名
            value: 字段值
        """
        # 兼容旧字段名
        if field_name == "extra":
            self.dimensions = value or {}
            return

        if hasattr(self, field_name) and field_name not in {"dimensions"}:
            setattr(self, field_name, value)
            return

        self.dimensions[field_name] = value

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        result: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_storage_token_hours": self.cache_storage_token_hours,
            "request_count": self.request_count,
        }
        if self.dimensions:
            # 新字段名
            result["dimensions"] = self.dimensions
            # 旧字段名（兼容）
            result["extra"] = self.dimensions
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StandardizedUsage:
        """从字典创建实例"""
        # 兼容：支持 extra / dimensions 两种键名
        extra = data.pop("extra", {}) if "extra" in data else {}
        dimensions = data.pop("dimensions", {}) if "dimensions" in data else {}
        merged_dimensions: dict[str, Any] = {}
        if isinstance(extra, dict):
            merged_dimensions.update(extra)
        if isinstance(dimensions, dict):
            merged_dimensions.update(dimensions)
        # 只取已知字段
        known_fields = {
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "reasoning_tokens",
            "cache_storage_token_hours",
            "request_count",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered, dimensions=merged_dimensions)


@dataclass
class CostBreakdown:
    """
    计费明细结果

    包含各维度的费用和总费用。
    """

    # 各维度费用 {"input": 0.01, "output": 0.02, "cache_read": 0.001, ...}
    costs: dict[str, float] = field(default_factory=dict)

    # 总费用
    total_cost: float = 0.0

    # 命中的阶梯索引（如果使用阶梯计费）
    tier_index: int | None = None

    # 货币单位
    currency: str = "USD"

    # 使用的价格（用于记录和审计）
    effective_prices: dict[str, float] = field(default_factory=dict)

    # =========================================================================
    # 兼容旧接口的属性（便于渐进式迁移）
    # =========================================================================

    @property
    def input_cost(self) -> float:
        """输入费用"""
        return self.costs.get("input", 0.0)

    @property
    def output_cost(self) -> float:
        """输出费用"""
        return self.costs.get("output", 0.0)

    @property
    def cache_creation_cost(self) -> float:
        """缓存创建费用"""
        return self.costs.get("cache_creation", 0.0)

    @property
    def cache_read_cost(self) -> float:
        """缓存读取费用"""
        return self.costs.get("cache_read", 0.0)

    @property
    def cache_cost(self) -> float:
        """总缓存费用（创建 + 读取）"""
        return self.cache_creation_cost + self.cache_read_cost

    @property
    def request_cost(self) -> float:
        """按次计费费用"""
        return self.costs.get("request", 0.0)

    @property
    def cache_storage_cost(self) -> float:
        """缓存存储费用（豆包等）"""
        return self.costs.get("cache_storage", 0.0)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "costs": self.costs,
            "total_cost": self.total_cost,
            "tier_index": self.tier_index,
            "currency": self.currency,
            "effective_prices": self.effective_prices,
            # 兼容字段
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "cache_creation_cost": self.cache_creation_cost,
            "cache_read_cost": self.cache_read_cost,
            "cache_cost": self.cache_cost,
            "request_cost": self.request_cost,
        }

    def to_legacy_tuple(self) -> tuple:
        """
        转换为旧接口的元组格式

        Returns:
            (input_cost, output_cost, cache_creation_cost, cache_read_cost,
             cache_cost, request_cost, total_cost, tier_index)
        """
        return (
            self.input_cost,
            self.output_cost,
            self.cache_creation_cost,
            self.cache_read_cost,
            self.cache_cost,
            self.request_cost,
            self.total_cost,
            self.tier_index,
        )
