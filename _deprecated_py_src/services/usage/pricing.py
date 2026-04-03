from __future__ import annotations

from sqlalchemy.orm import Session

from src.models.database import ProviderAPIKey
from src.services.model.cost import ModelCostService


class UsagePricingMixin:
    """定价相关方法"""

    @classmethod
    async def get_model_price_async(
        cls, db: Session, provider: str, model: str
    ) -> tuple[float, float]:
        """异步获取模型价格（输入价格，输出价格）每1M tokens

        查找逻辑：
        1. 直接通过 GlobalModel.name 匹配
        2. 查找该 Provider 的 Model 实现并获取价格
        3. 如果找不到则使用系统默认价格
        """

        service = ModelCostService(db)
        return await service.get_model_price_async(provider, model)

    @classmethod
    def get_model_price(cls, db: Session, provider: str, model: str) -> tuple[float, float]:
        """获取模型价格（输入价格，输出价格）每1M tokens

        查找逻辑：
        1. 直接通过 GlobalModel.name 匹配
        2. 查找该 Provider 的 Model 实现并获取价格
        3. 如果找不到则使用系统默认价格
        """

        service = ModelCostService(db)
        return service.get_model_price(provider, model)

    @classmethod
    async def get_cache_prices_async(
        cls, db: Session, provider: str, model: str, input_price: float
    ) -> tuple[float | None, float | None]:
        """异步获取模型缓存价格（缓存创建价格，缓存读取价格）每1M tokens"""
        service = ModelCostService(db)
        return await service.get_cache_prices_async(provider, model, input_price)

    @classmethod
    def get_cache_prices(
        cls, db: Session, provider: str, model: str, input_price: float
    ) -> tuple[float | None, float | None]:
        """获取模型缓存价格（缓存创建价格，缓存读取价格）每1M tokens"""
        service = ModelCostService(db)
        return service.get_cache_prices(provider, model, input_price)

    @classmethod
    async def get_request_price_async(cls, db: Session, provider: str, model: str) -> float | None:
        """异步获取模型按次计费价格"""
        service = ModelCostService(db)
        return await service.get_request_price_async(provider, model)

    @classmethod
    def get_request_price(cls, db: Session, provider: str, model: str) -> float | None:
        """获取模型按次计费价格"""
        service = ModelCostService(db)
        return service.get_request_price(provider, model)

    @staticmethod
    def calculate_cost(
        input_tokens: int,
        output_tokens: int,
        input_price_per_1m: float,
        output_price_per_1m: float,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_price_per_1m: float | None = None,
        cache_read_price_per_1m: float | None = None,
        price_per_request: float | None = None,
    ) -> tuple[float, float, float, float, float, float, float]:
        """计算成本（价格是每百万tokens）- 固定价格模式

        Returns:
            Tuple of (input_cost, output_cost, cache_creation_cost,
                     cache_read_cost, cache_cost, request_cost, total_cost)
        """
        return ModelCostService.compute_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_price_per_1m=input_price_per_1m,
            output_price_per_1m=output_price_per_1m,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_price_per_1m=cache_creation_price_per_1m,
            cache_read_price_per_1m=cache_read_price_per_1m,
            price_per_request=price_per_request,
        )

    @classmethod
    async def calculate_cost_with_strategy_async(
        cls,
        db: Session,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        api_format: str | None = None,
        cache_ttl_minutes: int | None = None,
    ) -> tuple[float, float, float, float, float, float, float, int | None]:
        """使用策略模式计算成本（支持阶梯计费）

        根据 api_format 选择对应的计费策略，支持阶梯计费和 TTL 差异化。

        Returns:
            Tuple of (input_cost, output_cost, cache_creation_cost,
                     cache_read_cost, cache_cost, request_cost, total_cost, tier_index)
        """
        service = ModelCostService(db)
        return await service.compute_cost_with_strategy_async(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            api_format=api_format,
            cache_ttl_minutes=cache_ttl_minutes,
        )

    @classmethod
    async def _get_rate_multiplier_and_free_tier(
        cls,
        db: Session,
        provider_api_key_id: str | None,
        provider_id: str | None,
        api_format: str | None = None,
    ) -> tuple[float, bool]:
        """获取费率倍数和是否免费套餐（使用缓存）"""
        from src.services.cache.provider_cache import ProviderCacheService

        return await ProviderCacheService.get_rate_multiplier_and_free_tier(
            db, provider_api_key_id, provider_id, api_format
        )

    @staticmethod
    def _get_rate_multiplier_sync(
        db: Session,
        provider_api_key_id: str,
        api_format: str | None = None,
    ) -> float | None:
        """
        同步获取 ProviderAPIKey 的 rate_multiplier

        Args:
            db: 数据库会话
            provider_api_key_id: ProviderAPIKey ID
            api_format: API 格式（可选），如 "CLAUDE"、"OPENAI"

        Returns:
            rate_multiplier 或 None
        """
        from src.services.cache.provider_cache import ProviderCacheService

        provider_key = (
            db.query(ProviderAPIKey.rate_multipliers)
            .filter(ProviderAPIKey.id == provider_api_key_id)
            .first()
        )

        if not provider_key:
            return None

        return ProviderCacheService.compute_rate_multiplier(
            provider_key.rate_multipliers, api_format
        )
