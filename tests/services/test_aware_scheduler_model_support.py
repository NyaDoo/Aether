"""
测试 aware_scheduler 的模型支持检查逻辑

场景：
- GlobalModel claude-haiku 配置了 model_aliases: ["haiku", "claude.*haiku.*"]
- Provider A：Model 表有记录关联到 claude-haiku
- Provider B：Model 表没有记录关联到 claude-haiku
- 用户请求 haiku 模型

期望：Provider B 应该被跳过，因为它不支持 haiku 模型
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.database import GlobalModel, Model, Provider
from src.services.cache.aware_scheduler import CacheAwareScheduler


class TestCheckModelSupportForGlobalModel:
    """测试 _check_model_support_for_global_model 方法"""

    @pytest.mark.asyncio
    async def test_provider_without_model_should_return_false(self):
        """Provider 没有配置对应的 Model 时应该返回 False"""
        scheduler = CacheAwareScheduler()

        # 创建 GlobalModel
        global_model = MagicMock(spec=GlobalModel)
        global_model.id = "gm-haiku"
        global_model.name = "claude-haiku"
        global_model.supported_capabilities = []

        # 创建 Provider（没有任何 Model）
        provider = MagicMock(spec=Provider)
        provider.name = "Provider-B"
        provider.models = []  # 空列表，没有任何模型

        # 创建 mock db session
        db = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.transient = False
        mock_inspect.detached = False

        with patch("sqlalchemy.inspect", return_value=mock_inspect):
            is_supported, skip_reason, caps, provider_model_names = (
                await scheduler._check_model_support_for_global_model(
                    db=db,
                    provider=provider,
                    global_model=global_model,
                    model_name="haiku",
                )
            )

        assert is_supported is False
        assert skip_reason == "Provider 未实现此模型"
        assert caps is None
        assert provider_model_names is None

    @pytest.mark.asyncio
    async def test_provider_with_different_model_should_return_false(self):
        """Provider 配置了其他模型但没有目标模型时应该返回 False"""
        scheduler = CacheAwareScheduler()

        # 创建 GlobalModel
        global_model = MagicMock(spec=GlobalModel)
        global_model.id = "gm-haiku"
        global_model.name = "claude-haiku"
        global_model.supported_capabilities = []

        # 创建另一个 GlobalModel
        other_global_model = MagicMock(spec=GlobalModel)
        other_global_model.id = "gm-sonnet"
        other_global_model.name = "claude-sonnet"

        # 创建 Model（关联到其他 GlobalModel）
        other_model = MagicMock(spec=Model)
        other_model.global_model_id = "gm-sonnet"  # 关联到 sonnet，不是 haiku
        other_model.is_active = True
        other_model.provider_model_name = "claude-3-sonnet-20240229"

        # 创建 Provider（只有 sonnet 模型，没有 haiku）
        provider = MagicMock(spec=Provider)
        provider.name = "Provider-B"
        provider.models = [other_model]

        # 创建 mock db session
        db = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.transient = False
        mock_inspect.detached = False

        with patch("sqlalchemy.inspect", return_value=mock_inspect):
            is_supported, skip_reason, caps, provider_model_names = (
                await scheduler._check_model_support_for_global_model(
                    db=db,
                    provider=provider,
                    global_model=global_model,
                    model_name="haiku",
                )
            )

        assert is_supported is False
        assert skip_reason == "Provider 未实现此模型"

    @pytest.mark.asyncio
    async def test_provider_with_matching_model_should_return_true(self):
        """Provider 配置了目标模型时应该返回 True"""
        scheduler = CacheAwareScheduler()

        # 创建 GlobalModel
        global_model = MagicMock(spec=GlobalModel)
        global_model.id = "gm-haiku"
        global_model.name = "claude-haiku"
        global_model.supported_capabilities = ["cache_1h"]

        # 创建 Model（关联到目标 GlobalModel）
        model = MagicMock(spec=Model)
        model.global_model_id = "gm-haiku"
        model.is_active = True
        model.provider_model_name = "claude-3-haiku-20240307"
        model.provider_model_mappings = None
        model.get_effective_supports_streaming = MagicMock(return_value=True)

        # 创建 Provider
        provider = MagicMock(spec=Provider)
        provider.name = "Provider-A"
        provider.models = [model]

        # 创建 mock db session
        db = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.transient = False
        mock_inspect.detached = False

        with patch("sqlalchemy.inspect", return_value=mock_inspect):
            is_supported, skip_reason, caps, provider_model_names = (
                await scheduler._check_model_support_for_global_model(
                    db=db,
                    provider=provider,
                    global_model=global_model,
                    model_name="haiku",
                )
            )

        assert is_supported is True
        assert skip_reason is None
        assert caps == ["cache_1h"]
        assert provider_model_names == {"claude-3-haiku-20240307"}

    @pytest.mark.asyncio
    async def test_provider_with_inactive_model_should_return_false(self):
        """Provider 的模型未激活时应该返回 False"""
        scheduler = CacheAwareScheduler()

        # 创建 GlobalModel
        global_model = MagicMock(spec=GlobalModel)
        global_model.id = "gm-haiku"
        global_model.name = "claude-haiku"
        global_model.supported_capabilities = []

        # 创建 Model（未激活）
        model = MagicMock(spec=Model)
        model.global_model_id = "gm-haiku"
        model.is_active = False  # 未激活
        model.provider_model_name = "claude-3-haiku-20240307"

        # 创建 Provider
        provider = MagicMock(spec=Provider)
        provider.name = "Provider-A"
        provider.models = [model]

        # 创建 mock db session
        db = MagicMock()
        mock_inspect = MagicMock()
        mock_inspect.transient = False
        mock_inspect.detached = False

        with patch("sqlalchemy.inspect", return_value=mock_inspect):
            is_supported, skip_reason, caps, provider_model_names = (
                await scheduler._check_model_support_for_global_model(
                    db=db,
                    provider=provider,
                    global_model=global_model,
                    model_name="haiku",
                )
            )

        assert is_supported is False
        assert skip_reason == "Provider 未实现此模型"
