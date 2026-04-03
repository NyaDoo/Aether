"""
格式兼容性检查

用于候选筛选时判断端点是否可以处理客户端请求格式。

三层开关优先级（从高到低）：
1. 全局开关 ON → 强制允许（跳过后续检查）
2. 全局开关 OFF → 看提供商开关
   - 提供商开关 ON → 强制允许（跳过端点检查）
   - 提供商开关 OFF → 看端点配置
3. 端点配置（format_acceptance_config）
   - enabled=true + 白名单/黑名单检查 → 允许
   - enabled=false 或未配置 → 禁止

转换逻辑：
1. 格式完全匹配 -> 透传（无需转换）
2. data_format_id 相同 -> 透传（无需数据转换，如 claude:chat / claude:cli）
3. 格式不同且 data_format_id 不同 -> 需要检查三层开关
   - 通过开关检查后，检查转换器能力
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.api_format.conversion.registry import FormatConversionRegistry

from src.core.api_format.metadata import can_passthrough_endpoint

logger = logging.getLogger(__name__)


def is_format_compatible(
    client_format: str,
    endpoint_api_format: str,
    endpoint_format_acceptance_config: dict | None,
    is_stream: bool,
    effective_conversion_enabled: bool,
    registry: FormatConversionRegistry | None = None,
    *,
    skip_endpoint_check: bool = False,
) -> tuple[bool, bool, str | None]:
    """
    检查端点是否兼容客户端格式

    Args:
        client_format: 客户端请求格式
        endpoint_api_format: 端点的 API 格式
        endpoint_format_acceptance_config: 端点的格式接受配置
        is_stream: 是否是流式请求
        effective_conversion_enabled: 格式转换总开关（通常来自环境变量/Feature Flag）
        registry: 转换器注册表（可选，默认使用全局单例）
        skip_endpoint_check: 是否跳过端点配置检查（当全局或提供商开关为 ON 时设为 True）

    Returns:
        (is_compatible, needs_conversion, skip_reason)
        - is_compatible: 是否兼容
        - needs_conversion: 是否需要转换
        - skip_reason: 不兼容时的原因
    """
    # 延迟导入避免循环依赖
    if registry is None:
        from src.core.api_format.conversion.registry import (
            format_conversion_registry,
            register_default_normalizers,
        )

        register_default_normalizers()
        registry = format_conversion_registry

    # 统一大写用于比较和 registry 查找（registry 以大写 key 索引 normalizer）
    client_key = client_format.upper()
    provider_key = endpoint_api_format.upper()

    # 1. 格式完全匹配 -> 透传（无需转换）
    if provider_key == client_key:
        return True, False, None

    # 2. data_format_id 相同 -> 透传（无需数据转换，也无需格式转换开关）
    # 例如：claude:chat / claude:cli 的 data_format_id 都是 “claude”，只是认证方式不同
    if can_passthrough_endpoint(client_key, provider_key):
        return True, False, None

    # 3. 格式不同且 data_format_id 不同 -> 需要检查格式转换开关（分层开关）
    #
    # 设计语义（与模块顶部注释一致）：
    # - 全局开关 ON  -> 强制允许跨格式（通常 caller 会传 skip_endpoint_check=True）
    # - 全局开关 OFF -> 不再”一刀切”拒绝，而是回退到 provider/endpoint 开关：
    #   - provider 开关 ON  -> 强制允许（skip_endpoint_check=True，跳过端点检查）
    #   - provider 开关 OFF -> 由端点 format_acceptance_config 决定（skip_endpoint_check=False）
    #
    # 说明：
    # - effective_conversion_enabled 表示”全局默认允许”，不是”全局总闸/kill switch”
    # - 当它为 False 时，我们仍然会继续执行后续检查（provider/endpoint），
    #   兼容”按 Provider/Endpoint 精细化开启转换”的场景。

    # 4. 如果全局或提供商开关为 ON，跳过端点配置检查
    if not skip_endpoint_check:
        # 检查端点配置（第三层开关）
        if endpoint_format_acceptance_config is None:
            return False, False, "端点未配置格式接受策略"

        config = endpoint_format_acceptance_config
        if not isinstance(config, dict):
            return False, False, "端点格式配置无效"
        if not config.get("enabled", False):
            return False, False, "端点格式接受未启用"

        # 检查 reject_formats（优先）
        reject_formats = config.get("reject_formats", [])
        if client_key in [f.upper() for f in reject_formats]:
            return False, False, f"端点拒绝 {client_format} 格式"

        # 检查 accept_formats
        accept_formats = config.get("accept_formats", [])
        if accept_formats and client_key not in [f.upper() for f in accept_formats]:
            return False, False, f"端点不接受 {client_format} 格式"

        # 检查流式转换
        if is_stream and not config.get("stream_conversion", True):
            return False, False, "端点不支持流式格式转换"

    # 5. 需要数据转换的情况（data_format_id 不同）
    # 检查转换器能力
    if not registry.can_convert_full(
        client_key,
        provider_key,
        require_stream=is_stream,
    ):
        return False, False, f"不存在 {client_format} <-> {endpoint_api_format} 的完整转换器"

    return True, True, None


__all__ = [
    "is_format_compatible",
]
