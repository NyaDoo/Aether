"""
视频/图像相关的纯工具函数。

从 api/handlers 层下沉到 core 层，消除 services→api 的反向依赖。
"""

from __future__ import annotations

import re

# 敏感信息匹配正则（预编译提升性能）
_SENSITIVE_PATTERN = re.compile(
    r"(api[_-]?key|token|bearer|authorization)[=:\s]+\S+",
    re.IGNORECASE,
)


def sanitize_error_message(message: str, max_length: int = 200) -> str:
    """
    移除错误消息中可能包含的敏感信息

    Args:
        message: 原始错误消息
        max_length: 最大长度，默认 200

    Returns:
        脱敏后的消息
    """
    if not message:
        return "Request failed"
    # 先脱敏再截断，确保敏感信息不会因截断位置而泄露
    sanitized = _SENSITIVE_PATTERN.sub("[REDACTED]", message)
    return sanitized[:max_length]


def extract_short_id_from_operation(operation_id: str) -> str:
    """
    从 operation ID 中提取短 ID

    我们对外暴露的 operation name 格式是:
    - models/{model}/operations/{short_id}

    此函数提取最后一部分作为 short_id，用于在数据库中查找任务。

    Args:
        operation_id: 原始 operation ID（如 "models/veo-3.1/operations/abc123"）

    Returns:
        short_id（如 "abc123"）
    """
    # 格式: models/{model}/operations/{short_id}
    # 或者直接是 short_id
    if "/" in operation_id:
        # 提取最后一部分
        return operation_id.rsplit("/", 1)[-1]
    return operation_id


def normalize_gemini_operation_id(operation_id: str) -> str:
    """
    规范化 Gemini operation ID（保留用于向后兼容）

    Args:
        operation_id: 原始 operation ID

    Returns:
        规范化后的 operation ID（原样返回）
    """
    return operation_id


def is_image_gen_model(model: str | None) -> bool:
    """判断是否为图像生成模型（模式匹配，覆盖 gemini-*-image / imagen-* 系列）"""
    if not model:
        return False
    m = model.lower()
    return "image" in m and ("gemini" in m or "imagen" in m)
