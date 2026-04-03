"""
API 格式转换子模块（Canonical）

对外提供：
- `format_conversion_registry`: 全局转换注册表（Hub-and-Spoke）
- `register_default_normalizers()`: 注册默认 Normalizers（OPENAI/CLAUDE/GEMINI）
- `StreamState`: 统一流式状态容器
"""

from src.core.api_format.conversion.compatibility import is_format_compatible
from src.core.api_format.conversion.exceptions import FormatConversionError
from src.core.api_format.conversion.registry import (
    FormatConversionRegistry,
    format_conversion_registry,
    register_default_normalizers,
)
from src.core.api_format.conversion.stream_state import StreamState

__all__ = [
    # Registry
    "FormatConversionRegistry",
    "format_conversion_registry",
    "register_default_normalizers",
    # Stream state
    "StreamState",
    # Exceptions
    "FormatConversionError",
    # Compatibility
    "is_format_compatible",
]
