"""
格式转换核心模块

该目录用于承载与「API 格式转换」相关的核心能力（不依赖 FastAPI/Handler 层），
以便 services/core 可复用并避免出现 services -> api 的反向依赖。

模块组成：
- registry.py: 转换器注册表，管理转换器实例和能力查询
- protocols.py: 转换器协议定义（Protocol）
- state.py: 流式转换状态类
- exceptions.py: 转换异常定义
- compatibility.py: 格式兼容性检查函数
- converters/: 内置格式转换器实现
"""

from src.core.api_format.conversion.compatibility import is_format_compatible
from src.core.api_format.conversion.converters import (
    ClaudeToGeminiConverter,
    ClaudeToOpenAIConverter,
    GeminiToClaudeConverter,
    GeminiToOpenAIConverter,
    OpenAIToClaudeConverter,
    OpenAIToGeminiConverter,
)
from src.core.api_format.conversion.exceptions import FormatConversionError
from src.core.api_format.conversion.protocols import (
    RequestConverter,
    ResponseConverter,
    StreamChunkConverter,
)
from src.core.api_format.conversion.registry import (
    FormatConverterRegistry,
    converter_registry,
)
from src.core.api_format.conversion.state import (
    ClaudeStreamConversionState,
    GeminiStreamConversionState,
    OpenAIStreamConversionState,
    StreamConversionState,
)
from src.core.logger import logger


def register_all_converters() -> None:
    """
    注册所有内置的格式转换器

    在应用启动时调用此函数
    """
    # Claude <-> OpenAI
    converter_registry.register("OPENAI", "CLAUDE", OpenAIToClaudeConverter())
    converter_registry.register("CLAUDE", "OPENAI", ClaudeToOpenAIConverter())

    # Claude <-> Gemini
    converter_registry.register("CLAUDE", "GEMINI", ClaudeToGeminiConverter())
    converter_registry.register("GEMINI", "CLAUDE", GeminiToClaudeConverter())

    # OpenAI <-> Gemini
    converter_registry.register("OPENAI", "GEMINI", OpenAIToGeminiConverter())
    converter_registry.register("GEMINI", "OPENAI", GeminiToOpenAIConverter())

    logger.info(f"[ConverterRegistry] 已注册 {len(converter_registry.list_converters())} 个格式转换器")


__all__ = [
    # Registry
    "FormatConverterRegistry",
    "converter_registry",
    "register_all_converters",
    # Protocols
    "RequestConverter",
    "ResponseConverter",
    "StreamChunkConverter",
    # State
    "StreamConversionState",
    "GeminiStreamConversionState",
    "ClaudeStreamConversionState",
    "OpenAIStreamConversionState",
    # Exceptions
    "FormatConversionError",
    # Compatibility
    "is_format_compatible",
    # Converters
    "OpenAIToClaudeConverter",
    "ClaudeToOpenAIConverter",
    "ClaudeToGeminiConverter",
    "GeminiToClaudeConverter",
    "OpenAIToGeminiConverter",
    "GeminiToOpenAIConverter",
]
