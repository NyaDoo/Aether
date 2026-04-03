"""
Handler 基类模块

提供 Adapter、Handler 的抽象基类，以及请求构建器和响应解析器。

注意：Handler 基类（ChatHandlerBase, CliMessageHandlerBase 等）不在这里导出，
因为它们依赖 services.usage.stream，而后者又需要导入 response_parser，
会形成循环导入。请直接从具体模块导入 Handler 基类。
"""

# Chat Adapter 基类（不会引起循环导入）
from src.api.handlers.base.chat_adapter_base import (
    ChatAdapterBase,
    get_adapter_class,
    get_adapter_instance,
    list_registered_formats,
    register_adapter,
)

# CLI Adapter 基类
from src.api.handlers.base.cli_adapter_base import (
    CliAdapterBase,
    get_cli_adapter_class,
    get_cli_adapter_instance,
    list_registered_cli_formats,
    register_cli_adapter,
)

# 请求构建器
from src.api.handlers.base.request_builder import (
    SENSITIVE_HEADERS,
    PassthroughRequestBuilder,
    RequestBuilder,
    build_passthrough_request,
)

# 响应解析器
from src.api.handlers.base.response_parser import (
    ParsedChunk,
    ParsedResponse,
    ResponseParser,
    StreamStats,
)

__all__ = [
    # Chat Adapter
    "ChatAdapterBase",
    "register_adapter",
    "get_adapter_class",
    "get_adapter_instance",
    "list_registered_formats",
    # CLI Adapter
    "CliAdapterBase",
    "register_cli_adapter",
    "get_cli_adapter_class",
    "get_cli_adapter_instance",
    "list_registered_cli_formats",
    # 请求构建器
    "RequestBuilder",
    "PassthroughRequestBuilder",
    "build_passthrough_request",
    "SENSITIVE_HEADERS",
    # 响应解析器
    "ResponseParser",
    "ParsedChunk",
    "ParsedResponse",
    "StreamStats",
]
