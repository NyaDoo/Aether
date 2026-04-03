"""
API Handlers - 请求处理器

按 API 格式组织的 Adapter 和 Handler：
- Adapter: 请求验证、格式转换、错误处理
- Handler: 业务逻辑、调用 Provider、记录用量

支持的格式：
- claude: Claude Chat API (/v1/messages)
- claude_cli: Claude CLI 透传模式
- openai: OpenAI Chat API (/v1/chat/completions)
- openai_cli: OpenAI CLI 透传模式

注意：Handler 基类和具体 Handler 使用延迟导入以避免循环依赖。
"""

# Adapter 基类（不会引起循环导入，可以直接导入）
from src.api.handlers.base import (
    ChatAdapterBase,
    CliAdapterBase,
)

__all__ = [
    # Adapter 基类
    "ChatAdapterBase",
    "CliAdapterBase",
    # Handler 基类（延迟导入）
    "ChatHandlerBase",
    "CliMessageHandlerBase",
    "BaseMessageHandler",
    "MessageHandlerProtocol",
    "MessageTelemetry",
    "StreamContext",
    # Claude
    "ClaudeChatAdapter",
    "ClaudeTokenCountAdapter",
    "build_claude_adapter",
    "ClaudeChatHandler",
    # Claude CLI
    "ClaudeCliAdapter",
    "ClaudeCliMessageHandler",
    # OpenAI
    "OpenAIChatAdapter",
    "OpenAIChatHandler",
    # OpenAI CLI
    "OpenAICliAdapter",
    "OpenAICliMessageHandler",
]

# 延迟导入映射表
_LAZY_IMPORTS = {
    # Handler 基类
    "ChatHandlerBase": ("src.api.handlers.base.chat_handler_base", "ChatHandlerBase"),
    "CliMessageHandlerBase": (
        "src.api.handlers.base.cli_handler_base",
        "CliMessageHandlerBase",
    ),
    "StreamContext": ("src.api.handlers.base.cli_handler_base", "StreamContext"),
    "BaseMessageHandler": ("src.api.handlers.base.base_handler", "BaseMessageHandler"),
    "MessageHandlerProtocol": (
        "src.api.handlers.base.base_handler",
        "MessageHandlerProtocol",
    ),
    "MessageTelemetry": ("src.api.handlers.base.base_handler", "MessageTelemetry"),
    # Claude
    "ClaudeChatAdapter": ("src.api.handlers.claude.adapter", "ClaudeChatAdapter"),
    "ClaudeTokenCountAdapter": (
        "src.api.handlers.claude.adapter",
        "ClaudeTokenCountAdapter",
    ),
    "build_claude_adapter": ("src.api.handlers.claude.adapter", "build_claude_adapter"),
    "ClaudeChatHandler": ("src.api.handlers.claude.handler", "ClaudeChatHandler"),
    # Claude CLI
    "ClaudeCliAdapter": ("src.api.handlers.claude_cli.adapter", "ClaudeCliAdapter"),
    "ClaudeCliMessageHandler": (
        "src.api.handlers.claude_cli.handler",
        "ClaudeCliMessageHandler",
    ),
    # OpenAI
    "OpenAIChatAdapter": ("src.api.handlers.openai.adapter", "OpenAIChatAdapter"),
    "OpenAIChatHandler": ("src.api.handlers.openai.handler", "OpenAIChatHandler"),
    # OpenAI CLI
    "OpenAICliAdapter": ("src.api.handlers.openai_cli.adapter", "OpenAICliAdapter"),
    "OpenAICliMessageHandler": (
        "src.api.handlers.openai_cli.handler",
        "OpenAICliMessageHandler",
    ),
}


def __getattr__(name: str) -> None:
    """延迟导入以避免循环依赖"""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
