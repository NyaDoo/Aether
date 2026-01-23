"""
流式转换状态类

用于在多个 chunk 之间维护转换上下文，例如：
- 是否已发送 message_start 事件
- 累积的文本（用于计算增量）
- 当前内容块索引
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StreamConversionState:
    """
    SSE 流式转换状态（Claude <-> OpenAI）

    用于跨 chunk 维护转换状态，确保生成正确的事件序列。
    """

    message_id: str = ""
    model: str = ""
    message_started: bool = False
    content_block_started: bool = False
    current_tool_index: int = 0

    def reset(self) -> None:
        """重置状态（重试时调用）"""
        self.message_started = False
        self.content_block_started = False
        self.current_tool_index = 0


@dataclass
class GeminiStreamConversionState:
    """
    Gemini 流式转换状态（JSON 数组格式）

    与 Claude/OpenAI 的 SSE 不同，Gemini 需要追踪额外状态：
    1. 累积文本（用于计算真正的增量）
    2. 内容块索引（用于工具调用）
    3. 是否已发送 message_start
    """

    message_id: str = ""
    model: str = ""
    accumulated_text: str = ""  # 累积的文本（用于计算增量）
    message_started: bool = False
    content_block_started: bool = False
    current_block_index: int = 0
    tool_call_index: int = 0  # 工具调用计数
    has_sent_usage: bool = False  # 是否已发送 usage

    def reset(self) -> None:
        """重置状态（重试时调用）"""
        self.accumulated_text = ""
        self.message_started = False
        self.content_block_started = False
        self.current_block_index = 0
        self.tool_call_index = 0
        self.has_sent_usage = False


@dataclass
class ClaudeStreamConversionState:
    """
    Claude -> Gemini 流式转换状态

    用于将 Claude SSE 事件流转换为 Gemini JSON 流式响应
    """

    message_id: str = ""
    model: str = ""
    current_block_type: str = ""  # 当前内容块类型（text/tool_use）
    current_block_index: int = 0
    current_tool_name: str = ""
    current_tool_id: str = ""
    accumulated_tool_input: str = ""  # 累积的工具输入 JSON

    def reset(self) -> None:
        """重置状态（重试时调用）"""
        self.current_block_type = ""
        self.current_block_index = 0
        self.current_tool_name = ""
        self.current_tool_id = ""
        self.accumulated_tool_input = ""


@dataclass
class OpenAIStreamConversionState:
    """
    OpenAI -> Gemini 流式转换状态

    用于将 OpenAI SSE 事件流转换为 Gemini JSON 流式响应
    """

    model: str = ""
    current_tool_name: str = ""
    accumulated_tool_args: str = ""  # 累积的工具参数 JSON

    def reset(self) -> None:
        """重置状态（重试时调用）"""
        self.current_tool_name = ""
        self.accumulated_tool_args = ""


__all__ = [
    "StreamConversionState",
    "GeminiStreamConversionState",
    "ClaudeStreamConversionState",
    "OpenAIStreamConversionState",
]
