"""
OpenAI -> Claude 格式转换器

将 OpenAI Chat Completions API 格式转换为 Claude Messages API 格式。
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from src.core.api_format.conversion.state import StreamConversionState


class OpenAIToClaudeConverter:
    """
    OpenAI -> Claude 格式转换器

    支持：
    - 请求转换：OpenAI Chat Request -> Claude Request
    - 响应转换：OpenAI Chat Response -> Claude Response
    - 流式转换：OpenAI SSE -> Claude SSE
    """

    # 内容类型常量
    CONTENT_TYPE_TEXT = "text"
    CONTENT_TYPE_IMAGE = "image"
    CONTENT_TYPE_TOOL_USE = "tool_use"
    CONTENT_TYPE_TOOL_RESULT = "tool_result"

    # 停止原因映射（OpenAI -> Claude）
    FINISH_REASON_MAP = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "end_turn",
    }

    def __init__(self, model_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            model_mapping: OpenAI 模型到 Claude 模型的映射
        """
        self._model_mapping = model_mapping or {}

    # ==================== 请求转换 ====================

    def convert_request(self, request: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        将 OpenAI 请求转换为 Claude 格式

        Args:
            request: OpenAI 请求（Dict 或 Pydantic 模型）

        Returns:
            Claude 格式的请求字典
        """
        if hasattr(request, "model_dump"):
            data = request.model_dump(exclude_none=True)
        else:
            data = dict(request)

        # 模型映射
        model = data.get("model", "")
        claude_model = self._model_mapping.get(model, model)

        # 处理消息
        system_content: Optional[str] = None
        claude_messages: List[Dict[str, Any]] = []

        for message in data.get("messages", []):
            role = message.get("role")

            # 提取 system 消息
            if role == "system":
                system_content = self._collapse_content(message.get("content"))
                continue

            # 转换其他消息
            converted = self._convert_message(message)
            if converted:
                claude_messages.append(converted)

        # 构建 Claude 请求
        result: Dict[str, Any] = {
            "model": claude_model,
            "messages": claude_messages,
            "max_tokens": data.get("max_tokens") or 4096,
        }

        # 可选参数
        if data.get("temperature") is not None:
            result["temperature"] = data["temperature"]
        if data.get("top_p") is not None:
            result["top_p"] = data["top_p"]
        if data.get("stream"):
            result["stream"] = data["stream"]
        if data.get("stop"):
            result["stop_sequences"] = self._convert_stop(data["stop"])
        if system_content:
            result["system"] = system_content

        # 工具转换
        tools = self._convert_tools(data.get("tools"))
        if tools:
            result["tools"] = tools

        tool_choice = self._convert_tool_choice(data.get("tool_choice"))
        if tool_choice:
            result["tool_choice"] = tool_choice

        return result

    def _convert_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """转换单条消息"""
        role = message.get("role")

        if role == "user":
            return self._convert_user_message(message)
        if role == "assistant":
            return self._convert_assistant_message(message)
        if role == "tool":
            return self._convert_tool_message(message)

        return None

    def _convert_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换用户消息"""
        content = message.get("content")

        if isinstance(content, str) or content is None:
            return {"role": "user", "content": content or ""}

        # 转换内容数组
        claude_content: List[Dict[str, Any]] = []
        for item in content:
            item_type = item.get("type")

            if item_type == "text":
                claude_content.append(
                    {"type": self.CONTENT_TYPE_TEXT, "text": item.get("text", "")}
                )
            elif item_type == "image_url":
                image_url = (item.get("image_url") or {}).get("url", "")
                claude_content.append(self._convert_image_url(image_url))

        return {"role": "user", "content": claude_content}

    def _convert_assistant_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换助手消息"""
        content_blocks: List[Dict[str, Any]] = []

        # 处理文本内容
        content = message.get("content")
        if isinstance(content, str):
            content_blocks.append({"type": self.CONTENT_TYPE_TEXT, "text": content})
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    content_blocks.append(
                        {"type": self.CONTENT_TYPE_TEXT, "text": part.get("text", "")}
                    )

        # 处理工具调用
        for tool_call in message.get("tool_calls") or []:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                try:
                    input_data = json.loads(arguments)
                except json.JSONDecodeError:
                    input_data = {"raw": arguments}

                content_blocks.append(
                    {
                        "type": self.CONTENT_TYPE_TOOL_USE,
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": input_data,
                    }
                )

        # 简化单文本内容
        if not content_blocks:
            return {"role": "assistant", "content": ""}
        if len(content_blocks) == 1 and content_blocks[0]["type"] == self.CONTENT_TYPE_TEXT:
            return {"role": "assistant", "content": content_blocks[0]["text"]}

        return {"role": "assistant", "content": content_blocks}

    def _convert_tool_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换工具结果消息"""
        tool_content = message.get("content", "")

        # 尝试解析 JSON
        parsed_content = tool_content
        if isinstance(tool_content, str):
            try:
                parsed_content = json.loads(tool_content)
            except json.JSONDecodeError:
                pass

        tool_block = {
            "type": self.CONTENT_TYPE_TOOL_RESULT,
            "tool_use_id": message.get("tool_call_id", ""),
            "content": parsed_content,
        }

        return {"role": "user", "content": [tool_block]}

    def _convert_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """转换工具定义"""
        if not tools:
            return None

        result: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                continue

            function = tool.get("function", {})
            result.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description"),
                    "input_schema": function.get("parameters") or {},
                }
            )

        return result if result else None

    def _convert_tool_choice(
        self, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        """转换工具选择"""
        if tool_choice is None:
            return None
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function", {})
            return {"type": "tool_use", "name": function.get("name", "")}

        return {"type": "auto"}

    def _convert_image_url(self, image_url: str) -> Dict[str, Any]:
        """转换图片 URL"""
        if image_url.startswith("data:"):
            header, _, data = image_url.partition(",")
            media_type = "image/jpeg"
            if ";" in header:
                media_type = header.split(";")[0].split(":")[-1]

            return {
                "type": self.CONTENT_TYPE_IMAGE,
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        return {"type": self.CONTENT_TYPE_TEXT, "text": f"[Image: {image_url}]"}

    def _convert_stop(self, stop: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """转换停止序列"""
        if stop is None:
            return None
        if isinstance(stop, str):
            return [stop]
        return stop

    # ==================== 响应转换 ====================

    def convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 OpenAI 响应转换为 Claude 格式

        Args:
            response: OpenAI 响应字典

        Returns:
            Claude 格式的响应字典
        """
        choices = response.get("choices", [])
        if not choices:
            return self._empty_claude_response(response)

        choice = choices[0]
        message = choice.get("message", {})

        # 构建 content 数组
        content: List[Dict[str, Any]] = []

        # 处理文本
        text_content = message.get("content")
        if text_content:
            content.append(
                {
                    "type": self.CONTENT_TYPE_TEXT,
                    "text": text_content,
                }
            )

        # 处理工具调用
        for tool_call in message.get("tool_calls") or []:
            if tool_call.get("type") == "function":
                function = tool_call.get("function", {})
                arguments = function.get("arguments", "{}")
                try:
                    input_data = json.loads(arguments)
                except json.JSONDecodeError:
                    input_data = {"raw": arguments}

                content.append(
                    {
                        "type": self.CONTENT_TYPE_TOOL_USE,
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": input_data,
                    }
                )

        # 转换 finish_reason
        finish_reason = choice.get("finish_reason")
        stop_reason = self.FINISH_REASON_MAP.get(finish_reason, "end_turn")

        # 转换 usage
        usage = response.get("usage", {})
        claude_usage = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        return {
            "id": f"msg_{response.get('id', uuid.uuid4().hex[:8])}",
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": claude_usage,
        }

    def _empty_claude_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """构建空的 Claude 响应"""
        return {
            "id": f"msg_{response.get('id', uuid.uuid4().hex[:8])}",
            "type": "message",
            "role": "assistant",
            "model": response.get("model", ""),
            "content": [],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["StreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 OpenAI SSE chunk 转换为 Claude SSE 事件

        Args:
            chunk: OpenAI SSE chunk
            state: 流式转换状态

        Returns:
            Claude SSE 事件列表
        """
        from src.core.api_format.conversion.state import StreamConversionState

        if state is None:
            state = StreamConversionState()

        events: List[Dict[str, Any]] = []

        choices = chunk.get("choices") or []
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        # 处理角色（第一个 chunk）
        role = delta.get("role")
        if role and not state.message_started:
            msg_id = state.message_id or f"msg_{uuid.uuid4().hex[:8]}"
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": role,
                        "model": state.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                    },
                }
            )
            state.message_started = True

        # 处理文本内容
        content_delta = delta.get("content")
        if isinstance(content_delta, str):
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content_delta},
                }
            )

        # 处理工具调用
        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            index = tool_call.get("index", 0)

            # 工具调用开始
            if "id" in tool_call:
                function = tool_call.get("function", {})
                events.append(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": self.CONTENT_TYPE_TOOL_USE,
                            "id": tool_call["id"],
                            "name": function.get("name", ""),
                        },
                    }
                )

            # 工具调用参数增量
            function = tool_call.get("function", {})
            if "arguments" in function:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": function.get("arguments", ""),
                        },
                    }
                )

        # 处理结束
        if finish_reason:
            stop_reason = self.FINISH_REASON_MAP.get(finish_reason, "end_turn")
            events.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                }
            )

        return events

    # ==================== 工具方法 ====================

    def _collapse_content(
        self, content: Optional[Union[str, List[Dict[str, Any]]]]
    ) -> Optional[str]:
        """折叠内容为字符串"""
        if isinstance(content, str):
            return content
        if not content:
            return None

        text_parts = [part.get("text", "") for part in content if part.get("type") == "text"]
        return "\n\n".join(filter(None, text_parts)) or None


__all__ = ["OpenAIToClaudeConverter"]
