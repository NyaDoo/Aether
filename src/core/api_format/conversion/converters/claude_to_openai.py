"""
Claude -> OpenAI 格式转换器

将 Claude Messages API 格式转换为 OpenAI Chat Completions API 格式。
"""

from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from src.core.api_format.conversion.state import StreamConversionState


class ClaudeToOpenAIConverter:
    """
    Claude -> OpenAI 格式转换器

    支持：
    - 请求转换：Claude Request -> OpenAI Chat Request
    - 响应转换：Claude Response -> OpenAI Chat Response
    - 流式转换：Claude SSE -> OpenAI SSE
    """

    # 内容类型常量
    CONTENT_TYPE_TEXT = "text"
    CONTENT_TYPE_IMAGE = "image"
    CONTENT_TYPE_TOOL_USE = "tool_use"
    CONTENT_TYPE_TOOL_RESULT = "tool_result"

    # 停止原因映射
    STOP_REASON_MAP = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }

    def __init__(self, model_mapping: Optional[Dict[str, str]] = None):
        """
        Args:
            model_mapping: Claude 模型到 OpenAI 模型的映射
        """
        self._model_mapping = model_mapping or {}

    # ==================== 请求转换 ====================

    def convert_request(self, request: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        将 Claude 请求转换为 OpenAI 格式

        Args:
            request: Claude 请求（Dict 或 Pydantic 模型）

        Returns:
            OpenAI 格式的请求字典
        """
        if hasattr(request, "model_dump"):
            data = request.model_dump(exclude_none=True)
        else:
            data = dict(request)

        # 模型映射
        model = data.get("model", "")
        openai_model = self._model_mapping.get(model, model)

        # 构建消息列表
        messages: List[Dict[str, Any]] = []

        # 处理 system 消息
        system_content = self._extract_text_content(data.get("system"))
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # 处理对话消息
        for message in data.get("messages", []):
            converted = self._convert_message(message)
            if converted:
                messages.append(converted)

        # 构建 OpenAI 请求
        result: Dict[str, Any] = {
            "model": openai_model,
            "messages": messages,
        }

        # 可选参数
        if data.get("max_tokens"):
            result["max_tokens"] = data["max_tokens"]
        if data.get("temperature") is not None:
            result["temperature"] = data["temperature"]
        if data.get("top_p") is not None:
            result["top_p"] = data["top_p"]
        if data.get("stream"):
            result["stream"] = data["stream"]
        if data.get("stop_sequences"):
            result["stop"] = data["stop_sequences"]

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

        return None

    def _convert_user_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换用户消息"""
        content = message.get("content")

        if isinstance(content, str):
            return {"role": "user", "content": content}

        openai_content: List[Dict[str, Any]] = []
        for block in content or []:
            block_type = block.get("type")

            if block_type == self.CONTENT_TYPE_TEXT:
                openai_content.append({"type": "text", "text": block.get("text", "")})
            elif block_type == self.CONTENT_TYPE_IMAGE:
                source = block.get("source", {})
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                openai_content.append(
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
                )
            elif block_type == self.CONTENT_TYPE_TOOL_RESULT:
                tool_content = block.get("content", "")
                rendered = self._render_tool_content(tool_content)
                openai_content.append({"type": "text", "text": f"Tool result: {rendered}"})

        # 简化单文本内容
        if len(openai_content) == 1 and openai_content[0]["type"] == "text":
            return {"role": "user", "content": openai_content[0]["text"]}

        return {"role": "user", "content": openai_content or ""}

    def _convert_assistant_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """转换助手消息"""
        content = message.get("content")
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content, str):
            text_parts.append(content)
        else:
            for idx, block in enumerate(content or []):
                block_type = block.get("type")

                if block_type == self.CONTENT_TYPE_TEXT:
                    text_parts.append(block.get("text", ""))
                elif block_type == self.CONTENT_TYPE_TOOL_USE:
                    tool_calls.append(
                        {
                            "id": block.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                            },
                        }
                    )

        result: Dict[str, Any] = {"role": "assistant"}

        message_content = "\n".join([p for p in text_parts if p]) or None
        if message_content:
            result["content"] = message_content

        if tool_calls:
            result["tool_calls"] = tool_calls

        return result

    def _convert_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """转换工具定义"""
        if not tools:
            return None

        result: List[Dict[str, Any]] = []
        for tool in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description"),
                        "parameters": tool.get("input_schema", {}),
                    },
                }
            )
        return result

    def _convert_tool_choice(
        self, tool_choice: Optional[Dict[str, Any]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        """转换工具选择"""
        if tool_choice is None:
            return None

        choice_type = tool_choice.get("type")
        if choice_type in ("tool", "tool_use"):
            return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
        if choice_type == "any":
            return "required"
        if choice_type == "auto":
            return "auto"

        return tool_choice

    # ==================== 响应转换 ====================

    def convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Claude 响应转换为 OpenAI 格式

        Args:
            response: Claude 响应字典

        Returns:
            OpenAI 格式的响应字典
        """
        # 提取内容
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for idx, block in enumerate(response.get("content", [])):
            block_type = block.get("type")

            if block_type == self.CONTENT_TYPE_TEXT:
                content_parts.append(block.get("text", ""))
            elif block_type == self.CONTENT_TYPE_TOOL_USE:
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        },
                    }
                )

        # 构建消息
        message: Dict[str, Any] = {"role": "assistant"}
        text_content = "\n".join([p for p in content_parts if p]) or None
        if text_content:
            message["content"] = text_content
        if tool_calls:
            message["tool_calls"] = tool_calls

        # 转换停止原因
        stop_reason = response.get("stop_reason")
        finish_reason = self.STOP_REASON_MAP.get(stop_reason, stop_reason) if stop_reason else None

        # 转换 usage
        usage = response.get("usage", {})
        openai_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
        }

        return {
            "id": f"chatcmpl-{response.get('id', uuid.uuid4().hex[:8])}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": openai_usage,
        }

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["StreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 Claude SSE 事件转换为 OpenAI 格式

        Args:
            chunk: Claude SSE 事件
            state: 流式转换状态

        Returns:
            OpenAI 格式的 SSE chunk 列表
        """
        from src.core.api_format.conversion.state import StreamConversionState

        if state is None:
            state = StreamConversionState()

        result = self._convert_single_event(chunk, state.model, state.message_id)
        if result is None:
            return []
        return [result]

    def _convert_single_event(
        self,
        event: Dict[str, Any],
        model: str = "",
        message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        转换单个 Claude SSE 事件为 OpenAI 格式

        Args:
            event: Claude SSE 事件
            model: 模型名称
            message_id: 消息 ID

        Returns:
            OpenAI 格式的 SSE chunk，如果无法转换返回 None
        """
        event_type = event.get("type")
        chunk_id = f"chatcmpl-{(message_id or 'stream')[-8:]}"

        if event_type == "message_start":
            message = event.get("message", {})
            return self._base_chunk(
                chunk_id,
                model or message.get("model", ""),
                {"role": "assistant"},
            )

        if event_type == "content_block_start":
            content_block = event.get("content_block", {})
            if content_block.get("type") == self.CONTENT_TYPE_TOOL_USE:
                delta = {
                    "tool_calls": [
                        {
                            "index": event.get("index", 0),
                            "id": content_block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": content_block.get("name", ""),
                                "arguments": "",
                            },
                        }
                    ]
                }
                return self._base_chunk(chunk_id, model, delta)
            return None

        if event_type == "content_block_delta":
            delta_payload = event.get("delta") or {}
            delta_type = delta_payload.get("type")

            if delta_type == "text_delta":
                delta = {"content": delta_payload.get("text", "")}
                return self._base_chunk(chunk_id, model, delta)

            if delta_type == "input_json_delta":
                delta = {
                    "tool_calls": [
                        {
                            "index": event.get("index", 0),
                            "function": {"arguments": delta_payload.get("partial_json", "")},
                        }
                    ]
                }
                return self._base_chunk(chunk_id, model, delta)
            return None

        if event_type == "message_delta":
            delta = event.get("delta") or {}
            stop_reason = delta.get("stop_reason")
            finish_reason = self.STOP_REASON_MAP.get(stop_reason, stop_reason)
            return self._base_chunk(chunk_id, model, {}, finish_reason=finish_reason)

        if event_type == "message_stop":
            return self._base_chunk(chunk_id, model, {}, finish_reason="stop")

        return None

    def _base_chunk(
        self,
        chunk_id: str,
        model: str,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建基础 OpenAI chunk"""
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    # ==================== 工具方法 ====================

    def _extract_text_content(
        self, content: Optional[Union[str, List[Dict[str, Any]]]]
    ) -> Optional[str]:
        """提取文本内容"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if block.get("type") == self.CONTENT_TYPE_TEXT
            ]
            return "\n\n".join(filter(None, parts)) or None
        return None

    def _render_tool_content(self, tool_content: Any) -> str:
        """渲染工具内容"""
        if isinstance(tool_content, list):
            return json.dumps(tool_content, ensure_ascii=False)
        return str(tool_content)


__all__ = ["ClaudeToOpenAIConverter"]
