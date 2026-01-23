"""
Gemini 格式转换器

提供 Gemini 与其他 API 格式（Claude、OpenAI）之间的转换
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from src.core.api_format.conversion.state import (
        ClaudeStreamConversionState,
        GeminiStreamConversionState,
        OpenAIStreamConversionState,
    )


class ClaudeToGeminiConverter:
    """
    Claude -> Gemini 请求转换器

    将 Claude Messages API 格式转换为 Gemini generateContent 格式
    """

    def convert_request(self, claude_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Claude 请求转换为 Gemini 请求

        Args:
            claude_request: Claude 格式的请求字典

        Returns:
            Gemini 格式的请求字典
        """
        gemini_request: Dict[str, Any] = {
            "contents": self._convert_messages(claude_request.get("messages", [])),
        }

        # 转换 system prompt
        system = claude_request.get("system")
        if system:
            gemini_request["system_instruction"] = self._convert_system(system)

        # 转换生成配置
        generation_config = self._build_generation_config(claude_request)
        if generation_config:
            gemini_request["generation_config"] = generation_config

        # 转换工具
        tools = claude_request.get("tools")
        if tools:
            gemini_request["tools"] = self._convert_tools(tools)

        return gemini_request

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息列表"""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            # Gemini 使用 "model" 而不是 "assistant"
            gemini_role = "model" if role == "assistant" else "user"

            content = msg.get("content", "")
            parts = self._convert_content_to_parts(content)

            contents.append(
                {
                    "role": gemini_role,
                    "parts": parts,
                }
            )
        return contents

    def _convert_content_to_parts(self, content: Any) -> List[Dict[str, Any]]:
        """将 Claude 内容转换为 Gemini parts"""
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, str):
                    parts.append({"text": block})
                elif isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        parts.append({"text": block.get("text", "")})
                    elif block_type == "image":
                        # 转换图片
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": source.get("media_type", "image/png"),
                                        "data": source.get("data", ""),
                                    }
                                }
                            )
                    elif block_type == "tool_use":
                        # 转换工具调用
                        parts.append(
                            {
                                "function_call": {
                                    "name": block.get("name", ""),
                                    "args": block.get("input", {}),
                                }
                            }
                        )
                    elif block_type == "tool_result":
                        # 转换工具结果
                        parts.append(
                            {
                                "function_response": {
                                    "name": block.get("tool_use_id", ""),
                                    "response": {"result": block.get("content", "")},
                                }
                            }
                        )
            return parts

        return [{"text": str(content)}]

    def _convert_system(self, system: Any) -> Dict[str, Any]:
        """转换 system prompt"""
        if isinstance(system, str):
            return {"parts": [{"text": system}]}

        if isinstance(system, list):
            parts = []
            for item in system:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
            return {"parts": parts}

        return {"parts": [{"text": str(system)}]}

    def _build_generation_config(self, claude_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """构建生成配置"""
        config: Dict[str, Any] = {}

        if "max_tokens" in claude_request:
            config["max_output_tokens"] = claude_request["max_tokens"]
        if "temperature" in claude_request:
            config["temperature"] = claude_request["temperature"]
        if "top_p" in claude_request:
            config["top_p"] = claude_request["top_p"]
        if "top_k" in claude_request:
            config["top_k"] = claude_request["top_k"]
        if "stop_sequences" in claude_request:
            config["stop_sequences"] = claude_request["stop_sequences"]

        return config if config else None

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换工具定义"""
        function_declarations = []
        for tool in tools:
            func_decl = {
                "name": tool.get("name", ""),
            }
            if "description" in tool:
                func_decl["description"] = tool["description"]
            if "input_schema" in tool:
                func_decl["parameters"] = tool["input_schema"]
            function_declarations.append(func_decl)

        return [{"function_declarations": function_declarations}]

    def convert_response(self, claude_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Claude 响应转换为 Gemini 响应

        Args:
            claude_response: Claude 格式的响应字典

        Returns:
            Gemini 格式的响应字典
        """
        content_blocks = claude_response.get("content", [])
        parts = self._convert_response_content_to_parts(content_blocks)

        # 转换停止原因
        stop_reason = claude_response.get("stop_reason")
        finish_reason = self._convert_stop_reason_to_gemini(stop_reason)

        # 转换使用量
        usage = claude_response.get("usage", {})

        return {
            "candidates": [
                {
                    "content": {
                        "parts": parts,
                        "role": "model",
                    },
                    "finishReason": finish_reason,
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": usage.get("input_tokens", 0),
                "candidatesTokenCount": usage.get("output_tokens", 0),
                "totalTokenCount": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            "modelVersion": claude_response.get("model", "claude"),
        }

    def _convert_response_content_to_parts(
        self, content_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将 Claude content blocks 转换为 Gemini parts"""
        parts = []
        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "text":
                parts.append({"text": block.get("text", "")})
            elif block_type == "tool_use":
                parts.append(
                    {
                        "functionCall": {
                            "name": block.get("name", ""),
                            "args": block.get("input", {}),
                        }
                    }
                )
        return parts if parts else [{"text": ""}]

    def _convert_stop_reason_to_gemini(self, stop_reason: Optional[str]) -> str:
        """转换停止原因为 Gemini 格式"""
        mapping = {
            "end_turn": "STOP",
            "max_tokens": "MAX_TOKENS",
            "stop_sequence": "STOP",
            "tool_use": "STOP",
        }
        return mapping.get(stop_reason or "", "STOP")

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["ClaudeStreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 Claude 流式响应转换为 Gemini 流式响应

        Args:
            chunk: Claude SSE 事件
            state: 流式转换状态

        Returns:
            Gemini 流式响应列表
        """
        from src.core.api_format.conversion.state import ClaudeStreamConversionState

        if state is None:
            state = ClaudeStreamConversionState()

        events: List[Dict[str, Any]] = []
        event_type = chunk.get("type")

        if event_type == "message_start":
            # 初始化状态
            message = chunk.get("message", {})
            state.model = message.get("model", "claude")
            state.message_id = message.get("id", "msg_claude")

        elif event_type == "content_block_start":
            # 记录内容块开始
            content_block = chunk.get("content_block", {})
            state.current_block_type = content_block.get("type", "text")
            state.current_block_index = chunk.get("index", 0)
            if state.current_block_type == "tool_use":
                state.current_tool_name = content_block.get("name", "")
                state.current_tool_id = content_block.get("id", "")
                state.accumulated_tool_input = ""

        elif event_type == "content_block_delta":
            delta = chunk.get("delta") or {}
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    # 发送 Gemini 流式响应
                    events.append(
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "parts": [{"text": text}],
                                        "role": "model",
                                    },
                                    "index": 0,
                                }
                            ],
                            "modelVersion": state.model,
                        }
                    )
            elif delta_type == "input_json_delta":
                # 累积工具输入
                state.accumulated_tool_input += delta.get("partial_json", "")

        elif event_type == "content_block_stop":
            # 如果是工具调用块结束，发送工具调用
            if state.current_block_type == "tool_use" and state.current_tool_name:
                try:
                    args = json.loads(state.accumulated_tool_input) if state.accumulated_tool_input else {}
                except json.JSONDecodeError:
                    args = {}
                events.append(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": state.current_tool_name,
                                                "args": args,
                                            }
                                        }
                                    ],
                                    "role": "model",
                                },
                                "index": 0,
                            }
                        ],
                        "modelVersion": state.model,
                    }
                )
                state.current_tool_name = ""
                state.accumulated_tool_input = ""

        elif event_type == "message_delta":
            # 消息结束
            delta = chunk.get("delta") or {}
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                finish_reason = self._convert_stop_reason_to_gemini(stop_reason)
                events.append(
                    {
                        "candidates": [
                            {
                                "content": {"parts": [], "role": "model"},
                                "finishReason": finish_reason,
                                "index": 0,
                            }
                        ],
                        "usageMetadata": chunk.get("usage", {}),
                        "modelVersion": state.model,
                    }
                )

        return events


class GeminiToClaudeConverter:
    """
    Gemini -> Claude 转换器

    - 请求转换：将 Gemini generateContent 请求转换为 Claude Messages API 格式
    - 响应转换：将 Gemini generateContent 响应转换为 Claude Messages API 格式
    """

    def convert_request(self, gemini_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Gemini 请求转换为 Claude 请求

        Args:
            gemini_request: Gemini 格式的请求字典

        Returns:
            Claude 格式的请求字典
        """
        claude_request: Dict[str, Any] = {
            "messages": self._convert_contents_to_messages(gemini_request.get("contents", [])),
        }

        # 转换 system instruction
        system_instruction = gemini_request.get("system_instruction")
        if system_instruction:
            parts = system_instruction.get("parts", [])
            system_text = "".join(p.get("text", "") for p in parts if "text" in p)
            if system_text:
                claude_request["system"] = system_text

        # 转换生成配置
        generation_config = gemini_request.get("generation_config", {})
        if "max_output_tokens" in generation_config:
            claude_request["max_tokens"] = generation_config["max_output_tokens"]
        else:
            claude_request["max_tokens"] = 4096  # Claude 需要 max_tokens
        if "temperature" in generation_config:
            claude_request["temperature"] = generation_config["temperature"]
        if "top_p" in generation_config:
            claude_request["top_p"] = generation_config["top_p"]
        if "top_k" in generation_config:
            claude_request["top_k"] = generation_config["top_k"]
        if "stop_sequences" in generation_config:
            claude_request["stop_sequences"] = generation_config["stop_sequences"]

        # 转换工具
        tools = gemini_request.get("tools", [])
        if tools:
            claude_request["tools"] = self._convert_tools_to_claude(tools)

        return claude_request

    def _convert_contents_to_messages(
        self, contents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """转换 Gemini contents 为 Claude messages"""
        messages = []
        for content in contents:
            role = content.get("role", "user")
            # Gemini 使用 "model"，Claude 使用 "assistant"
            claude_role = "assistant" if role == "model" else "user"

            parts = content.get("parts", [])
            claude_content = self._convert_parts_to_claude_content(parts)

            messages.append(
                {
                    "role": claude_role,
                    "content": claude_content,
                }
            )
        return messages

    def _convert_parts_to_claude_content(
        self, parts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """将 Gemini parts 转换为 Claude content blocks"""
        content = []
        for part in parts:
            if "text" in part:
                content.append(
                    {
                        "type": "text",
                        "text": part["text"],
                    }
                )
            elif "inline_data" in part:
                # 转换图片
                inline_data = part["inline_data"]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inline_data.get("mime_type", "image/png"),
                            "data": inline_data.get("data", ""),
                        },
                    }
                )
            elif "function_call" in part:
                # 转换工具调用
                func_call = part["function_call"]
                content.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{func_call.get('name', '')}",
                        "name": func_call.get("name", ""),
                        "input": func_call.get("args", {}),
                    }
                )
            elif "function_response" in part:
                # 转换工具结果
                func_response = part["function_response"]
                result = func_response.get("response", {}).get("result", "")
                content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": func_response.get("name", ""),
                        "content": result if isinstance(result, str) else json.dumps(result),
                    }
                )
        return content if content else [{"type": "text", "text": ""}]

    def _convert_tools_to_claude(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换 Gemini 工具为 Claude 格式"""
        claude_tools = []
        for tool in tools:
            function_declarations = tool.get("function_declarations", [])
            for func_decl in function_declarations:
                claude_tool = {
                    "name": func_decl.get("name", ""),
                }
                if "description" in func_decl:
                    claude_tool["description"] = func_decl["description"]
                if "parameters" in func_decl:
                    claude_tool["input_schema"] = func_decl["parameters"]
                claude_tools.append(claude_tool)
        return claude_tools

    def convert_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Gemini 响应转换为 Claude 响应

        Args:
            gemini_response: Gemini 格式的响应字典

        Returns:
            Claude 格式的响应字典
        """
        candidates = gemini_response.get("candidates", [])
        if not candidates:
            return self._create_empty_response()

        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        # 转换内容块
        claude_content = self._convert_parts_to_content(parts)

        # 转换使用量
        usage = self._convert_usage(gemini_response.get("usageMetadata", {}))

        # 转换停止原因
        stop_reason = self._convert_finish_reason(candidate.get("finishReason"))

        return {
            "id": f"msg_{gemini_response.get('modelVersion', 'gemini')}",
            "type": "message",
            "role": "assistant",
            "content": claude_content,
            "model": gemini_response.get("modelVersion", "gemini"),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }

    def _convert_parts_to_content(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将 Gemini parts 转换为 Claude content blocks"""
        content = []
        for part in parts:
            if "text" in part:
                content.append(
                    {
                        "type": "text",
                        "text": part["text"],
                    }
                )
            elif "functionCall" in part:
                func_call = part["functionCall"]
                content.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{func_call.get('name', '')}",
                        "name": func_call.get("name", ""),
                        "input": func_call.get("args", {}),
                    }
                )
        return content

    def _convert_usage(self, usage_metadata: Dict[str, Any]) -> Dict[str, int]:
        """转换使用量信息"""
        return {
            "input_tokens": usage_metadata.get("promptTokenCount", 0),
            "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": usage_metadata.get("cachedContentTokenCount", 0),
        }

    def _convert_finish_reason(self, finish_reason: Optional[str]) -> Optional[str]:
        """转换停止原因"""
        mapping = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "content_filtered",
            "RECITATION": "content_filtered",
            "OTHER": "stop_sequence",
        }
        if finish_reason is None:
            return "end_turn"
        return mapping.get(finish_reason, "end_turn")

    def _create_empty_response(self) -> Dict[str, Any]:
        """创建空响应"""
        return {
            "id": "msg_empty",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "gemini",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
            },
        }

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["GeminiStreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 Gemini 流式响应转换为 Claude SSE 事件

        Gemini 流式格式与 Claude/OpenAI 不同：
        - Gemini 返回完整累积文本，需要计算增量
        - 需要生成 message_start/content_block_delta/message_delta 事件序列

        Args:
            chunk: Gemini 流式响应块
            state: 流式转换状态（跨 chunk 追踪）

        Returns:
            Claude SSE 事件列表
        """
        from src.core.api_format.conversion.state import GeminiStreamConversionState

        if state is None:
            state = GeminiStreamConversionState()

        events: List[Dict[str, Any]] = []
        candidates = chunk.get("candidates") or []
        if not candidates:
            return events

        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []

        # 发送 message_start（首次）
        if not state.message_started:
            events.append(
                {
                    "type": "message_start",
                    "message": {
                        "id": state.message_id or "msg_gemini",
                        "type": "message",
                        "role": "assistant",
                        "model": state.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                    },
                }
            )
            state.message_started = True

        # 处理文本增量
        for part in parts:
            if "text" in part:
                full_text = part["text"]
                # Gemini 返回累积文本，计算增量
                delta_text = full_text[len(state.accumulated_text) :]
                if delta_text:
                    state.accumulated_text = full_text
                    # 首次文本需要发送 content_block_start
                    if not state.content_block_started:
                        events.append(
                            {
                                "type": "content_block_start",
                                "index": state.current_block_index,
                                "content_block": {"type": "text", "text": ""},
                            }
                        )
                        state.content_block_started = True
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": state.current_block_index,
                            "delta": {"type": "text_delta", "text": delta_text},
                        }
                    )
            elif "functionCall" in part:
                # 工具调用
                func_call = part["functionCall"]
                tool_id = f"toolu_{func_call.get('name', '')}_{state.tool_call_index}"
                state.tool_call_index += 1
                state.current_block_index += 1

                events.append(
                    {
                        "type": "content_block_start",
                        "index": state.current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": func_call.get("name", ""),
                            "input": {},
                        },
                    }
                )
                # 发送完整的 input 作为 JSON delta
                args = func_call.get("args", {})
                if args:
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": state.current_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
                events.append(
                    {
                        "type": "content_block_stop",
                        "index": state.current_block_index,
                    }
                )

        # 处理结束
        finish_reason = candidate.get("finishReason")
        if finish_reason:
            # 关闭当前内容块
            if state.content_block_started:
                events.append(
                    {
                        "type": "content_block_stop",
                        "index": state.current_block_index,
                    }
                )

            stop_reason = self._convert_finish_reason(finish_reason)
            events.append(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                    "usage": {"output_tokens": 0},
                }
            )

        return events


class OpenAIToGeminiConverter:
    """
    OpenAI -> Gemini 请求转换器

    将 OpenAI Chat Completions API 格式转换为 Gemini generateContent 格式
    """

    def convert_request(self, openai_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 OpenAI 请求转换为 Gemini 请求

        Args:
            openai_request: OpenAI 格式的请求字典

        Returns:
            Gemini 格式的请求字典
        """
        messages = openai_request.get("messages", [])

        # 分离 system 消息和其他消息
        system_messages = []
        other_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        gemini_request: Dict[str, Any] = {
            "contents": self._convert_messages(other_messages),
        }

        # 转换 system messages
        if system_messages:
            system_text = "\n".join(msg.get("content", "") for msg in system_messages)
            gemini_request["system_instruction"] = {"parts": [{"text": system_text}]}

        # 转换生成配置
        generation_config = self._build_generation_config(openai_request)
        if generation_config:
            gemini_request["generation_config"] = generation_config

        # 转换工具
        tools = openai_request.get("tools")
        if tools:
            gemini_request["tools"] = self._convert_tools(tools)

        return gemini_request

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换消息列表"""
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            gemini_role = "model" if role == "assistant" else "user"

            content = msg.get("content", "")
            parts = self._convert_content_to_parts(content)

            # 处理工具调用
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    parts.append(
                        {
                            "function_call": {
                                "name": func.get("name", ""),
                                "args": args,
                            }
                        }
                    )

            if parts:
                contents.append(
                    {
                        "role": gemini_role,
                        "parts": parts,
                    }
                )
        return contents

    def _convert_content_to_parts(self, content: Any) -> List[Dict[str, Any]]:
        """将 OpenAI 内容转换为 Gemini parts"""
        if content is None:
            return []

        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item_type == "image_url":
                        # OpenAI 图片 URL 格式
                        image_url = item.get("image_url", {})
                        url = image_url.get("url", "")
                        if url.startswith("data:"):
                            # base64 数据 URL
                            # 格式: data:image/png;base64,xxxxx
                            try:
                                header, data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": mime_type,
                                            "data": data,
                                        }
                                    }
                                )
                            except (ValueError, IndexError):
                                pass
            return parts

        return [{"text": str(content)}]

    def _build_generation_config(self, openai_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """构建生成配置"""
        config: Dict[str, Any] = {}

        if "max_tokens" in openai_request:
            config["max_output_tokens"] = openai_request["max_tokens"]
        if "temperature" in openai_request:
            config["temperature"] = openai_request["temperature"]
        if "top_p" in openai_request:
            config["top_p"] = openai_request["top_p"]
        if "stop" in openai_request:
            stop = openai_request["stop"]
            if isinstance(stop, str):
                config["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                config["stop_sequences"] = stop
        if "n" in openai_request:
            config["candidate_count"] = openai_request["n"]

        return config if config else None

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换工具定义"""
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                func_decl = {
                    "name": func.get("name", ""),
                }
                if "description" in func:
                    func_decl["description"] = func["description"]
                if "parameters" in func:
                    func_decl["parameters"] = func["parameters"]
                function_declarations.append(func_decl)

        return [{"function_declarations": function_declarations}]

    def convert_response(self, openai_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 OpenAI 响应转换为 Gemini 响应

        Args:
            openai_response: OpenAI 格式的响应字典

        Returns:
            Gemini 格式的响应字典
        """
        choices = openai_response.get("choices", [])
        candidates = []

        for i, choice in enumerate(choices):
            message = choice.get("message", {})
            parts = []

            # 转换文本内容
            content = message.get("content")
            if content:
                parts.append({"text": content})

            # 转换工具调用
            tool_calls = message.get("tool_calls", [])
            for tc in tool_calls:
                if tc.get("type") == "function":
                    func = tc.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    parts.append(
                        {
                            "functionCall": {
                                "name": func.get("name", ""),
                                "args": args,
                            }
                        }
                    )

            # 转换停止原因
            finish_reason = self._convert_finish_reason_to_gemini(choice.get("finish_reason"))

            candidates.append(
                {
                    "content": {
                        "parts": parts if parts else [{"text": ""}],
                        "role": "model",
                    },
                    "finishReason": finish_reason,
                    "index": i,
                }
            )

        # 转换使用量
        usage = openai_response.get("usage", {})

        return {
            "candidates": candidates,
            "usageMetadata": {
                "promptTokenCount": usage.get("prompt_tokens", 0),
                "candidatesTokenCount": usage.get("completion_tokens", 0),
                "totalTokenCount": usage.get("total_tokens", 0),
            },
            "modelVersion": openai_response.get("model", "gpt"),
        }

    def _convert_finish_reason_to_gemini(self, finish_reason: Optional[str]) -> str:
        """转换停止原因为 Gemini 格式"""
        mapping = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "function_call": "STOP",
        }
        return mapping.get(finish_reason or "", "STOP")

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["OpenAIStreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 OpenAI 流式响应转换为 Gemini 流式响应

        Args:
            chunk: OpenAI chat.completion.chunk
            state: 流式转换状态

        Returns:
            Gemini 流式响应列表
        """
        from src.core.api_format.conversion.state import OpenAIStreamConversionState

        if state is None:
            state = OpenAIStreamConversionState()

        events: List[Dict[str, Any]] = []
        choices = chunk.get("choices") or []

        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta") or {}
        finish_reason = choice.get("finish_reason")

        # 记录模型
        if chunk.get("model"):
            state.model = chunk["model"]

        # 处理文本增量
        content = delta.get("content")
        if content:
            events.append(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": content}],
                                "role": "model",
                            },
                            "index": 0,
                        }
                    ],
                    "modelVersion": state.model,
                }
            )

        # 处理工具调用
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if tc.get("function"):
                func = tc["function"]
                # 工具名称
                if func.get("name"):
                    state.current_tool_name = func["name"]
                    state.accumulated_tool_args = ""
                # 工具参数
                if func.get("arguments"):
                    state.accumulated_tool_args += func["arguments"]

        # 处理结束
        if finish_reason:
            # 如果有累积的工具调用，先发送
            if state.current_tool_name and state.accumulated_tool_args:
                try:
                    args = json.loads(state.accumulated_tool_args)
                except json.JSONDecodeError:
                    args = {}
                events.append(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": state.current_tool_name,
                                                "args": args,
                                            }
                                        }
                                    ],
                                    "role": "model",
                                },
                                "index": 0,
                            }
                        ],
                        "modelVersion": state.model,
                    }
                )
                state.current_tool_name = ""
                state.accumulated_tool_args = ""

            # 发送结束标记
            gemini_finish_reason = self._convert_finish_reason_to_gemini(finish_reason)
            events.append(
                {
                    "candidates": [
                        {
                            "content": {"parts": [], "role": "model"},
                            "finishReason": gemini_finish_reason,
                            "index": 0,
                        }
                    ],
                    "modelVersion": state.model,
                }
            )

        return events


class GeminiToOpenAIConverter:
    """
    Gemini -> OpenAI 转换器

    - 请求转换：将 Gemini generateContent 请求转换为 OpenAI Chat Completions API 格式
    - 响应转换：将 Gemini generateContent 响应转换为 OpenAI Chat Completions API 格式
    """

    def convert_request(self, gemini_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Gemini 请求转换为 OpenAI 请求

        Args:
            gemini_request: Gemini 格式的请求字典

        Returns:
            OpenAI 格式的请求字典
        """
        openai_request: Dict[str, Any] = {
            "messages": self._convert_contents_to_messages(gemini_request),
        }

        # 注意：stream 参数由调用方根据请求类型设置
        # Gemini 通过 URL 端点区分流式/非流式（streamGenerateContent vs generateContent）
        # OpenAI 通过请求体中的 stream 字段区分
        # 调用方（chat_handler_base）会在格式转换后设置 stream 参数

        # 转换生成配置
        generation_config = gemini_request.get("generation_config", {})
        if "max_output_tokens" in generation_config:
            openai_request["max_tokens"] = generation_config["max_output_tokens"]
        if "temperature" in generation_config:
            openai_request["temperature"] = generation_config["temperature"]
        if "top_p" in generation_config:
            openai_request["top_p"] = generation_config["top_p"]
        if "stop_sequences" in generation_config:
            openai_request["stop"] = generation_config["stop_sequences"]
        if "candidate_count" in generation_config:
            openai_request["n"] = generation_config["candidate_count"]

        # 转换工具
        tools = gemini_request.get("tools", [])
        if tools:
            openai_request["tools"] = self._convert_tools_to_openai(tools)

        return openai_request

    def _convert_contents_to_messages(
        self, gemini_request: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """转换 Gemini contents 为 OpenAI messages"""
        messages = []

        # 转换 system instruction
        system_instruction = gemini_request.get("system_instruction")
        if system_instruction:
            parts = system_instruction.get("parts", [])
            system_text = "".join(p.get("text", "") for p in parts if "text" in p)
            if system_text:
                messages.append({"role": "system", "content": system_text})

        # 转换 contents
        for content in gemini_request.get("contents", []):
            role = content.get("role", "user")
            # Gemini 使用 "model"，OpenAI 使用 "assistant"
            openai_role = "assistant" if role == "model" else "user"

            parts = content.get("parts", [])
            openai_content, tool_calls = self._convert_parts_to_openai_content(parts)

            message: Dict[str, Any] = {
                "role": openai_role,
            }

            if openai_content:
                message["content"] = openai_content
            if tool_calls:
                message["tool_calls"] = tool_calls

            messages.append(message)

        return messages

    def _convert_parts_to_openai_content(
        self, parts: List[Dict[str, Any]]
    ) -> tuple[Any, List[Dict[str, Any]]]:
        """将 Gemini parts 转换为 OpenAI content 和 tool_calls"""
        content_parts: List[Any] = []
        tool_calls: List[Dict[str, Any]] = []

        for part in parts:
            if "text" in part:
                content_parts.append({"type": "text", "text": part["text"]})
            elif "inline_data" in part:
                # 转换图片
                inline_data = part["inline_data"]
                mime_type = inline_data.get("mime_type", "image/png")
                data = inline_data.get("data", "")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{data}"},
                    }
                )
            elif "function_call" in part:
                # 转换工具调用
                func_call = part["function_call"]
                tool_calls.append(
                    {
                        "id": f"call_{func_call.get('name', '')}",
                        "type": "function",
                        "function": {
                            "name": func_call.get("name", ""),
                            "arguments": json.dumps(func_call.get("args", {})),
                        },
                    }
                )

        # 简化内容格式
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            content = content_parts[0]["text"]
        elif content_parts:
            content = content_parts
        else:
            content = None

        return content, tool_calls

    def _convert_tools_to_openai(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """转换 Gemini 工具为 OpenAI 格式"""
        openai_tools = []
        for tool in tools:
            function_declarations = tool.get("function_declarations", [])
            for func_decl in function_declarations:
                openai_tool: Dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": func_decl.get("name", ""),
                    },
                }
                if "description" in func_decl:
                    openai_tool["function"]["description"] = func_decl["description"]
                if "parameters" in func_decl:
                    openai_tool["function"]["parameters"] = func_decl["parameters"]
                openai_tools.append(openai_tool)
        return openai_tools

    def convert_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 Gemini 响应转换为 OpenAI 响应

        Args:
            gemini_response: Gemini 格式的响应字典

        Returns:
            OpenAI 格式的响应字典
        """
        candidates = gemini_response.get("candidates", [])
        choices = []

        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            # 提取文本内容
            text_parts = []
            tool_calls = []

            for part in parts:
                if "text" in part:
                    text_parts.append(part["text"])
                elif "functionCall" in part:
                    func_call = part["functionCall"]
                    tool_calls.append(
                        {
                            "id": f"call_{func_call.get('name', '')}_{i}",
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", ""),
                                "arguments": json.dumps(func_call.get("args", {})),
                            },
                        }
                    )

            message: Dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
            }

            if tool_calls:
                message["tool_calls"] = tool_calls

            finish_reason = self._convert_finish_reason(candidate.get("finishReason"))

            choices.append(
                {
                    "index": i,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            )

        # 转换使用量
        usage = self._convert_usage(gemini_response.get("usageMetadata", {}))

        return {
            "id": f"chatcmpl-{gemini_response.get('modelVersion', 'gemini')}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": gemini_response.get("modelVersion", "gemini"),
            "choices": choices,
            "usage": usage,
        }

    def _convert_usage(self, usage_metadata: Dict[str, Any]) -> Dict[str, int]:
        """转换使用量信息"""
        prompt_tokens = usage_metadata.get("promptTokenCount", 0)
        completion_tokens = usage_metadata.get("candidatesTokenCount", 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _convert_finish_reason(self, finish_reason: Optional[str]) -> str:
        """转换停止原因"""
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
        }
        if finish_reason is None:
            return "stop"
        return mapping.get(finish_reason, "stop")

    # ==================== 流式转换 ====================

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        state: Optional["GeminiStreamConversionState"] = None,
    ) -> List[Dict[str, Any]]:
        """
        将 Gemini 流式响应转换为 OpenAI SSE chunk

        Gemini 流式格式与 OpenAI 不同：
        - Gemini 返回完整累积文本，需要计算增量
        - 需要生成 OpenAI chat.completion.chunk 格式

        Args:
            chunk: Gemini 流式响应块
            state: 流式转换状态（跨 chunk 追踪）

        Returns:
            OpenAI SSE chunk 列表
        """
        from src.core.api_format.conversion.state import GeminiStreamConversionState

        if state is None:
            state = GeminiStreamConversionState()

        events: List[Dict[str, Any]] = []
        candidates = chunk.get("candidates") or []
        if not candidates:
            return events

        candidate = candidates[0]
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        finish_reason = candidate.get("finishReason")

        chunk_id = f"chatcmpl-{state.message_id or 'gemini'}"

        # 发送首个 chunk（带 role）
        if not state.message_started:
            events.append(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": state.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            state.message_started = True

        # 处理文本增量
        for part in parts:
            if "text" in part:
                full_text = part["text"]
                # Gemini 返回累积文本，计算增量
                delta_text = full_text[len(state.accumulated_text) :]
                if delta_text:
                    state.accumulated_text = full_text
                    events.append(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": state.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif "functionCall" in part:
                # 工具调用
                func_call = part["functionCall"]
                tool_id = f"call_{func_call.get('name', '')}_{state.tool_call_index}"

                # 工具调用开始
                events.append(
                    {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": state.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": state.tool_call_index,
                                            "id": tool_id,
                                            "type": "function",
                                            "function": {
                                                "name": func_call.get("name", ""),
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

                # 工具调用参数
                args = func_call.get("args", {})
                if args:
                    events.append(
                        {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": state.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": state.tool_call_index,
                                                "function": {
                                                    "arguments": json.dumps(
                                                        args, ensure_ascii=False
                                                    )
                                                },
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )

                state.tool_call_index += 1

        # 处理结束
        if finish_reason:
            openai_finish_reason = self._convert_finish_reason(finish_reason)
            events.append(
                {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": state.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": openai_finish_reason,
                        }
                    ],
                }
            )

        return events


__all__ = [
    "ClaudeToGeminiConverter",
    "GeminiToClaudeConverter",
    "OpenAIToGeminiConverter",
    "GeminiToOpenAIConverter",
]
