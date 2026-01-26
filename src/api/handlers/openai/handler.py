"""
OpenAI Chat Handler - 基于通用 Chat Handler 基类的简化实现

继承 ChatHandlerBase，只需覆盖格式特定的方法。
代码量从原来的 ~1315 行减少到 ~100 行。
"""

from typing import Any, Dict, Optional

from src.api.handlers.base.chat_handler_base import ChatHandlerBase


class OpenAIChatHandler(ChatHandlerBase):
    """
    OpenAI Chat Handler - 处理 OpenAI Chat Completions API 格式的请求

    格式特点：
    - 使用 prompt_tokens/completion_tokens
    - 不支持 cache tokens
    - 请求格式：OpenAIRequest
    """

    FORMAT_ID = "OPENAI"

    def extract_model_from_request(
        self,
        request_body: Dict[str, Any],
        path_params: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> str:
        """
        从请求中提取模型名 - OpenAI 格式实现

        OpenAI API 的 model 在请求体顶级字段。

        Args:
            request_body: 请求体
            path_params: URL 路径参数（OpenAI 不使用）

        Returns:
            模型名
        """
        model = request_body.get("model")
        return str(model) if model else "unknown"

    def apply_mapped_model(
        self,
        request_body: Dict[str, Any],
        mapped_model: str,
    ) -> Dict[str, Any]:
        """
        将映射后的模型名应用到请求体

        OpenAI API 的 model 在请求体顶级字段。

        Args:
            request_body: 原始请求体
            mapped_model: 映射后的模型名

        Returns:
            更新了 model 字段的请求体
        """
        result = dict(request_body)
        result["model"] = mapped_model
        return result

    async def _convert_request(self, request):
        """
        将请求转换为 OpenAI 格式

        Args:
            request: 原始请求对象

        Returns:
            OpenAIRequest 对象
        """
        from src.core.api_format.conversion.registry import (
            format_conversion_registry,
            register_default_normalizers,
        )
        from src.models.claude import ClaudeMessagesRequest
        from src.models.openai import OpenAIRequest

        register_default_normalizers()

        # 如果已经是 OpenAI 格式，直接返回
        if isinstance(request, OpenAIRequest):
            return request

        # 如果是 Claude 格式，转换为 OpenAI 格式
        if isinstance(request, ClaudeMessagesRequest):
            req_dict = request.model_dump() if hasattr(request, "model_dump") else request.dict()
            openai_dict = format_conversion_registry.convert_request(req_dict, "CLAUDE", "OPENAI")
            return OpenAIRequest(**openai_dict)

        # 如果是字典，尝试判断格式
        if isinstance(request, dict):
            try:
                return OpenAIRequest(**request)
            except Exception:
                try:
                    openai_dict = format_conversion_registry.convert_request(request, "CLAUDE", "OPENAI")
                    return OpenAIRequest(**openai_dict)
                except Exception:
                    return OpenAIRequest(**request)

        return request

    def _extract_usage(self, response: Dict) -> Dict[str, int]:
        """
        从 OpenAI 响应中提取 token 使用情况

        OpenAI 格式使用：
        - prompt_tokens / completion_tokens
        - 不支持 cache tokens
        """
        usage = response.get("usage", {})

        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    def _normalize_response(self, response: Dict) -> Dict:
        """
        规范化 OpenAI 响应

        Args:
            response: 原始响应

        Returns:
            规范化后的响应
        """
        # 作为中转站，直接透传响应，不做标准化处理
        return response
