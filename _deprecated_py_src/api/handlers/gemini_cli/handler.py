"""
Gemini CLI Message Handler - 基于通用 CLI Handler 基类的实现

继承 CliMessageHandlerBase，处理 Gemini CLI API 格式的请求。
"""

from typing import Any

from src.api.handlers.base.cli_handler_base import (
    CliMessageHandlerBase,
    StreamContext,
)
from src.core.api_format import ApiFamily, EndpointKind


class GeminiCliMessageHandler(CliMessageHandlerBase):
    """
    Gemini CLI Message Handler - 处理 Gemini CLI API 格式

    使用新三层架构 (Provider -> ProviderEndpoint -> ProviderAPIKey)
    通过 TaskService/FailoverEngine 实现自动故障转移、健康监控和并发控制

    响应格式特点：
    - Gemini 使用 JSON 数组格式流式响应（非 SSE）
    - 每个 chunk 包含 candidates、usageMetadata 等字段
    - finish_reason: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
    - Token 使用: promptTokenCount (输入), thoughtsTokenCount + candidatesTokenCount (输出), cachedContentTokenCount (缓存)

    Gemini API 特殊处理：
    - model 在 URL 路径中而非请求体，如 /v1beta/models/{model}:generateContent
    - 请求体中的 model 字段用于内部路由，不发送给 API
    """

    FORMAT_ID = "gemini:cli"
    API_FAMILY = ApiFamily.GEMINI
    ENDPOINT_KIND = EndpointKind.CLI

    def extract_model_from_request(
        self,
        request_body: dict[str, Any],  # noqa: ARG002 - 基类签名要求
        path_params: dict[str, Any] | None = None,
    ) -> str:
        """
        从请求中提取模型名 - Gemini 格式实现

        Gemini API 的 model 在 URL 路径中而非请求体：
        /v1beta/models/{model}:generateContent

        Args:
            request_body: 请求体（Gemini 不包含 model）
            path_params: URL 路径参数（包含 model）

        Returns:
            模型名，如果无法提取则返回 "unknown"
        """
        # Gemini: model 从 URL 路径参数获取
        if path_params and "model" in path_params:
            return str(path_params["model"])
        return "unknown"

    def prepare_provider_request_body(
        self,
        request_body: dict[str, Any],
    ) -> dict[str, Any]:
        """
        准备发送给 Gemini API 的请求体 - 移除 model 字段

        Gemini API 要求 model 只在 URL 路径中，请求体中的 model 字段
        会导致某些代理返回 404 错误。

        Args:
            request_body: 请求体

        Returns:
            不含 model 字段的请求体
        """
        result = dict(request_body)
        result.pop("model", None)
        return result

    def finalize_provider_request(
        self,
        request_body: dict[str, Any],
        *,
        mapped_model: str | None,
        provider_api_format: str | None,  # noqa: ARG002
    ) -> dict[str, Any]:
        from src.api.handlers.gemini.image_gen import (
            adapt_request_for_image_gen,
            is_image_gen_model,
        )

        # Sanitize Gemini contents: strip parts without a valid data-oneof
        # field and merge consecutive same-role entries.  This catches cases
        # missed by the normalizer (passthrough) or the antigravity envelope.
        from src.core.api_format.conversion.normalizers.gemini import (
            compact_gemini_contents,
        )

        contents = request_body.get("contents")
        if isinstance(contents, list):
            request_body["contents"] = compact_gemini_contents(contents)

        if not is_image_gen_model(mapped_model):
            return request_body
        return adapt_request_for_image_gen(request_body)

    def get_model_for_url(
        self,
        request_body: dict[str, Any],
        mapped_model: str | None,
    ) -> str | None:
        """
        Gemini 需要将 model 放入 URL 路径中

        Args:
            request_body: 请求体
            mapped_model: 映射后的模型名（如果有）

        Returns:
            用于 URL 路径的模型名
        """
        # 优先使用映射后的模型名，否则使用请求体中的
        return mapped_model or request_body.get("model")

    def _extract_usage_from_event(
        self,
        event: dict[str, Any],
        *,
        provider_type: str | None = None,
    ) -> dict[str, int]:
        """
        从 Gemini 事件中提取 token 使用情况

        调用 GeminiStreamParser.extract_usage 作为单一实现源

        Args:
            event: Gemini 流式响应事件
            provider_type: Provider 类型（用于 Antigravity 特判）

        Returns:
            包含 input_tokens, output_tokens, cached_tokens 的字典
        """
        from src.core.provider_types import ProviderType

        if str(provider_type or "").lower() == ProviderType.ANTIGRAVITY:
            return self._extract_antigravity_usage(event)

        from src.api.handlers.gemini.stream_parser import GeminiStreamParser

        usage = GeminiStreamParser().extract_usage(event)

        if not usage:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
            }

        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cached_tokens": usage.get("cached_tokens", 0),
        }

    def _extract_antigravity_usage(self, event: dict[str, Any]) -> dict[str, int]:
        """Antigravity 专用 usage 提取（宽松 + 边界保护）。

        Antigravity 的 usageMetadata 可能缺少 totalTokenCount，因此不能依赖
        GeminiStreamParser.extract_usage 的“totalTokenCount 必须存在”的严格判断。
        """
        usage_metadata = event.get("usageMetadata", {})
        if not isinstance(usage_metadata, dict) or not usage_metadata:
            return {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

        def _as_int(v: Any) -> int:
            try:
                return int(v or 0)
            except Exception:
                return 0

        prompt = _as_int(usage_metadata.get("promptTokenCount"))
        cached = _as_int(usage_metadata.get("cachedContentTokenCount"))
        candidates = _as_int(usage_metadata.get("candidatesTokenCount"))
        thoughts = _as_int(usage_metadata.get("thoughtsTokenCount"))

        return {
            # 注意：计费层会根据 api_family(GEMINI) 扣除 cache_read_tokens，
            # 因此这里保持 Gemini 口径：input_tokens=promptTokenCount（含缓存）。
            "input_tokens": max(0, prompt),
            "output_tokens": max(0, candidates + thoughts),
            "cached_tokens": max(0, cached),
        }

    def _process_event_data(
        self,
        ctx: StreamContext,
        _event_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        处理 Gemini CLI 格式的流式事件

        Gemini 的流式响应是 JSON 数组格式，每个元素结构如下:
        {
            "candidates": [{
                "content": {"parts": [{"text": "..."}], "role": "model"},
                "finishReason": "STOP",
                "safetyRatings": [...]
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
                "cachedContentTokenCount": 5
            },
            "modelVersion": "gemini-1.5-pro"
        }

        注意: Gemini 流解析器会将每个 JSON 对象作为一个"事件"传递
        event_type 在这里可能为空或是自定义的标记

        跨格式转换时（如 provider=claude:chat），原始事件数据是 Provider 格式而非 Gemini 格式。
        此时委托基类方法通过 Provider 格式解析器提取 usage。
        """
        # 跨格式转换时：原始事件是 Provider 格式，
        # 基类 _process_event_data 会自动选择正确的 Provider 解析器提取 usage/text
        if ctx.provider_api_format and ctx.provider_api_format != ctx.client_api_format:
            super()._process_event_data(ctx, _event_type, data)
            return

        # 以下是同格式（gemini:cli / gemini:chat）的处理逻辑

        # 提取候选响应
        candidates = data.get("candidates", [])
        if candidates:
            candidate = candidates[0]
            content = candidate.get("content", {})

            # 提取文本内容
            parts = content.get("parts", [])
            for part in parts:
                if "text" in part:
                    ctx.append_text(part["text"])

            # 检查结束原因
            finish_reason = candidate.get("finishReason")
            if finish_reason in ("STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"):
                ctx.has_completion = True
                ctx.final_response = data

        # 提取使用量信息（复用 GeminiStreamParser.extract_usage）
        usage = self._extract_usage_from_event(data, provider_type=ctx.provider_type)
        if usage["input_tokens"] > 0 or usage["output_tokens"] > 0:
            ctx.input_tokens = usage["input_tokens"]
            ctx.output_tokens = usage["output_tokens"]
            ctx.cached_tokens = usage["cached_tokens"]

        # 提取模型版本作为响应 ID
        model_version = data.get("modelVersion")
        if model_version:
            if not ctx.response_id:
                ctx.response_id = f"gemini-{model_version}"
            # 存储到 response_metadata 供 Usage 记录使用
            ctx.response_metadata["model_version"] = model_version

        # 检查错误
        if "error" in data:
            ctx.has_completion = True
            ctx.final_response = data

    def _extract_response_metadata(
        self,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        """
        从 Gemini 响应中提取元数据

        提取 modelVersion 字段，记录实际使用的模型版本。

        Args:
            response: Gemini API 响应

        Returns:
            包含 model_version 的元数据字典
        """
        metadata: dict[str, Any] = {}
        model_version = response.get("modelVersion")
        if model_version:
            metadata["model_version"] = model_version
        return metadata
