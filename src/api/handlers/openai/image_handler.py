"""
OpenAI Image Handler - 处理 /v1/images/generations 请求
"""

from __future__ import annotations

from src.api.handlers.base.image_handler_base import ImageHandlerBase
from src.core.api_format import ApiFamily, EndpointKind


class OpenAIImageHandler(ImageHandlerBase):
    """OpenAI 图像生成处理器（同步 + SSE 流式透传）。"""

    FORMAT_ID = "openai:image"
    API_FAMILY = ApiFamily.OPENAI
    ENDPOINT_KIND = EndpointKind.IMAGE

    DEFAULT_BASE_URL = "https://api.openai.com"

    def _build_upstream_url(
        self, base_url: str | None, *, path_suffix: str | None = None
    ) -> str:
        suffix = (path_suffix or "generations").strip().lower()
        if suffix not in {"generations", "edits"}:
            suffix = "generations"
        base = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/{suffix}"
        return f"{base}/v1/images/{suffix}"


__all__ = ["OpenAIImageHandler"]
