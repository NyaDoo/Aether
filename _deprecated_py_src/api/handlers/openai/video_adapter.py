"""
OpenAI Video Adapter - 基于 VideoAdapterBase 的 Sora 适配器
"""

from __future__ import annotations

from src.api.handlers.base.video_adapter_base import VideoAdapterBase
from src.api.handlers.base.video_handler_base import VideoHandlerBase
from src.core.api_format import ApiFamily


class OpenAIVideoAdapter(VideoAdapterBase):
    FORMAT_ID = "openai:video"
    API_FAMILY = ApiFamily.OPENAI
    name = "openai.video"

    @property
    def HANDLER_CLASS(self) -> type[VideoHandlerBase]:
        from src.api.handlers.openai.video_handler import OpenAIVideoHandler

        return OpenAIVideoHandler


__all__ = ["OpenAIVideoAdapter"]
