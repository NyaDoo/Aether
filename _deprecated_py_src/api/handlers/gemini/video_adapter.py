"""
Gemini Video Adapter - 基于 VideoAdapterBase 的 Veo 适配器
"""

from __future__ import annotations

from src.api.handlers.base.video_adapter_base import VideoAdapterBase
from src.api.handlers.base.video_handler_base import VideoHandlerBase
from src.core.api_format import ApiFamily


class GeminiVeoAdapter(VideoAdapterBase):
    FORMAT_ID = "gemini:video"
    API_FAMILY = ApiFamily.GEMINI
    name = "gemini.video"

    @property
    def HANDLER_CLASS(self) -> type[VideoHandlerBase]:
        from src.api.handlers.gemini.video_handler import GeminiVeoHandler

        return GeminiVeoHandler


__all__ = ["GeminiVeoAdapter"]
