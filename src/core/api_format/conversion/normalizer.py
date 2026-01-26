"""
格式标准化器接口（FormatNormalizer）

每个格式（OpenAI/Claude/Gemini）实现一个 Normalizer，将 provider 结构转换到 internal，
再从 internal 输出到目标格式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .internal import FormatCapabilities, InternalError, InternalRequest, InternalResponse
from .stream_events import InternalStreamEvent
from .stream_state import StreamState


class FormatNormalizer(ABC):
    """格式标准化器基类"""

    FORMAT_ID: str  # 如 "CLAUDE", "OPENAI", "GEMINI"
    capabilities: FormatCapabilities

    # ============ 请求转换 ============

    @abstractmethod
    def request_to_internal(self, request: Dict[str, Any]) -> InternalRequest:
        """将格式特定请求转换为内部表示"""
        raise NotImplementedError

    @abstractmethod
    def request_from_internal(self, internal: InternalRequest) -> Dict[str, Any]:
        """将内部表示转换为格式特定请求"""
        raise NotImplementedError

    # ============ 响应转换 ============

    @abstractmethod
    def response_to_internal(self, response: Dict[str, Any]) -> InternalResponse:
        """将格式特定响应转换为内部表示"""
        raise NotImplementedError

    @abstractmethod
    def response_from_internal(self, internal: InternalResponse) -> Dict[str, Any]:
        """将内部表示转换为格式特定响应"""
        raise NotImplementedError

    # ============ 流式转换（可选） ============

    def stream_chunk_to_internal(
        self,
        chunk: Dict[str, Any],
        state: StreamState,
    ) -> List[InternalStreamEvent]:
        """将格式特定流式块转换为内部事件"""
        raise NotImplementedError

    def stream_event_from_internal(
        self,
        event: InternalStreamEvent,
        state: StreamState,
    ) -> List[Dict[str, Any]]:
        """将内部事件转换为格式特定流式块"""
        raise NotImplementedError

    # ============ 错误转换（可选） ============

    def is_error_response(self, response: Dict[str, Any]) -> bool:
        """基于 body 的兜底判断（不可靠），子类可覆盖"""
        return False

    def error_to_internal(self, error_response: Dict[str, Any]) -> InternalError:
        """将格式特定错误转换为内部表示"""
        raise NotImplementedError

    def error_from_internal(self, internal: InternalError) -> Dict[str, Any]:
        """将内部错误表示转换为格式特定错误"""
        raise NotImplementedError


__all__ = [
    "FormatNormalizer",
]

