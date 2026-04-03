"""
响应解析器基类 - re-export from src.core.stream_types

实际定义已下沉到 src/core/stream_types.py，此文件保留向后兼容的 re-export。
"""

from src.core.stream_types import (
    ParsedChunk,
    ParsedResponse,
    ResponseParser,
    StreamStats,
)

__all__ = [
    "ParsedChunk",
    "ParsedResponse",
    "ResponseParser",
    "StreamStats",
]
