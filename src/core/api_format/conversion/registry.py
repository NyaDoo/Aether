"""
格式转换器注册表（核心层）

自动管理不同 API 格式之间的转换器，支持：
- 请求转换：客户端格式 → Provider 格式
- 响应转换：Provider 格式 → 客户端格式

说明：
- 该注册表位于 core 层，避免 services 依赖 api/handlers。
- 具体转换器的注册（例如 Claude/OpenAI/Gemini）应由应用启动层完成，
  或由 api 层的 bootstrap 逻辑完成，以保持依赖方向：api -> core。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Tuple, Union

from src.core.logger import logger
from src.core.metrics import format_conversion_duration_seconds, format_conversion_total

from .exceptions import FormatConversionError

if TYPE_CHECKING:
    from .state import (
        ClaudeStreamConversionState,
        GeminiStreamConversionState,
        OpenAIStreamConversionState,
        StreamConversionState,
    )


@contextmanager
def _track_conversion_metrics(
    direction: str, source: str, target: str
) -> Generator[None, None, None]:
    """
    跟踪转换指标的上下文管理器

    Args:
        direction: 转换方向（request/response/stream）
        source: 源格式（大写）
        target: 目标格式（大写）

    Yields:
        None - 执行转换逻辑
    """
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        format_conversion_total.labels(direction, source, target, status).inc()
        format_conversion_duration_seconds.labels(direction, source, target).observe(
            time.perf_counter() - start
        )


class FormatConverterRegistry:
    """
    格式转换器注册表

    管理不同 API 格式之间的双向转换器
    """

    def __init__(self) -> None:
        # key: (source_format, target_format), value: converter instance
        self._converters: Dict[Tuple[str, str], Any] = {}

    def register(
        self,
        source_format: str,
        target_format: str,
        converter: Any,
    ) -> None:
        """
        注册格式转换器

        Args:
            source_format: 源格式（如 "CLAUDE", "OPENAI", "GEMINI"）
            target_format: 目标格式
            converter: 转换器实例（需要有 convert_request/convert_response 方法）
        """
        key = (source_format.upper(), target_format.upper())
        self._converters[key] = converter
        logger.info(f"[ConverterRegistry] 注册转换器: {source_format} -> {target_format}")

    def get_converter(
        self,
        source_format: str,
        target_format: str,
    ) -> Optional[Any]:
        """
        获取转换器

        Args:
            source_format: 源格式
            target_format: 目标格式

        Returns:
            转换器实例，如果不存在返回 None
        """
        key = (source_format.upper(), target_format.upper())
        return self._converters.get(key)

    def has_converter(
        self,
        source_format: str,
        target_format: str,
    ) -> bool:
        """检查是否存在转换器"""
        key = (source_format.upper(), target_format.upper())
        return key in self._converters

    def convert_request(
        self,
        request: Dict[str, Any],
        source_format: str,
        target_format: str,
    ) -> Dict[str, Any]:
        """
        转换请求

        Args:
            request: 原始请求字典
            source_format: 源格式（客户端格式）
            target_format: 目标格式（Provider 格式）

        Returns:
            转换后的请求字典，如果无需转换或没有转换器则返回原始请求
        """
        # 同格式无需转换
        if source_format.upper() == target_format.upper():
            return request

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            logger.warning(
                f"[ConverterRegistry] 未找到请求转换器: {source_format} -> {target_format}，返回原始请求"
            )
            return request

        if not hasattr(converter, "convert_request"):
            logger.warning(
                f"[ConverterRegistry] 转换器缺少 convert_request 方法: {source_format} -> {target_format}"
            )
            return request

        try:
            converted: Dict[str, Any] = converter.convert_request(request)
            logger.debug(f"[ConverterRegistry] 请求转换成功: {source_format} -> {target_format}")
            return converted
        except Exception as e:
            logger.error(
                f"[ConverterRegistry] 请求转换失败: {source_format} -> {target_format}: {e}"
            )
            return request

    def convert_response(
        self,
        response: Dict[str, Any],
        source_format: str,
        target_format: str,
    ) -> Dict[str, Any]:
        """
        转换响应

        Args:
            response: 原始响应字典
            source_format: 源格式（Provider 格式）
            target_format: 目标格式（客户端格式）

        Returns:
            转换后的响应字典，如果无需转换或没有转换器则返回原始响应
        """
        # 同格式无需转换
        if source_format.upper() == target_format.upper():
            return response

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            logger.warning(
                f"[ConverterRegistry] 未找到响应转换器: {source_format} -> {target_format}，返回原始响应"
            )
            return response

        if not hasattr(converter, "convert_response"):
            logger.warning(
                f"[ConverterRegistry] 转换器缺少 convert_response 方法: {source_format} -> {target_format}"
            )
            return response

        try:
            converted: Dict[str, Any] = converter.convert_response(response)
            logger.debug(f"[ConverterRegistry] 响应转换成功: {source_format} -> {target_format}")
            return converted
        except Exception as e:
            logger.error(
                f"[ConverterRegistry] 响应转换失败: {source_format} -> {target_format}: {e}"
            )
            return response

    def convert_stream_chunk(
        self,
        chunk: Dict[str, Any],
        source_format: str,
        target_format: str,
        state: Optional[Union["StreamConversionState", "GeminiStreamConversionState"]] = None,
    ) -> list[Dict[str, Any]]:
        """
        转换流式响应块

        Args:
            chunk: 原始流式响应块
            source_format: 源格式（Provider 格式）
            target_format: 目标格式（客户端格式）
            state: 流式转换状态（StreamConversionState 或 GeminiStreamConversionState）

        Returns:
            转换后的事件列表（可能 0-N 个），失败时返回原始 chunk 的单元素列表
        """
        # 同格式无需转换
        if source_format.upper() == target_format.upper():
            return [chunk]

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            return [chunk]

        # 使用流式转换方法
        if hasattr(converter, "convert_stream_chunk"):
            try:
                result: list[Dict[str, Any]] = converter.convert_stream_chunk(chunk, state)
                return result
            except Exception as e:
                logger.error(
                    f"[ConverterRegistry] 流式块转换失败: {source_format} -> {target_format}: {e}"
                )
                return [chunk]

        # 降级到普通响应转换（作为单个事件返回）
        if hasattr(converter, "convert_response"):
            try:
                converted: Dict[str, Any] = converter.convert_response(chunk)
                return [converted]
            except Exception:
                return [chunk]

        return [chunk]

    def list_converters(self) -> list[Tuple[str, str]]:
        """列出所有已注册的转换器"""
        return list(self._converters.keys())

    # ========== 能力查询方法 ==========

    def can_convert_request(self, source: str, target: str) -> bool:
        """检查是否支持请求转换"""
        converter = self.get_converter(source, target)
        return converter is not None and hasattr(converter, "convert_request")

    def can_convert_response(self, source: str, target: str) -> bool:
        """检查是否支持响应转换"""
        converter = self.get_converter(source, target)
        return converter is not None and hasattr(converter, "convert_response")

    def can_convert_stream(self, source: str, target: str) -> bool:
        """检查是否支持流式转换"""
        converter = self.get_converter(source, target)
        if converter is None:
            return False
        return hasattr(converter, "convert_stream_chunk")

    def can_convert_full(
        self,
        source: str,
        target: str,
        require_stream: bool = False,
    ) -> bool:
        """
        检查是否支持完整的双向转换

        对于跨格式请求，需要：
        1. 请求转换：source -> target
        2. 响应转换：target -> source（注意方向相反）
        3. 流式转换（如果 require_stream=True）：target -> source

        Args:
            source: 客户端格式
            target: Provider 格式
            require_stream: 是否要求支持流式转换
        """
        # 请求：client -> provider
        if not self.can_convert_request(source, target):
            return False
        # 响应：provider -> client（方向相反）
        if not self.can_convert_response(target, source):
            return False
        # 流式：provider -> client（方向相反）
        if require_stream and not self.can_convert_stream(target, source):
            return False
        return True

    def get_supported_targets(self, source: str) -> list[str]:
        """获取指定源格式支持转换到的目标格式列表"""
        source_upper = source.upper()
        return [target for (src, target) in self._converters.keys() if src == source_upper]

    # ========== 严格模式方法 ==========

    def convert_request_strict(
        self,
        request: Dict[str, Any],
        source_format: str,
        target_format: str,
    ) -> Dict[str, Any]:
        """
        严格模式请求转换 - 失败时抛出异常

        用于需要故障转移的场景：转换失败会抛出 FormatConversionError，
        让 Orchestrator 可以尝试下一个候选。

        Raises:
            FormatConversionError: 转换失败时抛出
        """
        source_upper = source_format.upper()
        target_upper = target_format.upper()

        # 同格式无需转换
        if source_upper == target_upper:
            return request

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            raise FormatConversionError(source_format, target_format, "未找到转换器")

        if not hasattr(converter, "convert_request"):
            raise FormatConversionError(
                source_format, target_format, "转换器缺少 convert_request 方法"
            )

        with _track_conversion_metrics("request", source_upper, target_upper):
            try:
                converted: Dict[str, Any] = converter.convert_request(request)
                logger.debug(f"[ConverterRegistry] 请求转换成功: {source_format} -> {target_format}")
                return converted
            except FormatConversionError:
                raise
            except Exception as e:
                raise FormatConversionError(source_format, target_format, str(e)) from e

    def convert_response_strict(
        self,
        response: Dict[str, Any],
        source_format: str,
        target_format: str,
    ) -> Dict[str, Any]:
        """
        严格模式响应转换 - 失败时抛出异常

        Raises:
            FormatConversionError: 转换失败时抛出
        """
        source_upper = source_format.upper()
        target_upper = target_format.upper()

        if source_upper == target_upper:
            return response

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            raise FormatConversionError(source_format, target_format, "未找到转换器")

        if not hasattr(converter, "convert_response"):
            raise FormatConversionError(
                source_format, target_format, "转换器缺少 convert_response 方法"
            )

        with _track_conversion_metrics("response", source_upper, target_upper):
            try:
                converted: Dict[str, Any] = converter.convert_response(response)
                logger.debug(f"[ConverterRegistry] 响应转换成功: {source_format} -> {target_format}")
                return converted
            except FormatConversionError:
                raise
            except Exception as e:
                raise FormatConversionError(source_format, target_format, str(e)) from e

    def convert_stream_chunk_strict(
        self,
        chunk: Dict[str, Any],
        source_format: str,
        target_format: str,
        state: Optional[
            Union[
                "StreamConversionState",
                "GeminiStreamConversionState",
                "ClaudeStreamConversionState",
                "OpenAIStreamConversionState",
            ]
        ] = None,
    ) -> list[Dict[str, Any]]:
        """
        严格模式流式块转换 - 失败时抛出异常

        Args:
            chunk: 流式响应块
            source_format: 源格式
            target_format: 目标格式
            state: 流式转换状态（StreamConversionState 或 GeminiStreamConversionState）

        Returns:
            转换后的事件列表（可能 0-N 个）

        Raises:
            FormatConversionError: 转换失败时抛出
        """
        source_upper = source_format.upper()
        target_upper = target_format.upper()

        if source_upper == target_upper:
            return [chunk]

        converter = self.get_converter(source_format, target_format)
        if converter is None:
            raise FormatConversionError(source_format, target_format, "未找到转换器")

        if not hasattr(converter, "convert_stream_chunk"):
            raise FormatConversionError(
                source_format, target_format, "转换器缺少 convert_stream_chunk 方法"
            )

        with _track_conversion_metrics("stream", source_upper, target_upper):
            try:
                result: list[Dict[str, Any]] = converter.convert_stream_chunk(chunk, state)
                return result
            except FormatConversionError:
                raise
            except Exception as e:
                raise FormatConversionError(
                    source_format, target_format, f"流式块转换失败: {e}"
                ) from e


# 全局单例
converter_registry = FormatConverterRegistry()


__all__ = [
    "FormatConverterRegistry",
    "converter_registry",
    "FormatConversionError",
]
