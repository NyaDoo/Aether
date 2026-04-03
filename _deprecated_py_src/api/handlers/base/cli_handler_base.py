"""
CLI Message Handler 通用基类

将 CLI 格式处理器的通用逻辑（HTTP 请求、SSE 解析、统计记录）抽取到基类，
子类只需实现格式特定的事件解析逻辑。

设计目标：
1. 减少代码重复 - 原来每个 CLI Handler 900+ 行，抽取后子类只需 ~100 行
2. 统一错误处理 - 超时、空流、故障转移等逻辑集中管理
3. 简化新格式接入 - 只需实现 ResponseParser 和少量钩子方法
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from src.api.handlers.base.base_handler import BaseMessageHandler
from src.api.handlers.base.cli_event_mixin import CliEventMixin
from src.api.handlers.base.cli_monitor_mixin import CliMonitorMixin
from src.api.handlers.base.cli_prefetch_mixin import CliPrefetchMixin
from src.api.handlers.base.cli_request_mixin import CliRequestMixin
from src.api.handlers.base.cli_stream_mixin import CliStreamMixin
from src.api.handlers.base.cli_sync_mixin import CliSyncMixin
from src.api.handlers.base.parsers import get_parser_for_format
from src.api.handlers.base.request_builder import PassthroughRequestBuilder
from src.api.handlers.base.response_parser import ResponseParser
from src.api.handlers.base.stream_context import StreamContext
from src.models.database import ApiKey, User

__all__ = [
    "CliMessageHandlerBase",
    "StreamContext",
]


class CliMessageHandlerBase(
    CliRequestMixin,
    CliStreamMixin,
    CliPrefetchMixin,
    CliEventMixin,
    CliMonitorMixin,
    CliSyncMixin,
    BaseMessageHandler,
):
    """
    CLI 格式消息处理器基类

    提供 CLI 格式（直接透传请求）的通用处理逻辑：
    - 流式请求的 HTTP 连接管理
    - SSE 事件解析框架
    - 统计信息收集和记录
    - 错误处理和故障转移

    子类需要实现：
    - get_response_parser(): 返回格式特定的响应解析器
    - 可选覆盖 handle_sse_event() 自定义事件处理

    """

    # 子类可覆盖的配置
    FORMAT_ID: str = "UNKNOWN"  # API 格式标识
    DATA_TIMEOUT: int = 30  # 流数据超时时间（秒）
    EMPTY_CHUNK_THRESHOLD: int = 10  # 空流检测的 chunk 阈值

    def __init__(
        self,
        db: Session,
        user: User,
        api_key: ApiKey,
        request_id: str,
        client_ip: str,
        user_agent: str,
        start_time: float,
        allowed_api_formats: list | None = None,
        adapter_detector: None | (
            Callable[[dict[str, str], dict[str, Any] | None], dict[str, bool]]
        ) = None,
        perf_metrics: dict[str, Any] | None = None,
        api_family: str | None = None,
        endpoint_kind: str | None = None,
    ):
        allowed = allowed_api_formats or [self.FORMAT_ID]
        super().__init__(
            db=db,
            user=user,
            api_key=api_key,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
            start_time=start_time,
            allowed_api_formats=allowed,
            adapter_detector=adapter_detector,
            perf_metrics=perf_metrics,
            api_family=api_family,
            endpoint_kind=endpoint_kind,
        )
        self._parser: ResponseParser | None = None
        self._request_builder = PassthroughRequestBuilder()

    @property
    def parser(self) -> ResponseParser:
        """获取响应解析器（懒加载）"""
        if self._parser is None:
            self._parser = self.get_response_parser()
        return self._parser

    def get_response_parser(self) -> ResponseParser:
        """
        获取格式特定的响应解析器

        子类可覆盖此方法提供自定义解析器，
        默认从解析器注册表获取
        """
        return get_parser_for_format(self.FORMAT_ID)

    # _update_usage_to_streaming 方法已移至基类 BaseMessageHandler
