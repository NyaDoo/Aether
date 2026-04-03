"""Antigravity URL 可用性管理（带 TTL 自动恢复）。"""

from __future__ import annotations

import threading
import time

from src.services.provider.adapters.antigravity.constants import (
    DAILY_BASE_URL,
    PROD_BASE_URL,
    SANDBOX_BASE_URL,
    URL_UNAVAILABLE_TTL_SECONDS,
)


class URLAvailability:
    """管理 Antigravity API 端点可用性（进程内）。"""

    _instance: "URLAvailability | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "URLAvailability":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._unavailable: dict[str, float] = {}  # url -> recover_at(ts)
        self._last_success: str | None = None
        self._mu = threading.RLock()

    def _prune(self, now: float | None = None) -> None:
        now_ts = time.time() if now is None else now
        self._unavailable = {u: t for u, t in self._unavailable.items() if t > now_ts}

    def is_available(self, url: str) -> bool:
        with self._mu:
            self._prune()
            return url not in self._unavailable

    def get_ordered_urls(self, *, prefer_daily: bool = True) -> list[str]:
        """返回优先级排序的可用 URL 列表。

        - 默认 daily 优先（通常限流更宽松）
        - 最近成功的 URL 会被提升到最前
        - 若全部被标记不可用，则返回 base_order（允许继续尝试，等待 TTL 自动恢复）
        """
        with self._mu:
            self._prune()

            # Antigravity-Manager 顺序：sandbox → daily → prod
            base_order = (
                [SANDBOX_BASE_URL, DAILY_BASE_URL, PROD_BASE_URL]
                if prefer_daily
                else [PROD_BASE_URL, SANDBOX_BASE_URL, DAILY_BASE_URL]
            )

            if self._last_success and self._last_success in base_order:
                base_order.remove(self._last_success)
                base_order.insert(0, self._last_success)

            available = [u for u in base_order if u not in self._unavailable]
            return available if available else base_order

    def mark_success(self, url: str) -> None:
        with self._mu:
            self._last_success = url
            self._unavailable.pop(url, None)

    def mark_unavailable(self, url: str) -> None:
        with self._mu:
            self._unavailable[url] = time.time() + URL_UNAVAILABLE_TTL_SECONDS
            if self._last_success == url:
                self._last_success = None


url_availability = URLAvailability()

__all__ = ["URLAvailability", "url_availability"]
