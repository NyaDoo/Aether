from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RetryMode(str, Enum):
    """Retry mode for candidate attempts."""

    PRE_EXPAND = "pre_expand"  # pre-create retry slots (sync)
    ON_DEMAND = "on_demand"  # create retry record only when retry happens
    DISABLED = "disabled"  # no retry


@dataclass(frozen=True)
class RetryPolicy:
    """Unified retry policy."""

    mode: RetryMode = RetryMode.DISABLED
    max_retries: int = 1
    retry_on_cached_only: bool = True

    @classmethod
    def for_sync_task(cls) -> "RetryPolicy":
        return cls(mode=RetryMode.PRE_EXPAND, max_retries=2)

    @classmethod
    def for_async_task(cls) -> "RetryPolicy":
        return cls(mode=RetryMode.DISABLED, max_retries=1)

    @classmethod
    def for_async_submit_with_retry(cls) -> "RetryPolicy":
        return cls(mode=RetryMode.ON_DEMAND, max_retries=2)


@dataclass(frozen=True)
class SkipPolicy:
    """Rules for skipping unsupported candidates."""

    allow_format_conversion: bool = True
    supported_auth_types: set[str] | None = None
