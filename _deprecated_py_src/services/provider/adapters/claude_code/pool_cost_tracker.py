"""Backward compat shim -- canonical definitions moved to src.services.provider.pool.cost_tracker."""

from src.services.provider.pool.cost_tracker import (  # noqa: F401
    get_window_usage,
    is_approaching_limit,
    is_at_limit,
    record_usage,
)

__all__ = ["record_usage", "get_window_usage", "is_at_limit", "is_approaching_limit"]
