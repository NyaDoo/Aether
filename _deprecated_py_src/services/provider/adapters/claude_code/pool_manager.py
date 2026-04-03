"""Backward compat shim -- canonical definitions moved to src.services.provider.pool.manager."""

from src.services.provider.pool.manager import *  # noqa: F401,F403
from src.services.provider.pool.manager import PoolManager

ClaudeCodePoolManager = PoolManager  # noqa: F811

__all__ = ["PoolManager", "ClaudeCodePoolManager"]
