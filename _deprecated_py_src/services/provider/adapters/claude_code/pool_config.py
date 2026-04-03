"""Backward compat shim -- canonical definitions moved to src.services.provider.pool.config."""

from src.services.provider.pool.config import *  # noqa: F401,F403
from src.services.provider.pool.config import PoolConfig, UnschedulableRule, parse_pool_config

__all__ = ["PoolConfig", "UnschedulableRule", "parse_pool_config"]
