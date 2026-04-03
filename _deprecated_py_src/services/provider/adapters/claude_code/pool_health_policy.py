"""Backward compat shim -- canonical definitions moved to src.services.provider.pool.health_policy."""

from src.services.provider.pool.health_policy import *  # noqa: F401,F403
from src.services.provider.pool.health_policy import apply_health_policy  # noqa: F811

__all__ = ["apply_health_policy"]
