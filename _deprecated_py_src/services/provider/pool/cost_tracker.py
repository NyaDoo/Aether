"""Rolling-window cost tracking for the Account Pool.

Each key has a configurable token budget per rolling window (e.g. 5 hours).
When the budget is exhausted the key is marked as unschedulable by the pool
manager.  A "soft threshold" (default 80 %) causes the pool to *prefer*
other keys but still allows traffic if no alternatives exist.

All state is stored in Redis sorted sets via :mod:`redis_ops`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.provider.pool import redis_ops

if TYPE_CHECKING:
    from src.services.provider.pool.config import PoolConfig


async def record_usage(
    provider_id: str,
    key_id: str,
    tokens: int,
    config: PoolConfig,
) -> None:
    """Record *tokens* used by *key_id* in the rolling cost window."""
    if tokens <= 0:
        return
    if config.cost_limit_per_key_tokens is None:
        return  # cost tracking disabled
    await redis_ops.add_cost_entry(provider_id, key_id, tokens, config.cost_window_seconds)


async def get_window_usage(
    provider_id: str,
    key_id: str,
    config: PoolConfig,
) -> int:
    """Return total tokens used by *key_id* within the current window."""
    return await redis_ops.get_cost_window_total(provider_id, key_id, config.cost_window_seconds)


async def is_at_limit(
    provider_id: str,
    key_id: str,
    config: PoolConfig,
) -> bool:
    """Return ``True`` if the key has exhausted its budget."""
    if config.cost_limit_per_key_tokens is None:
        return False
    total = await get_window_usage(provider_id, key_id, config)
    return total >= config.cost_limit_per_key_tokens


async def is_approaching_limit(
    provider_id: str,
    key_id: str,
    config: PoolConfig,
) -> bool:
    """Return ``True`` if the key is above the soft threshold."""
    if config.cost_limit_per_key_tokens is None:
        return False
    total = await get_window_usage(provider_id, key_id, config)
    threshold = config.cost_limit_per_key_tokens * config.cost_soft_threshold_percent / 100
    return total >= threshold
