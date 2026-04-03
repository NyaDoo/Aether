"""Pluggable pool scheduling strategies.

Strategies allow customising pool-level candidate selection without
modifying the core :class:`PoolManager`.  Each strategy is an object
that implements one or more optional methods defined by the
:class:`PoolSchedulingStrategy` protocol.

Registration uses a thread-safe global registry (same pattern as
:mod:`~src.services.provider.pool.hooks`).

Usage::

    from src.services.provider.pool.strategy import register_pool_strategy

    class MyStrategy:
        name = "usage_weight"

        def compute_score(self, *, key_id, config, context):
            ...

    register_pool_strategy("usage_weight", MyStrategy())
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.services.provider.pool.config import PoolConfig
    from src.services.provider.pool.trace import PoolCandidateTrace


@runtime_checkable
class PoolSchedulingStrategy(Protocol):
    """Pluggable pool scheduling strategy.

    All methods are optional -- callers check via ``hasattr``.
    Strategies are activated per-provider through ``PoolConfig.strategies``.
    """

    name: str

    def on_before_select(
        self,
        *,
        provider_id: str,
        key_ids: list[str],
        config: PoolConfig,
        context: dict[str, Any],
    ) -> list[str] | None:
        """Filter / reorder *key_ids* before selection.

        Return ``None`` to leave the list unchanged.
        """
        ...

    def on_after_select(
        self,
        *,
        provider_id: str,
        selected_key_id: str,
        trace: PoolCandidateTrace,
        config: PoolConfig,
        context: dict[str, Any],
    ) -> None:
        """Called after a key has been selected (for logging / metrics)."""
        ...

    def compute_score(
        self,
        *,
        key_id: str,
        config: PoolConfig,
        context: dict[str, Any],
    ) -> float | None:
        """Return a custom sort score.  ``None`` means "do not override"."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_strategy_registry: dict[str, PoolSchedulingStrategy] = {}
_strategy_lock = threading.Lock()


def register_pool_strategy(name: str, strategy: PoolSchedulingStrategy) -> None:
    """Register a pool scheduling strategy globally."""
    with _strategy_lock:
        _strategy_registry[name] = strategy


def get_pool_strategy(name: str) -> PoolSchedulingStrategy | None:
    """Return a registered strategy by *name*, or ``None``."""
    return _strategy_registry.get(name)


def get_active_strategies(names: tuple[str, ...] | list[str]) -> list[PoolSchedulingStrategy]:
    """Return registered strategies whose names appear in *names*."""
    result: list[PoolSchedulingStrategy] = []
    for n in names:
        s = _strategy_registry.get(n)
        if s is not None:
            result.append(s)
    return result
