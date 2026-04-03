"""调度系统（候选/排序/亲和性/并发检查）。

原 `src.services.cache` 中与调度相关的代码已迁移到此包；
`src.services.cache` 现在只保留通用缓存 backend/sync/*_cache。
"""

from src.services.scheduling.affinity_manager import CacheAffinityManager, get_affinity_manager
from src.services.scheduling.aware_scheduler import (
    CacheAwareScheduler,
    ConcurrencySnapshot,
    ProviderCandidate,
    get_cache_aware_scheduler,
)
from src.services.scheduling.candidate_builder import CandidateBuilder
from src.services.scheduling.candidate_sorter import CandidateSorter
from src.services.scheduling.concurrency_checker import ConcurrencyChecker
from src.services.scheduling.scheduling_config import SchedulingConfig

__all__ = [
    "CacheAffinityManager",
    "CandidateBuilder",
    "CandidateSorter",
    "CacheAwareScheduler",
    "ConcurrencyChecker",
    "ConcurrencySnapshot",
    "ProviderCandidate",
    "SchedulingConfig",
    "get_affinity_manager",
    "get_cache_aware_scheduler",
]
