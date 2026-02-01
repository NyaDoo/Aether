from __future__ import annotations

from typing import Any, Protocol

from sqlalchemy.orm import Session

from src.services.cache.aware_scheduler import ProviderCandidate

from .policy import RetryPolicy, SkipPolicy
from .schema import CandidateKey, CandidateResult


class AttemptFunc(Protocol):
    async def __call__(self, candidate: ProviderCandidate) -> Any: ...


class FailoverEngine:
    """
    FailoverEngine executes candidate attempts under policies.

    Phase2 scaffolding: implementation will gradually replace legacy orchestrators.
    """

    def __init__(self, db: Session):
        self.db = db

    async def execute(
        self,
        *,
        candidates: list[ProviderCandidate],
        attempt_func: AttemptFunc,
        retry_policy: RetryPolicy,
        skip_policy: SkipPolicy,
        request_id: str | None = None,
        max_candidates: int | None = None,
    ) -> CandidateResult:
        # NOTE: intentionally minimal for now; legacy orchestrators still in use.
        # This will be implemented when migrating video/chat flows to CandidateService.
        _ = (retry_policy, skip_policy, request_id, max_candidates)
        candidate_keys: list[CandidateKey] = []
        for idx, cand in enumerate(candidates):
            candidate_keys.append(
                CandidateKey(
                    candidate_index=idx,
                    provider_id=str(cand.provider.id),
                    provider_name=str(cand.provider.name),
                    endpoint_id=str(cand.endpoint.id),
                    key_id=str(cand.key.id),
                    key_name=str(getattr(cand.key, "name", "") or ""),
                    auth_type=str(getattr(cand.key, "auth_type", "") or ""),
                    priority=int(getattr(cand.key, "priority", 0) or 0),
                    is_cached=bool(getattr(cand, "is_cached", False)),
                    status="available",
                )
            )

        raise NotImplementedError("FailoverEngine.execute is not implemented yet")
