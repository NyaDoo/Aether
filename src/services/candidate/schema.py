from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.services.cache.aware_scheduler import ProviderCandidate

CANDIDATE_KEY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CandidateKey:
    """Stable candidate key snapshot for audit."""

    schema_version: str = CANDIDATE_KEY_SCHEMA_VERSION

    candidate_index: int = 0
    retry_index: int = 0

    provider_id: str | None = None
    provider_name: str | None = None
    endpoint_id: str | None = None
    key_id: str | None = None
    key_name: str | None = None
    auth_type: str | None = None
    priority: int | None = None
    is_cached: bool = False

    status: str = "pending"  # pending/success/failed/skipped/available/...
    skip_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    status_code: int | None = None
    latency_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "candidate_index": self.candidate_index,
            "retry_index": self.retry_index,
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "endpoint_id": self.endpoint_id,
            "key_id": self.key_id,
            "key_name": self.key_name,
            "auth_type": self.auth_type,
            "priority": self.priority,
            "is_cached": self.is_cached,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
        }
        # drop Nones for compact audit payload
        return {k: v for k, v in data.items() if v is not None}


@dataclass(slots=True)
class CandidateResult:
    """Failover execution result."""

    success: bool
    selected: ProviderCandidate | None
    selected_index: int | None
    candidate_keys: list[CandidateKey]

    external_task_id: str | None = None
    error: Exception | None = None
    last_status_code: int | None = None
