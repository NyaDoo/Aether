from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from src.services.billing.rule_service import BillingRuleLookupResult
from src.services.cache.aware_scheduler import ProviderCandidate


@runtime_checkable
class SubmitFunc(Protocol):
    async def __call__(self, candidate: ProviderCandidate) -> httpx.Response: ...


@runtime_checkable
class ExtractExternalTaskIdFunc(Protocol):
    def __call__(self, payload: dict[str, Any]) -> str | None: ...


class UpstreamClientRequestError(RuntimeError):
    """可判定为客户端请求问题（不应 failover）的上游错误。"""

    def __init__(
        self,
        *,
        response: httpx.Response,
        candidate_keys: list[dict[str, Any]],
    ) -> None:
        self.response = response
        self.candidate_keys = candidate_keys
        super().__init__(f"Upstream client error: HTTP {response.status_code}")


class AllCandidatesFailedError(RuntimeError):
    def __init__(
        self,
        *,
        reason: str,
        candidate_keys: list[dict[str, Any]],
        last_status_code: int | None = None,
    ) -> None:
        self.reason = reason
        self.candidate_keys = candidate_keys
        self.last_status_code = last_status_code
        super().__init__(f"All candidates failed: {reason}")


class CandidateUnsupportedError(RuntimeError):
    """候选不被当前任务支持（如 auth_type/格式转换需求不支持）。"""


class CandidateSubmissionError(RuntimeError):
    """候选提交异常（网络/解密/解析等）。"""


@dataclass(slots=True)
class SubmitOutcome:
    candidate: ProviderCandidate
    candidate_keys: list[dict[str, Any]]
    external_task_id: str
    rule_lookup: BillingRuleLookupResult | None
    upstream_payload: dict[str, Any] | None = None
    upstream_headers: dict[str, str] | None = None
    upstream_status_code: int | None = None
