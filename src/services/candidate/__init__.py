"""
Candidate domain (Phase2)

This package centralizes:
- candidate resolving (Provider/Endpoint/Key combinations)
- request_candidates recording & audit
- failover execution policies
"""

from src.services.candidate.policy import RetryMode, RetryPolicy, SkipPolicy
from src.services.candidate.schema import (
    CANDIDATE_KEY_SCHEMA_VERSION,
    CandidateKey,
    CandidateResult,
)
from src.services.candidate.service import CandidateService

__all__ = [
    "CandidateService",
    # schema
    "CANDIDATE_KEY_SCHEMA_VERSION",
    "CandidateKey",
    "CandidateResult",
    # policies
    "RetryMode",
    "RetryPolicy",
    "SkipPolicy",
]
