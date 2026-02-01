"""
CandidateResolver facade import.

Phase2 keeps the implementation in `services/orchestration/` for compatibility,
and gradually migrates it into this package.
"""

from src.services.orchestration.candidate_resolver import CandidateResolver

__all__ = ["CandidateResolver"]
