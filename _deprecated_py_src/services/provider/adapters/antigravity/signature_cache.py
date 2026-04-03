"""Backward-compatible re-export for Antigravity thinking signature cache.

The implementation moved to `src.core.api_format.conversion.thinking_cache` to eliminate
core → services reverse dependencies.
"""

from src.core.api_format.conversion.thinking_cache import (
    MIN_SIGNATURE_LENGTH,
    ThinkingSignatureCache,
    signature_cache,
)

__all__ = ["MIN_SIGNATURE_LENGTH", "ThinkingSignatureCache", "signature_cache"]
