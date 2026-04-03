"""Public-facing API routers.

Keep the compatibility frontdoor surface explicit so Rust can take ownership of
that manifest later without re-auditing the entire Python public app shell.
"""

from __future__ import annotations

from fastapi import APIRouter

from .support import python_public_support_router

router = APIRouter()
router.include_router(python_public_support_router)

__all__ = ["frontdoor_compat_router", "python_public_support_router", "router"]


def __getattr__(name: str) -> object:
    if name == "frontdoor_compat_router":
        from .compat import frontdoor_compat_router

        return frontdoor_compat_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
