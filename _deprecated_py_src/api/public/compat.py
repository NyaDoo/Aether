"""Rust-frontdoor-owned public compatibility route definitions."""

from fastapi import APIRouter

from .claude import router as claude_router
from .gemini import router as gemini_router
from .gemini_files import router as gemini_files_router
from .openai import router as openai_router
from .videos import router as videos_router


def build_frontdoor_compat_router() -> APIRouter:
    """Return public compat routes that Rust frontdoor owns at host level."""
    compat_router = APIRouter()

    compat_router.include_router(videos_router, tags=["Video Generation"])
    compat_router.include_router(claude_router, tags=["Claude API"])
    compat_router.include_router(openai_router)
    compat_router.include_router(gemini_router, tags=["Gemini API"])
    compat_router.include_router(gemini_files_router, tags=["Gemini Files API"])
    return compat_router


frontdoor_compat_router = build_frontdoor_compat_router()

__all__ = ["build_frontdoor_compat_router", "frontdoor_compat_router"]
