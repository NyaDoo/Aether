"""Admin monitoring router合集。"""

from fastapi import APIRouter

from .audit import router as audit_router
from .cache import router as cache_router
from .trace import router as trace_router

router = APIRouter()
router.include_router(audit_router)
router.include_router(cache_router)
router.include_router(trace_router)

__all__ = ["router"]
