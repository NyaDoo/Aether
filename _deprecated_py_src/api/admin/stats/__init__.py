"""Stats admin routes export."""

from fastapi import APIRouter

from .comparison import router as comparison_router
from .cost import router as cost_router
from .errors import router as errors_router
from .leaderboard import router as leaderboard_router
from .performance import router as performance_router
from .quota import router as quota_router
from .time_series import router as time_series_router

router = APIRouter(prefix="/api/admin/stats", tags=["Admin - Stats"])
router.include_router(leaderboard_router)
router.include_router(time_series_router)
router.include_router(cost_router)
router.include_router(quota_router)
router.include_router(performance_router)
router.include_router(errors_router)
router.include_router(comparison_router)

__all__ = ["router"]
