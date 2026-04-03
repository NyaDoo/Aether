"""OAuth API 路由聚合。"""

from fastapi import APIRouter

from src.api.oauth.admin import router as admin_router
from src.api.oauth.public import router as public_router
from src.api.oauth.user import router as user_router

router = APIRouter()
router.include_router(public_router)
router.include_router(user_router)
router.include_router(admin_router)

__all__ = ["router"]
