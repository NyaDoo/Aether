"""Management Token 管理员路由模块"""

from fastapi import APIRouter

from .routes import router as management_tokens_router

router = APIRouter()
router.include_router(management_tokens_router)

__all__ = ["router"]
