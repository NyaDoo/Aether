"""
安全管理 API

提供 IP 黑白名单管理等安全功能
"""

from fastapi import APIRouter

from .ip_management import router as ip_router

router = APIRouter()
router.include_router(ip_router)

__all__ = ["router"]
