"""Public-facing API routers."""

from fastapi import APIRouter

from .capabilities import router as capabilities_router
from .catalog import router as catalog_router
from .claude import router as claude_router
from .gemini import router as gemini_router
from .gemini_files import router as gemini_files_router
from .models import router as models_router
from .modules import router as modules_router
from .openai import router as openai_router
from .system_catalog import router as system_catalog_router
from .videos import router as videos_router

router = APIRouter()
# Video API 路由需要在 Models API 之前注册，因为 Models API 有 /v1beta/models/{path} 通配符路由
# 会错误匹配 /v1beta/models/{model}/operations/{id}/content 等视频路由
router.include_router(videos_router, tags=["Video Generation"])
router.include_router(models_router)
router.include_router(claude_router, tags=["Claude API"])
router.include_router(openai_router)
router.include_router(gemini_router, tags=["Gemini API"])
router.include_router(gemini_files_router, tags=["Gemini Files API"])
router.include_router(system_catalog_router, tags=["System Catalog"])
router.include_router(catalog_router)
router.include_router(capabilities_router)
router.include_router(modules_router)

__all__ = ["router"]
