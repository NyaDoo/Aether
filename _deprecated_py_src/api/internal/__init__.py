"""Internal routers still surfaced by the Python host."""

from fastapi import APIRouter


def _build_python_internal_router() -> APIRouter:
    """Internal APIs that still belong to the Python host/runtime."""
    return APIRouter()


# Legacy internal gateway bridge 与 internal tunnel 模块仍保留在 `src.api.internal.*`
# 里给测试与过渡逻辑复用，但 Python host 已不再公开任何 `/api/internal/*` 路由。
python_internal_router = _build_python_internal_router()
router = python_internal_router

__all__ = [
    "python_internal_router",
    "router",
]
