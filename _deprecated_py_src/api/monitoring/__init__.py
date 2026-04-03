"""User monitoring routers."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .user import router as monitoring_router

_RUST_OWNED_MONITORING_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/monitoring/my-audit-logs"),
        ("GET", "/api/monitoring/rate-limit-status"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_MONITORING_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_monitoring_router() -> APIRouter:
    router = APIRouter()
    router.include_router(monitoring_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_monitoring_router = _build_python_monitoring_router()
router = python_monitoring_router

__all__ = ["python_monitoring_router", "router"]
