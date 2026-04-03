"""Dashboard API routers."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as dashboard_router

_RUST_OWNED_DASHBOARD_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/dashboard/stats"),
        ("GET", "/api/dashboard/recent-requests"),
        ("GET", "/api/dashboard/provider-status"),
        ("GET", "/api/dashboard/daily-stats"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_DASHBOARD_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_dashboard_router() -> APIRouter:
    router = APIRouter()
    router.include_router(dashboard_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_dashboard_router = _build_python_dashboard_router()
router = python_dashboard_router

__all__ = ["python_dashboard_router", "router"]
