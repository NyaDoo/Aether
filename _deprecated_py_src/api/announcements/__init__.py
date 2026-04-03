"""Announcement system routers."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as announcement_router

_RUST_OWNED_ANNOUNCEMENT_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/announcements"),
        ("GET", "/api/announcements/active"),
        ("POST", "/api/announcements"),
        ("GET", "/api/announcements/{announcement_id}"),
        ("PATCH", "/api/announcements/{announcement_id}/read-status"),
        ("PUT", "/api/announcements/{announcement_id}"),
        ("DELETE", "/api/announcements/{announcement_id}"),
        ("GET", "/api/announcements/users/me/unread-count"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_ANNOUNCEMENT_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_announcement_router() -> APIRouter:
    router = APIRouter()
    router.include_router(announcement_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_announcement_router = _build_python_announcement_router()
router = python_announcement_router

__all__ = ["python_announcement_router", "router"]
