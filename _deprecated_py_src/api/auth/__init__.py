"""Authentication route group."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as auth_router

_RUST_OWNED_AUTH_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/auth/registration-settings"),
        ("GET", "/api/auth/settings"),
        ("POST", "/api/auth/login"),
        ("POST", "/api/auth/refresh"),
        ("POST", "/api/auth/register"),
        ("GET", "/api/auth/me"),
        ("POST", "/api/auth/logout"),
        ("POST", "/api/auth/send-verification-code"),
        ("POST", "/api/auth/verify-email"),
        ("POST", "/api/auth/verification-status"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_AUTH_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_auth_router() -> APIRouter:
    router = APIRouter()
    router.include_router(auth_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_auth_router = _build_python_auth_router()
router = python_auth_router

__all__ = ["python_auth_router", "router"]
