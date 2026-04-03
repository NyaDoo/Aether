"""Payment API routes."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as payment_router

_RUST_OWNED_PAYMENT_ROUTE_SIGNATURES = frozenset(
    {
        ("POST", "/api/payment/callback/{payment_method}"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_PAYMENT_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_payment_router() -> APIRouter:
    router = APIRouter()
    router.include_router(payment_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_payment_router = _build_python_payment_router()
router = python_payment_router

__all__ = ["python_payment_router", "router"]
