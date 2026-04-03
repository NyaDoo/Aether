"""Wallet API routes."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as wallet_router

_RUST_OWNED_WALLET_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/wallet/balance"),
        ("GET", "/api/wallet/transactions"),
        ("GET", "/api/wallet/flow"),
        ("GET", "/api/wallet/today-cost"),
        ("GET", "/api/wallet/recharge"),
        ("POST", "/api/wallet/recharge"),
        ("GET", "/api/wallet/recharge/{order_id}"),
        ("GET", "/api/wallet/refunds"),
        ("POST", "/api/wallet/refunds"),
        ("GET", "/api/wallet/refunds/{refund_id}"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_WALLET_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_wallet_router() -> APIRouter:
    router = APIRouter()
    router.include_router(wallet_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_wallet_router = _build_python_wallet_router()
router = python_wallet_router

__all__ = ["python_wallet_router", "router"]
