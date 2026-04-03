"""Routes for authenticated user self-service APIs."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .routes import router as me_router

_RUST_OWNED_USER_ME_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/users/me"),
        ("PUT", "/api/users/me"),
        ("PATCH", "/api/users/me/password"),
        ("GET", "/api/users/me/sessions"),
        ("DELETE", "/api/users/me/sessions/others"),
        ("PATCH", "/api/users/me/sessions/{session_id}"),
        ("DELETE", "/api/users/me/sessions/{session_id}"),
        ("GET", "/api/users/me/api-keys"),
        ("POST", "/api/users/me/api-keys"),
        ("GET", "/api/users/me/api-keys/{key_id}"),
        ("DELETE", "/api/users/me/api-keys/{key_id}"),
        ("PUT", "/api/users/me/api-keys/{key_id}"),
        ("PATCH", "/api/users/me/api-keys/{key_id}"),
        ("GET", "/api/users/me/usage"),
        ("GET", "/api/users/me/usage/active"),
        ("GET", "/api/users/me/usage/interval-timeline"),
        ("GET", "/api/users/me/usage/heatmap"),
        ("GET", "/api/users/me/providers"),
        ("GET", "/api/users/me/available-models"),
        ("GET", "/api/users/me/endpoint-status"),
        ("PUT", "/api/users/me/api-keys/{api_key_id}/providers"),
        ("PUT", "/api/users/me/api-keys/{api_key_id}/capabilities"),
        ("GET", "/api/users/me/preferences"),
        ("PUT", "/api/users/me/preferences"),
        ("GET", "/api/users/me/model-capabilities"),
        ("PUT", "/api/users/me/model-capabilities"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_USER_ME_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_user_me_router() -> APIRouter:
    router = APIRouter()
    router.include_router(me_router)
    router.routes = [route for route in router.routes if not _route_is_rust_owned(route)]
    return router


python_user_me_router = _build_python_user_me_router()
router = python_user_me_router

# 注意：management_tokens_router 已迁移到模块系统，由 ModuleRegistry 动态注册
# 当 MANAGEMENT_TOKENS_AVAILABLE=true 时注册

__all__ = ["python_user_me_router", "router"]
