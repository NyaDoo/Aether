"""Middleware enforcing license-gated access to real APIs."""

from __future__ import annotations

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from src.core.logger import logger
from src.database.database import create_session
from src.services.license import LicenseService


class LicenseMiddleware:
    """Force unlicensed instances into demo-only behavior at the API boundary."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        if self._is_allowed_without_license(request):
            await self.app(scope, receive, send)
            return

        db = create_session()
        try:
            status = LicenseService.get_status(db)
        except Exception as exc:
            logger.warning("许可证状态检查失败: {}", exc)
            status = None
        finally:
            db.close()

        if status is not None and status.licensed:
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            status_code=403,
            content={
                "detail": "license_required",
                "code": "license_required",
                "message": "当前实例未授权，仅允许演示模式",
                "license": status.model_dump() if status is not None else None,
            },
            headers={
                "x-aether-license-mode": (
                    "licensed" if status is not None and status.licensed else "unlicensed"
                )
            },
        )
        await response(scope, receive, send)

    @staticmethod
    def _is_allowed_without_license(request: Request) -> bool:
        method = request.method.upper()
        path = request.url.path

        if method == "OPTIONS":
            return True

        if not path.startswith("/api/"):
            return True

        exact_allowed = {
            "/api/license/status": {"GET"},
            "/api/license/machine": {"GET"},
            "/api/license/activate": {"POST", "DELETE"},
            "/api/auth/settings": {"GET"},
            "/api/auth/registration-settings": {"GET"},
            "/api/public/site-info": {"GET"},
        }
        allowed_methods = exact_allowed.get(path)
        if allowed_methods and method in allowed_methods:
            return True

        return False
