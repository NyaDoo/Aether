from fastapi import HTTPException

from src.api.base.context import ApiRequestContext

from .adapter import ApiAdapter, ApiMode


class AuthenticatedApiAdapter(ApiAdapter):
    """通用需要登录的适配器基类。"""

    mode = ApiMode.USER

    def authorize(self, context: ApiRequestContext) -> None:  # type: ignore[override]
        if not context.user:
            raise HTTPException(status_code=401, detail="未登录")
