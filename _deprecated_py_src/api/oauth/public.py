"""OAuth 公开端点（无需登录）。"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from src.database import get_db

router = APIRouter(prefix="/api/oauth", tags=["OAuth"])
_OAUTH_PUBLIC_LEGACY_DETAIL = "OAuth public routes are retired; use Rust maintenance backend"


def _raise_oauth_public_legacy_unavailable() -> None:
    raise HTTPException(status_code=503, detail=_OAUTH_PUBLIC_LEGACY_DETAIL)


@router.get("/providers")
async def list_oauth_providers(db: Session = Depends(get_db)) -> dict[str, Any]:
    """
    获取可用 OAuth Providers 列表。

    模块未启用时返回空列表（前端友好）。
    """
    _ = db
    _raise_oauth_public_legacy_unavailable()


@router.get("/{provider_type}/authorize")
async def oauth_authorize(
    provider_type: str,
    client_device_id: str = Query(..., min_length=1, max_length=128),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """
    发起 OAuth 登录（login flow）。
    """
    _ = provider_type, client_device_id, db
    _raise_oauth_public_legacy_unavailable()


@router.get("/{provider_type}/callback")
async def oauth_callback(
    provider_type: str,
    request: Request,
    db: Session = Depends(get_db),
    code: str | None = Query(None),
    state: str | None = Query(None),
    error: str | None = Query(None),
    error_description: str | None = Query(None),
) -> RedirectResponse:
    """
    OAuth 回调端点。

    成功/失败都会重定向到前端回调页。
    """
    _ = provider_type, request, db, code, state, error, error_description
    _raise_oauth_public_legacy_unavailable()
