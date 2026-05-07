"""License status and activation endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database import get_db
from src.services.license import LicenseService

router = APIRouter(prefix="/api/license", tags=["License"])


class LicenseActivateRequest(BaseModel):
    license: dict[str, Any] | str = Field(..., description="签名许可证 JSON 或 JSON 字符串")


@router.get("/status")
async def get_license_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    return LicenseService.get_status(db).model_dump()


@router.get("/machine")
async def get_license_machine_binding() -> dict[str, Any]:
    return LicenseService.get_machine_binding()


@router.post("/activate")
async def activate_license(
    request: LicenseActivateRequest,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    try:
        status = LicenseService.activate(db, request.license)
    except ValueError as exc:
        detail = "license_update_unavailable" if str(exc) == "license_env_managed" else "license_invalid"
        raise HTTPException(status_code=400, detail=detail) from exc
    return status.model_dump()


@router.delete("/activate")
async def deactivate_license(db: Session = Depends(get_db)) -> dict[str, Any]:
    try:
        status = LicenseService.deactivate(db)
    except ValueError as exc:
        detail = "license_update_unavailable" if str(exc) == "license_env_managed" else "license_update_failed"
        raise HTTPException(status_code=400, detail=detail) from exc
    return status.model_dump()
