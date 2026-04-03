"""公开模块状态 API（供登录页等使用）"""

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.core.modules import get_module_registry
from src.database import get_db

router = APIRouter(prefix="/api/modules", tags=["Modules"])
pipeline = get_pipeline()


class AuthModuleInfo(BaseModel):
    """认证模块简要信息"""

    name: str
    display_name: str
    active: bool


class PublicModulesApiAdapter(ApiAdapter):
    mode = ApiMode.PUBLIC

    def authorize(self, context: ApiRequestContext) -> None:  # type: ignore[override]
        return None


class PublicAuthModulesStatusAdapter(PublicModulesApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        registry = get_module_registry()
        auth_modules = registry.get_auth_modules_status(context.db)
        return [
            AuthModuleInfo(
                name=status.name,
                display_name=status.display_name,
                active=status.active,
            )
            for status in auth_modules
        ]


@router.get("/auth-status", response_model=list[AuthModuleInfo])
async def get_auth_modules_status(request: Request, db: Session = Depends(get_db)) -> Any:
    """
    获取认证模块状态（公开接口）

    供登录页使用，返回所有可用的认证模块及其激活状态。
    不需要认证即可访问。

    **返回字段**:
    - `name`: 模块名称
    - `display_name`: 显示名称
    - `active`: 是否激活
    """
    adapter = PublicAuthModulesStatusAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)
