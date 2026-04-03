"""
models.dev 外部模型数据代理
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.api.base.admin_adapter import AdminApiAdapter
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.clients import get_redis_client
from src.core.logger import logger
from src.database import get_db
from src.models.database import User
from src.utils.auth_utils import require_admin

router = APIRouter()
pipeline = get_pipeline()

CACHE_KEY = "aether:external:models_dev"
CACHE_TTL = 15 * 60  # 15 分钟

# 标记官方/一手提供商，前端可据此过滤第三方转售商
OFFICIAL_PROVIDERS = {
    "anthropic",  # Claude 官方
    "openai",  # OpenAI 官方
    "google",  # Gemini 官方
    "google-vertex",  # Google Vertex AI
    "azure",  # Azure OpenAI
    "amazon-bedrock",  # AWS Bedrock
    "xai",  # Grok 官方
    "meta",  # Llama 官方
    "deepseek",  # DeepSeek 官方
    "mistral",  # Mistral 官方
    "cohere",  # Cohere 官方
    "zhipuai",  # 智谱 AI 官方
    "alibaba",  # 阿里云（通义千问）
    "minimax",  # MiniMax 官方
    "moonshot",  # 月之暗面（Kimi）
    "baichuan",  # 百川智能
    "ai21",  # AI21 Labs
}


async def _get_cached_data() -> dict[str, Any] | None:
    """从 Redis 获取缓存数据"""
    redis = await get_redis_client()
    if redis is None:
        return None
    try:
        cached = await redis.get(CACHE_KEY)
        if cached:
            result: dict[str, Any] = json.loads(cached)
            return result
    except Exception as e:
        logger.warning(f"读取 models.dev 缓存失败: {e}")
    return None


async def _set_cached_data(data: dict) -> None:
    """将数据写入 Redis 缓存"""
    redis = await get_redis_client()
    if redis is None:
        return
    try:
        await redis.setex(CACHE_KEY, CACHE_TTL, json.dumps(data, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"写入 models.dev 缓存失败: {e}")


def _mark_official_providers(data: dict[str, Any]) -> dict[str, Any]:
    """为每个提供商标记是否为官方"""
    result = {}
    for provider_id, provider_data in data.items():
        result[provider_id] = {
            **provider_data,
            "official": provider_id in OFFICIAL_PROVIDERS,
        }
    return result


async def _get_external_models_response() -> JSONResponse:
    """
    获取外部模型数据

    从 models.dev 获取第三方模型数据，用于导入新模型或参考定价信息。
    该接口作为代理请求解决跨域问题，并提供缓存优化。

    **功能特性**:
    - 代理 models.dev API，解决前端跨域问题
    - 使用 Redis 缓存 15 分钟，多 worker 共享缓存
    - 自动标记官方提供商（official 字段），前端可据此过滤第三方转售商

    **返回字段**:
    - 键为提供商 ID（如 "anthropic"、"openai"）
    - 值为提供商详细信息，包含：
      - `official`: 是否为官方提供商（true/false）
      - 其他 models.dev 提供的原始字段（模型列表、定价等）
    """
    # 检查缓存
    cached = await _get_cached_data()
    if cached is not None:
        # 兼容旧缓存：如果没有 official 字段则补全并回写
        try:
            needs_mark = False
            for provider_data in cached.values():
                if not isinstance(provider_data, dict) or "official" not in provider_data:
                    needs_mark = True
                    break
            if needs_mark:
                marked_cached = _mark_official_providers(cached)
                await _set_cached_data(marked_cached)
                return JSONResponse(content=marked_cached)
        except Exception as e:
            logger.warning(f"处理 models.dev 缓存数据失败，将直接返回原缓存: {e}")
        return JSONResponse(content=cached)

    raise HTTPException(status_code=503, detail="External models catalog requires Rust admin backend")


async def _clear_external_models_cache_response() -> dict[str, Any]:
    """
    清除外部模型数据缓存

    手动清除 models.dev 的 Redis 缓存，强制下次请求重新获取最新数据。
    通常用于需要立即更新外部模型数据的场景。

    **返回字段**:
    - `cleared`: 是否成功清除缓存（true/false）
    - `message`: 提示信息（仅在 Redis 未启用时返回）
    """
    redis = await get_redis_client()
    if redis is None:
        return {"cleared": False, "message": "Redis 未启用"}
    try:
        await redis.delete(CACHE_KEY)
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")


class ExternalModelsAdminAdapter(AdminApiAdapter):
    """models.dev 外部模型管理基类。"""


class AdminGetExternalModelsAdapter(ExternalModelsAdminAdapter):
    async def handle(self, context: ApiRequestContext) -> JSONResponse:  # type: ignore[override]
        del context
        return await _get_external_models_response()


class AdminClearExternalModelsCacheAdapter(ExternalModelsAdminAdapter):
    async def handle(self, context: ApiRequestContext) -> dict[str, Any]:  # type: ignore[override]
        del context
        return await _clear_external_models_cache_response()


@router.get("/external")
async def get_external_models(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> JSONResponse:
    adapter = AdminGetExternalModelsAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)


@router.delete("/external/cache")
async def clear_external_models_cache(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> dict[str, Any]:
    adapter = AdminClearExternalModelsCacheAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=adapter.mode)
