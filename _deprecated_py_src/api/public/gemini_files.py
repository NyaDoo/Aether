"""
Gemini Files API 代理端点

代理 Google Gemini Files API，支持文件的上传、查询、删除等操作。

端点列表：
- POST /upload/v1beta/files - 上传文件（可恢复上传）
- GET /v1beta/files - 列出文件
- GET /v1beta/files/{name} - 获取文件元数据
- DELETE /v1beta/files/{name} - 删除文件

认证方式：
- x-goog-api-key 请求头
- ?key= URL 参数

参考文档：
https://ai.google.dev/api/files

优化：HTTP 代理请求期间不持有数据库连接，避免阻塞其他请求。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.core.api_format import get_auth_handler, get_default_auth_method_for_endpoint
from src.core.crypto import crypto_service
from src.core.logger import logger
from src.database import create_session, get_db
from src.models.database import ApiKey, GlobalModel, Model, Provider, ProviderEndpoint, User
from src.services.auth.service import AuthService
from src.services.gemini_files_mapping import delete_file_key_mapping, store_file_key_mapping
from src.services.provider.provider_context import resolve_provider_proxy
from src.services.provider.transport import redact_url_for_log
from src.services.scheduling.aware_scheduler import CacheAwareScheduler, ProviderCandidate
from src.services.usage.service import UsageService


@dataclass
class UpstreamContext:
    """上游请求上下文（不依赖数据库会话）"""

    upstream_key: str
    base_url: str
    file_key_id: str
    user_id: str
    provider_id: str
    endpoint_id: str
    provider_proxy: dict[str, Any] | None = None
    key_proxy: dict[str, Any] | None = None
    proxy_config: dict[str, Any] | None = None
    delegate_config: dict[str, Any] | None = None
    proxy_snapshot: Any = None


router = APIRouter(tags=["Gemini Files API"])
pipeline = get_pipeline()


class PublicGeminiFilesApiAdapter(ApiAdapter):
    mode = ApiMode.PUBLIC

    def authorize(self, context: ApiRequestContext) -> None:  # type: ignore[override]
        return None

# Gemini Files API 基础 URL
GEMINI_FILES_BASE_URL = "https://generativelanguage.googleapis.com"

# Gemini Files API 无能力限制（任何 Gemini key 都可用）

# 需要从客户端请求中移除的头部（这些会由代理重新设置或不应转发）
HEADERS_TO_REMOVE = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "x-goog-api-key",
        "authorization",
    }
)


def _extract_gemini_api_key(request: Request) -> str | None:
    """
    从请求中提取 Gemini API Key

    优先级（与 Google SDK 行为一致）：
    1. URL 参数 ?key=
    2. x-goog-api-key 请求头
    """
    auth_method = get_default_auth_method_for_endpoint("gemini:chat")
    handler = get_auth_handler(auth_method)
    return handler.extract_credentials(request)


def _build_upstream_headers(
    original_headers: dict[str, str],
    upstream_api_key: str,
) -> dict[str, str]:
    """
    构建上游请求头

    Args:
        original_headers: 原始请求头
        upstream_api_key: 上游 API Key

    Returns:
        处理后的请求头字典
    """
    headers = {}

    # 透传非敏感头部
    for name, value in original_headers.items():
        if name.lower() not in HEADERS_TO_REMOVE:
            headers[name] = value

    # 设置认证头
    headers["x-goog-api-key"] = upstream_api_key

    return headers


def _ensure_balance_access(db: Session, user: User, api_key: ApiKey) -> None:
    access_ok, message = UsageService.check_request_balance(db, user, api_key=api_key)
    if access_ok:
        return
    raise HTTPException(
        status_code=429,
        detail={
            "error": {
                "code": 429,
                "message": message or "Insufficient balance",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def _build_upstream_url(
    base_url: str,
    path: str,
    query_params: dict[str, Any] | None = None,
    is_upload: bool = False,
) -> str:
    """
    构建上游 URL

    Args:
        base_url: 上游基础 URL
        path: API 路径
        query_params: 查询参数
        is_upload: 是否为上传端点

    Returns:
        完整的上游 URL
    """
    # 移除 key 参数（认证通过 header）
    effective_params = dict(query_params) if query_params else {}
    effective_params.pop("key", None)

    # 处理 base_url 可能包含 /v1beta 的情况，避免重复
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1beta"):
        normalized_base_url = normalized_base_url[: -len("/v1beta")]

    # 上传端点使用不同的路径前缀
    if is_upload:
        url = f"{normalized_base_url}/upload{path}"
    else:
        url = f"{normalized_base_url}{path}"

    if effective_params:
        query_string = urlencode(effective_params, doseq=True)
        url = f"{url}?{query_string}"

    return url


def _resolve_files_model_name(
    db: Session,
    user_api_key: ApiKey,
    user: User | None,
) -> str | None:
    """
    为 Files API 选择一个可用的模型名（用于 Key 选择与权限过滤）

    选择顺序:
    1. 用户/Key 的 allowed_models（取交集后选第一个）
    2. 任意支持 Gemini 格式的 GlobalModel
    """
    from src.core.model_permissions import merge_allowed_models

    allowed_models = merge_allowed_models(
        user_api_key.allowed_models,
        user.allowed_models if user else None,
    )
    if allowed_models is not None:
        if not allowed_models:
            return None
        return sorted(allowed_models)[0]

    row = (
        db.query(GlobalModel.name)
        .join(Model, Model.global_model_id == GlobalModel.id)
        .join(Provider, Provider.id == Model.provider_id)
        .join(ProviderEndpoint, ProviderEndpoint.provider_id == Provider.id)
        .filter(
            GlobalModel.is_active == True,
            Model.is_active == True,
            Provider.is_active == True,
            ProviderEndpoint.is_active == True,
            ProviderEndpoint.api_family == "gemini",
        )
        .distinct()
        .order_by(GlobalModel.name.asc())
        .first()
    )
    return row[0] if row else None


async def _select_provider_candidate(
    db: Session,
    user_api_key: ApiKey,
    model_name: str,
    require_files_capability: bool = True,
) -> ProviderCandidate | None:
    """
    选择可用的 Provider/Endpoint/Key 组合

    Args:
        db: 数据库会话
        user_api_key: 用户 API Key
        model_name: 模型名称
        require_files_capability: 是否要求 gemini_files 能力（默认 True）

    Returns:
        匹配的候选，如果没有则返回 None
    """
    scheduler = CacheAwareScheduler()

    # 要求 gemini_files 能力：只有 Google 官方 API 才支持 Files API
    capability_requirements = {"gemini_files": True} if require_files_capability else None

    candidates, _global_model_id, _provider_count = await scheduler.list_all_candidates(
        db=db,
        api_format="gemini:chat",
        model_name=model_name,
        affinity_key=str(user_api_key.id),
        user_api_key=user_api_key,
        max_candidates=10,
        capability_requirements=capability_requirements,
    )
    for candidate in candidates:
        auth_type = getattr(candidate.key, "auth_type", "api_key") or "api_key"
        if auth_type == "api_key":
            return candidate
    return None


async def _resolve_upstream_context(
    request: Request,
    db: Session,
) -> UpstreamContext:
    """
    解析上游 Key 与 Base URL（需要外部提供 db session）

    仅允许系统 API Key，选择可用的 Gemini Provider Key（无能力限制）。

    Args:
        request: HTTP 请求
        db: 数据库会话

    Returns:
        UpstreamContext
    """

    client_key = _extract_gemini_api_key(request)
    if not client_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": 401,
                    "message": "API key required. Provide via x-goog-api-key header or ?key= parameter.",
                    "status": "UNAUTHENTICATED",
                }
            },
        )

    auth_result = AuthService.authenticate_api_key(db, client_key)
    if not auth_result:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": 401,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "UNAUTHENTICATED",
                }
            },
        )

    user, user_api_key = auth_result
    _ensure_balance_access(db, user, user_api_key)
    model_name = _resolve_files_model_name(db, user_api_key, user)
    if not model_name:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "No available model for Gemini Files API routing",
                    "status": "UNAVAILABLE",
                }
            },
        )

    # 选择可用的 provider candidate（要求 gemini_files 能力）
    candidate = await _select_provider_candidate(
        db, user_api_key, model_name, require_files_capability=True
    )

    if not candidate:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "No available Gemini key with 'gemini_files' capability. "
                    "Please ensure at least one Provider Key has the 'gemini_files' capability enabled.",
                    "status": "UNAVAILABLE",
                }
            },
        )

    try:
        upstream_key = crypto_service.decrypt(candidate.key.api_key)
    except Exception as exc:
        logger.error("Failed to decrypt provider key for Gemini Files API: {}", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to decrypt provider key",
                    "status": "INTERNAL",
                }
            },
        )

    base_url = candidate.endpoint.base_url or GEMINI_FILES_BASE_URL
    return UpstreamContext(
        upstream_key,
        base_url,
        str(candidate.key.id),
        str(user.id),
        str(candidate.provider.id),
        str(candidate.endpoint.id),
        provider_proxy=resolve_provider_proxy(endpoint=candidate.endpoint, key=candidate.key),
        key_proxy=(
            candidate.key.proxy if isinstance(getattr(candidate.key, "proxy", None), dict) else None
        ),
    )


async def _enrich_upstream_context_proxy(ctx: UpstreamContext) -> UpstreamContext:
    from src.services.proxy_node.resolver import (
        build_proxy_url_async,
        get_system_proxy_config_async,
        resolve_delegate_config_async,
        resolve_effective_proxy,
        resolve_proxy_info_async,
    )
    from src.services.request.execution_runtime_plan import ExecutionProxySnapshot

    effective_proxy = resolve_effective_proxy(ctx.provider_proxy, ctx.key_proxy)
    if not effective_proxy or not effective_proxy.get("enabled", True):
        effective_proxy = await get_system_proxy_config_async()

    delegate_cfg = await resolve_delegate_config_async(effective_proxy)
    is_tunnel_delegate = bool(delegate_cfg and delegate_cfg.get("tunnel"))

    proxy_url: str | None = None
    if effective_proxy and not is_tunnel_delegate:
        proxy_url = await build_proxy_url_async(effective_proxy)

    proxy_info = await resolve_proxy_info_async(effective_proxy)
    ctx.proxy_config = effective_proxy
    ctx.delegate_config = delegate_cfg
    ctx.proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
        proxy_info,
        proxy_url=proxy_url,
        mode_override="tunnel" if is_tunnel_delegate else None,
        node_id_override=(
            str(delegate_cfg.get("node_id") or "").strip() or None if is_tunnel_delegate else None
        ),
    )
    return ctx


async def _resolve_upstream_context_standalone(request: Request) -> UpstreamContext:
    """
    解析上游上下文（自管理数据库连接，适用于 HTTP 代理场景）

    优化：在返回上下文后立即释放数据库连接，HTTP 请求期间不持有连接。

    Args:
        request: HTTP 请求

    Returns:
        UpstreamContext: 包含所有必要信息的上下文对象
    """
    with create_session() as db:
        ctx = await _resolve_upstream_context(request, db)
    return await _enrich_upstream_context_proxy(ctx)


def _build_response_headers(headers: dict[str, str]) -> dict[str, str]:
    response_headers = {}
    hop_by_hop = {"connection", "keep-alive", "transfer-encoding", "upgrade"}
    for name, value in headers.items():
        if name.lower() not in hop_by_hop:
            response_headers[name] = value
    return response_headers


def _build_rust_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": 503,
                "message": "Gemini Files requires Rust executor and is currently unavailable",
                "status": "UNAVAILABLE",
            }
        },
    )


async def _maybe_store_file_mapping_from_payload(
    *,
    status_code: int,
    headers: dict[str, str],
    content_bytes: bytes,
    file_key_id: str | None = None,
    user_id: str | None = None,
) -> None:
    if (
        not file_key_id
        or status_code >= 300
        or not headers.get("content-type", "").startswith("application/json")
    ):
        return

    try:
        payload = json.loads(content_bytes)
        file_name = None
        file_obj = None

        if isinstance(payload, dict):
            file_name = payload.get("name")
            file_obj = payload

            if not file_name and isinstance(payload.get("file"), dict):
                file_name = payload["file"].get("name")
                file_obj = payload["file"]

            if file_name and file_obj:
                display_name = file_obj.get("displayName") or file_obj.get("display_name")
                mime_type = file_obj.get("mimeType") or file_obj.get("mime_type")
                await store_file_key_mapping(
                    file_name,
                    file_key_id,
                    user_id=user_id,
                    display_name=display_name,
                    mime_type=mime_type,
                )
                logger.debug(f"Gemini file→key 映射已存储: {file_name} → key_id={file_key_id}")

            files_list = payload.get("files")
            if isinstance(files_list, list):
                mapped_count = 0
                for item in files_list:
                    if isinstance(item, dict) and item.get("name"):
                        item_display_name = item.get("displayName") or item.get("display_name")
                        item_mime_type = item.get("mimeType") or item.get("mime_type")
                        await store_file_key_mapping(
                            item["name"],
                            file_key_id,
                            user_id=user_id,
                            display_name=item_display_name,
                            mime_type=item_mime_type,
                        )
                        mapped_count += 1
                if mapped_count > 0:
                    logger.debug(
                        "Gemini list_files 批量映射已存储: {} 个文件 → key_id={}",
                        mapped_count,
                        file_key_id,
                    )
    except (ValueError, KeyError) as e:
        logger.debug("Failed to store Gemini file mapping: {}", e)


async def _try_rust_sync_proxy_request(
    method: str,
    upstream_url: str,
    headers: dict[str, str],
    *,
    content: bytes | None = None,
    json_body: dict[str, Any] | None = None,
    file_key_id: str | None = None,
    user_id: str | None = None,
    provider_id: str = "",
    endpoint_id: str = "",
    proxy: Any = None,
) -> Response:
    from src.config.settings import config
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
    )
    from src.services.request.execution_runtime_client import (
        ExecutionRuntimeClient,
        ExecutionRuntimeClientError,
    )

    if config.execution_runtime_backend != "rust":
        logger.error(
            "Gemini Files Rust sync proxy unavailable, rejecting request: {} {}",
            method.upper(),
            redact_url_for_log(upstream_url),
        )
        return _build_rust_unavailable_response()

    request_headers = dict(headers)
    content_type = str(request_headers.get("content-type") or "").strip() or None
    request_body = json_body if json_body is not None else content

    try:
        result = await ExecutionRuntimeClient().execute_sync_json(
            ExecutionPlan(
                request_id=f"gemini-files-{uuid4().hex}",
                candidate_id=None,
                provider_name="gemini",
                provider_id=provider_id,
                endpoint_id=endpoint_id,
                key_id=str(file_key_id or ""),
                method=method.upper(),
                url=upstream_url,
                headers=request_headers,
                body=build_execution_plan_body(request_body, content_type=content_type),
                stream=False,
                provider_api_format="gemini:files",
                client_api_format="gemini:files",
                model_name="gemini-files",
                proxy=proxy,
                content_type=content_type,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=30_000,
                    read_ms=300_000,
                    write_ms=300_000,
                    pool_ms=30_000,
                    total_ms=300_000,
                ),
            )
        )
    except (ExecutionRuntimeClientError, HTTPException, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Gemini Files Rust proxy unavailable: {}", redact_url_for_log(str(exc)))
        return _build_rust_unavailable_response()

    response_headers = _build_response_headers(dict(result.headers))
    if result.response_json is not None:
        response_headers.setdefault("content-type", "application/json")
        response_body = json.dumps(result.response_json, ensure_ascii=False).encode("utf-8")
    elif result.response_body_bytes is not None:
        response_body = result.response_body_bytes
    else:
        response_body = b""

    await _maybe_store_file_mapping_from_payload(
        status_code=result.status_code,
        headers=response_headers,
        content_bytes=response_body,
        file_key_id=file_key_id,
        user_id=user_id,
    )

    return Response(
        content=response_body,
        status_code=result.status_code,
        headers=response_headers,
        media_type=response_headers.get("content-type", "application/json"),
    )


async def _proxy_request(
    method: str,
    upstream_url: str,
    headers: dict[str, str],
    content: bytes | None = None,
    json_body: dict[str, Any] | None = None,
    file_key_id: str | None = None,
    user_id: str | None = None,
    provider_id: str = "",
    endpoint_id: str = "",
    proxy: Any = None,
    proxy_config: dict[str, Any] | None = None,
    delegate_config: dict[str, Any] | None = None,
) -> Response:
    """
    代理请求到上游 Gemini API

    Args:
        method: HTTP 方法
        upstream_url: 上游 URL
        headers: 请求头
        content: 原始请求体（二进制）
        json_body: JSON 请求体
        file_key_id: 上游 Provider Key ID，用于成功响应时存储 file→key 映射
        user_id: 用户 ID，用于文件映射的权限验证

    Returns:
        FastAPI Response 对象
    """
    response = await _try_rust_sync_proxy_request(
        method,
        upstream_url,
        headers,
        content=content,
        json_body=json_body,
        file_key_id=file_key_id,
        user_id=user_id,
        provider_id=provider_id,
        endpoint_id=endpoint_id,
        proxy=proxy,
    )
    if response is None:
        return _build_rust_unavailable_response()
    return response


# ==============================================================================
# 文件上传端点
# ==============================================================================


async def _upload_file_response(request: Request) -> Any:
    """
    上传文件到 Gemini Files API

    支持可恢复上传协议（Resumable Upload Protocol）：
    1. 初始请求：设置元数据，获取上传 URL
    2. 上传请求：上传实际文件内容

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **请求头（可恢复上传）**:
    - `X-Goog-Upload-Protocol: resumable`
    - `X-Goog-Upload-Command: start` | `upload, finalize`
    - `X-Goog-Upload-Header-Content-Length`: 文件大小
    - `X-Goog-Upload-Header-Content-Type`: 文件 MIME 类型

    **请求体（初始请求）**:
    ```json
    {
        "file": {
            "display_name": "文件名"
        }
    }
    ```

    优化：HTTP 代理期间不持有数据库连接
    """
    # 阶段 1：解析上下文（短暂持有数据库连接）
    ctx = await _resolve_upstream_context_standalone(request)

    # 阶段 2：读取请求体
    body = await request.body()

    # 阶段 3：代理请求（不持有数据库连接）
    upstream_url = _build_upstream_url(
        ctx.base_url,
        "/v1beta/files",
        dict(request.query_params),
        is_upload=True,
    )

    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)

    logger.debug("Gemini Files upload proxy: POST {}", redact_url_for_log(upstream_url))

    return await _proxy_request(
        "POST",
        upstream_url,
        headers,
        content=body,
        file_key_id=ctx.file_key_id,
        user_id=ctx.user_id,
        provider_id=ctx.provider_id,
        endpoint_id=ctx.endpoint_id,
        proxy=ctx.proxy_snapshot,
        proxy_config=ctx.proxy_config,
        delegate_config=ctx.delegate_config,
    )


# ==============================================================================
# 文件列表端点
# ==============================================================================


async def _list_files_response(
    request: Request,
    *,
    page_size: int | None,
    page_token: str | None,
) -> Any:
    """
    列出已上传的文件

    优化：HTTP 代理期间不持有数据库连接

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **查询参数**:
    - `pageSize`: 每页返回的文件数量（默认 10，最大 100）
    - `pageToken`: 分页令牌

    **响应格式**:
    ```json
    {
        "files": [
            {
                "name": "files/abc-123",
                "displayName": "文件名",
                "mimeType": "image/jpeg",
                "sizeBytes": "12345",
                "createTime": "2024-01-01T00:00:00Z",
                "updateTime": "2024-01-01T00:00:00Z",
                "expirationTime": "2024-01-03T00:00:00Z",
                "sha256Hash": "...",
                "uri": "https://...",
                "state": "ACTIVE"
            }
        ],
        "nextPageToken": "..."
    }
    ```
    """
    # 阶段 1：解析上下文（短暂持有数据库连接）
    ctx = await _resolve_upstream_context_standalone(request)

    # 阶段 2：代理请求（不持有数据库连接）
    query_params = dict(request.query_params)
    if page_size is not None:
        query_params["pageSize"] = page_size
    if page_token is not None:
        query_params["pageToken"] = page_token

    upstream_url = _build_upstream_url(ctx.base_url, "/v1beta/files", query_params)
    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)

    logger.debug("Gemini Files list proxy: GET {}", redact_url_for_log(upstream_url))

    return await _proxy_request(
        "GET",
        upstream_url,
        headers,
        file_key_id=ctx.file_key_id,
        user_id=ctx.user_id,
        provider_id=ctx.provider_id,
        endpoint_id=ctx.endpoint_id,
        proxy=ctx.proxy_snapshot,
        proxy_config=ctx.proxy_config,
        delegate_config=ctx.delegate_config,
    )


class PublicGeminiFilesUploadAdapter(PublicGeminiFilesApiAdapter):
    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _upload_file_response(context.request)


@dataclass
class PublicGeminiFilesListAdapter(PublicGeminiFilesApiAdapter):
    page_size: int | None
    page_token: str | None

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _list_files_response(
            context.request,
            page_size=self.page_size,
            page_token=self.page_token,
        )


@router.post("/upload/v1beta/files")
async def upload_file(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = PublicGeminiFilesUploadAdapter()
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


@router.get("/v1beta/files")
async def list_files(
    request: Request,
    pageSize: int | None = None,
    pageToken: str | None = None,
    db: Session = Depends(get_db),
) -> Any:
    adapter = PublicGeminiFilesListAdapter(page_size=pageSize, page_token=pageToken)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


# ==============================================================================
# 下载文件内容端点（用于视频等媒体文件）
# 注意：必须在 /v1beta/files/{file_name:path} 之前注册，否则会被通配符路由捕获
# ==============================================================================


async def _find_video_task_by_id(
    db: Session, short_id: str, user_id: str
) -> tuple[str | None, str | None]:
    """
    通过短 ID 查找视频任务，返回其 provider key 和 video_url

    Args:
        db: 数据库会话
        short_id: 视频任务的短 ID（VideoTask.short_id，Gemini 风格）
        user_id: 用户 ID（用于权限验证）

    Returns:
        (upstream_key, video_url) - 如果找到任务返回 key 和 url，否则返回 (None, None)
    """
    from src.models.database import ProviderAPIKey, VideoTask

    logger.debug(
        "[Files Download] Searching video task: short_id={}, user_id={}", short_id, user_id
    )

    # 通过 short_id 查找，同时验证用户权限
    task = (
        db.query(VideoTask)
        .filter(VideoTask.short_id == short_id, VideoTask.user_id == user_id)
        .first()
    )

    if not task:
        logger.debug("[Files Download] No video task found: short_id={}", short_id)
        return None, None

    if not task.video_url:
        logger.debug("[Files Download] Task found but no video_url: short_id={}", short_id)
        return None, None

    if not task.key_id:
        logger.debug("[Files Download] Task found but no key_id: short_id={}", short_id)
        return None, task.video_url

    # 获取 provider key
    provider_key = db.query(ProviderAPIKey).filter(ProviderAPIKey.id == task.key_id).first()
    if not provider_key or not provider_key.api_key:
        logger.debug("[Files Download] Provider key not found: key_id={}", task.key_id)
        return None, task.video_url

    try:
        upstream_key = crypto_service.decrypt(provider_key.api_key)
        logger.debug("[Files Download] Found key for task: short_id={}", short_id)
        return upstream_key, task.video_url
    except Exception as e:
        logger.error("[Files Download] Failed to decrypt key: {}", e)
        return None, task.video_url


async def _download_file_response(file_id: str, request: Request) -> Any:
    """
    下载文件（官方 Gemini API 格式）

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **路径参数**:
    - `file_id`: 文件 ID
      - 以 `aev_` 开头：视频任务下载（如 `aev_sknuzqlo8sds`，Gemini 风格短 ID）
      - 其他：普通 Gemini 文件下载（透传到上游）

    **查询参数**:
    - `alt=media`: 可选，保持与官方 API 兼容

    **示例**:
    ```
    GET /v1beta/files/aev_{short_id}:download?alt=media  # 视频任务
    GET /v1beta/files/{gemini_file_id}:download?alt=media  # 普通文件
    ```

    优化：HTTP 下载期间不持有数据库连接
    """
    from src.config.settings import config
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
        ExecutionProxySnapshot,
    )
    from src.services.request.execution_runtime_client import (
        ExecutionRuntimeClient,
        ExecutionRuntimeClientError,
    )

    # ========== 阶段 1：数据库操作（短暂持有连接）==========
    client_key = _extract_gemini_api_key(request)
    if not client_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {"code": 401, "message": "API key required", "status": "UNAUTHENTICATED"}
            },
        )

    regular_file_ctx: UpstreamContext | None = None

    # 在数据库会话内完成所有查询
    with create_session() as db:
        auth_result = AuthService.authenticate_api_key(db, client_key)
        if not auth_result:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": 401,
                        "message": "API key not valid",
                        "status": "UNAUTHENTICATED",
                    }
                },
            )

        user, _user_api_key = auth_result
        _ensure_balance_access(db, user, _user_api_key)

        # 根据前缀判断处理方式
        if file_id.startswith("aev_"):
            # 视频任务下载：使用短 ID 查找
            short_id = file_id[4:]  # 去掉 "aev_" 前缀
            logger.debug("[Files Download] Video task: short_id={}, user_id={}", short_id, user.id)
            upstream_key, video_url = await _find_video_task_by_id(db, short_id, user.id)
            if not upstream_key or not video_url:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": {
                            "code": 404,
                            "message": f"Video not found or not ready: {file_id}",
                            "status": "NOT_FOUND",
                        }
                    },
                )
            upstream_url = video_url
            file_key_id = ""
            provider_id = ""
            endpoint_id = ""
        else:
            # 普通文件下载：透传到 Gemini
            try:
                regular_file_ctx = await _resolve_upstream_context(request, db)
            except HTTPException:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": {
                            "code": 404,
                            "message": f"File not found: {file_id}",
                            "status": "NOT_FOUND",
                        }
                    },
                )

    if regular_file_ctx is not None:
        ctx = await _enrich_upstream_context_proxy(regular_file_ctx)
        upstream_key = ctx.upstream_key
        file_key_id = ctx.file_key_id
        provider_id = ctx.provider_id
        endpoint_id = ctx.endpoint_id
        file_name = f"files/{file_id}" if not file_id.startswith("files/") else file_id
        upstream_url = _build_upstream_url(
            ctx.base_url,
            f"/v1beta/{file_name}:download",
            dict(request.query_params),
        )

    # ========== 阶段 2：HTTP 下载（不持有数据库连接）==========
    headers = _build_upstream_headers(dict(request.headers), upstream_key)

    logger.debug("Gemini Files download proxy: GET {}", redact_url_for_log(upstream_url))

    proxy_snapshot = None
    if regular_file_ctx is not None:
        proxy_snapshot = ctx.proxy_snapshot

    if config.execution_runtime_backend != "rust":
        logger.error(
            "Gemini Files download rejected because Rust backend is disabled: {}",
            redact_url_for_log(upstream_url),
        )
        return _build_rust_unavailable_response()

    try:
        if proxy_snapshot is None and file_id.startswith("aev_"):
            from src.services.proxy_node.resolver import (
                build_proxy_url_async,
                get_system_proxy_config_async,
                resolve_delegate_config_async,
                resolve_proxy_info_async,
            )

            system_proxy = await get_system_proxy_config_async()
            delegate_cfg = await resolve_delegate_config_async(system_proxy)
            proxy_url: str | None = None
            if system_proxy and not (delegate_cfg and delegate_cfg.get("tunnel")):
                proxy_url = await build_proxy_url_async(system_proxy)
            proxy_info = await resolve_proxy_info_async(system_proxy)
            proxy_snapshot = ExecutionProxySnapshot.from_proxy_info(
                proxy_info,
                proxy_url=proxy_url,
                mode_override="tunnel" if delegate_cfg and delegate_cfg.get("tunnel") else None,
                node_id_override=(
                    str(delegate_cfg.get("node_id") or "").strip() or None
                    if delegate_cfg and delegate_cfg.get("tunnel")
                    else None
                ),
            )
    except Exception as exc:
        logger.warning("Gemini Files download proxy snapshot build failed: {}", exc)

    try:
        rust_stream = await ExecutionRuntimeClient().execute_stream(
            ExecutionPlan(
                request_id=f"gemini-files-download-{uuid4().hex}",
                candidate_id=None,
                provider_name="gemini",
                provider_id=provider_id,
                endpoint_id=endpoint_id,
                key_id=str(file_key_id or ""),
                method="GET",
                url=upstream_url,
                headers=headers,
                body=ExecutionPlanBody(),
                stream=True,
                provider_api_format="gemini:files",
                client_api_format="gemini:files",
                model_name="gemini-files",
                proxy=proxy_snapshot,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=30_000,
                    read_ms=300_000,
                    write_ms=300_000,
                    pool_ms=30_000,
                    total_ms=None,
                ),
            )
        )
    except (ExecutionRuntimeClientError, ValueError) as exc:
        logger.warning("Gemini Files Rust download unavailable: {}", exc)
        return _build_rust_unavailable_response()

    safe_headers = _build_response_headers(dict(rust_stream.headers))
    if rust_stream.status_code >= 400:
        error_chunks: list[bytes] = []
        try:
            async for chunk in rust_stream.byte_iterator:
                if chunk:
                    error_chunks.append(chunk)
                if sum(len(item) for item in error_chunks) >= 16_384:
                    break
        finally:
            await rust_stream.response_ctx.__aexit__(None, None, None)

        raw_error = b"".join(error_chunks)
        if safe_headers.get("content-type", "").startswith("application/json"):
            try:
                return JSONResponse(
                    content=json.loads(raw_error.decode("utf-8")),
                    status_code=rust_stream.status_code,
                )
            except Exception:
                pass
        return JSONResponse(
            content={"error": raw_error.decode("utf-8", errors="replace")},
            status_code=rust_stream.status_code,
        )

    async def _iter_bytes() -> AsyncIterator[bytes]:
        try:
            async for chunk in rust_stream.byte_iterator:
                yield chunk
        finally:
            await rust_stream.response_ctx.__aexit__(None, None, None)

    return StreamingResponse(
        _iter_bytes(),
        status_code=rust_stream.status_code,
        headers=safe_headers,
        media_type=safe_headers.get("content-type", "application/octet-stream"),
    )


@dataclass
class PublicGeminiFilesDownloadAdapter(PublicGeminiFilesApiAdapter):
    file_id: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _download_file_response(self.file_id, context.request)


@router.get("/v1beta/files/{file_id}:download")
async def download_file(
    file_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = PublicGeminiFilesDownloadAdapter(file_id=file_id)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


# ==============================================================================
# 获取文件元数据端点
# ==============================================================================


async def _get_file_response(file_name: str, request: Request) -> Any:
    """
    获取指定文件的元数据

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **路径参数**:
    - `file_name`: 文件名（格式：files/xxx 或 xxx）

    **响应格式**:
    ```json
    {
        "name": "files/abc-123",
        "displayName": "文件名",
        "mimeType": "image/jpeg",
        "sizeBytes": "12345",
        "createTime": "2024-01-01T00:00:00Z",
        "updateTime": "2024-01-01T00:00:00Z",
        "expirationTime": "2024-01-03T00:00:00Z",
        "sha256Hash": "...",
        "uri": "https://...",
        "state": "ACTIVE"
    }
    ```

    优化：HTTP 代理期间不持有数据库连接
    """
    # 阶段 1：解析上下文（短暂持有数据库连接）
    ctx = await _resolve_upstream_context_standalone(request)

    # 阶段 2：代理请求（不持有数据库连接）
    # 规范化文件名（确保以 files/ 开头）
    if not file_name.startswith("files/"):
        file_name = f"files/{file_name}"

    upstream_url = _build_upstream_url(
        ctx.base_url,
        f"/v1beta/{file_name}",
        dict(request.query_params),
    )
    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)

    logger.debug("Gemini Files get proxy: GET {}", redact_url_for_log(upstream_url))

    return await _proxy_request(
        "GET",
        upstream_url,
        headers,
        file_key_id=ctx.file_key_id,
        user_id=ctx.user_id,
        provider_id=ctx.provider_id,
        endpoint_id=ctx.endpoint_id,
        proxy=ctx.proxy_snapshot,
        proxy_config=ctx.proxy_config,
        delegate_config=ctx.delegate_config,
    )


@dataclass
class PublicGeminiFilesGetAdapter(PublicGeminiFilesApiAdapter):
    file_name: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _get_file_response(self.file_name, context.request)


@router.get("/v1beta/files/{file_name:path}")
async def get_file(
    file_name: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = PublicGeminiFilesGetAdapter(file_name=file_name)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)


# ==============================================================================
# 删除文件端点
# ==============================================================================


async def _delete_file_response(file_name: str, request: Request) -> Any:
    """
    删除指定文件

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **路径参数**:
    - `file_name`: 文件名（格式：files/xxx 或 xxx）

    **响应格式**:
    成功时返回空 JSON 对象：`{}`

    优化：HTTP 代理期间不持有数据库连接
    """
    # 阶段 1：解析上下文（短暂持有数据库连接）
    ctx = await _resolve_upstream_context_standalone(request)

    # 阶段 2：代理请求（不持有数据库连接）
    # 规范化文件名（确保以 files/ 开头）
    if not file_name.startswith("files/"):
        file_name = f"files/{file_name}"

    upstream_url = _build_upstream_url(
        ctx.base_url,
        f"/v1beta/{file_name}",
        dict(request.query_params),
    )
    headers = _build_upstream_headers(dict(request.headers), ctx.upstream_key)

    logger.debug("Gemini Files delete proxy: DELETE {}", redact_url_for_log(upstream_url))

    response = await _proxy_request(
        "DELETE",
        upstream_url,
        headers,
        provider_id=ctx.provider_id,
        endpoint_id=ctx.endpoint_id,
        proxy=ctx.proxy_snapshot,
        proxy_config=ctx.proxy_config,
        delegate_config=ctx.delegate_config,
    )
    if response.status_code < 300:
        await delete_file_key_mapping(file_name)
    else:
        logger.debug(
            "Gemini Files delete failed, skip mapping cleanup: status={}", response.status_code
        )
    return response


@dataclass
class PublicGeminiFilesDeleteAdapter(PublicGeminiFilesApiAdapter):
    file_name: str

    async def handle(self, context: ApiRequestContext) -> Any:  # type: ignore[override]
        return await _delete_file_response(self.file_name, context.request)


@router.delete("/v1beta/files/{file_name:path}")
async def delete_file(
    file_name: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    adapter = PublicGeminiFilesDeleteAdapter(file_name=file_name)
    return await pipeline.run(adapter=adapter, http_request=request, db=db, mode=ApiMode.PUBLIC)
