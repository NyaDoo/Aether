"""
Image Adapter 基类

支持两个端点：
- POST /v1/images/generations  (application/json)
- POST /v1/images/edits        (multipart/form-data)

multipart 路径：adapter 读原始字节、解析出一份 dict 用于 Codex 分支的
body transformer；同时保留原始字节和 content-type 用于透传。
"""

from __future__ import annotations

import json
import time
from typing import Any, ClassVar

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.context import ApiRequestContext
from src.api.handlers.base.image_handler_base import ImageHandlerBase
from src.core.api_format import (
    ApiFamily,
    EndpointKind,
    get_auth_handler,
    get_default_auth_method_for_endpoint,
)
from src.core.logger import logger
from src.core.provider_types import ProviderType


class ImageAdapterBase(ApiAdapter):
    """图像生成适配器基类。"""

    FORMAT_ID: str = "UNKNOWN"
    HANDLER_CLASS: type[ImageHandlerBase]

    API_FAMILY: ClassVar[ApiFamily | None] = None
    ENDPOINT_KIND: ClassVar[EndpointKind] = EndpointKind.IMAGE

    name: str = "image.base"
    mode = ApiMode.STANDARD
    # /edits 路径是 multipart，不能 eager 反序列化；/generations 是 JSON。
    # 统一由 handle() 内部决定如何读 body。
    eager_request_body = False

    # 指示这是哪个端点：由具体路由适配器覆盖。
    # "generations" | "edits"
    ENDPOINT_OPERATION: ClassVar[str] = "generations"

    def __init__(
        self,
        allowed_api_formats: list[str] | None = None,
        *,
        endpoint_operation: str | None = None,
    ):
        self.allowed_api_formats = allowed_api_formats or [self.FORMAT_ID]
        self.endpoint_operation = (
            (endpoint_operation or self.ENDPOINT_OPERATION).strip().lower()
            or "generations"
        )

    def extract_api_key(self, request: Request) -> str | None:
        auth_method = get_default_auth_method_for_endpoint(self.FORMAT_ID)
        handler = get_auth_handler(auth_method)
        return handler.extract_credentials(request)

    async def handle(
        self, context: ApiRequestContext
    ) -> Response | StreamingResponse | JSONResponse:
        http_request = context.request
        if context.api_key is None or context.user is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        if http_request.method.upper() != "POST":
            raise HTTPException(status_code=405, detail="Method Not Allowed")

        content_type = str(
            context.original_headers.get("content-type")
            or context.original_headers.get("Content-Type")
            or ""
        ).lower()
        is_multipart = content_type.startswith("multipart/form-data")

        if self.endpoint_operation == "edits" and not is_multipart:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "/v1/images/edits requires multipart/form-data body",
                    }
                },
            )

        if is_multipart:
            return await self._handle_multipart(context, http_request, content_type)
        return await self._handle_json(context, http_request)

    async def _handle_json(
        self,
        context: ApiRequestContext,
        http_request: Request,
    ) -> Response | StreamingResponse | JSONResponse:
        original_request_body = await context.ensure_json_body_async()
        if not isinstance(original_request_body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Request body must be a JSON object",
                    }
                },
            )

        missing = [f for f in ("model", "prompt") if not original_request_body.get(f)]
        if missing:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"Missing required fields: {', '.join(missing)}",
                    }
                },
            )

        stream = bool(original_request_body.get("stream"))
        model = str(original_request_body.get("model") or "unknown")
        self._record_audit(context, model, stream, original_request_body)

        handler = self._create_handler(context)

        params: dict[str, Any] = {
            "http_request": http_request,
            "original_headers": context.original_headers,
            "original_request_body": original_request_body,
            "query_params": context.query_params,
            "upstream_path_suffix": self.endpoint_operation,
        }
        if stream:
            return await handler.process_stream(**params)
        return await handler.process_sync(**params)

    async def _handle_multipart(
        self,
        context: ApiRequestContext,
        http_request: Request,
        content_type: str,
    ) -> Response | StreamingResponse | JSONResponse:
        # 读原始字节 — 透传路径需要它
        raw_body = await http_request.body()
        if not raw_body:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Empty multipart body",
                    }
                },
            )

        # 解析出结构化字段，用于 Codex 分支的 body transformer
        from src.services.provider.adapters.codex.image_transform import (
            parse_multipart_image_edit,
        )

        parsed = parse_multipart_image_edit(raw_body, content_type) or {}

        # 最小校验：edits 必须有 image（至少一张）和 prompt
        if self.endpoint_operation == "edits":
            images = parsed.get("images") or []
            prompt = (parsed.get("prompt") or "").strip() if isinstance(
                parsed.get("prompt"), str
            ) else ""
            if not images:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "/v1/images/edits requires at least one image",
                        }
                    },
                )
            if not prompt:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "/v1/images/edits requires prompt",
                        }
                    },
                )

        # stream 字段在 multipart 表单里以字符串形式出现
        stream_raw = str(parsed.get("stream") or "").strip().lower()
        stream = stream_raw in {"true", "1", "yes"}

        model = str(parsed.get("model") or "unknown")
        # 为 handler 构造一个最小 JSON-like 字段视图（用于审计 + 非 Codex 分支的 body 字段读取）
        # 注意：透传路径实际用 raw_body 字节转发，这里的 dict 只给 Codex 用
        surface_body: dict[str, Any] = {
            "model": model,
            "prompt": parsed.get("prompt") or "",
        }
        for key in ("n", "size", "quality", "response_format", "user", "background"):
            if key in parsed:
                surface_body[key] = parsed[key]

        self._record_audit(context, model, stream, surface_body)

        handler = self._create_handler(context)
        params: dict[str, Any] = {
            "http_request": http_request,
            "original_headers": context.original_headers,
            "original_request_body": surface_body,
            "query_params": context.query_params,
            "multipart_context": parsed,
            "raw_body": raw_body,
            "raw_content_type": content_type,
            "upstream_path_suffix": self.endpoint_operation,
        }
        if stream:
            return await handler.process_stream(**params)
        return await handler.process_sync(**params)

    def _record_audit(
        self,
        context: ApiRequestContext,
        model: str,
        stream: bool,
        body_view: dict[str, Any],
    ) -> None:
        context.add_audit_metadata(
            action=f"openai_image_{self.endpoint_operation}",
            model=model,
            stream=stream,
            image_count=body_view.get("n", 1),
            size=body_view.get("size"),
            quality=body_view.get("quality"),
        )
        logger.info(
            "[REQ] {} | {} | {} | {} | {} | {}",
            context.request_id[:8],
            self.FORMAT_ID,
            getattr(context.api_key, "name", "unknown"),
            self.endpoint_operation,
            model,
            "stream" if stream else "sync",
        )

    def _create_handler(self, context: ApiRequestContext) -> ImageHandlerBase:
        return self.HANDLER_CLASS(
            db=context.db,
            user=context.user,
            api_key=context.api_key,
            request_id=context.request_id,
            client_ip=context.client_ip,
            user_agent=context.user_agent,
            start_time=context.start_time,
            allowed_api_formats=self.allowed_api_formats,
        )

    # ------------------------------------------------------------------ #
    # 端点测试（后台“测试模型”功能调用）
    # ------------------------------------------------------------------ #

    @classmethod
    async def check_endpoint(
        cls,
        client: httpx.AsyncClient | None,
        base_url: str,
        api_key: str,
        request_data: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
        body_rules: list[dict[str, Any]] | None = None,
        header_rules: list[dict[str, Any]] | None = None,
        db: Any | None = None,
        user: Any | None = None,
        provider_name: str | None = None,
        provider_id: str | None = None,
        api_key_id: str | None = None,
        model_name: str | None = None,
        auth_type: str | None = None,
        provider_type: str | None = None,
        decrypted_auth_config: dict[str, Any] | None = None,
        provider_endpoint: Any | None = None,
        provider_api_key: Any | None = None,
        proxy_config: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """后台测试模型 —— 对 image 端点发一次最小请求。

        Codex 反代：走 `/responses` + `image_generation` tool。
        其他 provider：走 `/v1/images/generations`。
        """
        _ = (client, body_rules, header_rules, db, user, provider_id, api_key_id)
        _ = (provider_api_key,)

        pt = str(provider_type or "").strip().lower()
        is_codex = pt == ProviderType.CODEX.value or pt == "codex"

        test_model = str(model_name or request_data.get("model") or "").strip()
        prompt = "test"

        timeout = httpx.Timeout(float(timeout_seconds or 60.0))

        async with httpx.AsyncClient(
            timeout=timeout,
            proxy=(proxy_config or {}).get("url") if isinstance(proxy_config, dict) else None,
        ) as http:
            try:
                if is_codex:
                    result = await cls._check_codex_image_endpoint(
                        http=http,
                        base_url=base_url,
                        api_key=api_key,
                        extra_headers=extra_headers,
                        test_model=test_model,
                        prompt=prompt,
                        decrypted_auth_config=decrypted_auth_config,
                    )
                else:
                    result = await cls._check_standard_image_endpoint(
                        http=http,
                        base_url=base_url,
                        api_key=api_key,
                        extra_headers=extra_headers,
                        test_model=test_model,
                        prompt=prompt,
                    )
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"connection_failed: {exc}"[:500],
                    "status_code": 0,
                    "provider": {"name": provider_name},
                    "model": test_model,
                }

        return result

    @staticmethod
    async def _check_standard_image_endpoint(
        *,
        http: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        extra_headers: dict[str, str] | None,
        test_model: str,
        prompt: str,
    ) -> dict[str, Any]:
        normalized = (base_url or "").rstrip("/")
        url = (
            f"{normalized}/images/generations"
            if normalized.endswith("/v1")
            else f"{normalized}/v1/images/generations"
        )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        body: dict[str, Any] = {
            "model": test_model,
            "prompt": prompt,
            "n": 1,
        }
        start = time.time()
        response = await http.post(url, headers=headers, json=body)
        latency_ms = int((time.time() - start) * 1000)
        return _normalize_test_response(response, latency_ms)

    @staticmethod
    async def _check_codex_image_endpoint(
        *,
        http: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        extra_headers: dict[str, str] | None,
        test_model: str,
        prompt: str,
        decrypted_auth_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        # Codex: POST /responses with image_generation tool
        normalized = (base_url or "").rstrip("/")
        if normalized.endswith("/responses"):
            url = normalized
        else:
            url = f"{normalized}/responses"
        body: dict[str, Any] = {
            "model": test_model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "image_generation"}],
            "tool_choice": "auto",
            "instructions": "you are a helpful assistant",
            "stream": True,
            "store": False,
        }
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        account_id = (decrypted_auth_config or {}).get("account_id")
        if account_id:
            headers["chatgpt-account-id"] = str(account_id)
        if extra_headers:
            headers.update({k: v for k, v in extra_headers.items() if k.lower() != "content-type"})

        start = time.time()
        async with http.stream("POST", url, headers=headers, json=body) as response:
            latency_ms = int((time.time() - start) * 1000)
            status_code = response.status_code
            if status_code >= 400:
                err_text = (await response.aread()).decode("utf-8", errors="ignore")[:500]
                return {
                    "success": False,
                    "status_code": status_code,
                    "error": err_text or f"HTTP {status_code}",
                    "latency_ms": latency_ms,
                }
            # 不需要读完整流，确认握手成功即可
            return {
                "success": True,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "content": "codex image handshake ok",
            }


def _normalize_test_response(
    response: httpx.Response,
    latency_ms: int,
) -> dict[str, Any]:
    status_code = response.status_code
    content_type = response.headers.get("content-type", "")
    if status_code >= 400:
        err: Any
        try:
            err = response.json()
        except Exception:
            err = response.text[:500]
        return {
            "success": False,
            "status_code": status_code,
            "error": json.dumps(err)[:500] if not isinstance(err, str) else err,
            "latency_ms": latency_ms,
        }
    if "application/json" in content_type:
        try:
            payload = response.json()
        except Exception:
            payload = {}
    else:
        payload = {"raw": response.text[:500]}
    return {
        "success": True,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "content": "image endpoint ok",
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
    }


# =========================================================================
# Image Adapter 注册表
# =========================================================================

_IMAGE_ADAPTER_REGISTRY: dict[str, type[ImageAdapterBase]] = {}
_IMAGE_ADAPTERS_LOADED = False


def register_image_adapter(
    adapter_class: type[ImageAdapterBase],
) -> type[ImageAdapterBase]:
    """注册 Image Adapter 类。"""
    format_id = getattr(adapter_class, "FORMAT_ID", None)
    if format_id and format_id != "UNKNOWN":
        _IMAGE_ADAPTER_REGISTRY[format_id.upper()] = adapter_class
    return adapter_class


def _ensure_image_adapters_loaded() -> None:
    global _IMAGE_ADAPTERS_LOADED
    if _IMAGE_ADAPTERS_LOADED:
        return
    try:
        from src.api.handlers.openai import image_adapter as _  # noqa: F401
    except ImportError:
        pass
    _IMAGE_ADAPTERS_LOADED = True


def get_image_adapter_class(
    api_format: str,
) -> type[ImageAdapterBase] | None:
    """根据 API format 获取 Image Adapter 类。"""
    _ensure_image_adapters_loaded()
    return _IMAGE_ADAPTER_REGISTRY.get(api_format.upper()) if api_format else None


__all__ = [
    "ImageAdapterBase",
    "register_image_adapter",
    "get_image_adapter_class",
]
