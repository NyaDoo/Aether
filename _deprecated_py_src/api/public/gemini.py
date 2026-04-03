"""
Gemini API 专属端点

托管 Gemini API 相关路由:
- /v1beta/models/{model}:generateContent
- /v1beta/models/{model}:streamGenerateContent

注意:
- Gemini API 的 model 在 URL 路径中，而不是请求体中
- /v1beta/models (列表) 和 /v1beta/models/{model} (详情) 由 models.py 统一处理
"""

from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from src.api.base.adapter import ApiAdapter, ApiMode
from src.api.base.context import ApiRequestContext
from src.api.base.pipeline import get_pipeline
from src.database import get_db

router = APIRouter(tags=["Gemini API"])
pipeline = get_pipeline()


def _is_cli_request(request: Request) -> bool:
    """
    判断是否为 CLI 请求

    检查顺序:
    1. x-app header 包含 "cli"
    2. user-agent 包含 "GeminiCLI" 或 "gemini-cli"
    """
    # 检查 x-app header
    x_app = request.headers.get("x-app", "")
    if "cli" in x_app.lower():
        return True

    # 检查 user-agent
    user_agent = request.headers.get("user-agent", "")
    user_agent_lower = user_agent.lower()
    if "geminicli" in user_agent_lower or "gemini-cli" in user_agent_lower:
        return True

    return False


def _build_adapter_for_request(request: Request) -> Any:
    """按请求类型懒加载 Gemini 适配器，降低模块导入开销。"""
    if _is_cli_request(request):
        from src.api.handlers.gemini_cli import build_gemini_cli_adapter

        return build_gemini_cli_adapter()

    from src.api.handlers.gemini import build_gemini_adapter

    return build_gemini_adapter()


@dataclass
class PublicGeminiContentAdapter(ApiAdapter):
    model: str
    stream: bool

    name = "public.gemini.content"
    mode = ApiMode.STANDARD

    def _delegate(self, request: Request) -> Any:
        return _build_adapter_for_request(request)

    def extract_api_key(self, request: Request) -> str | None:
        return self._delegate(request).extract_api_key(request)

    def authorize(self, context: ApiRequestContext) -> None:
        return self._delegate(context.request).authorize(context)

    def detect_capability_requirements(
        self,
        headers: dict[str, str],
        request_body: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        adapter = _build_adapter_for_request(
            Request(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/",
                    "raw_path": b"/",
                    "query_string": b"",
                    "headers": [
                        (str(key).lower().encode(), str(value).encode()) for key, value in headers.items()
                    ],
                    "client": ("127.0.0.1", 0),
                    "server": ("testserver", 80),
                }
            )
        )
        return adapter.detect_capability_requirements(headers, request_body)

    def get_audit_metadata(
        self,
        context: ApiRequestContext,
        *,
        success: bool,
        status_code: int | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return self._delegate(context.request).get_audit_metadata(
            context,
            success=success,
            status_code=status_code,
            error=error,
        )

    def api_format_hint_for_request(self, request: Request) -> str:
        return self._delegate(request).allowed_api_formats[0]

    def path_params(self) -> dict[str, Any]:
        return {"model": self.model, "stream": self.stream}

    async def handle(self, context: ApiRequestContext) -> Any:
        return await self._delegate(context.request).handle(context)


@router.post("/v1beta/models/{model}:generateContent")
async def generate_content(
    model: str,
    http_request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Gemini generateContent API

    兼容 Google Gemini API 格式的代理接口（非流式）。

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **请求格式**:
    ```json
    {
        "contents": [{"parts": [{"text": "Hello"}]}]
    }
    ```

    **路径参数**:
    - `model`: 模型名称，如 gemini-2.0-flash
    """
    adapter = PublicGeminiContentAdapter(model=model, stream=False)

    return await pipeline.run(
        adapter=adapter,
        http_request=http_request,
        db=db,
        mode=adapter.mode,
        api_format_hint=adapter.api_format_hint_for_request(http_request),
        path_params=adapter.path_params(),
    )


@router.post("/v1beta/models/{model}:streamGenerateContent")
async def stream_generate_content(
    model: str,
    http_request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Gemini streamGenerateContent API

    兼容 Google Gemini API 格式的代理接口（流式）。

    **认证方式**:
    - `x-goog-api-key` 请求头，或
    - `?key=` URL 参数

    **路径参数**:
    - `model`: 模型名称，如 gemini-2.0-flash

    注意: Gemini API 通过 URL 端点区分流式/非流式，不需要在请求体中添加 stream 字段
    """
    adapter = PublicGeminiContentAdapter(model=model, stream=True)

    return await pipeline.run(
        adapter=adapter,
        http_request=http_request,
        db=db,
        mode=adapter.mode,
        api_format_hint=adapter.api_format_hint_for_request(http_request),
        path_params=adapter.path_params(),
    )


# 兼容 v1 路径（部分 SDK 可能使用 generateContent）
@router.post("/v1/models/{model}:generateContent")
async def generate_content_v1(
    model: str,
    http_request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Gemini generateContent API (v1 兼容)

    v1 版本 API 端点，兼容部分使用旧版路径的 SDK。
    """
    adapter = PublicGeminiContentAdapter(model=model, stream=False)
    return await pipeline.run(
        adapter=adapter,
        http_request=http_request,
        db=db,
        mode=adapter.mode,
        api_format_hint=adapter.api_format_hint_for_request(http_request),
        path_params=adapter.path_params(),
    )


@router.post("/v1/models/{model}:streamGenerateContent")
async def stream_generate_content_v1(
    model: str,
    http_request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """
    Gemini streamGenerateContent API (v1 兼容)

    v1 版本流式 API 端点，兼容部分使用旧版路径的 SDK。
    """
    adapter = PublicGeminiContentAdapter(model=model, stream=True)
    return await pipeline.run(
        adapter=adapter,
        http_request=http_request,
        db=db,
        mode=adapter.mode,
        api_format_hint=adapter.api_format_hint_for_request(http_request),
        path_params=adapter.path_params(),
    )
