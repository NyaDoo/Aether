"""Shared Rust executor HTTP helper for Gemini CLI side calls."""

from __future__ import annotations

import json
from typing import Any

import httpx

from src.config.settings import config
from src.core.logger import logger


async def execute_gemini_cli_rust_http_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: Any,
    proxy_config: dict[str, Any] | None,
    request_id: str,
    provider_api_format: str,
    timeout_seconds: float,
    content_type: str | None = None,
) -> httpx.Response | None:
    from src.services.request.execution_runtime_plan import (
        ExecutionPlan,
        ExecutionPlanBody,
        ExecutionPlanTimeouts,
        build_execution_plan_body,
        build_proxy_snapshot,
    )
    from src.services.request.execution_runtime_client import (
        ExecutionRuntimeClient,
        ExecutionRuntimeClientError,
    )

    if config.execution_runtime_backend != "rust":
        return None

    final_headers = dict(headers)
    if (
        body is not None
        and content_type
        and not any(str(key).lower() == "content-type" for key in final_headers)
    ):
        final_headers["content-type"] = content_type

    timeout_ms = max(int(timeout_seconds * 1000), 1_000)

    try:
        proxy_snapshot = await build_proxy_snapshot(proxy_config, label="GeminiCLI")
        result = await ExecutionRuntimeClient().execute_sync_json(
            ExecutionPlan(
                request_id=request_id,
                candidate_id=None,
                provider_name="gemini_cli",
                provider_id="",
                endpoint_id="",
                key_id="",
                method=method,
                url=url,
                headers=final_headers,
                body=(
                    build_execution_plan_body(body, content_type=content_type)
                    if body is not None
                    else ExecutionPlanBody()
                ),
                stream=False,
                provider_api_format=provider_api_format,
                client_api_format=provider_api_format,
                model_name="gemini_cli",
                content_type=content_type,
                proxy=proxy_snapshot,
                timeouts=ExecutionPlanTimeouts(
                    connect_ms=timeout_ms,
                    read_ms=timeout_ms,
                    write_ms=timeout_ms,
                    pool_ms=timeout_ms,
                    total_ms=timeout_ms,
                ),
            )
        )
    except (ExecutionRuntimeClientError, httpx.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("GeminiCLI Rust HTTP fallback {} {}: {}", method, url, exc)
        return None
    except Exception as exc:
        logger.warning("GeminiCLI Rust HTTP unexpected fallback {} {}: {}", method, url, exc)
        return None

    response_headers = dict(result.headers)
    if result.response_json is not None:
        response_headers.setdefault("content-type", "application/json")
        response_body = json.dumps(result.response_json, ensure_ascii=False).encode("utf-8")
    elif result.response_body_bytes is not None:
        response_body = result.response_body_bytes
    else:
        response_body = b""

    return httpx.Response(
        status_code=result.status_code,
        request=httpx.Request(method, url, headers=final_headers),
        headers=response_headers,
        content=response_body,
    )


__all__ = ["execute_gemini_cli_rust_http_request"]
