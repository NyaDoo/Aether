from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

_GEMINI_MODEL_ROUTE_RE = re.compile(
    r"^/(v1|v1beta)/models/[^/]+:(generateContent|streamGenerateContent|predictLongRunning)$"
)
_GEMINI_MODEL_OPERATION_CANCEL_RE = re.compile(r"^/v1beta/models/[^/]+/operations/[^/]+:cancel$")
_GEMINI_OPERATION_CANCEL_RE = re.compile(r"^/v1beta/operations/[^/]+:cancel$")
_GEMINI_MODEL_OPERATION_ROUTE_RE = re.compile(r"^/v1beta/models/[^/]+/operations/[^/]+$")
_GEMINI_FILES_ROUTE_RE = re.compile(r"^/v1beta/files(?:/.+)?$")
_GEMINI_FILES_DOWNLOAD_ROUTE_RE = re.compile(r"^/v1beta/files/(?P<file_id>[^/]+):download$")
_GEMINI_FILES_RESOURCE_ROUTE_RE = re.compile(r"^/v1beta/files/(?P<file_name>.+)$")
_GEMINI_SYNC_ROUTE_RE = re.compile(
    r"^/(?P<version>v1|v1beta)/models/(?P<model>[^/]+):(?P<action>generateContent|streamGenerateContent)$"
)
_OPENAI_VIDEO_CANCEL_ROUTE_RE = re.compile(r"^/v1/videos/(?P<task_id>[^/]+)/cancel$")
_OPENAI_VIDEO_REMIX_ROUTE_RE = re.compile(r"^/v1/videos/(?P<task_id>[^/]+)/remix$")
_OPENAI_VIDEO_CONTENT_ROUTE_RE = re.compile(r"^/v1/videos/(?P<task_id>[^/]+)/content$")
_OPENAI_VIDEO_TASK_ROUTE_RE = re.compile(r"^/v1/videos/(?P<task_id>[^/]+)$")
_GEMINI_VIDEO_CREATE_ROUTE_RE = re.compile(r"^/v1beta/models/(?P<model>[^/]+):predictLongRunning$")
_GEMINI_VIDEO_MODEL_OPERATION_ANY_RE = re.compile(
    r"^/v1beta/models/(?P<model>[^/]+)/operations/(?P<operation_id>[^/]+)(?P<cancel>:cancel)?$"
)

CONTROL_EXECUTED_HEADER = "x-aether-control-executed"
CONTROL_ACTION_HEADER = "x-aether-control-action"
CONTROL_ACTION_PROXY_PUBLIC = "proxy_public"


class GatewayResolveRequest(BaseModel):
    trace_id: str | None = Field(None, max_length=128)
    method: str = Field(..., min_length=1, max_length=16)
    path: str = Field(..., min_length=1, max_length=2048)
    query_string: str | None = Field(None, max_length=8192)
    headers: dict[str, str] = Field(default_factory=dict)
    has_body: bool = False
    content_type: str | None = Field(None, max_length=512)
    content_length: int | None = Field(None, ge=0)


class GatewayAuthContextRequest(BaseModel):
    trace_id: str | None = Field(None, max_length=128)
    query_string: str | None = Field(None, max_length=8192)
    headers: dict[str, str] = Field(default_factory=dict)
    auth_endpoint_signature: str = Field(..., min_length=1, max_length=128)


class GatewayRouteDecision(BaseModel):
    action: str = "proxy_public"
    route_class: str
    public_path: str
    public_query_string: str | None = None
    route_family: str | None = None
    route_kind: str | None = None
    auth_endpoint_signature: str | None = None
    executor_candidate: bool = False
    auth_context: dict[str, Any] | None = None


class GatewayAuthContext(BaseModel):
    user_id: str
    api_key_id: str
    balance_remaining: float | None = None
    access_allowed: bool = True


class GatewayExecuteRequest(BaseModel):
    trace_id: str | None = Field(None, max_length=128)
    method: str = Field(..., min_length=1, max_length=16)
    path: str = Field(..., min_length=1, max_length=2048)
    query_string: str | None = Field(None, max_length=8192)
    headers: dict[str, str] = Field(default_factory=dict)
    body_json: dict[str, Any] = Field(default_factory=dict)
    body_base64: str | None = None
    auth_context: GatewayAuthContext | None = None


class GatewayExecutionPlanResponse(BaseModel):
    action: str
    plan_kind: str
    plan: dict[str, Any]
    report_kind: str | None = None
    report_context: dict[str, Any] | None = None
    auth_context: GatewayAuthContext | None = None


class GatewayExecutionDecisionResponse(BaseModel):
    action: str
    decision_kind: str
    request_id: str
    candidate_id: str | None = None
    provider_name: str
    provider_id: str
    endpoint_id: str
    key_id: str
    upstream_base_url: str
    upstream_url: str | None = None
    provider_request_method: str | None = None
    auth_header: str
    auth_value: str
    provider_api_format: str
    client_api_format: str
    model_name: str
    mapped_model: str | None = None
    prompt_cache_key: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)
    provider_request_headers: dict[str, str] | None = None
    provider_request_body: dict[str, Any] | None = None
    content_type: str | None = None
    proxy: dict[str, Any] | None = None
    tls_profile: str | None = None
    timeouts: dict[str, Any] | None = None
    upstream_is_stream: bool | None = None
    report_kind: str | None = None
    report_context: dict[str, Any] | None = None
    auth_context: GatewayAuthContext | None = None


class GatewaySyncReportRequest(BaseModel):
    trace_id: str | None = Field(None, max_length=128)
    report_kind: str = Field(..., min_length=1, max_length=128)
    report_context: dict[str, Any] = Field(default_factory=dict)
    status_code: int = Field(..., ge=100, le=599)
    headers: dict[str, str] = Field(default_factory=dict)
    body_json: Any = None
    client_body_json: Any = None
    body_base64: str | None = None
    telemetry: dict[str, Any] | None = None


class GatewayStreamReportRequest(BaseModel):
    trace_id: str | None = Field(None, max_length=128)
    report_kind: str = Field(..., min_length=1, max_length=128)
    report_context: dict[str, Any] = Field(default_factory=dict)
    status_code: int = Field(..., ge=100, le=599)
    headers: dict[str, str] = Field(default_factory=dict)
    body_base64: str | None = None
    telemetry: dict[str, Any] | None = None


def classify_gateway_route(
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
) -> GatewayRouteDecision:
    normalized_method = str(method or "").strip().upper() or "GET"
    normalized_path = str(path or "").strip() or "/"
    normalized_headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    if normalized_method == "POST" and normalized_path == "/v1/chat/completions":
        return _ai_route(
            normalized_path,
            family="openai",
            kind="chat",
            auth_endpoint_signature="openai:chat",
        )

    if normalized_method == "POST" and normalized_path in {
        "/v1/responses",
        "/v1/responses/compact",
    }:
        route_kind = "compact" if normalized_path.endswith("/compact") else "cli"
        auth_endpoint_signature = "openai:compact" if route_kind == "compact" else "openai:cli"
        return _ai_route(
            normalized_path,
            family="openai",
            kind=route_kind,
            auth_endpoint_signature=auth_endpoint_signature,
        )

    if normalized_method == "POST" and normalized_path == "/v1/messages":
        is_claude_cli = _is_claude_cli_request(normalized_headers)
        return _ai_route(
            normalized_path,
            family="claude",
            kind="cli" if is_claude_cli else "chat",
            auth_endpoint_signature="claude:cli" if is_claude_cli else "claude:chat",
        )

    if normalized_path.startswith("/v1/videos"):
        return _ai_route(
            normalized_path,
            family="openai",
            kind="video",
            auth_endpoint_signature="openai:video",
        )

    if _GEMINI_MODEL_ROUTE_RE.match(normalized_path):
        if normalized_path.endswith(":predictLongRunning"):
            return _ai_route(
                normalized_path,
                family="gemini",
                kind="video",
                auth_endpoint_signature="gemini:video",
            )
        is_gemini_cli = _is_gemini_cli_request(normalized_headers)
        return _ai_route(
            normalized_path,
            family="gemini",
            kind="cli" if is_gemini_cli else "chat",
            auth_endpoint_signature="gemini:cli" if is_gemini_cli else "gemini:chat",
        )

    if (
        _GEMINI_MODEL_OPERATION_CANCEL_RE.match(normalized_path)
        or _GEMINI_OPERATION_CANCEL_RE.match(normalized_path)
        or _GEMINI_MODEL_OPERATION_ROUTE_RE.match(normalized_path)
        or normalized_path == "/v1beta/operations"
        or normalized_path.startswith("/v1beta/operations/")
    ):
        return _ai_route(
            normalized_path,
            family="gemini",
            kind="video",
            auth_endpoint_signature="gemini:video",
        )

    if normalized_method == "POST" and normalized_path == "/upload/v1beta/files":
        return _ai_route(
            normalized_path,
            family="gemini",
            kind="files",
            auth_endpoint_signature="gemini:chat",
        )

    if _GEMINI_FILES_ROUTE_RE.match(normalized_path):
        return _ai_route(
            normalized_path,
            family="gemini",
            kind="files",
            auth_endpoint_signature="gemini:chat",
        )

    return GatewayRouteDecision(
        route_class="passthrough",
        public_path=normalized_path,
        executor_candidate=False,
    )


def _ai_route(
    path: str,
    *,
    family: str,
    kind: str,
    auth_endpoint_signature: str,
) -> GatewayRouteDecision:
    return GatewayRouteDecision(
        route_class="ai_public",
        public_path=path,
        route_family=family,
        route_kind=kind,
        auth_endpoint_signature=auth_endpoint_signature,
        executor_candidate=True,
    )


def _is_claude_cli_request(headers: dict[str, str]) -> bool:
    auth_header = str(headers.get("authorization") or "").strip().lower()
    has_bearer = auth_header.startswith("bearer ")
    has_api_key = bool(str(headers.get("x-api-key") or "").strip())
    return has_bearer and not has_api_key


def _is_gemini_cli_request(headers: dict[str, str]) -> bool:
    x_app = str(headers.get("x-app") or "").lower()
    if "cli" in x_app:
        return True

    user_agent = str(headers.get("user-agent") or "").lower()
    return "geminicli" in user_agent or "gemini-cli" in user_agent
