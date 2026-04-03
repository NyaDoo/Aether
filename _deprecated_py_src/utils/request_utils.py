"""
请求处理工具函数
提供统一的HTTP请求信息提取功能
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from fastapi import Request

TRACE_ID_HEADER = "x-trace-id"
_MISSING = object()


@dataclass(frozen=True)
class RequestIdentityMetadata:
    """请求身份元数据（最小集合）。"""

    request_id: str | None
    client_ip: str
    user_agent: str


def get_client_ip(request: Request) -> str:
    """
    获取客户端真实IP地址

    按优先级检查：
    1. X-Real-IP 头（最可靠，由最外层可信 Nginx 直接设置）
    2. X-Forwarded-For 头的第一个 IP（原始客户端）
    3. 直接客户端IP

    安全说明：
    - X-Real-IP 优先级最高，因为它通常由最外层 Nginx 设置为 $remote_addr，
      Nginx 会直接覆盖这个头，不会传递客户端伪造的值
    - 只要最外层 Nginx 配置了 proxy_set_header X-Real-IP $remote_addr; 即可正确获取真实 IP

    Args:
        request: FastAPI Request 对象

    Returns:
        str: 客户端IP地址，如果无法获取则返回 "unknown"
    """
    # 优先检查 X-Real-IP 头（由最外层 Nginx 设置，最可靠）
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # 检查 X-Forwarded-For 头，取第一个 IP（原始客户端）
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For 格式: "client, proxy1, proxy2"
        ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if ips:
            return ips[0]

    # 回退到直接客户端IP
    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def get_user_agent(request: Request) -> str:
    """
    获取用户代理字符串

    Args:
        request: FastAPI Request 对象

    Returns:
        str: User-Agent 字符串，如果无法获取则返回 "unknown"
    """
    return request.headers.get("User-Agent", "unknown")


def get_request_id(request: Request) -> str | None:
    """
    获取请求ID（如果存在）

    Args:
        request: FastAPI Request 对象

    Returns:
        Optional[str]: 请求ID，如果不存在则返回 None
    """
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id

    trace_id = request.headers.get(TRACE_ID_HEADER)
    if trace_id:
        return trace_id.strip() or None

    return None


def update_request_state(
    request: Request,
    *,
    request_id: object = _MISSING,
    user_id: object = _MISSING,
    api_key_id: object = _MISSING,
    management_token_id: object = _MISSING,
    user_session_id: object = _MISSING,
    prefetched_balance_remaining: object = _MISSING,
    gateway_execution_path: object = _MISSING,
    rate_limit_scope: object = _MISSING,
) -> None:
    """集中维护请求级 runtime state，减少散点赋值。"""

    if request_id is not _MISSING:
        request.state.request_id = request_id
    if user_id is not _MISSING:
        request.state.user_id = user_id
    if api_key_id is not _MISSING:
        request.state.api_key_id = api_key_id
    if management_token_id is not _MISSING:
        request.state.management_token_id = management_token_id
    if user_session_id is not _MISSING:
        request.state.user_session_id = user_session_id
    if prefetched_balance_remaining is not _MISSING:
        request.state.prefetched_balance_remaining = prefetched_balance_remaining
    if gateway_execution_path is not _MISSING:
        request.state.gateway_execution_path = gateway_execution_path
    if rate_limit_scope is not _MISSING:
        request.state.rate_limit_scope = rate_limit_scope


def get_request_metadata(request: Request) -> dict:
    """
    获取请求的完整元数据

    Args:
        request: FastAPI Request 对象

    Returns:
        dict: 包含请求元数据的字典
    """
    identity = get_request_identity_metadata(request)
    return {
        **asdict(identity),
        "method": request.method,
        "path": request.url.path,
        "query_params": str(request.query_params) if request.query_params else None,
        "content_type": request.headers.get("Content-Type"),
        "content_length": request.headers.get("Content-Length"),
    }


def get_request_identity_metadata(request: Request) -> RequestIdentityMetadata:
    """集中读取 request_id/client_ip/user_agent，避免散点访问。"""

    return RequestIdentityMetadata(
        request_id=get_request_id(request),
        client_ip=get_client_ip(request),
        user_agent=get_user_agent(request),
    )


def extract_ip_from_headers(headers: dict) -> str:
    """
    从HTTP头字典中提取IP地址（用于中间件等场景）

    Args:
        headers: HTTP头字典

    Returns:
        str: 客户端IP地址
    """
    # 优先检查 X-Real-IP（由最外层 Nginx 设置，最可靠）
    real_ip = headers.get("x-real-ip", "")
    if real_ip:
        return real_ip.strip()

    # 检查 X-Forwarded-For，取第一个 IP
    forwarded_for = headers.get("x-forwarded-for", "")
    if forwarded_for:
        ips = [ip.strip() for ip in forwarded_for.split(",") if ip.strip()]
        if ips:
            return ips[0]

    return "unknown"
