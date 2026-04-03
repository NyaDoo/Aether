"""
Tunnel httpx transport。

统一通过 gateway relay 转发 tunnel 请求。
"""

from __future__ import annotations

from typing import Any

import httpx


def is_tunnel_node(node_info: dict[str, Any] | None) -> bool:
    """检查节点是否为 tunnel 模式且已连接"""
    if not node_info:
        return False
    return bool(node_info.get("tunnel_mode")) and bool(node_info.get("tunnel_connected"))


def create_tunnel_transport(node_id: str, timeout: float = 60.0) -> httpx.AsyncBaseTransport:
    """创建统一的 gateway tunnel relay transport。"""
    from .gateway_tunnel_transport import GatewayTunnelTransport

    return GatewayTunnelTransport(node_id, timeout=timeout)
