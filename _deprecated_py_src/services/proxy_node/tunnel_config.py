"""
Tunnel relay 配置。

控制 worker 是否通过本机 gateway relay 转发 tunnel 帧。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote

_DEFAULT_DOCKER_TUNNEL_URL = "http://127.0.0.1:8084"
_DOCKER_TUNNEL_CONNECT_TIMEOUT_SECONDS = 5.0
_DOCKER_TUNNEL_URL_ENV_KEYS = ("AETHER_TUNNEL_BASE_URL",)


@dataclass(frozen=True)
class TunnelRelayConfig:
    enabled: bool
    url: str
    connect_timeout_seconds: float

    @property
    def local_relay_base_url(self) -> str:
        return f"{self.url.rstrip('/')}/api/internal/tunnel/relay"

    def local_relay_url(self, node_id: str) -> str:
        return f"{self.local_relay_base_url}/{quote(node_id, safe='')}"


_tunnel_relay_config: TunnelRelayConfig | None = None


def _is_docker_runtime() -> bool:
    if os.getenv("DOCKER_CONTAINER", "").strip().lower() == "true":
        return True
    return os.path.exists("/.dockerenv")


def _resolve_docker_tunnel_url() -> str:
    for key in _DOCKER_TUNNEL_URL_ENV_KEYS:
        value = os.getenv(key, "").strip()
        if value:
            return value.rstrip("/")
    return _DEFAULT_DOCKER_TUNNEL_URL


def get_tunnel_relay_config() -> TunnelRelayConfig:
    """读取 tunnel relay 配置（进程内缓存）。"""
    global _tunnel_relay_config
    if _tunnel_relay_config is not None:
        return _tunnel_relay_config

    docker_runtime = _is_docker_runtime()
    _tunnel_relay_config = TunnelRelayConfig(
        enabled=docker_runtime,
        url=_resolve_docker_tunnel_url(),
        connect_timeout_seconds=_DOCKER_TUNNEL_CONNECT_TIMEOUT_SECONDS,
    )
    return _tunnel_relay_config


def reset_tunnel_relay_config_cache() -> None:
    """测试或热更新场景下清理配置缓存。"""
    global _tunnel_relay_config
    _tunnel_relay_config = None
