"""
代理节点模块

提供海外 VPS 代理节点的注册、心跳、管理功能。
aether-proxy 部署在海外 VPS 上，通过 WebSocket 隧道连接 Aether 转发 API 请求。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from starlette.routing import BaseRoute

from src.core.modules.base import (
    ModuleCategory,
    ModuleDefinition,
    ModuleHealth,
    ModuleMetadata,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


_proxy_node_task_coordinator: Any | None = None
_RUST_OWNED_PROXY_NODE_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/admin/proxy-nodes"),
        ("GET", "/api/admin/proxy-nodes/"),
        ("POST", "/api/admin/proxy-nodes/register"),
        ("POST", "/api/admin/proxy-nodes/heartbeat"),
        ("POST", "/api/admin/proxy-nodes/unregister"),
        ("POST", "/api/admin/proxy-nodes/manual"),
        ("POST", "/api/admin/proxy-nodes/upgrade"),
        ("POST", "/api/admin/proxy-nodes/test-url"),
        ("PATCH", "/api/admin/proxy-nodes/{node_id}"),
        ("DELETE", "/api/admin/proxy-nodes/{node_id}"),
        ("POST", "/api/admin/proxy-nodes/{node_id}/test"),
        ("PUT", "/api/admin/proxy-nodes/{node_id}/config"),
        ("GET", "/api/admin/proxy-nodes/{node_id}/events"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_PROXY_NODE_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _reset_tunnel_connected_on_startup() -> None:
    """服务端启动时将所有 tunnel_connected=True 的节点重置为 False/OFFLINE。

    服务端重启后 DB 中可能残留 tunnel_connected=True 的记录。
    如果不重置，节点会在 tunnel 状态广播到来前短暂显示 ONLINE。
    """
    from datetime import datetime, timezone

    from src.core.logger import logger
    from src.database import create_session
    from src.models.database import ProxyNode, ProxyNodeStatus

    db = create_session()
    try:
        now = datetime.now(timezone.utc)
        stale_nodes = (
            db.query(ProxyNode)
            .filter(
                ProxyNode.tunnel_connected == True,  # noqa: E712
                ProxyNode.is_manual == False,  # noqa: E712
            )
            .all()
        )
        if stale_nodes:
            for node in stale_nodes:
                node.tunnel_connected = False
                node.tunnel_connected_at = now
                node.status = ProxyNodeStatus.OFFLINE
                node.updated_at = now
            db.commit()
            logger.info(
                "重置 {} 个节点的残留 tunnel 连接状态 (tunnel_connected -> False)",
                len(stale_nodes),
            )
    except Exception as e:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("重置 tunnel 连接状态失败: {}", e)
    finally:
        db.close()


def _get_router() -> Any:
    """延迟导入路由"""
    from src.api.admin.proxy_nodes import router

    filtered_router = APIRouter()
    filtered_router.include_router(router)
    filtered_router.routes = [
        route for route in filtered_router.routes if not _route_is_rust_owned(route)
    ]
    return filtered_router


async def _on_startup() -> None:
    """启动心跳检测调度器"""
    import logging

    from src.config import config
    from src.services.proxy_node.health_scheduler import get_proxy_node_health_scheduler
    from src.utils.task_coordinator import StartupTaskCoordinator

    logger = logging.getLogger("aether.modules.proxy_nodes")

    if config.worker_processes > 1:
        logger.info(
            "检测到 WEB_CONCURRENCY={}，隧道模式允许多 worker 共享 tunnel。",
            config.worker_processes,
        )

    from src.clients import get_redis_client

    global _proxy_node_task_coordinator
    redis_client = await get_redis_client()
    task_coordinator = StartupTaskCoordinator(redis_client)
    _proxy_node_task_coordinator = task_coordinator

    proxy_node_health_scheduler = get_proxy_node_health_scheduler()
    active = await task_coordinator.acquire("proxy_node_health")
    if active:
        # 仅 leader worker 执行启动重置，避免多 worker 并发启动/重启时
        # 把其他 worker 已建立的 tunnel 状态错误重置为 OFFLINE。
        _reset_tunnel_connected_on_startup()
    else:
        logger.info("检测到其他 worker 已运行 ProxyNode 心跳检测，本实例跳过")

    if active:
        logger.info("启动 ProxyNode 心跳检测调度器...")
        await proxy_node_health_scheduler.start()

        async def _on_lock_lost(_name: str) -> None:
            await _stop_proxy_node_scheduler_on_lock_lost(logger)

        task_coordinator.register_lock_lost_callback("proxy_node_health", _on_lock_lost)


async def _on_shutdown() -> None:
    """停止心跳检测调度器"""
    import logging

    from src.services.proxy_node.health_scheduler import get_proxy_node_health_scheduler
    from src.utils.task_coordinator import StartupTaskCoordinator

    logger = logging.getLogger("aether.modules.proxy_nodes")

    from src.clients import get_redis_client

    global _proxy_node_task_coordinator
    if _proxy_node_task_coordinator is None:
        redis_client = await get_redis_client()
        _proxy_node_task_coordinator = StartupTaskCoordinator(redis_client)

    scheduler = get_proxy_node_health_scheduler()
    if scheduler.running:
        logger.info("停止 ProxyNode 心跳检测调度器...")
        await scheduler.stop()
    await _proxy_node_task_coordinator.release("proxy_node_health")
    _proxy_node_task_coordinator = None


async def _stop_proxy_node_scheduler_on_lock_lost(logger: Any) -> None:
    from src.services.proxy_node.health_scheduler import get_proxy_node_health_scheduler

    scheduler = get_proxy_node_health_scheduler()
    if not scheduler.running:
        return

    logger.warning("检测到 proxy_node_health 的 leader 锁已丢失，停止本实例的 ProxyNode 心跳检测")
    await scheduler.stop()


async def _health_check() -> ModuleHealth:
    """健康检查 - 检查是否有在线节点"""
    return ModuleHealth.HEALTHY


def _validate_config(db: Session) -> tuple[bool, str]:
    """验证配置（tunnel 模式无需额外密钥配置）"""
    return True, ""


proxy_nodes_module = ModuleDefinition(
    metadata=ModuleMetadata(
        name="proxy_nodes",
        display_name="代理节点",
        description="添加Http/Socket代理节点, 或使用Aether-Proxy自动连接代理节点.",
        category=ModuleCategory.INTEGRATION,
        env_key="PROXY_NODES_AVAILABLE",
        default_available=True,
        required_packages=[],
        api_prefix="/api/admin/proxy-nodes",
        admin_route="/admin/proxy-nodes",
        admin_menu_icon="Server",
        admin_menu_group="system",
        admin_menu_order=60,
    ),
    router_factory=_get_router,
    on_startup=_on_startup,
    on_shutdown=_on_shutdown,
    health_check=_health_check,
    validate_config=_validate_config,
)
