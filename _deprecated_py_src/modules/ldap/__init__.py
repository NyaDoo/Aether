"""
LDAP 认证模块

提供 LDAP/Active Directory 用户认证支持
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.modules.base import (
    ModuleCategory,
    ModuleDefinition,
    ModuleHealth,
    ModuleMetadata,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def _get_router() -> Any:
    """延迟导入路由（避免启动时加载重依赖）"""
    from src.api.admin.ldap import router

    return router


async def _health_check() -> ModuleHealth:
    """健康检查 - 简化版，不依赖数据库连接"""
    # 健康检查在启动时调用，此时可能没有数据库会话
    # 返回 UNKNOWN 表示需要进一步检查
    return ModuleHealth.UNKNOWN


def _validate_config(db: Session) -> tuple[bool, str]:
    """
    验证 LDAP 配置是否可以启用模块

    检查项：
    1. 配置是否存在
    2. 必填字段是否完整
    3. 绑定密码是否可解密

    注意：不在此处执行连接测试，因为 validate_config 会在每次查询模块状态时调用，
    同步阻塞等待 LDAP 服务器响应会严重影响性能。连接测试应在专门的测试接口中进行。
    """
    from src.core.crypto import crypto_service
    from src.models.database import LDAPConfig

    config = db.query(LDAPConfig).first()
    if not config:
        return False, "请先配置 LDAP 连接信息"

    # 检查必填字段
    if not config.server_url:
        return False, "请配置 LDAP 服务器地址"
    if not config.bind_dn:
        return False, "请配置绑定 DN"
    if not config.base_dn:
        return False, "请配置搜索基准 DN"
    if not config.bind_password_encrypted:
        return False, "请配置绑定密码"

    # 尝试解密密码（仅验证可解密，不执行连接测试）
    try:
        bind_password = crypto_service.decrypt(config.bind_password_encrypted)
        if not bind_password:
            return False, "绑定密码为空，请重新设置"
    except Exception:
        return False, "绑定密码解密失败，请重新设置"

    return True, ""


# ==================== 钩子实现 ====================


def _hook_get_auth_methods(db: Session) -> list[dict[str, Any]]:
    """auth.get_methods: 返回 LDAP 认证方法信息"""
    from src.services.auth.ldap import LDAPService

    if not LDAPService.is_ldap_enabled(db):
        return []
    is_exclusive = LDAPService.is_ldap_exclusive(db)
    return [
        {
            "type": "ldap",
            "enabled": True,
            "exclusive": is_exclusive,
        }
    ]


async def _hook_authenticate(db: Session, email: str, password: str, auth_type: str) -> Any:
    """auth.authenticate: LDAP 认证

    仅当 auth_type == "ldap" 时处理，否则返回 None 让其他模块尝试。
    """
    if auth_type != "ldap":
        return None

    import asyncio

    from starlette.concurrency import run_in_threadpool

    from src.core.logger import logger
    from src.services.auth.ldap import LDAPService

    # 预取配置，避免将 Session 传递到线程池
    config_data = LDAPService.get_config_data(db)
    if not config_data:
        logger.warning("登录失败 - LDAP 未启用或配置无效")
        return None

    # 计算总体超时
    single_timeout = config_data.get("connect_timeout", 10)
    total_timeout = max(20, min(int(single_timeout * 4 * 1.1), 60))

    try:
        ldap_user = await asyncio.wait_for(
            run_in_threadpool(LDAPService.authenticate_with_config, config_data, email, password),
            timeout=total_timeout,
        )
    except TimeoutError:
        logger.error("LDAP 认证总体超时({}秒): {}", total_timeout, email)
        return None

    if not ldap_user:
        return None

    # 获取或创建本地用户
    from src.services.auth.service import AuthService

    user = await AuthService.get_or_create_ldap_user(db, ldap_user)
    if not user:
        return None
    if user.is_deleted:
        logger.warning("登录失败 - 用户已删除: {}", email)
        return None
    if not user.is_active:
        logger.warning("登录失败 - 用户已禁用: {}", email)
        return None
    return user


def _hook_check_exclusive_mode(db: Session) -> bool | None:
    """auth.check_exclusive_mode: 检查 LDAP 排他登录模式"""
    from src.services.auth.ldap import LDAPService

    if LDAPService.is_ldap_exclusive(db):
        return True
    return None


def _hook_check_registration(db: Session) -> dict[str, Any] | None:
    """auth.check_registration: LDAP 排他模式下阻止本地注册"""
    from src.services.auth.ldap import LDAPService

    if LDAPService.is_ldap_exclusive(db):
        return {"blocked": True, "reason": "系统已启用 LDAP 专属登录，禁止本地注册"}
    return None


# LDAP 模块定义
ldap_module = ModuleDefinition(
    metadata=ModuleMetadata(
        name="ldap",
        display_name="LDAP 认证",
        description="支持通过 LDAP/Active Directory 进行用户认证",
        category=ModuleCategory.AUTH,
        # 可用性控制
        env_key="LDAP_AVAILABLE",
        default_available=True,
        required_packages=["ldap3"],
        # 路由配置（使用现有路由，不改变路径）
        api_prefix="/api/admin/ldap",
        # 前端配置
        admin_route="/admin/ldap",
        admin_menu_icon="Users",
        admin_menu_group="system",
        admin_menu_order=50,
    ),
    router_factory=_get_router,
    health_check=_health_check,
    validate_config=_validate_config,
    hooks={
        "auth.get_methods": _hook_get_auth_methods,
        "auth.authenticate": _hook_authenticate,
        "auth.check_exclusive_mode": _hook_check_exclusive_mode,
        "auth.check_registration": _hook_check_registration,
    },
)
