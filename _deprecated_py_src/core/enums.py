"""
统一的枚举定义
避免重复定义造成的不一致

注意：APIFormat 架构已移除，统一使用 endpoint signature（family:kind）。
"""

from enum import Enum


class UserRole(Enum):
    """用户角色枚举"""

    ADMIN = "admin"
    USER = "user"


class ProviderBillingType(Enum):
    """提供商计费类型"""

    MONTHLY_QUOTA = "monthly_quota"  # 月卡额度
    PAY_AS_YOU_GO = "pay_as_you_go"  # 按量付费
    FREE_TIER = "free_tier"  # 免费额度


class AuthSource(str, Enum):
    """认证来源枚举"""

    LOCAL = "local"  # 本地认证
    LDAP = "ldap"  # LDAP 认证
    OAUTH = "oauth"  # OAuth 认证（账号首创来源）


class ErrorCategory(str, Enum):
    """错误分类枚举"""

    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
    CONTENT_FILTER = "content_filter"
    CONTEXT_LENGTH = "context_length"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    NETWORK = "network"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
