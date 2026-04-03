"""
Provider 认证相关的数据类型。

从 api/handlers/base/request_builder.py 下沉到 core 层，
消除 services→api 的反向依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderAuthInfo:
    """Provider 认证信息（用于 Service Account 等异步认证场景）"""

    auth_header: str
    auth_value: str
    # 解密后的认证配置（用于 URL 构建等场景，避免重复解密）
    decrypted_auth_config: dict[str, Any] | None = None

    def as_tuple(self) -> tuple[str, str]:
        """返回 (auth_header, auth_value) 元组"""
        return (self.auth_header, self.auth_value)
