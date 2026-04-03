"""
Provider Key 重复校验规则。
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from src.core.crypto import crypto_service
from src.core.exceptions import InvalidRequestException
from src.models.database import ProviderAPIKey


def check_duplicate_key(
    db: Session,
    provider_id: str,
    auth_type: str,
    new_api_key: str | None = None,
    new_auth_config: dict | None = None,
    exclude_key_id: str | None = None,
) -> None:
    """
    检查密钥是否与其他现有密钥重复

    对于不同的认证类型，使用不同的比较方式：
    - api_key: 比较 API Key 的哈希值
    - service_account: 比较 Service Account 的 client_email

    Args:
        db: 数据库会话
        provider_id: Provider ID
        auth_type: 认证类型 (api_key, service_account, oauth)
        new_api_key: 新的 API Key（用于 api_key 类型）
        new_auth_config: 新的认证配置（用于 service_account 类型）
        exclude_key_id: 要排除的 Key ID（用于更新场景）
    """
    if auth_type == "api_key" and new_api_key:
        # 跳过占位符
        if new_api_key == "__placeholder__":
            return

        # 仅查询同 auth_type 的 Keys，减少不必要的解密操作
        query = db.query(ProviderAPIKey).filter(
            ProviderAPIKey.provider_id == provider_id,
            ProviderAPIKey.auth_type == "api_key",
        )
        if exclude_key_id:
            query = query.filter(ProviderAPIKey.id != exclude_key_id)

        new_key_hash = crypto_service.hash_api_key(new_api_key)
        for existing_key in query:
            try:
                decrypted_key = crypto_service.decrypt(existing_key.api_key, silent=True)
                if decrypted_key == "__placeholder__":
                    continue
                existing_hash = crypto_service.hash_api_key(decrypted_key)
                if new_key_hash == existing_hash:
                    raise InvalidRequestException(
                        f"该 API Key 已存在于当前 Provider 中（名称: {existing_key.name}）"
                    )
            except InvalidRequestException:
                raise
            except Exception:
                # 解密失败时跳过该 Key
                continue

    elif auth_type in ("service_account", "vertex_ai") and new_auth_config:
        new_client_email = (
            new_auth_config.get("client_email") if isinstance(new_auth_config, dict) else None
        )
        if not new_client_email:
            return

        # 仅查询同 auth_type 且有 auth_config 的 Keys
        query = db.query(ProviderAPIKey).filter(
            ProviderAPIKey.provider_id == provider_id,
            ProviderAPIKey.auth_type.in_(["service_account", "vertex_ai"]),
            ProviderAPIKey.auth_config.isnot(None),
        )
        if exclude_key_id:
            query = query.filter(ProviderAPIKey.id != exclude_key_id)

        for existing_key in query:
            try:
                decrypted_config = json.loads(
                    crypto_service.decrypt(existing_key.auth_config, silent=True)
                )
                existing_email = decrypted_config.get("client_email")
                if existing_email and existing_email == new_client_email:
                    raise InvalidRequestException(
                        f"该 Service Account ({new_client_email}) 已存在于当前 Provider 中"
                        f"（名称: {existing_key.name}）"
                    )
            except InvalidRequestException:
                raise
            except Exception:
                # 解密失败时跳过该 Key
                continue
