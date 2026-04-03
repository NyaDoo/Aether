"""Vertex AI 认证处理。

- Service Account: GCP SA JSON → JWT → Access Token → Bearer header
- API Key: 通过 URL ?key= 查询参数认证，auth 层返回 None（由 transport hook 处理）
"""

from __future__ import annotations

import json
from typing import Any

from src.core.provider_auth_types import ProviderAuthInfo


async def _auth_service_account(key: Any, endpoint: Any | None = None) -> ProviderAuthInfo:
    """Service Account 认证：SA JSON → JWT → Access Token。"""
    from src.core.crypto import crypto_service
    from src.core.exceptions import InvalidRequestException
    from src.core.vertex_auth import VertexAuthError, VertexAuthService

    try:
        # 优先从 auth_config 读取，兼容从 api_key 读取（过渡期）
        encrypted_auth_config = getattr(key, "auth_config", None)
        if encrypted_auth_config:
            if isinstance(encrypted_auth_config, dict):
                sa_json = encrypted_auth_config
            else:
                decrypted_config = crypto_service.decrypt(encrypted_auth_config)
                sa_json = json.loads(decrypted_config)
        else:
            # 兼容旧数据：从 api_key 读取
            decrypted_key = crypto_service.decrypt(key.api_key)
            if decrypted_key == "__placeholder__":
                raise InvalidRequestException("认证配置丢失，请重新添加该密钥。")
            sa_json = json.loads(decrypted_key)

        if not isinstance(sa_json, dict):
            raise InvalidRequestException("Service Account JSON 无效，请重新添加该密钥。")

        # 获取 Access Token（注入代理配置）
        from src.services.provider.auth import _get_proxy_config
        from src.services.proxy_node.resolver import build_proxy_client_kwargs

        effective_proxy = _get_proxy_config(key, endpoint)

        service = VertexAuthService(sa_json)
        access_token = await service.get_access_token(
            httpx_client_kwargs=build_proxy_client_kwargs(effective_proxy, timeout=30),
        )

        return ProviderAuthInfo(
            auth_header="Authorization",
            auth_value=f"Bearer {access_token}",
            decrypted_auth_config=sa_json,
        )
    except InvalidRequestException:
        raise
    except VertexAuthError as e:
        raise InvalidRequestException(f"Vertex AI 认证失败：{e}")
    except json.JSONDecodeError:
        raise InvalidRequestException("Service Account JSON 格式无效，请重新添加该密钥。")
    except Exception:
        raise InvalidRequestException("Vertex AI 认证失败，请检查 Key 的 auth_config")
