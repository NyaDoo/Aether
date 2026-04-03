from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from src.core.logger import logger
from src.services.auth.oauth.base import OAuthProviderBase
from src.services.auth.oauth.models import OAuthFlowError, OAuthToken, OAuthUserInfo

if TYPE_CHECKING:
    from src.models.database import OAuthProvider


class LinuxDoOAuthProvider(OAuthProviderBase):
    """
    LinuxDo OAuth Provider。

    基于论坛信任等级（trust_level 0-4）的 OAuth2 认证，
    用于通过用户等级进行额度配给和频率限制。

    参考：https://linux.do/t/topic/329408

    返回的用户信息示例：
    {
        "id": 1,
        "username": "neo",
        "name": "Neo",
        "active": true,
        "trust_level": 4,
        "email": "u1@linux.do",
        "avatar_url": "https://linux.do/xxxx",
        "silenced": false
    }
    """

    provider_type = "linuxdo"
    display_name = "Linux Do"

    allowed_domains = ("linux.do", "connect.linux.do", "connect.linuxdo.org")

    # 默认端点
    authorization_url = "https://connect.linux.do/oauth2/authorize"
    token_url = "https://connect.linux.do/oauth2/token"
    userinfo_url = "https://connect.linux.do/api/user"
    backup_token_url = "https://connect.linuxdo.org/oauth2/token"
    backup_userinfo_url = "https://connect.linuxdo.org/api/user"

    # LinuxDo 不需要 scope
    default_scopes = ()

    @staticmethod
    def _build_basic_auth_header(client_id: str, client_secret: str) -> str:
        credentials = f"{client_id}:{client_secret}".encode("utf-8")
        return f"Basic {base64.b64encode(credentials).decode('ascii')}"

    @staticmethod
    def _build_candidate_urls(primary_url: str, backup_url: str) -> list[str]:
        parsed = urlparse(primary_url)
        host = (parsed.hostname or "").lower().rstrip(".")
        backup_path = urlparse(backup_url).path
        urls = [primary_url]
        if parsed.scheme == "https" and host == "connect.linux.do" and parsed.path == backup_path:
            urls.append(backup_url)
        return urls

    @staticmethod
    async def _request_with_fallback(
        candidate_urls: list[str],
        request_fn: Callable[[str], Awaitable[httpx.Response]],
        error_code: str,
        label: str,
    ) -> httpx.Response:
        resp: httpx.Response | None = None
        for idx, url in enumerate(candidate_urls):
            try:
                resp = await request_fn(url)
                break
            except httpx.HTTPError as exc:
                if idx < len(candidate_urls) - 1:
                    logger.warning("LinuxDo {} 端点不可达，尝试备用端点: {} ({})", label, url, exc)
                    continue
                logger.warning("LinuxDo {} 请求失败: {} ({})", label, url, exc)
                raise OAuthFlowError(error_code, "transport_error") from exc
        if resp is None:
            raise OAuthFlowError(error_code, "no_response")
        return resp

    async def exchange_code(self, config: OAuthProvider, code: str) -> OAuthToken:
        client_secret = config.get_client_secret()
        if not client_secret:
            raise OAuthFlowError("provider_unavailable", "client_secret 未配置")

        redirect_uri = config.redirect_uri
        client_id = config.client_id
        if not redirect_uri or not client_id:
            raise OAuthFlowError("provider_unavailable", "redirect_uri/client_id 未配置")

        candidate_urls = self._build_candidate_urls(
            self.get_effective_token_url(config), self.backup_token_url
        )
        headers = {
            "Authorization": self._build_basic_auth_header(client_id, client_secret),
            "Accept": "application/json",
        }
        resp = await self._request_with_fallback(
            candidate_urls,
            lambda url: self._http_post_form(
                url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers=headers,
            ),
            error_code="token_exchange_failed",
            label="token",
        )

        if resp.status_code >= 400:
            logger.warning("LinuxDo token 兑换失败: status={}", resp.status_code)
            raise OAuthFlowError("token_exchange_failed", f"status={resp.status_code}")

        data = resp.json()
        access_token = data.get("access_token")
        if not access_token:
            raise OAuthFlowError("token_exchange_failed", "missing access_token")

        return OAuthToken(
            access_token=str(access_token),
            token_type=str(data.get("token_type") or "bearer"),
            refresh_token=(str(data["refresh_token"]) if data.get("refresh_token") else None),
            expires_in=(int(data["expires_in"]) if data.get("expires_in") is not None else None),
            id_token=(str(data["id_token"]) if data.get("id_token") else None),
            scope=(str(data["scope"]) if data.get("scope") else None),
            raw=data,
        )

    async def get_user_info(self, config: OAuthProvider, access_token: str) -> OAuthUserInfo:
        candidate_urls = self._build_candidate_urls(
            self.get_effective_userinfo_url(config), self.backup_userinfo_url
        )
        resp = await self._request_with_fallback(
            candidate_urls,
            lambda url: self._http_get(url, headers={"Authorization": f"Bearer {access_token}"}),
            error_code="userinfo_fetch_failed",
            label="userinfo",
        )

        if resp.status_code >= 400:
            logger.warning("LinuxDo userinfo 获取失败: status={}", resp.status_code)
            raise OAuthFlowError("userinfo_fetch_failed", f"status={resp.status_code}")

        data: dict[str, Any] = resp.json()

        # LinuxDo 返回的 id 是数字类型
        provider_user_id = data.get("id")
        if provider_user_id is None:
            raise OAuthFlowError("userinfo_fetch_failed", "missing user id")

        return OAuthUserInfo(
            id=str(provider_user_id),
            username=data.get("username"),
            email=str(data["email"]).lower() if data.get("email") else None,
            email_verified=None,  # LinuxDo 不返回此字段
            raw=data,  # 包含 trust_level, active, silenced, avatar_url, name 等
        )
