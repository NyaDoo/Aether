from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api.handlers.base.request_builder import get_provider_auth


@pytest.mark.asyncio
async def test_get_provider_auth_oauth_returns_decrypted_auth_config() -> None:
    now = int(time.time())
    token_meta = {
        "provider_type": "antigravity",
        "expires_at": now + 3600,
        "refresh_token": "rt-1",
        "project_id": "project-1",
    }

    endpoint = SimpleNamespace(api_format="gemini:chat")
    key = SimpleNamespace(
        id="k1",
        auth_type="oauth",
        api_key="enc_access",
        auth_config="enc_cfg",
        provider=None,
    )

    def _decrypt(v: str) -> str:
        if v == "enc_access":
            return "access-token"
        if v == "enc_cfg":
            return json.dumps(token_meta)
        return ""

    with patch(
        "src.api.handlers.base.request_builder.crypto_service.decrypt", side_effect=_decrypt
    ):
        auth = await get_provider_auth(endpoint, key)  # type: ignore[arg-type]

    assert auth is not None
    assert auth.auth_header == "Authorization"
    assert auth.auth_value == "Bearer access-token"
    assert auth.decrypted_auth_config == token_meta
