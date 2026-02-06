from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from src.services.provider.adapters.antigravity.constants import PROD_BASE_URL
from src.services.provider.transport import build_provider_url, get_antigravity_base_url


@dataclass
class _DummyEndpoint:
    base_url: str
    api_format: str
    custom_path: str | None = None
    provider: object | None = None


def test_antigravity_uses_v1internal_path_and_sets_contextvar() -> None:
    endpoint = _DummyEndpoint(
        base_url="https://ignored.example.com",
        api_format="gemini:chat",
        provider=SimpleNamespace(provider_type="antigravity"),
    )

    with patch(
        "src.services.provider.adapters.antigravity.plugin.url_availability.get_ordered_urls",
        return_value=[PROD_BASE_URL],
    ):
        url = build_provider_url(
            endpoint,  # type: ignore[arg-type]
            path_params={"model": "gemini-2.0-flash"},
            is_stream=True,
        )

    assert url.startswith(f"{PROD_BASE_URL}/v1internal:streamGenerateContent")
    assert "alt=sse" in url
    assert get_antigravity_base_url() == PROD_BASE_URL


def test_gemini_cli_non_antigravity_uses_v1beta_path_and_clears_contextvar() -> None:
    endpoint = _DummyEndpoint(
        base_url="https://generativelanguage.googleapis.com",
        api_format="gemini:cli",
        provider=SimpleNamespace(provider_type="gemini_cli"),
    )

    url = build_provider_url(
        endpoint,  # type: ignore[arg-type]
        path_params={"model": "gemini-2.0-flash"},
        is_stream=False,
    )

    assert "/v1beta/models/gemini-2.0-flash:generateContent" in url
    assert get_antigravity_base_url() is None
