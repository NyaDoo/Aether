import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.api_format import APIFormat
from src.core.api_format.conversion import register_default_normalizers
from src.services.cache.aware_scheduler import CacheAwareScheduler


def _mock_key(key_id: str, api_formats: list[str]) -> MagicMock:
    key = MagicMock()
    key.id = key_id
    key.is_active = True
    key.api_formats = api_formats
    key.cache_ttl_minutes = 1
    key.internal_priority = 1
    return key


def _mock_endpoint(api_format: str, config: dict | None = None) -> MagicMock:
    endpoint = MagicMock()
    endpoint.id = f"ep_{api_format.lower()}"
    endpoint.is_active = True
    endpoint.api_format = api_format
    endpoint.format_acceptance_config = config
    return endpoint


@pytest.mark.asyncio
async def test_build_candidates_blocks_cross_format_when_global_switch_off() -> None:
    register_default_normalizers()

    scheduler = CacheAwareScheduler()
    scheduler._check_model_support = AsyncMock(return_value=(True, None, None, {"m"}))  # type: ignore[attr-defined]
    scheduler._check_key_availability = MagicMock(return_value=(True, None, None))  # type: ignore[attr-defined]

    provider = MagicMock()
    provider.name = "p1"
    provider.endpoints = [
        _mock_endpoint(
            "OPENAI",
            {"enabled": True, "accept_formats": ["CLAUDE"], "stream_conversion": True},
        )
    ]
    provider.api_keys = [_mock_key("k1", ["OPENAI"])]

    candidates = await scheduler._build_candidates(
        db=MagicMock(),
        providers=[provider],
        client_format=APIFormat.CLAUDE,
        model_name="dummy-model",
        affinity_key=None,
        global_conversion_enabled=False,
    )

    assert candidates == []


@pytest.mark.asyncio
async def test_build_candidates_includes_cross_format_when_enabled() -> None:
    register_default_normalizers()

    scheduler = CacheAwareScheduler()
    scheduler._check_model_support = AsyncMock(return_value=(True, None, None, {"m"}))  # type: ignore[attr-defined]
    scheduler._check_key_availability = MagicMock(return_value=(True, None, None))  # type: ignore[attr-defined]

    provider = MagicMock()
    provider.name = "p1"
    provider.endpoints = [
        _mock_endpoint(
            "OPENAI",
            {"enabled": True, "accept_formats": ["CLAUDE"], "stream_conversion": True},
        )
    ]
    provider.api_keys = [_mock_key("k1", ["OPENAI"])]

    candidates = await scheduler._build_candidates(
        db=MagicMock(),
        providers=[provider],
        client_format=APIFormat.CLAUDE,
        model_name="dummy-model",
        affinity_key=None,
        global_conversion_enabled=True,
    )

    assert len(candidates) == 1
    assert candidates[0].needs_conversion is True
    assert candidates[0].provider_api_format == "OPENAI"


@pytest.mark.asyncio
async def test_exact_matches_rank_before_convertible() -> None:
    register_default_normalizers()

    scheduler = CacheAwareScheduler()
    scheduler._check_model_support = AsyncMock(return_value=(True, None, None, {"m"}))  # type: ignore[attr-defined]
    scheduler._check_key_availability = MagicMock(return_value=(True, None, None))  # type: ignore[attr-defined]

    provider = MagicMock()
    provider.name = "p1"
    # 故意把 OPENAI 放在 endpoints[0]，验证排序仍然是 CLAUDE（exact）在前
    provider.endpoints = [
        _mock_endpoint(
            "OPENAI",
            {"enabled": True, "accept_formats": ["CLAUDE"], "stream_conversion": True},
        ),
        _mock_endpoint("CLAUDE", None),
    ]
    provider.api_keys = [
        _mock_key("k_openai", ["OPENAI"]),
        _mock_key("k_claude", ["CLAUDE"]),
    ]

    candidates = await scheduler._build_candidates(
        db=MagicMock(),
        providers=[provider],
        client_format=APIFormat.CLAUDE,
        model_name="dummy-model",
        affinity_key=None,
        global_conversion_enabled=True,
    )

    assert len(candidates) == 2
    assert candidates[0].needs_conversion is False
    assert candidates[0].provider_api_format == "CLAUDE"
    assert candidates[1].needs_conversion is True
    assert candidates[1].provider_api_format == "OPENAI"
