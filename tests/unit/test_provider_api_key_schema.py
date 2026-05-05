from __future__ import annotations

from sqlalchemy import BigInteger

from src.models.database import ProviderAPIKey


def test_provider_api_key_lifetime_counters_use_bigint() -> None:
    columns = ProviderAPIKey.__table__.c

    for column_name in (
        "request_count",
        "success_count",
        "error_count",
        "total_response_time_ms",
    ):
        assert isinstance(columns[column_name].type, BigInteger)
