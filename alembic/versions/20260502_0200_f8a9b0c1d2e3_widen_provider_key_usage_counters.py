"""widen provider api key usage counters

Revision ID: f8a9b0c1d2e3
Revises: e7f8a9b0c1d2
Create Date: 2026-05-02 02:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "f8a9b0c1d2e3"
down_revision: str | None = "e7f8a9b0c1d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE_NAME = "provider_api_keys"
_COUNTER_COLUMNS = (
    "request_count",
    "success_count",
    "error_count",
    "total_response_time_ms",
)


def _column_info(column_name: str) -> dict[str, Any] | None:
    inspector = inspect(op.get_bind())
    for column in inspector.get_columns(_TABLE_NAME):
        if column["name"] == column_name:
            return column
    return None


def _alter_counter_column(column_name: str, target_type: sa.types.TypeEngine[Any]) -> None:
    column = _column_info(column_name)
    if column is None:
        return

    op.alter_column(
        _TABLE_NAME,
        column_name,
        existing_type=column["type"],
        type_=target_type,
        existing_nullable=column.get("nullable", True),
    )


def upgrade() -> None:
    for column_name in _COUNTER_COLUMNS:
        _alter_counter_column(column_name, sa.BigInteger())


def downgrade() -> None:
    for column_name in _COUNTER_COLUMNS:
        _alter_counter_column(column_name, sa.Integer())
