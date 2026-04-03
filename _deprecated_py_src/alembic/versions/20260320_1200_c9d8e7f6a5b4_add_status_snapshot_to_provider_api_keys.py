"""add status_snapshot column to provider_api_keys

Revision ID: c9d8e7f6a5b4
Revises: f6e7d8c9b0a1
Create Date: 2026-03-20 12:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d8e7f6a5b4"
down_revision: str | None = "f6e7d8c9b0a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not column_exists("provider_api_keys", "status_snapshot"):
        op.add_column(
            "provider_api_keys",
            sa.Column("status_snapshot", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    if column_exists("provider_api_keys", "status_snapshot"):
        op.drop_column("provider_api_keys", "status_snapshot")
