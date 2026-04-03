"""Add proxy column to provider_api_keys for per-key proxy configuration

Revision ID: 6d7e8f9a0b1c
Revises: 5c6d7e8f9a0b
Create Date: 2026-02-08 15:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6d7e8f9a0b1c"
down_revision: str | None = "5c6d7e8f9a0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not column_exists("provider_api_keys", "proxy"):
        op.add_column(
            "provider_api_keys",
            sa.Column(
                "proxy",
                sa.JSON(),
                nullable=True,
                comment="Key 级别代理配置（覆盖 Provider 级别代理），如 {node_id, enabled}",
            ),
        )


def downgrade() -> None:
    if column_exists("provider_api_keys", "proxy"):
        op.drop_column("provider_api_keys", "proxy")
