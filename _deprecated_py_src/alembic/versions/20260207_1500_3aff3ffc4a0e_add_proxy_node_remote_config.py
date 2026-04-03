"""Add remote_config and config_version to proxy_nodes

Revision ID: 3aff3ffc4a0e
Revises: e1b2c3d4f5a6
Create Date: 2026-02-07 15:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3aff3ffc4a0e"
down_revision: str | None = "e1b2c3d4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not column_exists("proxy_nodes", "remote_config"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "remote_config",
                sa.JSON(),
                nullable=True,
                comment="管理端下发的远程配置 (allowed_ports, log_level, heartbeat_interval, timestamp_tolerance)",
            ),
        )

    if not column_exists("proxy_nodes", "config_version"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "config_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
                comment="远程配置版本号，每次更新 +1",
            ),
        )


def downgrade() -> None:
    if column_exists("proxy_nodes", "config_version"):
        op.drop_column("proxy_nodes", "config_version")
    if column_exists("proxy_nodes", "remote_config"):
        op.drop_column("proxy_nodes", "remote_config")
