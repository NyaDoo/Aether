"""Add hardware_info and estimated_max_concurrency to proxy_nodes

Revision ID: 5c6d7e8f9a0b
Revises: 4b5c6d7e8f9a
Create Date: 2026-02-08 12:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5c6d7e8f9a0b"
down_revision: str | None = "4b5c6d7e8f9a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not column_exists("proxy_nodes", "hardware_info"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "hardware_info",
                sa.JSON(),
                nullable=True,
                comment="硬件信息 (cpu_cores, total_memory_mb, os_info, fd_limit, ...)",
            ),
        )

    if not column_exists("proxy_nodes", "estimated_max_concurrency"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "estimated_max_concurrency",
                sa.Integer(),
                nullable=True,
                comment="基于硬件估算的最大并发连接数",
            ),
        )


def downgrade() -> None:
    if column_exists("proxy_nodes", "estimated_max_concurrency"):
        op.drop_column("proxy_nodes", "estimated_max_concurrency")
    if column_exists("proxy_nodes", "hardware_info"):
        op.drop_column("proxy_nodes", "hardware_info")
