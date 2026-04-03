"""Add tunnel mode fields and remove IP forwarding fields

Revision ID: 9a0b1c2d3e4f
Revises: 8f9a0b1c2d3e
Create Date: 2026-02-24 17:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "9a0b1c2d3e4f"
down_revision: str | None = "8f9a0b1c2d3e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # 添加 tunnel 模式字段
    if not column_exists("proxy_nodes", "tunnel_mode"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tunnel_mode",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
                comment="是否使用 WebSocket 隧道模式",
            ),
        )
    if not column_exists("proxy_nodes", "tunnel_connected"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tunnel_connected",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
                comment="隧道是否已连接",
            ),
        )
    if not column_exists("proxy_nodes", "tunnel_connected_at"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tunnel_connected_at",
                sa.DateTime(timezone=True),
                nullable=True,
                comment="隧道最近一次建立时间",
            ),
        )

    # tunnel 模式节点不需要 port，将其置零
    op.execute("UPDATE proxy_nodes SET port = 0 WHERE tunnel_mode = true")

    # 移除旧的 IP 转发字段
    if column_exists("proxy_nodes", "tls_enabled"):
        op.drop_column("proxy_nodes", "tls_enabled")
    if column_exists("proxy_nodes", "tls_cert_fingerprint"):
        op.drop_column("proxy_nodes", "tls_cert_fingerprint")


def downgrade() -> None:
    # 恢复 IP 转发字段
    if not column_exists("proxy_nodes", "tls_cert_fingerprint"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tls_cert_fingerprint",
                sa.String(128),
                nullable=True,
                comment="TLS 证书 SHA-256 指纹（hex）",
            ),
        )
    if not column_exists("proxy_nodes", "tls_enabled"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tls_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
                comment="是否启用 TLS 加密",
            ),
        )

    # 移除 tunnel 模式字段
    if column_exists("proxy_nodes", "tunnel_connected_at"):
        op.drop_column("proxy_nodes", "tunnel_connected_at")
    if column_exists("proxy_nodes", "tunnel_connected"):
        op.drop_column("proxy_nodes", "tunnel_connected")
    if column_exists("proxy_nodes", "tunnel_mode"):
        op.drop_column("proxy_nodes", "tunnel_mode")
