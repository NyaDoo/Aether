"""Add tls_enabled and tls_cert_fingerprint to proxy_nodes

Revision ID: 4b5c6d7e8f9a
Revises: 3aff3ffc4a0e
Create Date: 2026-02-07 18:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b5c6d7e8f9a"
down_revision: str | None = "3aff3ffc4a0e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [c["name"] for c in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    if not column_exists("proxy_nodes", "tls_enabled"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "tls_enabled",
                sa.Boolean(),
                nullable=False,
                server_default="false",
                comment="是否启用 TLS 加密",
            ),
        )

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


def downgrade() -> None:
    if column_exists("proxy_nodes", "tls_cert_fingerprint"):
        op.drop_column("proxy_nodes", "tls_cert_fingerprint")
    if column_exists("proxy_nodes", "tls_enabled"):
        op.drop_column("proxy_nodes", "tls_enabled")
