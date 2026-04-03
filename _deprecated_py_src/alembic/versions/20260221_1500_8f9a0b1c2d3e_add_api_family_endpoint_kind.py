"""Add api_family and endpoint_kind columns to usage table

Revision ID: 8f9a0b1c2d3e
Revises: 7e8f9a0b1c2d
Create Date: 2026-02-21 15:00:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect, text

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f9a0b1c2d3e"
down_revision: str | None = "7e8f9a0b1c2d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Usage 表新增列
NEW_COLUMNS = [
    ("api_family", sa.String(50)),
    ("endpoint_kind", sa.String(50)),
    ("provider_api_family", sa.String(50)),
    ("provider_endpoint_kind", sa.String(50)),
]

# 新增索引
NEW_INDEXES = [
    ("idx_usage_api_family", "usage", ["api_family"]),
    ("idx_usage_endpoint_kind", "usage", ["endpoint_kind"]),
    ("idx_usage_family_kind", "usage", ["api_family", "endpoint_kind"]),
]


def upgrade() -> None:
    conn = op.get_bind()

    # 使用 PostgreSQL 原生 IF NOT EXISTS，比 inspect 更可靠（避免同一事务内缓存问题）
    col_definitions = {
        "api_family": "VARCHAR(50)",
        "endpoint_kind": "VARCHAR(50)",
        "provider_api_family": "VARCHAR(50)",
        "provider_endpoint_kind": "VARCHAR(50)",
    }
    for col_name, col_type_sql in col_definitions.items():
        conn.execute(text(f"ALTER TABLE usage ADD COLUMN IF NOT EXISTS {col_name} {col_type_sql}"))

    # 数据迁移：从 api_format 解析 api_family + endpoint_kind
    conn.execute(text("""
            UPDATE usage SET
                api_family = lower(split_part(api_format, ':', 1)),
                endpoint_kind = lower(split_part(api_format, ':', 2))
            WHERE api_format IS NOT NULL
              AND api_format LIKE '%%:%%'
              AND api_family IS NULL
        """))
    conn.execute(text("""
            UPDATE usage SET
                provider_api_family = lower(split_part(endpoint_api_format, ':', 1)),
                provider_endpoint_kind = lower(split_part(endpoint_api_format, ':', 2))
            WHERE endpoint_api_format IS NOT NULL
              AND endpoint_api_format LIKE '%%:%%'
              AND provider_api_family IS NULL
        """))

    # 创建索引
    inspector = inspect(conn)
    existing_indexes = {idx["name"] for idx in inspector.get_indexes("usage")}
    for idx_name, table, columns in NEW_INDEXES:
        if idx_name not in existing_indexes:
            op.create_index(idx_name, table, columns)


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)

    existing_indexes = {idx["name"] for idx in inspector.get_indexes("usage")}
    for idx_name, _, _ in reversed(NEW_INDEXES):
        if idx_name in existing_indexes:
            op.drop_index(idx_name, table_name="usage")

    existing_columns = {col["name"] for col in inspector.get_columns("usage")}
    for col_name, _ in reversed(NEW_COLUMNS):
        if col_name in existing_columns:
            op.drop_column("usage", col_name)
