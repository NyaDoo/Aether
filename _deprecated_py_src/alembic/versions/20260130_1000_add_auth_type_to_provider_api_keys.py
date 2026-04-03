"""Add auth_type and auth_config fields to provider_api_keys table

Revision ID: 7f6f8065f517
Revises: 364680d1bc99
Create Date: 2026-01-30 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "7f6f8065f517"
down_revision: Union[str, None] = "364680d1bc99"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否已存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # 添加 auth_type 字段，默认值为 "api_key"
    if not column_exists("provider_api_keys", "auth_type"):
        op.add_column(
            "provider_api_keys",
            sa.Column("auth_type", sa.String(20), nullable=False, server_default="api_key"),
        )

    # 添加 auth_config 字段（Text，存储加密后的认证配置）
    if not column_exists("provider_api_keys", "auth_config"):
        op.add_column(
            "provider_api_keys",
            sa.Column("auth_config", sa.Text, nullable=True),
        )


def downgrade() -> None:
    if column_exists("provider_api_keys", "auth_config"):
        op.drop_column("provider_api_keys", "auth_config")

    if column_exists("provider_api_keys", "auth_type"):
        op.drop_column("provider_api_keys", "auth_type")
