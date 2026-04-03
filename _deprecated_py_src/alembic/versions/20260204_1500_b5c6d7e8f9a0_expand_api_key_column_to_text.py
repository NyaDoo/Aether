"""Add provider_type, upstream_metadata, oauth_invalid fields and expand string columns to TEXT

- Add providers.provider_type (String(20), server_default="custom")
- Add provider_api_keys.upstream_metadata (JSON, nullable)
- Add provider_api_keys.oauth_invalid_at (DateTime, nullable) - OAuth Token 失效时间
- Add provider_api_keys.oauth_invalid_reason (String(255), nullable) - OAuth Token 失效原因
- Expand multiple VARCHAR columns to TEXT for long values (OAuth tokens, LDAP DN, URLs, etc.)

Revision ID: b5c6d7e8f9a0
Revises: c4e8f9a1b2c3
Create Date: 2026-02-04 15:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b5c6d7e8f9a0"
down_revision: Union[str, None] = "c4e8f9a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# 需要扩展为 TEXT 的列（表名, 列名, 原始类型长度）
COLUMNS_TO_EXPAND = [
    ("provider_api_keys", "api_key", 500),  # OAuth tokens can be very long
    (
        "provider_api_keys",
        "auth_config",
        None,
    ),  # 确保 auth_config 是 TEXT 类型（可能从 JSON 迁移过来）
    ("ldap_configs", "bind_dn", 255),  # LDAP DN can be deeply nested
    ("ldap_configs", "base_dn", 255),  # LDAP DN can be deeply nested
    ("ldap_configs", "user_search_filter", 500),  # Complex LDAP filters
    ("oauth_providers", "client_id", 255),  # Some OAuth providers use JWT client_id
]


def column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否已存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name: str) -> bool:
    """检查表是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def is_sqlite() -> bool:
    """检查是否为 SQLite 数据库"""
    bind = op.get_bind()
    return bind.dialect.name == "sqlite"


def get_column_type(table_name: str, column_name: str) -> str | None:
    """获取列的数据类型"""
    bind = op.get_bind()
    inspector = inspect(bind)
    for col in inspector.get_columns(table_name):
        if col["name"] == column_name:
            return str(col["type"]).upper()
    return None


def expand_column_to_text(table_name: str, column_name: str, original_length: int | None) -> None:
    """将 VARCHAR 列扩展为 TEXT（兼容 SQLite）"""
    if not table_exists(table_name):
        return
    if not column_exists(table_name, column_name):
        return

    # 检查当前列类型，如果已经是 TEXT 则跳过
    col_type = get_column_type(table_name, column_name)
    if col_type and "TEXT" in col_type:
        return

    # 如果是 JSON 类型（可能是历史遗留），先将 JSON 数据转为文本表示再变更类型
    is_json_col = col_type and "JSON" in col_type

    if is_json_col and not is_sqlite():
        # PostgreSQL: 先用 CAST 把 JSON 值转为 TEXT，保留数据
        op.execute(
            sa.text(
                f"ALTER TABLE {table_name} ALTER COLUMN {column_name} "
                f"TYPE TEXT USING {column_name}::TEXT"
            )
        )
        return

    if is_sqlite():
        # SQLite 不支持直接 ALTER COLUMN，需要用 batch 模式
        # batch 模式会自动处理 JSON->TEXT 的数据迁移
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                column_name,
                type_=sa.Text(),
                existing_type=sa.String(original_length) if original_length else sa.Text(),
            )
    else:
        op.alter_column(
            table_name,
            column_name,
            type_=sa.Text(),
            existing_type=sa.String(original_length) if original_length else sa.Text(),
            existing_nullable=True,
        )


def shrink_column_to_varchar(
    table_name: str, column_name: str, target_length: int, nullable: bool = False
) -> None:
    """将 TEXT 列缩小为 VARCHAR（兼容 SQLite）
    WARNING: 如果数据超过 target_length 会失败
    """
    if not table_exists(table_name):
        return
    if not column_exists(table_name, column_name):
        return

    if is_sqlite():
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                column_name,
                type_=sa.String(target_length),
                existing_type=sa.Text(),
                existing_nullable=nullable,
            )
    else:
        op.alter_column(
            table_name,
            column_name,
            type_=sa.String(target_length),
            existing_type=sa.Text(),
            existing_nullable=nullable,
        )


def upgrade() -> None:
    # Add providers.provider_type
    if not column_exists("providers", "provider_type"):
        op.add_column(
            "providers",
            sa.Column("provider_type", sa.String(20), nullable=False, server_default="custom"),
        )

    # Add provider_api_keys.upstream_metadata
    if not column_exists("provider_api_keys", "upstream_metadata"):
        op.add_column(
            "provider_api_keys",
            sa.Column("upstream_metadata", sa.JSON(), nullable=True),
        )

    # Add provider_api_keys.oauth_invalid_at
    if not column_exists("provider_api_keys", "oauth_invalid_at"):
        op.add_column(
            "provider_api_keys",
            sa.Column("oauth_invalid_at", sa.DateTime(timezone=True), nullable=True),
        )

    # Add provider_api_keys.oauth_invalid_reason
    if not column_exists("provider_api_keys", "oauth_invalid_reason"):
        op.add_column(
            "provider_api_keys",
            sa.Column("oauth_invalid_reason", sa.String(255), nullable=True),
        )

    # Expand VARCHAR columns to TEXT
    for table_name, column_name, original_length in COLUMNS_TO_EXPAND:
        expand_column_to_text(table_name, column_name, original_length)


def downgrade() -> None:
    # Shrink TEXT columns back to VARCHAR
    # WARNING: Downgrade may fail if any values exceed original length
    for table_name, column_name, original_length in reversed(COLUMNS_TO_EXPAND):
        # 跳过没有原始长度的列（如 auth_config，由其他迁移创建）
        if original_length is None:
            continue
        shrink_column_to_varchar(table_name, column_name, original_length)

    # Drop provider_api_keys.oauth_invalid_reason
    if column_exists("provider_api_keys", "oauth_invalid_reason"):
        op.drop_column("provider_api_keys", "oauth_invalid_reason")

    # Drop provider_api_keys.oauth_invalid_at
    if column_exists("provider_api_keys", "oauth_invalid_at"):
        op.drop_column("provider_api_keys", "oauth_invalid_at")

    # Drop provider_api_keys.upstream_metadata
    if column_exists("provider_api_keys", "upstream_metadata"):
        op.drop_column("provider_api_keys", "upstream_metadata")

    # Drop providers.provider_type
    if column_exists("providers", "provider_type"):
        op.drop_column("providers", "provider_type")
