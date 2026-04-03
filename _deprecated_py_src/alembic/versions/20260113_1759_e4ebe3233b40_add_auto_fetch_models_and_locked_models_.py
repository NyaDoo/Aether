"""add auto_fetch_models and locked_models to provider_api_keys

Revision ID: e4ebe3233b40
Revises: m4n5o6p7q8r9
Create Date: 2026-01-13 17:59:53.119479+00:00

为 provider_api_keys 表添加自动获取模型相关字段:
1. auto_fetch_models: 是否启用自动获取模型
2. last_models_fetch_at: 最后获取时间
3. last_models_fetch_error: 最后获取错误信息
4. locked_models: 被锁定的模型列表（刷新时不会被删除）

注意: downgrade 操作会永久删除 auto_fetch_models 配置和 locked_models 数据
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


def _index_exists(index_name: str) -> bool:
    """Check if an index exists"""
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = inspector.get_indexes("provider_api_keys")
    return any(idx["name"] == index_name for idx in indexes)


# revision identifiers, used by Alembic.
revision = 'e4ebe3233b40'
down_revision = 'm4n5o6p7q8r9'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in the table"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """添加自动获取模型相关字段"""
    if not _column_exists("provider_api_keys", "auto_fetch_models"):
        op.add_column(
            "provider_api_keys",
            sa.Column("auto_fetch_models", sa.Boolean(), nullable=False, server_default="false"),
        )

    if not _column_exists("provider_api_keys", "last_models_fetch_at"):
        op.add_column(
            "provider_api_keys",
            sa.Column("last_models_fetch_at", sa.DateTime(timezone=True), nullable=True),
        )

    if not _column_exists("provider_api_keys", "last_models_fetch_error"):
        op.add_column(
            "provider_api_keys",
            sa.Column("last_models_fetch_error", sa.Text(), nullable=True),
        )

    if not _column_exists("provider_api_keys", "locked_models"):
        op.add_column(
            "provider_api_keys",
            sa.Column("locked_models", sa.JSON(), nullable=True),
        )

    # 添加复合索引以优化调度器查询
    if not _index_exists("ix_provider_api_keys_auto_fetch_active"):
        op.create_index(
            "ix_provider_api_keys_auto_fetch_active",
            "provider_api_keys",
            ["auto_fetch_models", "is_active"],
            postgresql_where=sa.text("auto_fetch_models = true AND is_active = true"),
        )


def downgrade() -> None:
    """移除自动获取模型相关字段"""
    # 先删除索引
    if _index_exists("ix_provider_api_keys_auto_fetch_active"):
        op.drop_index("ix_provider_api_keys_auto_fetch_active", table_name="provider_api_keys")

    if _column_exists("provider_api_keys", "locked_models"):
        op.drop_column("provider_api_keys", "locked_models")

    if _column_exists("provider_api_keys", "last_models_fetch_error"):
        op.drop_column("provider_api_keys", "last_models_fetch_error")

    if _column_exists("provider_api_keys", "last_models_fetch_at"):
        op.drop_column("provider_api_keys", "last_models_fetch_at")

    if _column_exists("provider_api_keys", "auto_fetch_models"):
        op.drop_column("provider_api_keys", "auto_fetch_models")
