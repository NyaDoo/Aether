"""add_keep_priority_on_conversion_to_providers

Revision ID: 364680d1bc99
Revises: f7c8d9e0a1b2
Create Date: 2026-01-28 12:00:00+00:00

Changes:
1. providers 表: 添加 keep_priority_on_conversion 字段
   - 格式转换时是否保持提供商原优先级
   - 默认 False：需要格式转换时，候选会被降级到不需要转换的候选之后
   - 设为 True：即使需要格式转换，也保持原优先级排名
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "364680d1bc99"
down_revision = "f7c8d9e0a1b2"
branch_labels = None
depends_on = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    # === providers 表: 添加格式转换优先级保持配置 ===
    if table_exists("providers"):
        if not column_exists("providers", "keep_priority_on_conversion"):
            op.add_column(
                "providers",
                sa.Column(
                    "keep_priority_on_conversion",
                    sa.Boolean(),
                    nullable=False,
                    server_default="false",
                ),
            )


def downgrade() -> None:
    # === providers 表: 移除格式转换优先级保持配置 ===
    if table_exists("providers"):
        if column_exists("providers", "keep_priority_on_conversion"):
            op.drop_column("providers", "keep_priority_on_conversion")
