"""add first_byte_time_ms to usage table

Revision ID: 180e63a9c83a
Revises: e9b3d63f0cbf
Create Date: 2025-12-15 17:07:44.631032+00:00

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = '180e63a9c83a'
down_revision = 'e9b3d63f0cbf'
branch_labels = None
depends_on = None


def column_exists(bind, table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    result = bind.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table_name AND column_name = :column_name
            )
            """
        ),
        {"table_name": table_name, "column_name": column_name},
    )
    return result.scalar()


def upgrade() -> None:
    """应用迁移：升级到新版本"""
    bind = op.get_bind()

    # 添加首字时间字段到 usage 表（如果不存在）
    if not column_exists(bind, "usage", "first_byte_time_ms"):
        op.add_column('usage', sa.Column('first_byte_time_ms', sa.Integer(), nullable=True))


def downgrade() -> None:
    """回滚迁移：降级到旧版本"""
    # 删除首字时间字段
    op.drop_column('usage', 'first_byte_time_ms')
