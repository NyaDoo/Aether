"""add proxy field to provider_endpoints

Revision ID: f30f9936f6a2
Revises: 1cc6942cf06f
Create Date: 2025-12-18 06:31:58.451112+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'f30f9936f6a2'
down_revision = '1cc6942cf06f'
branch_labels = None
depends_on = None


def column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def get_column_type(table_name: str, column_name: str) -> str:
    """获取列的类型"""
    bind = op.get_bind()
    inspector = inspect(bind)
    for col in inspector.get_columns(table_name):
        if col['name'] == column_name:
            return str(col['type']).upper()
    return ''


def upgrade() -> None:
    """添加 proxy 字段到 provider_endpoints 表"""
    if not column_exists('provider_endpoints', 'proxy'):
        # 字段不存在，直接添加 JSONB 类型
        op.add_column('provider_endpoints', sa.Column('proxy', JSONB(), nullable=True))
    else:
        # 字段已存在，检查是否需要转换类型
        col_type = get_column_type('provider_endpoints', 'proxy')
        if 'JSONB' not in col_type:
            # 如果是 JSON 类型，转换为 JSONB
            op.execute(
                'ALTER TABLE provider_endpoints '
                'ALTER COLUMN proxy TYPE JSONB USING proxy::jsonb'
            )


def downgrade() -> None:
    """移除 proxy 字段"""
    if column_exists('provider_endpoints', 'proxy'):
        op.drop_column('provider_endpoints', 'proxy')
