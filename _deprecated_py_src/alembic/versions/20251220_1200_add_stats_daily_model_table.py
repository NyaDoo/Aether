"""add stats_daily_model table and rename provider_model_aliases

Revision ID: a1b2c3d4e5f6
Revises: f30f9936f6a2
Create Date: 2025-12-20 12:00:00.000000+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = 'f30f9936f6a2'
branch_labels = None
depends_on = None


def table_exists(table_name: str) -> bool:
    """检查表是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """创建 stats_daily_model 表，重命名 provider_model_aliases 为 provider_model_mappings"""
    # 1. 创建 stats_daily_model 表
    if not table_exists('stats_daily_model'):
        op.create_table(
            'stats_daily_model',
            sa.Column('id', sa.String(36), primary_key=True),
            sa.Column('date', sa.DateTime(timezone=True), nullable=False),
            sa.Column('model', sa.String(100), nullable=False),
            sa.Column('total_requests', sa.Integer(), nullable=False, default=0),
            sa.Column('input_tokens', sa.BigInteger(), nullable=False, default=0),
            sa.Column('output_tokens', sa.BigInteger(), nullable=False, default=0),
            sa.Column('cache_creation_tokens', sa.BigInteger(), nullable=False, default=0),
            sa.Column('cache_read_tokens', sa.BigInteger(), nullable=False, default=0),
            sa.Column('total_cost', sa.Float(), nullable=False, default=0.0),
            sa.Column('avg_response_time_ms', sa.Float(), nullable=False, default=0.0),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                      server_default=sa.func.now()),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                      server_default=sa.func.now(), onupdate=sa.func.now()),
            sa.UniqueConstraint('date', 'model', name='uq_stats_daily_model'),
        )

        # 创建索引
        op.create_index('idx_stats_daily_model_date', 'stats_daily_model', ['date'])
        op.create_index('idx_stats_daily_model_date_model', 'stats_daily_model', ['date', 'model'])

    # 2. 重命名 models 表的 provider_model_aliases 为 provider_model_mappings
    if column_exists('models', 'provider_model_aliases') and not column_exists('models', 'provider_model_mappings'):
        op.alter_column('models', 'provider_model_aliases', new_column_name='provider_model_mappings')


def index_exists(table_name: str, index_name: str) -> bool:
    """检查索引是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = [idx['name'] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def downgrade() -> None:
    """删除 stats_daily_model 表，恢复 provider_model_aliases 列名"""
    # 恢复列名
    if column_exists('models', 'provider_model_mappings') and not column_exists('models', 'provider_model_aliases'):
        op.alter_column('models', 'provider_model_mappings', new_column_name='provider_model_aliases')

    # 删除表
    if table_exists('stats_daily_model'):
        if index_exists('stats_daily_model', 'idx_stats_daily_model_date_model'):
            op.drop_index('idx_stats_daily_model_date_model', table_name='stats_daily_model')
        if index_exists('stats_daily_model', 'idx_stats_daily_model_date'):
            op.drop_index('idx_stats_daily_model_date', table_name='stats_daily_model')
        op.drop_table('stats_daily_model')
