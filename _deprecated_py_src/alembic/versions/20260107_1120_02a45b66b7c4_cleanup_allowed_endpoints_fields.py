"""cleanup ambiguous database fields

Revision ID: 02a45b66b7c4
Revises: ad55f1d008b7
Create Date: 2026-01-07 11:20:12.684426+00:00

变更内容：
1. users 表：重命名 allowed_endpoints 为 allowed_api_formats（修正历史命名错误）
2. api_keys 表：删除 allowed_endpoints 字段（未使用的功能）
3. providers 表：删除 rate_limit 字段（与 rpm_limit 功能重复，且未使用）
4. usage 表：重命名 provider 为 provider_name（避免与 provider_id 外键混淆）
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = '02a45b66b7c4'
down_revision = 'ad55f1d008b7'
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade() -> None:
    """
    1. users.allowed_endpoints -> allowed_api_formats（重命名）
    2. api_keys.allowed_endpoints 删除
    3. providers.rate_limit 删除（与 rpm_limit 重复）
    4. usage.provider -> provider_name（重命名）
    """
    # 1. users 表：重命名 allowed_endpoints 为 allowed_api_formats
    if _column_exists('users', 'allowed_endpoints'):
        op.alter_column('users', 'allowed_endpoints', new_column_name='allowed_api_formats')

    # 2. api_keys 表：删除 allowed_endpoints 字段
    if _column_exists('api_keys', 'allowed_endpoints'):
        op.drop_column('api_keys', 'allowed_endpoints')

    # 3. providers 表：删除 rate_limit 字段（与 rpm_limit 功能重复）
    if _column_exists('providers', 'rate_limit'):
        op.drop_column('providers', 'rate_limit')

    # 4. usage 表：重命名 provider 为 provider_name
    if _column_exists('usage', 'provider'):
        op.alter_column('usage', 'provider', new_column_name='provider_name')


def downgrade() -> None:
    """回滚：恢复原字段"""
    # 4. usage 表：将 provider_name 改回 provider
    if _column_exists('usage', 'provider_name'):
        op.alter_column('usage', 'provider_name', new_column_name='provider')

    # 3. providers 表：恢复 rate_limit 字段
    if not _column_exists('providers', 'rate_limit'):
        op.add_column('providers', sa.Column('rate_limit', sa.Integer(), nullable=True))

    # 2. api_keys 表：恢复 allowed_endpoints 字段
    if not _column_exists('api_keys', 'allowed_endpoints'):
        op.add_column('api_keys', sa.Column('allowed_endpoints', sa.JSON(), nullable=True))

    # 1. users 表：将 allowed_api_formats 改回 allowed_endpoints
    if _column_exists('users', 'allowed_api_formats'):
        op.alter_column('users', 'allowed_api_formats', new_column_name='allowed_endpoints')
