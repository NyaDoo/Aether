"""add_management_tokens_table

Revision ID: ad55f1d008b7
Revises: c3d4e5f6g7h8
Create Date: 2026-01-06 15:24:10.660394+00:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'ad55f1d008b7'
down_revision = 'c3d4e5f6g7h8'
branch_labels = None
depends_on = None


def table_exists(table_name: str) -> bool:
    """检查表是否存在"""
    conn = op.get_bind()
    inspector = inspect(conn)
    return table_name in inspector.get_table_names()


def index_exists(table_name: str, index_name: str) -> bool:
    """检查索引是否存在"""
    conn = op.get_bind()
    inspector = inspect(conn)
    try:
        indexes = inspector.get_indexes(table_name)
        return any(idx["name"] == index_name for idx in indexes)
    except Exception:
        return False


def constraint_exists(table_name: str, constraint_name: str) -> bool:
    """检查约束是否存在"""
    conn = op.get_bind()
    inspector = inspect(conn)
    try:
        constraints = inspector.get_unique_constraints(table_name)
        if any(c["name"] == constraint_name for c in constraints):
            return True
        # 也检查 check 约束
        check_constraints = inspector.get_check_constraints(table_name)
        if any(c["name"] == constraint_name for c in check_constraints):
            return True
        return False
    except Exception:
        return False


def upgrade() -> None:
    """应用迁移：创建 management_tokens 表"""
    # 幂等性检查
    if table_exists("management_tokens"):
        # 表已存在，检查是否需要添加约束
        if not constraint_exists("management_tokens", "uq_management_tokens_user_name"):
            op.create_unique_constraint(
                "uq_management_tokens_user_name",
                "management_tokens",
                ["user_id", "name"],
            )
        # 添加 IP 白名单非空检查约束
        if not constraint_exists("management_tokens", "check_allowed_ips_not_empty"):
            op.create_check_constraint(
                "check_allowed_ips_not_empty",
                "management_tokens",
                "allowed_ips IS NULL OR allowed_ips::text = 'null' OR json_array_length(allowed_ips) > 0",
            )
        return

    op.create_table('management_tokens',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('token_hash', sa.String(length=64), nullable=False),
        sa.Column('token_prefix', sa.String(length=12), nullable=True),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('allowed_ips', sa.JSON(), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_used_ip', sa.String(length=45), nullable=True),
        sa.Column('usage_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('is_active', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_management_tokens_is_active', 'management_tokens', ['is_active'], unique=False)
    op.create_index('idx_management_tokens_user_id', 'management_tokens', ['user_id'], unique=False)
    op.create_index(op.f('ix_management_tokens_token_hash'), 'management_tokens', ['token_hash'], unique=True)
    # 添加用户名称唯一约束
    op.create_unique_constraint(
        "uq_management_tokens_user_name",
        "management_tokens",
        ["user_id", "name"],
    )
    # 添加 IP 白名单非空检查约束
    # 注意：JSON 类型的 NULL 可能被序列化为 JSON 'null'，需要同时处理
    op.create_check_constraint(
        "check_allowed_ips_not_empty",
        "management_tokens",
        "allowed_ips IS NULL OR allowed_ips::text = 'null' OR json_array_length(allowed_ips) > 0",
    )


def downgrade() -> None:
    """回滚迁移：删除 management_tokens 表"""
    # 幂等性检查
    if not table_exists("management_tokens"):
        return

    # 删除约束
    if constraint_exists("management_tokens", "check_allowed_ips_not_empty"):
        op.drop_constraint("check_allowed_ips_not_empty", "management_tokens", type_="check")
    if constraint_exists("management_tokens", "uq_management_tokens_user_name"):
        op.drop_constraint("uq_management_tokens_user_name", "management_tokens", type_="unique")

    # 删除索引
    if index_exists("management_tokens", "ix_management_tokens_token_hash"):
        op.drop_index(op.f('ix_management_tokens_token_hash'), table_name='management_tokens')
    if index_exists("management_tokens", "idx_management_tokens_user_id"):
        op.drop_index('idx_management_tokens_user_id', table_name='management_tokens')
    if index_exists("management_tokens", "idx_management_tokens_is_active"):
        op.drop_index('idx_management_tokens_is_active', table_name='management_tokens')

    # 删除表
    op.drop_table('management_tokens')
