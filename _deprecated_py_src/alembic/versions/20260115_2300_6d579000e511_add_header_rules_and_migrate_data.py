"""add header_rules to provider_endpoints and is_locked to api_keys

Revision ID: 6d579000e511
Revises: e4ebe3233b40
Create Date: 2026-01-15 23:00:00.000000+00:00

变更:
1. provider_endpoints 表: 添加 header_rules 字段，迁移 headers 数据
2. api_keys 表: 添加 is_locked 字段（管理员锁定标志）
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = '6d579000e511'
down_revision = 'e4ebe3233b40'
branch_labels = None
depends_on = None


def _column_exists(connection, table: str, column: str) -> bool:
    """检查列是否存在"""
    result = connection.execute(
        sa.text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = :table AND column_name = :column
        """),
        {"table": table, "column": column}
    )
    return result.fetchone() is not None


def upgrade() -> None:
    """添加 header_rules 字段并迁移现有 headers 数据；添加 is_locked 字段"""
    connection = op.get_bind()

    # ========== provider_endpoints.header_rules ==========
    # 1. 添加 header_rules 列（幂等）
    if not _column_exists(connection, 'provider_endpoints', 'header_rules'):
        op.add_column('provider_endpoints', sa.Column('header_rules', JSON, nullable=True))

    # 2. 批量迁移：headers -> header_rules
    # 使用纯 SQL 将 {"k1":"v1", "k2":"v2"} 转换为 [{"action":"set","key":"k1","value":"v1"}, ...]
    if _column_exists(connection, 'provider_endpoints', 'headers'):
        connection.execute(
            sa.text("""
                UPDATE provider_endpoints
                SET header_rules = (
                    SELECT jsonb_agg(
                        jsonb_build_object('action', 'set', 'key', key, 'value', value)
                    )
                    FROM jsonb_each_text(headers::jsonb)
                )
                WHERE headers IS NOT NULL
                  AND headers::text != '{}'
                  AND jsonb_typeof(headers::jsonb) = 'object'
                  AND header_rules IS NULL
            """)
        )

        # 3. 删除旧列
        op.drop_column('provider_endpoints', 'headers')

    # ========== api_keys.is_locked ==========
    if not _column_exists(connection, 'api_keys', 'is_locked'):
        op.add_column(
            'api_keys',
            sa.Column('is_locked', sa.Boolean(), nullable=False, server_default='false')
        )


def downgrade() -> None:
    """移除 header_rules 字段，恢复 headers 字段；移除 is_locked 字段"""
    connection = op.get_bind()

    # ========== api_keys.is_locked ==========
    if _column_exists(connection, 'api_keys', 'is_locked'):
        op.drop_column('api_keys', 'is_locked')

    # ========== provider_endpoints.header_rules ==========
    # 1. 添加 headers 列（幂等）
    if not _column_exists(connection, 'provider_endpoints', 'headers'):
        op.add_column('provider_endpoints', sa.Column('headers', JSON, nullable=True))

    # 2. 批量迁移：header_rules -> headers（仅提取 set 操作）
    if _column_exists(connection, 'provider_endpoints', 'header_rules'):
        connection.execute(
            sa.text("""
                UPDATE provider_endpoints
                SET headers = (
                    SELECT jsonb_object_agg(rule->>'key', rule->>'value')
                    FROM jsonb_array_elements(header_rules::jsonb) AS rule
                    WHERE rule->>'action' = 'set'
                      AND rule->>'key' IS NOT NULL
                )
                WHERE header_rules IS NOT NULL
                  AND jsonb_typeof(header_rules::jsonb) = 'array'
                  AND jsonb_array_length(header_rules::jsonb) > 0
            """)
        )

        # 3. 删除 header_rules 列
        op.drop_column('provider_endpoints', 'header_rules')
