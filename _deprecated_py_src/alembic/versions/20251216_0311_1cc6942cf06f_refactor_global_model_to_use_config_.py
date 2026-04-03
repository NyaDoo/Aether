"""refactor global_model to use config json field

Revision ID: 1cc6942cf06f
Revises: 180e63a9c83a
Create Date: 2025-12-16 03:11:32.480976+00:00

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '1cc6942cf06f'
down_revision = '180e63a9c83a'
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
    """应用迁移：升级到新版本

    1. 添加 config 列
    2. 把旧数据迁移到 config
    3. 删除旧列
    """
    bind = op.get_bind()

    # 检查是否已经迁移过（config 列存在且旧列不存在）
    has_config = column_exists(bind, "global_models", "config")
    has_old_columns = column_exists(bind, "global_models", "default_supports_streaming")

    if has_config and not has_old_columns:
        # 已完成迁移，跳过
        return

    # 1. 添加 config 列（使用 JSONB 类型，支持索引和更高效的查询）
    if not has_config:
        op.add_column('global_models', sa.Column('config', postgresql.JSONB(), nullable=True))

    # 2. 迁移数据：把旧字段合并到 config JSON（仅当旧列存在时）
    if has_old_columns:
        op.execute("""
            UPDATE global_models
            SET config = jsonb_strip_nulls(jsonb_build_object(
                'streaming', COALESCE(default_supports_streaming, true),
                'vision', CASE WHEN COALESCE(default_supports_vision, false) THEN true ELSE NULL END,
                'function_calling', CASE WHEN COALESCE(default_supports_function_calling, false) THEN true ELSE NULL END,
                'extended_thinking', CASE WHEN COALESCE(default_supports_extended_thinking, false) THEN true ELSE NULL END,
                'image_generation', CASE WHEN COALESCE(default_supports_image_generation, false) THEN true ELSE NULL END,
                'description', description,
                'icon_url', icon_url,
                'official_url', official_url
            ))
        """)

        # 3. 删除旧列
        op.drop_column('global_models', 'default_supports_streaming')
        op.drop_column('global_models', 'default_supports_vision')
        op.drop_column('global_models', 'default_supports_function_calling')
        op.drop_column('global_models', 'default_supports_extended_thinking')
        op.drop_column('global_models', 'default_supports_image_generation')
        op.drop_column('global_models', 'description')
        op.drop_column('global_models', 'icon_url')
        op.drop_column('global_models', 'official_url')


def downgrade() -> None:
    """回滚迁移：降级到旧版本"""
    # 1. 添加旧列
    op.add_column('global_models', sa.Column('icon_url', sa.VARCHAR(length=500), nullable=True))
    op.add_column('global_models', sa.Column('official_url', sa.VARCHAR(length=500), nullable=True))
    op.add_column('global_models', sa.Column('description', sa.TEXT(), nullable=True))
    op.add_column('global_models', sa.Column('default_supports_streaming', sa.BOOLEAN(), nullable=True))
    op.add_column('global_models', sa.Column('default_supports_vision', sa.BOOLEAN(), nullable=True))
    op.add_column('global_models', sa.Column('default_supports_function_calling', sa.BOOLEAN(), nullable=True))
    op.add_column('global_models', sa.Column('default_supports_extended_thinking', sa.BOOLEAN(), nullable=True))
    op.add_column('global_models', sa.Column('default_supports_image_generation', sa.BOOLEAN(), nullable=True))

    # 2. 从 config 恢复数据
    op.execute("""
        UPDATE global_models
        SET
            default_supports_streaming = COALESCE((config->>'streaming')::boolean, true),
            default_supports_vision = COALESCE((config->>'vision')::boolean, false),
            default_supports_function_calling = COALESCE((config->>'function_calling')::boolean, false),
            default_supports_extended_thinking = COALESCE((config->>'extended_thinking')::boolean, false),
            default_supports_image_generation = COALESCE((config->>'image_generation')::boolean, false),
            description = config->>'description',
            icon_url = config->>'icon_url',
            official_url = config->>'official_url'
    """)

    # 3. 删除 config 列
    op.drop_column('global_models', 'config')
