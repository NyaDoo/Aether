"""add_format_conversion_tracking_and_model_filter_patterns_and_provider_timeout

Revision ID: f7c8d9e0a1b2
Revises: 4b4c7b0df1a2
Create Date: 2026-01-27 10:00:00+00:00

Changes:
1. usage 表: 添加 endpoint_api_format 和 has_format_conversion 字段
2. provider_api_keys 表: 添加 model_include_patterns 和 model_exclude_patterns 字段
   - 支持通配符规则自动过滤从上游获取的模型列表
   - 包含规则和排除规则（支持 * 和 ? 通配符）
3. providers 表: 添加 stream_first_byte_timeout 和 request_timeout 字段
   - 允许每个提供商单独配置超时时间
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "f7c8d9e0a1b2"
down_revision = "4b4c7b0df1a2"
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
    # === usage 表: 格式转换追踪 ===
    if table_exists("usage"):
        # 添加 endpoint_api_format 字段（端点原生 API 格式）
        if not column_exists("usage", "endpoint_api_format"):
            op.add_column(
                "usage",
                sa.Column("endpoint_api_format", sa.String(50), nullable=True),
            )

        # 添加 has_format_conversion 字段（是否发生了格式转换）
        if not column_exists("usage", "has_format_conversion"):
            op.add_column(
                "usage",
                sa.Column("has_format_conversion", sa.Boolean(), nullable=True, server_default="false"),
            )

    # === provider_api_keys 表: 模型过滤规则 ===
    if table_exists("provider_api_keys"):
        # 添加 model_include_patterns 字段（包含规则，支持 * 和 ? 通配符）
        if not column_exists("provider_api_keys", "model_include_patterns"):
            op.add_column(
                "provider_api_keys",
                sa.Column("model_include_patterns", sa.JSON(), nullable=True),
            )

        # 添加 model_exclude_patterns 字段（排除规则，支持 * 和 ? 通配符）
        if not column_exists("provider_api_keys", "model_exclude_patterns"):
            op.add_column(
                "provider_api_keys",
                sa.Column("model_exclude_patterns", sa.JSON(), nullable=True),
            )

    # === providers 表: 超时配置 ===
    if table_exists("providers"):
        # 添加 stream_first_byte_timeout 字段（流式请求首字节超时）
        if not column_exists("providers", "stream_first_byte_timeout"):
            op.add_column(
                "providers",
                sa.Column("stream_first_byte_timeout", sa.Float(), nullable=True),
            )

        # 添加 request_timeout 字段（非流式请求整体超时）
        if not column_exists("providers", "request_timeout"):
            op.add_column(
                "providers",
                sa.Column("request_timeout", sa.Float(), nullable=True),
            )


def downgrade() -> None:
    # === providers 表: 移除超时配置 ===
    if table_exists("providers"):
        if column_exists("providers", "request_timeout"):
            op.drop_column("providers", "request_timeout")

        if column_exists("providers", "stream_first_byte_timeout"):
            op.drop_column("providers", "stream_first_byte_timeout")

    # === provider_api_keys 表: 移除模型过滤规则 ===
    if table_exists("provider_api_keys"):
        if column_exists("provider_api_keys", "model_exclude_patterns"):
            op.drop_column("provider_api_keys", "model_exclude_patterns")

        if column_exists("provider_api_keys", "model_include_patterns"):
            op.drop_column("provider_api_keys", "model_include_patterns")

    # === usage 表: 移除格式转换追踪 ===
    if table_exists("usage"):
        if column_exists("usage", "has_format_conversion"):
            op.drop_column("usage", "has_format_conversion")

        if column_exists("usage", "endpoint_api_format"):
            op.drop_column("usage", "endpoint_api_format")
