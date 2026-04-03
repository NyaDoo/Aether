"""Add billing system tables and video_tasks.request_metadata

Revision ID: c8d2e4f6a1b3
Revises: b6f1a2c5d8e9
Create Date: 2026-01-31 12:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8d2e4f6a1b3"
down_revision: Union[str, None] = "b6f1a2c5d8e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    try:
        indexes = inspector.get_indexes(table_name)
    except Exception:
        return False
    return any(idx.get("name") == index_name for idx in indexes)


def upgrade() -> None:
    # ==================== video_tasks.request_metadata ====================
    if not column_exists("video_tasks", "request_metadata"):
        op.add_column(
            "video_tasks",
            sa.Column("request_metadata", sa.JSON(), nullable=True),
        )

    # ==================== billing_rules ====================
    if not table_exists("billing_rules"):
        op.create_table(
            "billing_rules",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "global_model_id",
                sa.String(36),
                sa.ForeignKey("global_models.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column(
                "model_id",
                sa.String(36),
                sa.ForeignKey("models.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("name", sa.String(100), nullable=False),
            sa.Column("task_type", sa.String(20), nullable=False, server_default="chat"),
            sa.Column("expression", sa.Text(), nullable=False),
            sa.Column("variables", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column(
                "dimension_mappings", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
            ),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.CheckConstraint(
                "(global_model_id IS NOT NULL AND model_id IS NULL) OR "
                "(global_model_id IS NULL AND model_id IS NOT NULL)",
                name="chk_billing_rules_model_ref",
            ),
        )

    # Partial unique indexes for enabled rules
    if table_exists("billing_rules"):
        if not index_exists("billing_rules", "uq_billing_rules_global_model_task"):
            op.create_index(
                "uq_billing_rules_global_model_task",
                "billing_rules",
                ["global_model_id", "task_type"],
                unique=True,
                postgresql_where=sa.text("is_enabled = TRUE AND global_model_id IS NOT NULL"),
            )
        if not index_exists("billing_rules", "uq_billing_rules_model_task"):
            op.create_index(
                "uq_billing_rules_model_task",
                "billing_rules",
                ["model_id", "task_type"],
                unique=True,
                postgresql_where=sa.text("is_enabled = TRUE AND model_id IS NOT NULL"),
            )

    # ==================== dimension_collectors ====================
    if not table_exists("dimension_collectors"):
        op.create_table(
            "dimension_collectors",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("api_format", sa.String(50), nullable=False),
            sa.Column("task_type", sa.String(20), nullable=False),
            sa.Column("dimension_name", sa.String(100), nullable=False),
            sa.Column("source_type", sa.String(20), nullable=False),
            sa.Column("source_path", sa.String(200), nullable=True),
            sa.Column("value_type", sa.String(20), nullable=False, server_default="float"),
            sa.Column("transform_expression", sa.Text(), nullable=True),
            sa.Column("default_value", sa.String(100), nullable=True),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.CheckConstraint(
                "(source_type = 'computed' AND source_path IS NULL AND transform_expression IS NOT NULL) OR "
                "(source_type != 'computed' AND source_path IS NOT NULL)",
                name="chk_dimension_collectors_source_config",
            ),
        )

    if table_exists("dimension_collectors"):
        if not index_exists("dimension_collectors", "uq_dimension_collectors_enabled"):
            op.create_index(
                "uq_dimension_collectors_enabled",
                "dimension_collectors",
                ["api_format", "task_type", "dimension_name", "priority"],
                unique=True,
                postgresql_where=sa.text("is_enabled = TRUE"),
            )


def downgrade() -> None:
    # Drop in reverse order
    if table_exists("dimension_collectors"):
        if index_exists("dimension_collectors", "uq_dimension_collectors_enabled"):
            op.drop_index("uq_dimension_collectors_enabled", table_name="dimension_collectors")
        op.drop_table("dimension_collectors")

    if table_exists("billing_rules"):
        if index_exists("billing_rules", "uq_billing_rules_model_task"):
            op.drop_index("uq_billing_rules_model_task", table_name="billing_rules")
        if index_exists("billing_rules", "uq_billing_rules_global_model_task"):
            op.drop_index("uq_billing_rules_global_model_task", table_name="billing_rules")
        op.drop_table("billing_rules")

    if column_exists("video_tasks", "request_metadata"):
        op.drop_column("video_tasks", "request_metadata")
