"""Add video_tasks table

Revision ID: b6f1a2c5d8e9
Revises: 7f6f8065f517
Create Date: 2026-01-30 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b6f1a2c5d8e9"
down_revision: Union[str, None] = "7f6f8065f517"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if table_exists("video_tasks"):
        return

    op.create_table(
        "video_tasks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("external_task_id", sa.String(200), nullable=True, index=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("api_key_id", sa.String(36), sa.ForeignKey("api_keys.id"), nullable=True),
        sa.Column("provider_id", sa.String(36), sa.ForeignKey("providers.id"), nullable=True),
        sa.Column(
            "endpoint_id", sa.String(36), sa.ForeignKey("provider_endpoints.id"), nullable=True
        ),
        sa.Column("key_id", sa.String(36), sa.ForeignKey("provider_api_keys.id"), nullable=True),
        sa.Column("client_api_format", sa.String(50), nullable=False),
        sa.Column("provider_api_format", sa.String(50), nullable=False),
        sa.Column("format_converted", sa.Boolean(), server_default=sa.false()),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("original_request_body", sa.JSON(), nullable=True),
        sa.Column("converted_request_body", sa.JSON(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), server_default=sa.text("4")),
        sa.Column("resolution", sa.String(20), server_default=sa.text("'720p'")),
        sa.Column("aspect_ratio", sa.String(10), server_default=sa.text("'16:9'")),
        sa.Column("size", sa.String(20), nullable=True),
        sa.Column("status", sa.String(20), server_default=sa.text("'pending'")),
        sa.Column("progress_percent", sa.Integer(), server_default=sa.text("0")),
        sa.Column("progress_message", sa.String(500), nullable=True),
        sa.Column("video_url", sa.String(2000), nullable=True),
        sa.Column("video_urls", sa.JSON(), nullable=True),
        sa.Column("thumbnail_url", sa.String(2000), nullable=True),
        sa.Column("video_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("video_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("stored_video_path", sa.String(500), nullable=True),
        sa.Column("storage_provider", sa.String(50), nullable=True),
        sa.Column("error_code", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("max_retries", sa.Integer(), server_default=sa.text("3")),
        sa.Column("poll_interval_seconds", sa.Integer(), server_default=sa.text("10")),
        sa.Column("next_poll_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("poll_count", sa.Integer(), server_default=sa.text("0")),
        sa.Column("max_poll_count", sa.Integer(), server_default=sa.text("360")),
        sa.Column(
            "remixed_from_task_id",
            sa.String(36),
            sa.ForeignKey("video_tasks.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )

    op.create_index("idx_video_tasks_user_status", "video_tasks", ["user_id", "status"])
    op.create_index("idx_video_tasks_next_poll", "video_tasks", ["next_poll_at"])
    op.create_index("idx_video_tasks_external_id", "video_tasks", ["external_task_id"])
    # 唯一约束：同一用户不能有重复的 external_task_id
    op.create_unique_constraint(
        "uq_video_tasks_user_external_id",
        "video_tasks",
        ["user_id", "external_task_id"],
    )


def downgrade() -> None:
    if not table_exists("video_tasks"):
        return

    op.drop_constraint("uq_video_tasks_user_external_id", "video_tasks", type_="unique")
    op.drop_index("idx_video_tasks_external_id", table_name="video_tasks")
    op.drop_index("idx_video_tasks_next_poll", table_name="video_tasks")
    op.drop_index("idx_video_tasks_user_status", table_name="video_tasks")
    op.drop_table("video_tasks")
