"""add_stats_hourly_and_daily_complete_flag

Revision ID: c4e8f9a1b2c3
Revises: b3c4d5e6f7a8
Create Date: 2026-02-04 12:00:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e8f9a1b2c3"
down_revision: Union[str, None] = "b3c4d5e6f7a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    indexes = [idx["name"] for idx in inspector.get_indexes(table_name)]
    return index_name in indexes


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    # Use information_schema for more reliable detection (inspector can have caching issues)
    result = bind.execute(
        sa.text(
            "SELECT EXISTS ("
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
            ")"
        ),
        {"table": table_name, "column": column_name},
    )
    return bool(result.scalar())


def upgrade() -> None:
    if _table_exists("stats_daily"):
        if not _column_exists("stats_daily", "is_complete"):
            op.add_column(
                "stats_daily",
                sa.Column("is_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
            )
            op.execute("UPDATE stats_daily SET is_complete = true")
        if not _column_exists("stats_daily", "aggregated_at"):
            op.add_column(
                "stats_daily",
                sa.Column("aggregated_at", sa.DateTime(timezone=True), nullable=True),
            )

    if not _table_exists("stats_hourly"):
        op.create_table(
            "stats_hourly",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("hour_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("total_requests", sa.Integer(), nullable=False),
            sa.Column("success_requests", sa.Integer(), nullable=False),
            sa.Column("error_requests", sa.Integer(), nullable=False),
            sa.Column("input_tokens", sa.BigInteger(), nullable=False),
            sa.Column("output_tokens", sa.BigInteger(), nullable=False),
            sa.Column("cache_creation_tokens", sa.BigInteger(), nullable=False),
            sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False),
            sa.Column("total_cost", sa.Float(), nullable=False),
            sa.Column("actual_total_cost", sa.Float(), nullable=False),
            sa.Column("avg_response_time_ms", sa.Float(), nullable=False),
            sa.Column("is_complete", sa.Boolean(), nullable=False),
            sa.Column("aggregated_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("hour_utc", name="uq_stats_hourly_hour"),
        )
        op.create_index("idx_stats_hourly_hour", "stats_hourly", ["hour_utc"], unique=False)

    if not _table_exists("stats_hourly_user"):
        op.create_table(
            "stats_hourly_user",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("hour_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("total_requests", sa.Integer(), nullable=False),
            sa.Column("success_requests", sa.Integer(), nullable=False),
            sa.Column("error_requests", sa.Integer(), nullable=False),
            sa.Column("input_tokens", sa.BigInteger(), nullable=False),
            sa.Column("output_tokens", sa.BigInteger(), nullable=False),
            sa.Column("total_cost", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("hour_utc", "user_id", name="uq_stats_hourly_user"),
        )
        op.create_index(
            "idx_stats_hourly_user_hour", "stats_hourly_user", ["hour_utc"], unique=False
        )
        op.create_index(
            "idx_stats_hourly_user_user_hour",
            "stats_hourly_user",
            ["user_id", "hour_utc"],
            unique=False,
        )

    if not _table_exists("stats_hourly_model"):
        op.create_table(
            "stats_hourly_model",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("hour_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("model", sa.String(length=100), nullable=False),
            sa.Column("total_requests", sa.Integer(), nullable=False),
            sa.Column("input_tokens", sa.BigInteger(), nullable=False),
            sa.Column("output_tokens", sa.BigInteger(), nullable=False),
            sa.Column("total_cost", sa.Float(), nullable=False),
            sa.Column("avg_response_time_ms", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("hour_utc", "model", name="uq_stats_hourly_model"),
        )
        op.create_index(
            "idx_stats_hourly_model_hour", "stats_hourly_model", ["hour_utc"], unique=False
        )
        op.create_index(
            "idx_stats_hourly_model_model_hour",
            "stats_hourly_model",
            ["model", "hour_utc"],
            unique=False,
        )

    if not _table_exists("stats_hourly_provider"):
        op.create_table(
            "stats_hourly_provider",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("hour_utc", sa.DateTime(timezone=True), nullable=False),
            sa.Column("provider_name", sa.String(length=100), nullable=False),
            sa.Column("total_requests", sa.Integer(), nullable=False),
            sa.Column("input_tokens", sa.BigInteger(), nullable=False),
            sa.Column("output_tokens", sa.BigInteger(), nullable=False),
            sa.Column("total_cost", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("hour_utc", "provider_name", name="uq_stats_hourly_provider"),
        )
        op.create_index(
            "idx_stats_hourly_provider_hour",
            "stats_hourly_provider",
            ["hour_utc"],
            unique=False,
        )

    if not _table_exists("stats_daily_api_key"):
        op.create_table(
            "stats_daily_api_key",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "api_key_id",
                sa.String(length=36),
                sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("total_requests", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("success_requests", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("error_requests", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("input_tokens", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("output_tokens", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("cache_creation_tokens", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False, server_default="0"),
            sa.Column("total_cost", sa.Float(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.UniqueConstraint("api_key_id", "date", name="uq_stats_daily_api_key"),
        )

    if _table_exists("stats_daily_api_key"):
        if not _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date"):
            op.create_index("idx_stats_daily_api_key_date", "stats_daily_api_key", ["date"])
        if not _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_key_date"):
            op.create_index(
                "idx_stats_daily_api_key_key_date",
                "stats_daily_api_key",
                ["api_key_id", "date"],
            )
        if not _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date_requests"):
            op.create_index(
                "idx_stats_daily_api_key_date_requests",
                "stats_daily_api_key",
                ["date", "total_requests"],
            )
        if not _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date_cost"):
            op.create_index(
                "idx_stats_daily_api_key_date_cost",
                "stats_daily_api_key",
                ["date", "total_cost"],
            )

    if _table_exists("usage"):
        if not _column_exists("usage", "error_category"):
            op.add_column(
                "usage",
                sa.Column("error_category", sa.String(length=50), nullable=True),
            )
            op.create_index("idx_usage_error_category", "usage", ["error_category"], unique=False)

    if _table_exists("stats_daily"):
        for name in (
            "p50_response_time_ms",
            "p90_response_time_ms",
            "p99_response_time_ms",
            "p50_first_byte_time_ms",
            "p90_first_byte_time_ms",
            "p99_first_byte_time_ms",
        ):
            if not _column_exists("stats_daily", name):
                op.add_column("stats_daily", sa.Column(name, sa.Integer(), nullable=True))

    if not _table_exists("stats_daily_error"):
        op.create_table(
            "stats_daily_error",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("date", sa.DateTime(timezone=True), nullable=False),
            sa.Column("error_category", sa.String(length=50), nullable=False),
            sa.Column("provider_name", sa.String(length=100), nullable=True),
            sa.Column("model", sa.String(length=100), nullable=True),
            sa.Column("count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.UniqueConstraint(
                "date",
                "error_category",
                "provider_name",
                "model",
                name="uq_stats_daily_error",
            ),
        )

    if _table_exists("stats_daily_error"):
        if not _index_exists("stats_daily_error", "idx_stats_daily_error_date"):
            op.create_index("idx_stats_daily_error_date", "stats_daily_error", ["date"])
        if not _index_exists("stats_daily_error", "idx_stats_daily_error_category"):
            op.create_index(
                "idx_stats_daily_error_category",
                "stats_daily_error",
                ["date", "error_category"],
            )


def downgrade() -> None:
    if _table_exists("stats_daily_error"):
        if _index_exists("stats_daily_error", "idx_stats_daily_error_category"):
            op.drop_index("idx_stats_daily_error_category", table_name="stats_daily_error")
        if _index_exists("stats_daily_error", "idx_stats_daily_error_date"):
            op.drop_index("idx_stats_daily_error_date", table_name="stats_daily_error")
        op.drop_table("stats_daily_error")

    if _table_exists("stats_daily"):
        for name in (
            "p50_response_time_ms",
            "p90_response_time_ms",
            "p99_response_time_ms",
            "p50_first_byte_time_ms",
            "p90_first_byte_time_ms",
            "p99_first_byte_time_ms",
        ):
            if _column_exists("stats_daily", name):
                op.drop_column("stats_daily", name)

    if _table_exists("usage") and _column_exists("usage", "error_category"):
        if _index_exists("usage", "idx_usage_error_category"):
            op.drop_index("idx_usage_error_category", table_name="usage")
        op.drop_column("usage", "error_category")

    if _table_exists("stats_daily_api_key"):
        if _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date_cost"):
            op.drop_index("idx_stats_daily_api_key_date_cost", table_name="stats_daily_api_key")
        if _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date_requests"):
            op.drop_index("idx_stats_daily_api_key_date_requests", table_name="stats_daily_api_key")
        if _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_key_date"):
            op.drop_index("idx_stats_daily_api_key_key_date", table_name="stats_daily_api_key")
        if _index_exists("stats_daily_api_key", "idx_stats_daily_api_key_date"):
            op.drop_index("idx_stats_daily_api_key_date", table_name="stats_daily_api_key")
        op.drop_table("stats_daily_api_key")

    if _table_exists("stats_hourly_provider"):
        if _index_exists("stats_hourly_provider", "idx_stats_hourly_provider_hour"):
            op.drop_index("idx_stats_hourly_provider_hour", table_name="stats_hourly_provider")
        op.drop_table("stats_hourly_provider")

    if _table_exists("stats_hourly_model"):
        if _index_exists("stats_hourly_model", "idx_stats_hourly_model_model_hour"):
            op.drop_index("idx_stats_hourly_model_model_hour", table_name="stats_hourly_model")
        if _index_exists("stats_hourly_model", "idx_stats_hourly_model_hour"):
            op.drop_index("idx_stats_hourly_model_hour", table_name="stats_hourly_model")
        op.drop_table("stats_hourly_model")

    if _table_exists("stats_hourly_user"):
        if _index_exists("stats_hourly_user", "idx_stats_hourly_user_user_hour"):
            op.drop_index("idx_stats_hourly_user_user_hour", table_name="stats_hourly_user")
        if _index_exists("stats_hourly_user", "idx_stats_hourly_user_hour"):
            op.drop_index("idx_stats_hourly_user_hour", table_name="stats_hourly_user")
        op.drop_table("stats_hourly_user")

    if _table_exists("stats_hourly"):
        if _index_exists("stats_hourly", "idx_stats_hourly_hour"):
            op.drop_index("idx_stats_hourly_hour", table_name="stats_hourly")
        op.drop_table("stats_hourly")

    if _table_exists("stats_daily"):
        if _column_exists("stats_daily", "aggregated_at"):
            op.drop_column("stats_daily", "aggregated_at")
        if _column_exists("stats_daily", "is_complete"):
            op.drop_column("stats_daily", "is_complete")
