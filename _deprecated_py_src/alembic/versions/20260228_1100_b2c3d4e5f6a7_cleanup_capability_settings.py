"""Add cache_creation columns, clean up capability settings, add user_model_usage_counts,
enforce global_model_id NOT NULL

1. Add cache_creation_input_tokens_5m and cache_creation_input_tokens_1h to usage table.
2. Clean up cache_1h/context_1m/gemini_files from user-configurable settings
   (now auto-detected via REQUEST_PARAM mode).
3. Create user_model_usage_counts table for per-user per-model atomic usage counters.
4. Enforce models.global_model_id NOT NULL (delete orphan models without global model).

Revision ID: b2c3d4e5f6a7
Revises: 9a0b1c2d3e4f
Create Date: 2026-02-28 14:00:00.000000

"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "9a0b1c2d3e4f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    columns = [c["name"] for c in insp.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return table_name in insp.get_table_names()


def index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return any(idx["name"] == index_name for idx in insp.get_indexes(table_name))


def upgrade() -> None:
    # --- 1. Add cache_creation columns ---
    if not column_exists("usage", "cache_creation_input_tokens_5m"):
        op.add_column(
            "usage",
            sa.Column(
                "cache_creation_input_tokens_5m",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
                comment="5min TTL cache creation input tokens",
            ),
        )
    if not column_exists("usage", "cache_creation_input_tokens_1h"):
        op.add_column(
            "usage",
            sa.Column(
                "cache_creation_input_tokens_1h",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
                comment="1h TTL cache creation input tokens",
            ),
        )

    # --- 2. Clean up stale capability settings (pure Python, DB-agnostic) ---
    stale_keys = {"cache_1h", "context_1m", "gemini_files"}
    conn = op.get_bind()

    # ApiKey.force_capabilities: dict-like JSON, remove stale keys
    rows = conn.execute(
        sa.text("SELECT id, force_capabilities FROM api_keys WHERE force_capabilities IS NOT NULL")
    ).fetchall()
    for row in rows:
        raw = row[1]
        if raw is None:
            continue
        data = raw if isinstance(raw, dict) else json.loads(raw)
        cleaned = {k: v for k, v in data.items() if k not in stale_keys}
        new_val = json.dumps(cleaned) if cleaned else None
        conn.execute(
            sa.text("UPDATE api_keys SET force_capabilities = :val WHERE id = :id"),
            {"val": new_val, "id": row[0]},
        )

    # User.model_capability_settings: nested dict {model_key: {cap: val}}, remove stale keys
    rows = conn.execute(
        sa.text(
            "SELECT id, model_capability_settings FROM users"
            " WHERE model_capability_settings IS NOT NULL"
        )
    ).fetchall()
    for row in rows:
        raw = row[1]
        if raw is None:
            continue
        data = raw if isinstance(raw, dict) else json.loads(raw)
        cleaned = {}
        for model_key, caps in data.items():
            cap_cleaned = {k: v for k, v in caps.items() if k not in stale_keys}
            if cap_cleaned:
                cleaned[model_key] = cap_cleaned
        new_val = json.dumps(cleaned) if cleaned else None
        conn.execute(
            sa.text("UPDATE users SET model_capability_settings = :val WHERE id = :id"),
            {"val": new_val, "id": row[0]},
        )

    # GlobalModel.supported_capabilities: JSON array, remove stale entries
    rows = conn.execute(
        sa.text(
            "SELECT id, supported_capabilities FROM global_models"
            " WHERE supported_capabilities IS NOT NULL"
        )
    ).fetchall()
    for row in rows:
        raw = row[1]
        if raw is None:
            continue
        data = raw if isinstance(raw, list) else json.loads(raw)
        cleaned = [c for c in data if c not in stale_keys]
        new_val = json.dumps(cleaned) if cleaned else None
        conn.execute(
            sa.text("UPDATE global_models SET supported_capabilities = :val WHERE id = :id"),
            {"val": new_val, "id": row[0]},
        )

    # --- 3. Create user_model_usage_counts table ---
    if not table_exists("user_model_usage_counts"):
        op.create_table(
            "user_model_usage_counts",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("model", sa.String(100), nullable=False),
            sa.Column("usage_count", sa.Integer, nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.UniqueConstraint("user_id", "model", name="uq_user_model_usage_count"),
        )
    if not index_exists("user_model_usage_counts", "idx_user_model_usage_user"):
        op.create_index("idx_user_model_usage_user", "user_model_usage_counts", ["user_id"])
    if not index_exists("user_model_usage_counts", "idx_user_model_usage_model"):
        op.create_index("idx_user_model_usage_model", "user_model_usage_counts", ["model"])

    # Backfill from existing usage records (truncate first for idempotency)
    conn.execute(sa.text("DELETE FROM user_model_usage_counts"))
    rows = conn.execute(
        sa.text(
            "SELECT user_id, model, COUNT(*) AS cnt FROM usage"
            " WHERE user_id IS NOT NULL GROUP BY user_id, model"
        )
    ).fetchall()
    now = datetime.now(timezone.utc)
    for row in rows:
        conn.execute(
            sa.text(
                "INSERT INTO user_model_usage_counts"
                " (id, user_id, model, usage_count, created_at, updated_at)"
                " VALUES (:id, :user_id, :model, :cnt, :now, :now)"
            ),
            {
                "id": str(uuid.uuid4()),
                "user_id": row[0],
                "model": row[1],
                "cnt": row[2],
                "now": now,
            },
        )

    # --- 4. Enforce models.global_model_id NOT NULL ---
    conn = op.get_bind()
    insp = inspect(conn)
    model_cols = {c["name"]: c for c in insp.get_columns("models")}
    if model_cols.get("global_model_id", {}).get("nullable", True):
        op.execute("DELETE FROM models WHERE global_model_id IS NULL")
        op.alter_column("models", "global_model_id", existing_type=sa.String(36), nullable=False)


def downgrade() -> None:
    # Revert models.global_model_id to nullable
    if column_exists("models", "global_model_id"):
        op.alter_column("models", "global_model_id", existing_type=sa.String(36), nullable=True)

    # Drop user_model_usage_counts
    if table_exists("user_model_usage_counts"):
        if index_exists("user_model_usage_counts", "idx_user_model_usage_model"):
            op.drop_index("idx_user_model_usage_model", table_name="user_model_usage_counts")
        if index_exists("user_model_usage_counts", "idx_user_model_usage_user"):
            op.drop_index("idx_user_model_usage_user", table_name="user_model_usage_counts")
        op.drop_table("user_model_usage_counts")

    # Drop cache_creation columns
    if column_exists("usage", "cache_creation_input_tokens_1h"):
        op.drop_column("usage", "cache_creation_input_tokens_1h")
    if column_exists("usage", "cache_creation_input_tokens_5m"):
        op.drop_column("usage", "cache_creation_input_tokens_5m")
    # capability settings cleanup is not reversible
