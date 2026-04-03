"""add_idx_usage_status_user_created

Add composite index on usage(status, user_id, created_at) to speed up
interval timeline and active usage analytics queries.

Revision ID: 5f1d2e3c4b5a
Revises: 0ba031f328de
Create Date: 2026-03-03 17:30:00.000000+00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "5f1d2e3c4b5a"
down_revision = "0ba031f328de"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_usage_status_user_created"
TABLE = "usage"
COLUMNS = ["status", "user_id", "created_at"]


def upgrade() -> None:
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
        {"name": INDEX_NAME},
    ).fetchone()
    if result:
        return
    op.create_index(INDEX_NAME, TABLE, COLUMNS)


def downgrade() -> None:
    bind = op.get_bind()
    result = bind.execute(
        sa.text("SELECT 1 FROM pg_indexes WHERE indexname = :name"),
        {"name": INDEX_NAME},
    ).fetchone()
    if not result:
        return
    op.drop_index(INDEX_NAME, table_name=TABLE)
