"""add_idx_usage_provider_key

Add composite index on usage(provider_id, provider_api_key_id) to support
the pool management page's per-key usage stats aggregation query.

Revision ID: 0ba031f328de
Revises: dd0278c0a28c
Create Date: 2026-03-03 10:00:00.000000+00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0ba031f328de"
down_revision = "dd0278c0a28c"
branch_labels = None
depends_on = None

INDEX_NAME = "idx_usage_provider_key"
TABLE = "usage"
COLUMNS = ["provider_id", "provider_api_key_id"]


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
