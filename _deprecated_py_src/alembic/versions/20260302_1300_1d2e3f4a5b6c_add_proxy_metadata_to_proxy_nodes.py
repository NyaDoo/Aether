"""add_proxy_metadata_to_proxy_nodes

Revision ID: 1d2e3f4a5b6c
Revises: f0c3a7b9d1e2
Create Date: 2026-03-02 13:00:00.000000+00:00
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1d2e3f4a5b6c"
down_revision: str | None = "f0c3a7b9d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE public.proxy_nodes ADD COLUMN IF NOT EXISTS proxy_metadata json")
    op.execute(
        "COMMENT ON COLUMN public.proxy_nodes.proxy_metadata IS 'aether-proxy 上报元数据（版本等）'"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE public.proxy_nodes DROP COLUMN IF EXISTS proxy_metadata")
