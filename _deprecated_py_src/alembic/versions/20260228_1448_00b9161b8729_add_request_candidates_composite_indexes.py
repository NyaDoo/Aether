"""add_request_candidates_composite_indexes

Revision ID: 00b9161b8729
Revises: 48afe197cc15
Create Date: 2026-02-28 14:48:00.000000+00:00

"""

from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision = "00b9161b8729"
down_revision = "48afe197cc15"
branch_labels = None
depends_on = None


def _index_exists(index_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    indexes = insp.get_indexes("request_candidates")
    return any(idx["name"] == index_name for idx in indexes)


def upgrade() -> None:
    # (request_id, status) - fallback/retry 查询优化
    if not _index_exists("idx_rc_request_id_status"):
        op.create_index(
            "idx_rc_request_id_status",
            "request_candidates",
            ["request_id", "status"],
        )

    # (provider_id, status, created_at) - provider 聚合统计优化
    if not _index_exists("idx_rc_provider_status_created"):
        op.create_index(
            "idx_rc_provider_status_created",
            "request_candidates",
            ["provider_id", "status", "created_at"],
        )


def downgrade() -> None:
    if _index_exists("idx_rc_provider_status_created"):
        op.drop_index("idx_rc_provider_status_created", table_name="request_candidates")
    if _index_exists("idx_rc_request_id_status"):
        op.drop_index("idx_rc_request_id_status", table_name="request_candidates")
