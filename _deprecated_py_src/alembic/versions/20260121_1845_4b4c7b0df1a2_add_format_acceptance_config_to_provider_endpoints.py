"""add_format_acceptance_config_to_provider_endpoints

Revision ID: 4b4c7b0df1a2
Revises: c868729753ad
Create Date: 2026-01-21 18:45:00+00:00

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "4b4c7b0df1a2"
down_revision = "c868729753ad"
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
    if not table_exists("provider_endpoints"):
        return
    if column_exists("provider_endpoints", "format_acceptance_config"):
        return
    op.add_column(
        "provider_endpoints",
        sa.Column("format_acceptance_config", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    if not table_exists("provider_endpoints"):
        return
    if not column_exists("provider_endpoints", "format_acceptance_config"):
        return
    op.drop_column("provider_endpoints", "format_acceptance_config")

