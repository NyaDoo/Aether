"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    """应用迁移：升级到新版本"""
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    """回滚迁移：降级到旧版本"""
    ${downgrades if downgrades else "pass"}
