"""usage stats retention: SET NULL on delete and add name snapshots

Revision ID: 45b118150a78
Revises: 2d932114930d
Create Date: 2026-03-08 03:48:49.622091+00:00

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '45b118150a78'
down_revision = '2d932114930d'
branch_labels = None
depends_on = None


# ---------------------------------------------------------------------------
# Idempotent helpers
# ---------------------------------------------------------------------------

def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :col"
        ),
        {"table": table_name, "col": column_name},
    )
    return result.scalar() is not None


def _constraint_exists(table_name: str, constraint_name: str) -> bool:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM information_schema.table_constraints "
            "WHERE table_name = :table AND constraint_name = :name"
        ),
        {"table": table_name, "name": constraint_name},
    )
    return result.scalar() is not None


def _fk_ondelete(table_name: str, constraint_name: str) -> str | None:
    """Return the ON DELETE action for a foreign key, or None if it doesn't exist."""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT rc.delete_rule "
            "FROM information_schema.referential_constraints rc "
            "JOIN information_schema.table_constraints tc "
            "  ON rc.constraint_name = tc.constraint_name "
            "WHERE tc.table_name = :table AND tc.constraint_name = :name"
        ),
        {"table": table_name, "name": constraint_name},
    )
    row = result.first()
    return row[0] if row else None


def _replace_fk_if_needed(
    constraint_name: str,
    table_name: str,
    ref_table: str,
    local_cols: list[str],
    remote_cols: list[str],
    desired_ondelete: str,
) -> None:
    """Drop and recreate a FK only if the current ON DELETE rule differs."""
    current = _fk_ondelete(table_name, constraint_name)
    if current and current.upper() == desired_ondelete.upper():
        return  # already correct
    if current:
        op.drop_constraint(constraint_name, table_name, type_='foreignkey')
    op.create_foreign_key(
        constraint_name, table_name, ref_table,
        local_cols, remote_cols, ondelete=desired_ondelete,
    )


def upgrade() -> None:
    # --- Usage: add name snapshot columns ---
    if not _column_exists('usage', 'username'):
        op.add_column('usage', sa.Column('username', sa.String(100), nullable=True,
                                         comment='用户名快照'))
    if not _column_exists('usage', 'api_key_name'):
        op.add_column('usage', sa.Column('api_key_name', sa.String(200), nullable=True,
                                         comment='API Key 名称快照'))

    # --- StatsUserDaily: CASCADE -> SET NULL, add username snapshot ---
    _replace_fk_if_needed(
        'stats_user_daily_user_id_fkey', 'stats_user_daily',
        'users', ['user_id'], ['id'], 'SET NULL',
    )
    op.alter_column('stats_user_daily', 'user_id', existing_type=sa.String(36), nullable=True)
    if not _column_exists('stats_user_daily', 'username'):
        op.add_column('stats_user_daily', sa.Column('username', sa.String(100), nullable=True,
                                                    comment='用户名快照（删除用户后仍可追溯）'))

    # --- StatsDailyApiKey: CASCADE -> SET NULL, add api_key_name snapshot ---
    _replace_fk_if_needed(
        'stats_daily_api_key_api_key_id_fkey', 'stats_daily_api_key',
        'api_keys', ['api_key_id'], ['id'], 'SET NULL',
    )
    op.alter_column('stats_daily_api_key', 'api_key_id', existing_type=sa.String(36),
                    nullable=True)
    if not _column_exists('stats_daily_api_key', 'api_key_name'):
        op.add_column('stats_daily_api_key', sa.Column('api_key_name', sa.String(200),
                                                       nullable=True,
                                                       comment='API Key 名称快照（删除 Key 后仍可追溯）'))

    # --- Backfill: populate snapshots from existing FK joins ---
    # Single UPDATE per table using LEFT JOINs to fill both columns in one pass.
    # (WHERE ... IS NULL makes these inherently idempotent)
    op.execute("""
        UPDATE usage u
        SET username     = COALESCE(u.username, usr.username),
            api_key_name = COALESCE(u.api_key_name, ak.name)
        FROM users usr, api_keys ak
        WHERE usr.id = u.user_id
          AND ak.id = u.api_key_id
          AND (u.username IS NULL OR u.api_key_name IS NULL)
    """)
    # Catch rows that have user_id but no api_key_id (or vice versa)
    op.execute("""
        UPDATE usage u
        SET username = usr.username
        FROM users usr
        WHERE u.user_id = usr.id AND u.username IS NULL
    """)
    op.execute("""
        UPDATE usage u
        SET api_key_name = ak.name
        FROM api_keys ak
        WHERE u.api_key_id = ak.id AND u.api_key_name IS NULL
    """)
    op.execute("""
        UPDATE stats_user_daily s
        SET username = usr.username
        FROM users usr
        WHERE s.user_id = usr.id AND s.username IS NULL
    """)
    op.execute("""
        UPDATE stats_daily_api_key s
        SET api_key_name = ak.name
        FROM api_keys ak
        WHERE s.api_key_id = ak.id AND s.api_key_name IS NULL
    """)


def downgrade() -> None:
    # --- Remove snapshot columns ---
    if _column_exists('stats_daily_api_key', 'api_key_name'):
        op.drop_column('stats_daily_api_key', 'api_key_name')
    if _column_exists('stats_user_daily', 'username'):
        op.drop_column('stats_user_daily', 'username')
    if _column_exists('usage', 'api_key_name'):
        op.drop_column('usage', 'api_key_name')
    if _column_exists('usage', 'username'):
        op.drop_column('usage', 'username')

    # --- StatsDailyApiKey: SET NULL -> CASCADE ---
    _replace_fk_if_needed(
        'stats_daily_api_key_api_key_id_fkey', 'stats_daily_api_key',
        'api_keys', ['api_key_id'], ['id'], 'CASCADE',
    )
    op.alter_column('stats_daily_api_key', 'api_key_id', existing_type=sa.String(36),
                    nullable=False)

    # --- StatsUserDaily: SET NULL -> CASCADE ---
    _replace_fk_if_needed(
        'stats_user_daily_user_id_fkey', 'stats_user_daily',
        'users', ['user_id'], ['id'], 'CASCADE',
    )
    op.alter_column('stats_user_daily', 'user_id', existing_type=sa.String(36), nullable=False)
