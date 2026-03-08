"""request_candidates/video_tasks retention: SET NULL and add snapshots

Revision ID: 13a4c8f6d9e0
Revises: 45b118150a78
Create Date: 2026-03-08 12:15:00.000000+00:00

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "13a4c8f6d9e0"
down_revision = "45b118150a78"
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
        op.drop_constraint(constraint_name, table_name, type_="foreignkey")
    op.create_foreign_key(
        constraint_name, table_name, ref_table,
        local_cols, remote_cols, ondelete=desired_ondelete,
    )


def upgrade() -> None:
    # --- request_candidates: add snapshot columns ---
    if not _column_exists("request_candidates", "username"):
        op.add_column(
            "request_candidates",
            sa.Column("username", sa.String(length=100), nullable=True, comment="用户名快照"),
        )
    if not _column_exists("request_candidates", "api_key_name"):
        op.add_column(
            "request_candidates",
            sa.Column(
                "api_key_name",
                sa.String(length=200),
                nullable=True,
                comment="API Key 名称快照",
            ),
        )

    # --- request_candidates: CASCADE -> SET NULL ---
    _replace_fk_if_needed(
        "request_candidates_user_id_fkey", "request_candidates",
        "users", ["user_id"], ["id"], "SET NULL",
    )
    _replace_fk_if_needed(
        "request_candidates_api_key_id_fkey", "request_candidates",
        "api_keys", ["api_key_id"], ["id"], "SET NULL",
    )

    # --- video_tasks: add snapshot columns ---
    if not _column_exists("video_tasks", "username"):
        op.add_column(
            "video_tasks",
            sa.Column("username", sa.String(length=100), nullable=True, comment="用户名快照"),
        )
    if not _column_exists("video_tasks", "api_key_name"):
        op.add_column(
            "video_tasks",
            sa.Column(
                "api_key_name",
                sa.String(length=200),
                nullable=True,
                comment="API Key 名称快照",
            ),
        )

    # --- video_tasks: CASCADE -> SET NULL, user_id nullable ---
    op.alter_column("video_tasks", "user_id", existing_type=sa.String(length=36), nullable=True)
    _replace_fk_if_needed(
        "video_tasks_user_id_fkey", "video_tasks",
        "users", ["user_id"], ["id"], "SET NULL",
    )
    _replace_fk_if_needed(
        "video_tasks_api_key_id_fkey", "video_tasks",
        "api_keys", ["api_key_id"], ["id"], "SET NULL",
    )

    # --- Backfill: populate snapshots from existing FK joins ---
    # Single UPDATE per table using JOINs to fill both columns in one pass.
    # (WHERE ... IS NULL makes these inherently idempotent)

    # request_candidates: fill both columns where both FKs exist
    op.execute(
        """
        UPDATE request_candidates rc
        SET username     = COALESCE(rc.username, usr.username),
            api_key_name = COALESCE(rc.api_key_name, ak.name)
        FROM users usr, api_keys ak
        WHERE usr.id = rc.user_id
          AND ak.id = rc.api_key_id
          AND (rc.username IS NULL OR rc.api_key_name IS NULL)
        """
    )
    # Catch rows with only user_id or only api_key_id
    op.execute(
        """
        UPDATE request_candidates rc
        SET username = usr.username
        FROM users usr
        WHERE usr.id = rc.user_id AND rc.username IS NULL
        """
    )
    op.execute(
        """
        UPDATE request_candidates rc
        SET api_key_name = ak.name
        FROM api_keys ak
        WHERE ak.id = rc.api_key_id AND rc.api_key_name IS NULL
        """
    )

    # video_tasks: fill both columns where both FKs exist
    op.execute(
        """
        UPDATE video_tasks vt
        SET username     = COALESCE(vt.username, usr.username),
            api_key_name = COALESCE(vt.api_key_name, ak.name)
        FROM users usr, api_keys ak
        WHERE usr.id = vt.user_id
          AND ak.id = vt.api_key_id
          AND (vt.username IS NULL OR vt.api_key_name IS NULL)
        """
    )
    # Catch rows with only user_id or only api_key_id
    op.execute(
        """
        UPDATE video_tasks vt
        SET username = usr.username
        FROM users usr
        WHERE usr.id = vt.user_id AND vt.username IS NULL
        """
    )
    op.execute(
        """
        UPDATE video_tasks vt
        SET api_key_name = ak.name
        FROM api_keys ak
        WHERE ak.id = vt.api_key_id AND vt.api_key_name IS NULL
        """
    )


def downgrade() -> None:
    # --- video_tasks: SET NULL -> default (no action), restore NOT NULL ---
    _replace_fk_if_needed(
        "video_tasks_api_key_id_fkey", "video_tasks",
        "api_keys", ["api_key_id"], ["id"], "NO ACTION",
    )
    _replace_fk_if_needed(
        "video_tasks_user_id_fkey", "video_tasks",
        "users", ["user_id"], ["id"], "NO ACTION",
    )
    op.alter_column("video_tasks", "user_id", existing_type=sa.String(length=36), nullable=False)
    if _column_exists("video_tasks", "api_key_name"):
        op.drop_column("video_tasks", "api_key_name")
    if _column_exists("video_tasks", "username"):
        op.drop_column("video_tasks", "username")

    # --- request_candidates: SET NULL -> CASCADE ---
    _replace_fk_if_needed(
        "request_candidates_api_key_id_fkey", "request_candidates",
        "api_keys", ["api_key_id"], ["id"], "CASCADE",
    )
    _replace_fk_if_needed(
        "request_candidates_user_id_fkey", "request_candidates",
        "users", ["user_id"], ["id"], "CASCADE",
    )
    if _column_exists("request_candidates", "api_key_name"):
        op.drop_column("request_candidates", "api_key_name")
    if _column_exists("request_candidates", "username"):
        op.drop_column("request_candidates", "username")
