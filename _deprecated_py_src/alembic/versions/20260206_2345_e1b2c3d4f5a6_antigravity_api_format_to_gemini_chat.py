"""Antigravity endpoint signature to gemini:chat & add proxy_nodes table (with manual fields)

Revision ID: e1b2c3d4f5a6
Revises: b5c6d7e8f9a0
Create Date: 2026-02-06 23:45:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1b2c3d4f5a6"
down_revision: str | None = "b5c6d7e8f9a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    conn = op.get_bind()

    # =========================================================================
    # Part 1: Antigravity endpoint signature migration (gemini:cli -> gemini:chat)
    # =========================================================================

    # --- provider_endpoints ---
    # Update only when there is no conflicting gemini:chat endpoint for the same provider
    # (provider_endpoints has a unique constraint on (provider_id, api_format)).
    conn.execute(text("""
            UPDATE provider_endpoints pe
            SET
                api_format = 'gemini:chat',
                api_family = 'gemini',
                endpoint_kind = 'chat'
            WHERE pe.api_format = 'gemini:cli'
              AND pe.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
              )
              AND NOT EXISTS (
                SELECT 1 FROM provider_endpoints pe2
                WHERE pe2.provider_id = pe.provider_id
                  AND pe2.api_format = 'gemini:chat'
              )
            """))

    # Best-effort normalization for already-existing Antigravity gemini:chat endpoints.
    conn.execute(text("""
            UPDATE provider_endpoints pe
            SET
                api_family = 'gemini',
                endpoint_kind = 'chat'
            WHERE pe.api_format = 'gemini:chat'
              AND pe.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
              )
            """))

    # --- provider_api_keys.api_formats (JSON array) ---
    # Replace "gemini:cli" with "gemini:chat" in the JSON array for Antigravity keys.
    # Uses text-level replace on the serialized JSON -- safe because the value is a
    # simple string with no special characters that could cause ambiguous replacements.
    conn.execute(text("""
            UPDATE provider_api_keys pak
            SET api_formats = replace(pak.api_formats::text, '"gemini:cli"', '"gemini:chat"')::json
            WHERE pak.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
            )
              AND pak.api_formats IS NOT NULL
              AND pak.api_formats::text LIKE '%"gemini:cli"%'
            """))

    # =========================================================================
    # Part 2: Create proxy_nodes table with manual proxy fields (idempotent)
    # =========================================================================

    # Create ENUM type (idempotent)
    op.execute(
        "DO $$ BEGIN "
        "CREATE TYPE proxynodestatus AS ENUM ('online', 'unhealthy', 'offline'); "
        "EXCEPTION WHEN duplicate_object THEN NULL; "
        "END $$"
    )

    if table_exists("proxy_nodes"):
        # Table already exists — ensure manual proxy columns are present
        inspector = inspect(conn)
        existing_columns = {c["name"] for c in inspector.get_columns("proxy_nodes")}

        # ip 列扩容：手动节点的 ip 存储 "socks5://hostname" 形式，45 字符可能不够
        ip_col = next((c for c in inspector.get_columns("proxy_nodes") if c["name"] == "ip"), None)
        if ip_col and hasattr(ip_col["type"], "length") and (ip_col["type"].length or 0) < 512:
            op.alter_column("proxy_nodes", "ip", type_=sa.String(512), existing_nullable=False)

        manual_columns = [
            ("is_manual", sa.Boolean(), False, sa.text("false"), "是否为手动添加的代理节点"),
            ("proxy_url", sa.String(500), True, None, "手动节点的完整代理 URL"),
            ("proxy_username", sa.String(255), True, None, "手动节点的代理用户名"),
            ("proxy_password", sa.String(500), True, None, "手动节点的代理密码"),
        ]
        for col_name, col_type, nullable, default, comment in manual_columns:
            if col_name not in existing_columns:
                op.add_column(
                    "proxy_nodes",
                    sa.Column(
                        col_name,
                        col_type,  # type: ignore[arg-type]
                        nullable=nullable,
                        server_default=default,
                        comment=comment,
                    ),
                )
        return

    op.create_table(
        "proxy_nodes",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("ip", sa.String(512), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("region", sa.String(100), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "online",
                "unhealthy",
                "offline",
                name="proxynodestatus",
                create_type=False,
            ),
            nullable=False,
            server_default=sa.text("'online'"),
        ),
        sa.Column(
            "registered_by",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("last_heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_interval", sa.Integer(), nullable=False, server_default=sa.text("30")),
        sa.Column("active_connections", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_requests", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        # --- Manual proxy node fields ---
        sa.Column(
            "is_manual",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
            comment="是否为手动添加的代理节点",
        ),
        sa.Column(
            "proxy_url",
            sa.String(500),
            nullable=True,
            comment="手动节点的完整代理 URL",
        ),
        sa.Column(
            "proxy_username",
            sa.String(255),
            nullable=True,
            comment="手动节点的代理用户名",
        ),
        sa.Column(
            "proxy_password",
            sa.String(500),
            nullable=True,
            comment="手动节点的代理密码",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.UniqueConstraint("ip", "port", name="uq_proxy_node_ip_port"),
    )


def downgrade() -> None:
    conn = op.get_bind()

    # =========================================================================
    # Part 2 rollback: Drop proxy_nodes table (and manual columns if present)
    # =========================================================================
    if table_exists("proxy_nodes"):
        op.drop_table("proxy_nodes")

    # Best-effort: drop type (only used by proxy_nodes)
    op.execute("DROP TYPE IF EXISTS proxynodestatus")

    # =========================================================================
    # Part 1 rollback: Revert Antigravity endpoint signature (gemini:chat -> gemini:cli)
    # =========================================================================

    # --- provider_endpoints ---
    conn.execute(text("""
            UPDATE provider_endpoints pe
            SET
                api_format = 'gemini:cli',
                api_family = 'gemini',
                endpoint_kind = 'cli'
            WHERE pe.api_format = 'gemini:chat'
              AND pe.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
              )
              AND NOT EXISTS (
                SELECT 1 FROM provider_endpoints pe2
                WHERE pe2.provider_id = pe.provider_id
                  AND pe2.api_format = 'gemini:cli'
              )
            """))

    # Best-effort normalization for already-existing Antigravity gemini:cli endpoints.
    conn.execute(text("""
            UPDATE provider_endpoints pe
            SET
                api_family = 'gemini',
                endpoint_kind = 'cli'
            WHERE pe.api_format = 'gemini:cli'
              AND pe.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
              )
            """))

    # --- provider_api_keys.api_formats (JSON array) ---
    conn.execute(text("""
            UPDATE provider_api_keys pak
            SET api_formats = replace(pak.api_formats::text, '"gemini:chat"', '"gemini:cli"')::json
            WHERE pak.provider_id IN (
                SELECT id FROM providers WHERE lower(provider_type) = 'antigravity'
            )
              AND pak.api_formats IS NOT NULL
              AND pak.api_formats::text LIKE '%"gemini:chat"%'
            """))
