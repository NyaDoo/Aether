"""Update Antigravity endpoint signature to gemini:chat

Revision ID: e1b2c3d4f5a6
Revises: b5c6d7e8f9a0
Create Date: 2026-02-06 23:45:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1b2c3d4f5a6"
down_revision: str | None = "b5c6d7e8f9a0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()

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
    # Uses text-level replace on the serialized JSON — safe because the value is a
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


def downgrade() -> None:
    conn = op.get_bind()

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
