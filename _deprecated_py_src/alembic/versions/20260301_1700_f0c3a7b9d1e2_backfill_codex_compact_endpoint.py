"""backfill_codex_compact_endpoint

Backfill Codex reverse-proxy endpoints:
- ensure `openai:cli` endpoint is pinned to force_stream
- ensure `openai:compact` endpoint exists

Revision ID: f0c3a7b9d1e2
Revises: 2a624af8dd3a
Create Date: 2026-03-01 17:00:00.000000+00:00
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "f0c3a7b9d1e2"
down_revision = "2a624af8dd3a"
branch_labels = None
depends_on = None

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
_COMPACT_FORMAT = "openai:compact"
_CLI_FORMAT = "openai:cli"
_FORCE_STREAM = "force_stream"


def _find_codex_provider_ids(conn: sa.Connection) -> list[str]:
    """Find Codex providers (by provider_type or legacy base_url pattern)."""
    rows = conn.execute(sa.text("""
        SELECT DISTINCT p.id
        FROM providers p
        LEFT JOIN provider_endpoints pe ON pe.provider_id = p.id
        WHERE lower(COALESCE(p.provider_type, '')) = 'codex'
           OR (
                lower(COALESCE(pe.api_format, '')) = 'openai:cli'
            AND lower(COALESCE(pe.base_url, '')) LIKE '%/backend-api/codex%'
           )
    """))
    return [str(r[0]) for r in rows if r[0]]


def _get_cli_endpoint(conn: sa.Connection, provider_id: str) -> dict[str, Any] | None:
    """Load existing openai:cli endpoint for the provider."""
    row = (
        conn.execute(
            sa.text("""
                SELECT base_url, header_rules, body_rules, max_retries, proxy, config
                FROM provider_endpoints
                WHERE provider_id = :pid AND api_format = :fmt
                LIMIT 1
            """),
            {"pid": provider_id, "fmt": _CLI_FORMAT},
        )
        .mappings()
        .first()
    )
    return dict(row) if row else None


def _pin_cli_force_stream(conn: sa.Connection, provider_id: str, cli: dict[str, Any]) -> None:
    """Set upstream_stream_policy=force_stream on existing cli endpoint."""
    cfg = dict(cli.get("config") or {}) if isinstance(cli.get("config"), dict) else {}
    cfg.pop("upstreamStreamPolicy", None)
    cfg.pop("upstream_stream", None)
    cfg["upstream_stream_policy"] = _FORCE_STREAM
    conn.execute(
        sa.text("""
            UPDATE provider_endpoints
            SET api_family = 'openai',
                endpoint_kind = 'cli',
                config = CAST(:config AS json),
                updated_at = CURRENT_TIMESTAMP
            WHERE provider_id = :pid AND api_format = :fmt
        """),
        {
            "pid": provider_id,
            "fmt": _CLI_FORMAT,
            "config": json.dumps(cfg, ensure_ascii=False),
        },
    )


def _ensure_compact_endpoint(conn: sa.Connection, provider_id: str, cli: dict[str, Any]) -> None:
    """Create openai:compact endpoint if missing (clone from cli)."""
    exists = conn.execute(
        sa.text(
            "SELECT 1 FROM provider_endpoints WHERE provider_id = :pid AND api_format = :fmt LIMIT 1"
        ),
        {"pid": provider_id, "fmt": _COMPACT_FORMAT},
    ).first()
    if exists:
        # Already exists, just ensure api_family/endpoint_kind are set.
        conn.execute(
            sa.text("""
                UPDATE provider_endpoints
                SET api_family = 'openai', endpoint_kind = 'compact',
                    updated_at = CURRENT_TIMESTAMP
                WHERE provider_id = :pid AND api_format = :fmt
            """),
            {"pid": provider_id, "fmt": _COMPACT_FORMAT},
        )
        return

    # Clone from cli endpoint, strip stream policy.
    cfg = dict(cli.get("config") or {}) if isinstance(cli.get("config"), dict) else {}
    for k in ("upstream_stream_policy", "upstreamStreamPolicy", "upstream_stream"):
        cfg.pop(k, None)

    def _json(val: Any) -> str | None:
        return json.dumps(val, ensure_ascii=False) if val is not None else None

    conn.execute(
        sa.text("""
            INSERT INTO provider_endpoints (
                id, provider_id, api_format, api_family, endpoint_kind,
                base_url, custom_path, header_rules, body_rules,
                max_retries, is_active, config, format_acceptance_config,
                proxy, created_at, updated_at
            ) VALUES (
                :id, :pid, :fmt, 'openai', 'compact',
                :base_url, NULL, CAST(:header_rules AS json), CAST(:body_rules AS json),
                :max_retries, TRUE, CAST(:config AS json), NULL,
                CAST(:proxy AS jsonb), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            )
        """),
        {
            "id": str(uuid.uuid4()),
            "pid": provider_id,
            "fmt": _COMPACT_FORMAT,
            "base_url": cli.get("base_url") or _CODEX_BASE_URL,
            "header_rules": _json(cli.get("header_rules")),
            "body_rules": _json(cli.get("body_rules")),
            "max_retries": cli.get("max_retries") or 2,
            "config": _json(cfg or None),
            "proxy": _json(cli.get("proxy")),
        },
    )


def _add_compact_to_key_formats(conn: sa.Connection, provider_id: str) -> None:
    """Ensure provider keys include openai:compact in api_formats."""
    rows = (
        conn.execute(
            sa.text("SELECT id, api_formats FROM provider_api_keys WHERE provider_id = :pid"),
            {"pid": provider_id},
        )
        .mappings()
        .all()
    )
    for row in rows:
        raw = row["api_formats"]
        formats: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                v = str(item or "").strip().lower()
                if v and v not in formats:
                    formats.append(v)

        if _COMPACT_FORMAT in formats:
            continue

        # Insert compact right after cli, or at end.
        if _CLI_FORMAT in formats:
            idx = formats.index(_CLI_FORMAT) + 1
            formats.insert(idx, _COMPACT_FORMAT)
        else:
            formats.append(_COMPACT_FORMAT)

        conn.execute(
            sa.text("""
                UPDATE provider_api_keys
                SET api_formats = CAST(:fmts AS json), updated_at = CURRENT_TIMESTAMP
                WHERE id = :id
            """),
            {"id": row["id"], "fmts": json.dumps(formats, ensure_ascii=False)},
        )


def upgrade() -> None:
    conn = op.get_bind()
    for provider_id in _find_codex_provider_ids(conn):
        cli = _get_cli_endpoint(conn, provider_id)
        if not cli:
            continue  # No cli endpoint to clone from; skip.
        _pin_cli_force_stream(conn, provider_id, cli)
        _ensure_compact_endpoint(conn, provider_id, cli)
        _add_compact_to_key_formats(conn, provider_id)


def downgrade() -> None:
    # Data backfill: no-op to avoid deleting user-managed data.
    return
