"""Add api_family/endpoint_kind and migrate api_format to endpoint signature keys

Revision ID: cf40e6a5c5b1
Revises: c8d2e4f6a1b3
Create Date: 2026-01-31 15:30:00.000000

"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Sequence, Union
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import inspect, text

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cf40e6a5c5b1"
down_revision: Union[str, None] = "c8d2e4f6a1b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json_loads(val):
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def _json_dumps(val):
    """将 dict/list 转为 JSON 字符串，None 保持 None"""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return json.dumps(val)


def _normalize_signature(value: str | None) -> str | None:
    """
    Normalize legacy api_format / signature-ish strings to canonical signature key.

    - canonical: `<family>:<kind>` (lowercase)
    - legacy examples: "OPENAI", "OPENAI_CLI", "GEMINI_VIDEO"
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    if ":" in raw:
        fam, kind = raw.split(":", 1)
        fam = fam.strip().lower()
        kind = kind.strip().lower()
        if not fam or not kind:
            return None
        return f"{fam}:{kind}"

    upper = raw.upper()
    if upper.startswith("CLAUDE"):
        fam = "claude"
    elif upper.startswith("OPENAI"):
        fam = "openai"
    elif upper.startswith("GEMINI"):
        fam = "gemini"
    else:
        return None

    kind = "chat"
    if upper.endswith("_CLI"):
        kind = "cli"
    elif upper.endswith("_VIDEO"):
        kind = "video"

    return f"{fam}:{kind}"


def _normalize_signature_list(values) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = _json_loads(values)
    if not isinstance(values, list):
        return None

    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        sig = _normalize_signature(str(v) if v is not None else None)
        if not sig:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(sig)
    return out


def _normalize_signature_dict(values) -> dict | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = _json_loads(values)
    if not isinstance(values, dict):
        return None

    out: dict = {}
    for k, v in values.items():
        sig = _normalize_signature(str(k) if k is not None else None)
        if not sig:
            continue
        out[sig] = v
    return out


def _add_video_variants(formats: list[str]) -> list[str]:
    """
    保持原有格式，不自动补齐 video 变体。
    """
    return formats


def table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    try:
        indexes = inspector.get_indexes(table_name)
    except Exception:
        return False
    return any(idx.get("name") == index_name for idx in indexes)


def _migrate_format_acceptance_config(cfg) -> dict | None:
    cfg_obj = _json_loads(cfg)
    if not isinstance(cfg_obj, dict):
        return cfg_obj if cfg_obj is None else None

    for key in ("accept_formats", "reject_formats"):
        raw = cfg_obj.get(key)
        if not isinstance(raw, list):
            continue
        normalized = _normalize_signature_list(raw) or []
        cfg_obj[key] = normalized

    return cfg_obj


def migrate_provider_endpoints(connection) -> None:
    """
    - 将 provider_endpoints.api_format 统一迁移为 signature key（小写）
    - 填充/校准 api_family / endpoint_kind
    - 迁移 format_acceptance_config 中的 accept/reject formats
    """
    rows = connection.execute(text("""
            SELECT
                id,
                api_format,
                api_family,
                endpoint_kind,
                format_acceptance_config
            FROM provider_endpoints
            """)).fetchall()

    for row in rows:
        sig = _normalize_signature(row.api_format)
        if not sig:
            continue
        fam, kind = sig.split(":", 1)

        cfg = _migrate_format_acceptance_config(row.format_acceptance_config)

        connection.execute(
            text("""
                UPDATE provider_endpoints
                SET
                    api_format = :api_format,
                    api_family = :api_family,
                    endpoint_kind = :endpoint_kind,
                    format_acceptance_config = CAST(:format_acceptance_config AS json)
                WHERE id = :id
                """),
            {
                "id": row.id,
                "api_format": sig,
                "api_family": fam,
                "endpoint_kind": kind,
                "format_acceptance_config": _json_dumps(cfg),
            },
        )


def create_video_endpoints(connection) -> None:
    """
    不再自动创建 video endpoint，保持原有配置。
    """
    pass


def migrate_provider_api_keys(connection) -> None:
    """
    迁移 provider_api_keys:
    - api_formats -> signature keys（并补齐 video 变体）
    - dict 字段 key -> signature keys（rate_multipliers/global_priority/health/circuit_breaker）
    - rate_multipliers/global_priority_by_format 复制 chat -> video（如 openai:chat -> openai:video）
    """
    rows = connection.execute(text("""
            SELECT
                id,
                api_formats,
                rate_multipliers,
                global_priority_by_format,
                health_by_format,
                circuit_breaker_by_format
            FROM provider_api_keys
            """)).fetchall()

    for row in rows:
        api_formats = _normalize_signature_list(row.api_formats)
        if api_formats is not None:
            api_formats = _add_video_variants(api_formats)

        rate_multipliers = _normalize_signature_dict(row.rate_multipliers)
        global_priority_by_format = _normalize_signature_dict(row.global_priority_by_format)

        health_by_format = _normalize_signature_dict(row.health_by_format)
        circuit_breaker_by_format = _normalize_signature_dict(row.circuit_breaker_by_format)

        connection.execute(
            text("""
                UPDATE provider_api_keys
                SET
                    api_formats = CAST(:api_formats AS json),
                    rate_multipliers = CAST(:rate_multipliers AS json),
                    global_priority_by_format = CAST(:global_priority_by_format AS json),
                    health_by_format = CAST(:health_by_format AS json),
                    circuit_breaker_by_format = CAST(:circuit_breaker_by_format AS json)
                WHERE id = :id
                """),
            {
                "id": row.id,
                "api_formats": _json_dumps(api_formats),
                "rate_multipliers": _json_dumps(rate_multipliers),
                "global_priority_by_format": _json_dumps(global_priority_by_format),
                "health_by_format": _json_dumps(health_by_format),
                "circuit_breaker_by_format": _json_dumps(circuit_breaker_by_format),
            },
        )


def migrate_allowed_api_formats(connection, *, table_name: str) -> None:
    """迁移 users/api_keys.allowed_api_formats 为 signature keys（并补齐 video 变体）。"""
    if not table_exists(table_name):
        return
    rows = connection.execute(text(f"""
            SELECT id, allowed_api_formats
            FROM {table_name}
            """)).fetchall()

    for row in rows:
        allowed = _normalize_signature_list(row.allowed_api_formats)
        if allowed is None:
            continue
        allowed = _add_video_variants(allowed)
        connection.execute(
            text(f"""
                UPDATE {table_name}
                SET allowed_api_formats = CAST(:allowed_api_formats AS json)
                WHERE id = :id
                """),
            {"id": row.id, "allowed_api_formats": _json_dumps(allowed)},
        )


def migrate_video_tasks(connection) -> None:
    """
    video_tasks.*_api_format 迁移为 signature keys。

    注意：video_tasks 表天然是 video 任务，因此将 openai/gemini 的 kind 强制归一为 video，
    以兼容历史上复用 chat 格式存储的旧记录。
    """
    if not table_exists("video_tasks"):
        return

    rows = connection.execute(text("""
            SELECT id, client_api_format, provider_api_format
            FROM video_tasks
            """)).fetchall()

    for row in rows:
        client_sig = _normalize_signature(row.client_api_format) or ""
        provider_sig = _normalize_signature(row.provider_api_format) or ""

        def _force_video(sig: str) -> str:
            if not sig or ":" not in sig:
                return sig
            fam, _kind = sig.split(":", 1)
            fam = fam.strip().lower()
            if fam in ("openai", "gemini"):
                return f"{fam}:video"
            return sig

        client_sig = _force_video(client_sig)
        provider_sig = _force_video(provider_sig)

        if not client_sig or not provider_sig:
            continue

        connection.execute(
            text("""
                UPDATE video_tasks
                SET client_api_format = :client_api_format,
                    provider_api_format = :provider_api_format
                WHERE id = :id
                """),
            {
                "id": row.id,
                "client_api_format": client_sig,
                "provider_api_format": provider_sig,
            },
        )


def migrate_model_provider_mappings(connection) -> None:
    """迁移 models.provider_model_mappings[*].api_formats 为 signature keys。"""
    if not table_exists("models"):
        return

    rows = connection.execute(text("""
            SELECT id, provider_model_mappings
            FROM models
            WHERE provider_model_mappings IS NOT NULL
            """)).fetchall()

    for row in rows:
        mappings = _json_loads(row.provider_model_mappings)
        if not isinstance(mappings, list):
            continue

        changed = False
        new_mappings: list = []
        for item in mappings:
            if not isinstance(item, dict):
                new_mappings.append(item)
                continue
            raw_formats = item.get("api_formats")
            if isinstance(raw_formats, list):
                normalized = _normalize_signature_list(raw_formats) or []
                # 内容比较（而非引用比较），避免已迁移数据被无意义地重复 UPDATE
                if set(normalized) != set(raw_formats):
                    changed = True
                item = dict(item)
                item["api_formats"] = normalized
            new_mappings.append(item)

        if not changed:
            continue

        connection.execute(
            text("""
                UPDATE models
                SET provider_model_mappings = CAST(:provider_model_mappings AS json)
                WHERE id = :id
                """),
            {"id": row.id, "provider_model_mappings": _json_dumps(new_mappings)},
        )


def migrate_dimension_collectors(connection) -> None:
    """迁移 dimension_collectors.api_format 为 signature keys（如果存在历史数据）。"""
    if not table_exists("dimension_collectors"):
        return

    rows = connection.execute(text("""
            SELECT id, api_format
            FROM dimension_collectors
            WHERE api_format IS NOT NULL
            """)).fetchall()

    for row in rows:
        sig = _normalize_signature(row.api_format)
        if not sig:
            continue
        connection.execute(
            text("""
                UPDATE dimension_collectors
                SET api_format = :api_format
                WHERE id = :id
                """),
            {"id": row.id, "api_format": sig},
        )


def upgrade() -> None:
    if not table_exists("provider_endpoints"):
        return

    # ==================== provider_endpoints.api_family / endpoint_kind ====================
    if not column_exists("provider_endpoints", "api_family"):
        op.add_column("provider_endpoints", sa.Column("api_family", sa.String(50), nullable=True))
    if not column_exists("provider_endpoints", "endpoint_kind"):
        op.add_column(
            "provider_endpoints", sa.Column("endpoint_kind", sa.String(50), nullable=True)
        )

    # ==================== idx_provider_family_kind ====================
    if not index_exists("provider_endpoints", "idx_provider_family_kind"):
        op.create_index(
            "idx_provider_family_kind",
            "provider_endpoints",
            ["provider_id", "api_family", "endpoint_kind"],
        )

    # ==================== data migrations (idempotent) ====================
    conn = op.get_bind()

    migrate_provider_endpoints(conn)
    create_video_endpoints(conn)

    if table_exists("provider_api_keys"):
        migrate_provider_api_keys(conn)

    migrate_allowed_api_formats(conn, table_name="users")
    migrate_allowed_api_formats(conn, table_name="api_keys")
    migrate_video_tasks(conn)
    migrate_model_provider_mappings(conn)
    migrate_dimension_collectors(conn)


def downgrade() -> None:
    # Drop index/columns only; data changes are intentionally kept (safe rollback strategy).
    if table_exists("provider_endpoints"):
        if index_exists("provider_endpoints", "idx_provider_family_kind"):
            op.drop_index("idx_provider_family_kind", table_name="provider_endpoints")
        if column_exists("provider_endpoints", "endpoint_kind"):
            op.drop_column("provider_endpoints", "endpoint_kind")
        if column_exists("provider_endpoints", "api_family"):
            op.drop_column("provider_endpoints", "api_family")
