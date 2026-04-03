"""proxy_node_metrics_and_events

Revision ID: 48afe197cc15
Revises: b2c3d4e5f6a7
Create Date: 2026-02-28 04:33:11.201185+00:00

"""

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

# revision identifiers, used by Alembic.
revision = "48afe197cc15"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    columns = [c["name"] for c in insp.get_columns(table_name)]
    return column_name in columns


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    insp = inspect(bind)
    return table_name in insp.get_table_names()


def _enum_has_value(enum_name: str, value: str) -> bool:
    """检查 PostgreSQL 枚举类型是否包含指定值"""
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM pg_enum e JOIN pg_type t ON e.enumtypid = t.oid"
            " WHERE t.typname = :enum_name AND e.enumlabel = :value"
        ),
        {"enum_name": enum_name, "value": value},
    )
    return result.fetchone() is not None


def upgrade() -> None:
    # proxy_nodes: 将已废弃的 unhealthy 状态迁移为 offline，然后从枚举中移除
    if _enum_has_value("proxynodestatus", "unhealthy"):
        op.execute("UPDATE proxy_nodes SET status = 'offline' WHERE status = 'unhealthy'")
        op.execute("ALTER TYPE proxynodestatus RENAME TO proxynodestatus_old")
        op.execute("CREATE TYPE proxynodestatus AS ENUM ('online', 'offline')")
        # 必须先移除旧枚举类型的 DEFAULT，否则 ALTER TYPE 会因无法转换默认值而报错
        op.execute("ALTER TABLE proxy_nodes ALTER COLUMN status DROP DEFAULT")
        op.execute(
            "ALTER TABLE proxy_nodes ALTER COLUMN status TYPE proxynodestatus"
            " USING status::text::proxynodestatus"
        )
        op.execute("ALTER TABLE proxy_nodes ALTER COLUMN status SET DEFAULT 'online'::proxynodestatus")
        op.execute("DROP TYPE proxynodestatus_old")

    # proxy_nodes: 新增错误指标字段
    if not _column_exists("proxy_nodes", "failed_requests"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "failed_requests",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
                comment="累计失败请求数",
            ),
        )
    if not _column_exists("proxy_nodes", "dns_failures"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "dns_failures",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
                comment="累计 DNS 失败数",
            ),
        )
    if not _column_exists("proxy_nodes", "stream_errors"):
        op.add_column(
            "proxy_nodes",
            sa.Column(
                "stream_errors",
                sa.BigInteger(),
                nullable=False,
                server_default="0",
                comment="累计流错误数",
            ),
        )

    # proxy_node_events: 连接事件表
    if not _table_exists("proxy_node_events"):
        op.create_table(
            "proxy_node_events",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("node_id", sa.String(length=36), nullable=False),
            sa.Column(
                "event_type",
                sa.String(length=20),
                nullable=False,
                comment="事件类型: connected, disconnected, error",
            ),
            sa.Column(
                "detail",
                sa.String(length=500),
                nullable=True,
                comment="事件详情（如断开原因）",
            ),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["node_id"], ["proxy_nodes.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "idx_proxy_node_events_node_created",
            "proxy_node_events",
            ["node_id", "created_at"],
        )
        op.create_index(
            op.f("ix_proxy_node_events_node_id"),
            "proxy_node_events",
            ["node_id"],
        )


def downgrade() -> None:
    # 恢复 proxynodestatus 枚举，加回 unhealthy
    if not _enum_has_value("proxynodestatus", "unhealthy"):
        op.execute("ALTER TYPE proxynodestatus RENAME TO proxynodestatus_old")
        op.execute("CREATE TYPE proxynodestatus AS ENUM ('online', 'unhealthy', 'offline')")
        op.execute("ALTER TABLE proxy_nodes ALTER COLUMN status DROP DEFAULT")
        op.execute(
            "ALTER TABLE proxy_nodes ALTER COLUMN status TYPE proxynodestatus"
            " USING status::text::proxynodestatus"
        )
        op.execute("ALTER TABLE proxy_nodes ALTER COLUMN status SET DEFAULT 'online'::proxynodestatus")
        op.execute("DROP TYPE proxynodestatus_old")

    if _table_exists("proxy_node_events"):
        op.drop_index(op.f("ix_proxy_node_events_node_id"), table_name="proxy_node_events")
        op.drop_index("idx_proxy_node_events_node_created", table_name="proxy_node_events")
        op.drop_table("proxy_node_events")
    if _column_exists("proxy_nodes", "stream_errors"):
        op.drop_column("proxy_nodes", "stream_errors")
    if _column_exists("proxy_nodes", "dns_failures"):
        op.drop_column("proxy_nodes", "dns_failures")
    if _column_exists("proxy_nodes", "failed_requests"):
        op.drop_column("proxy_nodes", "failed_requests")
