"""make users email/password nullable add email_verified and oauth tables

Revision ID: 33e347f97c0c
Revises: ddd59cdf0349
Create Date: 2026-01-18 11:18:15.940559+00:00

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "33e347f97c0c"
down_revision = "ddd59cdf0349"
branch_labels = None
depends_on = None


def column_exists(table_name: str, column_name: str) -> bool:
    """检查列是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    columns = [col["name"] for col in inspector.get_columns(table_name)]
    return column_name in columns


def table_exists(table_name: str) -> bool:
    """检查表是否存在"""
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def column_is_nullable(table_name: str, column_name: str) -> bool:
    """检查列是否允许 NULL"""
    bind = op.get_bind()
    inspector = inspect(bind)
    for col in inspector.get_columns(table_name):
        if col["name"] == column_name:
            return col["nullable"]
    return False


def enum_value_exists(enum_name: str, value: str) -> bool:
    """检查 PostgreSQL ENUM 是否包含指定值"""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return True  # 非 PostgreSQL 跳过检查
    result = bind.execute(
        sa.text(
            "SELECT 1 FROM pg_enum WHERE enumlabel = :value "
            "AND enumtypid = (SELECT oid FROM pg_type WHERE typname = :enum_name)"
        ),
        {"value": value, "enum_name": enum_name},
    ).first()
    return result is not None


def upgrade() -> None:
    """应用迁移：升级到新版本"""
    bind = op.get_bind()

    # ========== Part 1: users 表修改 ==========

    # 1) 新增 email_verified
    if not column_exists("users", "email_verified"):
        op.add_column("users", sa.Column("email_verified", sa.Boolean(), nullable=True))
        # 历史数据回填：已有邮箱的用户默认视为已验证
        op.execute(sa.text("UPDATE users SET email_verified = true WHERE email IS NOT NULL"))
        op.execute(sa.text("UPDATE users SET email_verified = false WHERE email IS NULL"))
        # 收紧约束
        op.alter_column("users", "email_verified", existing_type=sa.Boolean(), nullable=False)

    # 2) email 放宽为可空
    if not column_is_nullable("users", "email"):
        op.alter_column(
            "users",
            "email",
            existing_type=sa.String(length=255),
            nullable=True,
        )

    # 3) password_hash 放宽为可空
    if not column_is_nullable("users", "password_hash"):
        op.alter_column(
            "users",
            "password_hash",
            existing_type=sa.String(length=255),
            nullable=True,
        )

    # ========== Part 2: OAuth 相关 ==========

    # 4) 扩展 authsource enum
    if bind.dialect.name == "postgresql" and not enum_value_exists("authsource", "oauth"):
        ctx = op.get_context()
        with ctx.autocommit_block():
            op.execute("ALTER TYPE authsource ADD VALUE IF NOT EXISTS 'oauth'")

    # 5) OAuth provider 配置表
    if not table_exists("oauth_providers"):
        op.create_table(
            "oauth_providers",
            sa.Column("provider_type", sa.String(length=50), primary_key=True),
            sa.Column("display_name", sa.String(length=100), nullable=False),
            sa.Column("client_id", sa.String(length=255), nullable=False),
            sa.Column("client_secret_encrypted", sa.Text(), nullable=True),
            sa.Column("authorization_url_override", sa.String(length=500), nullable=True),
            sa.Column("token_url_override", sa.String(length=500), nullable=True),
            sa.Column("userinfo_url_override", sa.String(length=500), nullable=True),
            sa.Column("scopes", sa.JSON(), nullable=True),
            sa.Column("redirect_uri", sa.String(length=500), nullable=False),
            sa.Column("frontend_callback_url", sa.String(length=500), nullable=False),
            sa.Column("attribute_mapping", sa.JSON(), nullable=True),
            sa.Column("extra_config", sa.JSON(), nullable=True),
            sa.Column(
                "is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
        )

    # 6) 用户 OAuth 绑定关系表
    if not table_exists("user_oauth_links"):
        op.create_table(
            "user_oauth_links",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(length=36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "provider_type",
                sa.String(length=50),
                sa.ForeignKey("oauth_providers.provider_type", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("provider_user_id", sa.String(length=255), nullable=False),
            sa.Column("provider_username", sa.String(length=255), nullable=True),
            sa.Column("provider_email", sa.String(length=255), nullable=True),
            sa.Column("extra_data", sa.JSON(), nullable=True),
            sa.Column(
                "linked_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.UniqueConstraint(
                "provider_type", "provider_user_id", name="uq_oauth_provider_user"
            ),
            sa.UniqueConstraint("user_id", "provider_type", name="uq_user_oauth_provider"),
        )
        op.create_index("ix_user_oauth_links_user_id", "user_oauth_links", ["user_id"])
        op.create_index(
            "ix_user_oauth_links_provider_type", "user_oauth_links", ["provider_type"]
        )


def downgrade() -> None:
    """回滚迁移：降级到旧版本"""
    bind = op.get_bind()

    # ========== Part 2: OAuth 相关（先删除，因为有外键依赖） ==========

    if table_exists("user_oauth_links"):
        op.drop_index("ix_user_oauth_links_provider_type", table_name="user_oauth_links")
        op.drop_index("ix_user_oauth_links_user_id", table_name="user_oauth_links")
        op.drop_table("user_oauth_links")

    if table_exists("oauth_providers"):
        op.drop_table("oauth_providers")

    # 注意：Postgres 不支持从 ENUM 删除值，authsource 不回退

    # ========== Part 1: users 表修改 ==========

    # 降级前检查：避免把包含 NULL 的列强制改回 NOT NULL
    has_null_email = bind.execute(
        sa.text("SELECT 1 FROM users WHERE email IS NULL LIMIT 1")
    ).first()
    if has_null_email:
        raise RuntimeError("Cannot downgrade: users.email contains NULL values")

    has_null_password = bind.execute(
        sa.text("SELECT 1 FROM users WHERE password_hash IS NULL LIMIT 1")
    ).first()
    if has_null_password:
        raise RuntimeError("Cannot downgrade: users.password_hash contains NULL values")

    # 恢复 NOT NULL 约束
    if column_is_nullable("users", "email"):
        op.alter_column(
            "users",
            "email",
            existing_type=sa.String(length=255),
            nullable=False,
        )

    if column_is_nullable("users", "password_hash"):
        op.alter_column(
            "users",
            "password_hash",
            existing_type=sa.String(length=255),
            nullable=False,
        )

    if column_exists("users", "email_verified"):
        op.drop_column("users", "email_verified")
