"""Baseline migration - all tables consolidated

Revision ID: 20251210_baseline
Revises:
Create Date: 2024-12-10

This is the consolidated baseline migration that creates all tables from scratch.
Includes all schema changes up to circuit breaker v2.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "20251210_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create ENUM types (with IF NOT EXISTS for idempotency)
    op.execute("DO $$ BEGIN CREATE TYPE userrole AS ENUM ('admin', 'user'); EXCEPTION WHEN duplicate_object THEN NULL; END $$")
    op.execute(
        "DO $$ BEGIN CREATE TYPE providerbillingtype AS ENUM ('monthly_quota', 'pay_as_you_go', 'free_tier'); EXCEPTION WHEN duplicate_object THEN NULL; END $$"
    )

    # ==================== users ====================
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("email", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("username", sa.String(100), unique=True, index=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column(
            "role",
            postgresql.ENUM("admin", "user", name="userrole", create_type=False),
            nullable=False,
            server_default="user",
        ),
        sa.Column("allowed_providers", sa.JSON, nullable=True),
        sa.Column("allowed_endpoints", sa.JSON, nullable=True),
        sa.Column("allowed_models", sa.JSON, nullable=True),
        sa.Column("model_capability_settings", sa.JSON, nullable=True),
        sa.Column("quota_usd", sa.Float, nullable=True),
        sa.Column("used_usd", sa.Float, server_default="0.0"),
        sa.Column("total_usd", sa.Float, server_default="0.0"),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("is_deleted", sa.Boolean, server_default="false", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ==================== providers ====================
    op.create_table(
        "providers",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("name", sa.String(100), unique=True, index=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("website", sa.String(500), nullable=True),
        sa.Column(
            "billing_type",
            postgresql.ENUM(
                "monthly_quota", "pay_as_you_go", "free_tier", name="providerbillingtype", create_type=False
            ),
            nullable=False,
            server_default="pay_as_you_go",
        ),
        sa.Column("monthly_quota_usd", sa.Float, nullable=True),
        sa.Column("monthly_used_usd", sa.Float, server_default="0.0"),
        sa.Column("quota_reset_day", sa.Integer, server_default="30"),
        sa.Column("quota_last_reset_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("quota_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rpm_limit", sa.Integer, nullable=True),
        sa.Column("rpm_used", sa.Integer, server_default="0"),
        sa.Column("rpm_reset_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_priority", sa.Integer, server_default="100"),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("rate_limit", sa.Integer, nullable=True),
        sa.Column("concurrent_limit", sa.Integer, nullable=True),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== global_models ====================
    op.create_table(
        "global_models",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("name", sa.String(100), unique=True, index=True, nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("icon_url", sa.String(500), nullable=True),
        sa.Column("official_url", sa.String(500), nullable=True),
        sa.Column("default_price_per_request", sa.Float, nullable=True),
        sa.Column("default_tiered_pricing", sa.JSON, nullable=False),
        sa.Column("default_supports_vision", sa.Boolean, server_default="false", nullable=True),
        sa.Column("default_supports_function_calling", sa.Boolean, server_default="false", nullable=True),
        sa.Column("default_supports_streaming", sa.Boolean, server_default="true", nullable=True),
        sa.Column("default_supports_extended_thinking", sa.Boolean, server_default="false", nullable=True),
        sa.Column("default_supports_image_generation", sa.Boolean, server_default="false", nullable=True),
        sa.Column("supported_capabilities", sa.JSON, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("usage_count", sa.Integer, server_default="0", nullable=False, index=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== api_keys ====================
    op.create_table(
        "api_keys",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("key_hash", sa.String(64), unique=True, index=True, nullable=False),
        sa.Column("key_encrypted", sa.Text, nullable=True),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("total_requests", sa.Integer, server_default="0"),
        sa.Column("total_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("balance_used_usd", sa.Float, server_default="0.0"),
        sa.Column("current_balance_usd", sa.Float, nullable=True),
        sa.Column("is_standalone", sa.Boolean, server_default="false", nullable=False),
        sa.Column("allowed_providers", sa.JSON, nullable=True),
        sa.Column("allowed_endpoints", sa.JSON, nullable=True),
        sa.Column("allowed_api_formats", sa.JSON, nullable=True),
        sa.Column("allowed_models", sa.JSON, nullable=True),
        sa.Column("rate_limit", sa.Integer, server_default="100"),
        sa.Column("concurrent_limit", sa.Integer, server_default="5", nullable=True),
        sa.Column("force_capabilities", sa.JSON, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("auto_delete_on_expiry", sa.Boolean, server_default="false", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== provider_endpoints ====================
    op.create_table(
        "provider_endpoints",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "provider_id",
            sa.String(36),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("api_format", sa.String(50), nullable=False),
        sa.Column("base_url", sa.String(500), nullable=False),
        sa.Column("headers", sa.JSON, nullable=True),
        sa.Column("timeout", sa.Integer, server_default="300"),
        sa.Column("max_retries", sa.Integer, server_default="3"),
        sa.Column("max_concurrent", sa.Integer, nullable=True),
        sa.Column("rate_limit", sa.Integer, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("custom_path", sa.String(200), nullable=True),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("provider_id", "api_format", name="uq_provider_api_format"),
    )
    op.create_index(
        "idx_endpoint_format_active", "provider_endpoints", ["api_format", "is_active"]
    )

    # ==================== models ====================
    op.create_table(
        "models",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "provider_id", sa.String(36), sa.ForeignKey("providers.id"), nullable=False
        ),
        sa.Column(
            "global_model_id",
            sa.String(36),
            sa.ForeignKey("global_models.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("provider_model_name", sa.String(200), nullable=False),
        sa.Column("price_per_request", sa.Float, nullable=True),
        sa.Column("tiered_pricing", sa.JSON, nullable=True),
        sa.Column("supports_vision", sa.Boolean, nullable=True),
        sa.Column("supports_function_calling", sa.Boolean, nullable=True),
        sa.Column("supports_streaming", sa.Boolean, nullable=True),
        sa.Column("supports_extended_thinking", sa.Boolean, nullable=True),
        sa.Column("supports_image_generation", sa.Boolean, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("is_available", sa.Boolean, server_default="true"),
        sa.Column("config", sa.JSON, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("provider_id", "provider_model_name", name="uq_provider_model"),
    )

    # ==================== model_mappings ====================
    op.create_table(
        "model_mappings",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("source_model", sa.String(200), nullable=False, index=True),
        sa.Column(
            "target_global_model_id",
            sa.String(36),
            sa.ForeignKey("global_models.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "provider_id", sa.String(36), sa.ForeignKey("providers.id"), nullable=True, index=True
        ),
        sa.Column("mapping_type", sa.String(20), nullable=False, server_default="alias", index=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("source_model", "provider_id", name="uq_model_mapping_source_provider"),
    )

    # ==================== provider_api_keys ====================
    op.create_table(
        "provider_api_keys",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "endpoint_id",
            sa.String(36),
            sa.ForeignKey("provider_endpoints.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("api_key", sa.String(500), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("note", sa.String(500), nullable=True),
        sa.Column("rate_multiplier", sa.Float, server_default="1.0", nullable=False),
        sa.Column("internal_priority", sa.Integer, server_default="50"),
        sa.Column("global_priority", sa.Integer, nullable=True),
        sa.Column("max_concurrent", sa.Integer, nullable=True),
        sa.Column("rate_limit", sa.Integer, nullable=True),
        sa.Column("daily_limit", sa.Integer, nullable=True),
        sa.Column("monthly_limit", sa.Integer, nullable=True),
        sa.Column("allowed_models", sa.JSON, nullable=True),
        sa.Column("capabilities", sa.JSON, nullable=True),
        sa.Column("learned_max_concurrent", sa.Integer, nullable=True),
        sa.Column("concurrent_429_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("rpm_429_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("last_429_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_429_type", sa.String(50), nullable=True),
        sa.Column("last_concurrent_peak", sa.Integer, nullable=True),
        sa.Column("adjustment_history", sa.JSON, nullable=True),
        # Sliding window fields (replaces high_utilization_start)
        sa.Column("utilization_samples", sa.JSON, nullable=True),
        sa.Column("last_probe_increase_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_score", sa.Float, server_default="1.0"),
        sa.Column("consecutive_failures", sa.Integer, server_default="0"),
        sa.Column("last_failure_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cache_ttl_minutes", sa.Integer, server_default="5", nullable=False),
        sa.Column("max_probe_interval_minutes", sa.Integer, server_default="32", nullable=False),
        sa.Column("circuit_breaker_open", sa.Boolean, server_default="false", nullable=False),
        sa.Column("circuit_breaker_open_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_probe_at", sa.DateTime(timezone=True), nullable=True),
        # Circuit breaker v2 fields
        sa.Column("request_results_window", sa.JSON, nullable=True),
        sa.Column("half_open_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("half_open_successes", sa.Integer, server_default="0", nullable=True),
        sa.Column("half_open_failures", sa.Integer, server_default="0", nullable=True),
        sa.Column("request_count", sa.Integer, server_default="0"),
        sa.Column("success_count", sa.Integer, server_default="0"),
        sa.Column("error_count", sa.Integer, server_default="0"),
        sa.Column("total_response_time_ms", sa.Integer, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error_msg", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, server_default="true", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== usage ====================
    op.create_table(
        "usage",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "api_key_id",
            sa.String(36),
            sa.ForeignKey("api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("request_id", sa.String(100), unique=True, index=True, nullable=False),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("target_model", sa.String(100), nullable=True),
        sa.Column(
            "provider_id",
            sa.String(36),
            sa.ForeignKey("providers.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "provider_endpoint_id",
            sa.String(36),
            sa.ForeignKey("provider_endpoints.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "provider_api_key_id",
            sa.String(36),
            sa.ForeignKey("provider_api_keys.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("input_tokens", sa.Integer, server_default="0"),
        sa.Column("output_tokens", sa.Integer, server_default="0"),
        sa.Column("total_tokens", sa.Integer, server_default="0"),
        sa.Column("cache_creation_input_tokens", sa.Integer, server_default="0"),
        sa.Column("cache_read_input_tokens", sa.Integer, server_default="0"),
        sa.Column("input_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("output_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("cache_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("cache_creation_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("cache_read_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("request_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("total_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_input_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_output_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_cache_creation_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_cache_read_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_request_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("actual_total_cost_usd", sa.Float, server_default="0.0"),
        sa.Column("rate_multiplier", sa.Float, server_default="1.0"),
        sa.Column("input_price_per_1m", sa.Float, nullable=True),
        sa.Column("output_price_per_1m", sa.Float, nullable=True),
        sa.Column("cache_creation_price_per_1m", sa.Float, nullable=True),
        sa.Column("cache_read_price_per_1m", sa.Float, nullable=True),
        sa.Column("price_per_request", sa.Float, nullable=True),
        sa.Column("request_type", sa.String(50), nullable=True),
        sa.Column("api_format", sa.String(50), nullable=True),
        sa.Column("is_stream", sa.Boolean, server_default="false"),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("response_time_ms", sa.Integer, nullable=True),
        sa.Column("status", sa.String(20), server_default="completed", nullable=False, index=True),
        sa.Column("request_headers", sa.JSON, nullable=True),
        sa.Column("request_body", sa.JSON, nullable=True),
        sa.Column("provider_request_headers", sa.JSON, nullable=True),
        sa.Column("response_headers", sa.JSON, nullable=True),
        sa.Column("response_body", sa.JSON, nullable=True),
        sa.Column("request_body_compressed", sa.LargeBinary, nullable=True),
        sa.Column("response_body_compressed", sa.LargeBinary, nullable=True),
        sa.Column("request_metadata", sa.JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            index=True,
        ),
    )
    # usage 表复合索引（优化常见查询）
    op.create_index("idx_usage_user_created", "usage", ["user_id", "created_at"])
    op.create_index("idx_usage_apikey_created", "usage", ["api_key_id", "created_at"])
    op.create_index("idx_usage_provider_model_created", "usage", ["provider", "model", "created_at"])

    # ==================== user_quotas ====================
    op.create_table(
        "user_quotas",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("quota_type", sa.String(50), nullable=False),
        sa.Column("quota_usd", sa.Float, nullable=False),
        sa.Column("period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_usd", sa.Float, server_default="0.0"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== system_configs ====================
    op.create_table(
        "system_configs",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("key", sa.String(100), unique=True, nullable=False),
        sa.Column("value", sa.JSON, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== user_preferences ====================
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("bio", sa.Text, nullable=True),
        sa.Column(
            "default_provider_id", sa.String(36), sa.ForeignKey("providers.id"), nullable=True
        ),
        sa.Column("theme", sa.String(20), server_default="light"),
        sa.Column("language", sa.String(10), server_default="zh-CN"),
        sa.Column("timezone", sa.String(50), server_default="Asia/Shanghai"),
        sa.Column("email_notifications", sa.Boolean, server_default="true"),
        sa.Column("usage_alerts", sa.Boolean, server_default="true"),
        sa.Column("announcement_notifications", sa.Boolean, server_default="true"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== announcements ====================
    op.create_table(
        "announcements",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("type", sa.String(20), server_default="info"),
        sa.Column("priority", sa.Integer, server_default="0"),
        sa.Column(
            "author_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean, server_default="true", index=True),
        sa.Column("is_pinned", sa.Boolean, server_default="false"),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== announcement_reads ====================
    op.create_table(
        "announcement_reads",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "announcement_id", sa.String(36), sa.ForeignKey("announcements.id"), nullable=False
        ),
        sa.Column(
            "read_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("user_id", "announcement_id", name="uq_user_announcement"),
    )

    # ==================== audit_logs ====================
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column("event_type", sa.String(50), nullable=False, index=True),
        sa.Column(
            "user_id",
            sa.String(36),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("api_key_id", sa.String(36), nullable=True),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(500), nullable=True),
        sa.Column("request_id", sa.String(100), nullable=True, index=True),
        sa.Column("event_metadata", sa.JSON, nullable=True),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            index=True,
        ),
    )

    # ==================== request_candidates ====================
    op.create_table(
        "request_candidates",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("request_id", sa.String(100), nullable=False, index=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column(
            "api_key_id",
            sa.String(36),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("candidate_index", sa.Integer, nullable=False),
        sa.Column("retry_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "provider_id",
            sa.String(36),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "endpoint_id",
            sa.String(36),
            sa.ForeignKey("provider_endpoints.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "key_id",
            sa.String(36),
            sa.ForeignKey("provider_api_keys.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("skip_reason", sa.Text, nullable=True),
        sa.Column("is_cached", sa.Boolean, server_default="false"),
        sa.Column("status_code", sa.Integer, nullable=True),
        sa.Column("error_type", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("concurrent_requests", sa.Integer, nullable=True),
        sa.Column("extra_data", sa.JSON, nullable=True),
        sa.Column("required_capabilities", sa.JSON, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "request_id", "candidate_index", "retry_index", name="uq_request_candidate_with_retry"
        ),
    )
    op.create_index("idx_request_candidates_request_id", "request_candidates", ["request_id"])
    op.create_index("idx_request_candidates_status", "request_candidates", ["status"])
    op.create_index("idx_request_candidates_provider_id", "request_candidates", ["provider_id"])

    # ==================== stats_daily ====================
    op.create_table(
        "stats_daily",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False, unique=True, index=True),
        sa.Column("total_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("success_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("error_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("input_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("output_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("cache_creation_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("cache_read_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("total_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("actual_total_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("input_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("output_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("cache_creation_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("cache_read_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("avg_response_time_ms", sa.Float, server_default="0.0", nullable=False),
        sa.Column("fallback_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("unique_models", sa.Integer, server_default="0", nullable=False),
        sa.Column("unique_providers", sa.Integer, server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== stats_summary ====================
    op.create_table(
        "stats_summary",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("cutoff_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("all_time_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("all_time_success_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("all_time_error_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("all_time_input_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("all_time_output_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column(
            "all_time_cache_creation_tokens", sa.BigInteger, server_default="0", nullable=False
        ),
        sa.Column("all_time_cache_read_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("all_time_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("all_time_actual_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column("total_users", sa.Integer, server_default="0", nullable=False),
        sa.Column("active_users", sa.Integer, server_default="0", nullable=False),
        sa.Column("total_api_keys", sa.Integer, server_default="0", nullable=False),
        sa.Column("active_api_keys", sa.Integer, server_default="0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )

    # ==================== stats_user_daily ====================
    op.create_table(
        "stats_user_daily",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("total_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("success_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("error_requests", sa.Integer, server_default="0", nullable=False),
        sa.Column("input_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("output_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("cache_creation_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("cache_read_tokens", sa.BigInteger, server_default="0", nullable=False),
        sa.Column("total_cost", sa.Float, server_default="0.0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("user_id", "date", name="uq_stats_user_daily"),
    )
    op.create_index("idx_stats_user_daily_user_date", "stats_user_daily", ["user_id", "date"])

    # ==================== api_key_provider_mappings ====================
    op.create_table(
        "api_key_provider_mappings",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "api_key_id",
            sa.String(36),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "provider_id",
            sa.String(36),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("priority_adjustment", sa.Integer, server_default="0"),
        sa.Column("weight_multiplier", sa.Float, server_default="1.0"),
        sa.Column("is_enabled", sa.Boolean, server_default="true", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("api_key_id", "provider_id", name="uq_apikey_provider"),
    )
    op.create_index(
        "idx_apikey_provider_enabled", "api_key_provider_mappings", ["api_key_id", "is_enabled"]
    )

    # ==================== provider_usage_tracking ====================
    op.create_table(
        "provider_usage_tracking",
        sa.Column("id", sa.String(36), primary_key=True, index=True),
        sa.Column(
            "provider_id",
            sa.String(36),
            sa.ForeignKey("providers.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("window_start", sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column("window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_requests", sa.Integer, server_default="0"),
        sa.Column("successful_requests", sa.Integer, server_default="0"),
        sa.Column("failed_requests", sa.Integer, server_default="0"),
        sa.Column("avg_response_time_ms", sa.Float, server_default="0.0"),
        sa.Column("total_response_time_ms", sa.Float, server_default="0.0"),
        sa.Column("total_cost_usd", sa.Float, server_default="0.0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "idx_provider_window", "provider_usage_tracking", ["provider_id", "window_start"]
    )
    op.create_index("idx_window_time", "provider_usage_tracking", ["window_start", "window_end"])


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign key dependencies)
    op.drop_table("provider_usage_tracking")
    op.drop_table("api_key_provider_mappings")
    op.drop_table("stats_user_daily")
    op.drop_table("stats_summary")
    op.drop_table("stats_daily")
    op.drop_table("request_candidates")
    op.drop_table("audit_logs")
    op.drop_table("announcement_reads")
    op.drop_table("announcements")
    op.drop_table("user_preferences")
    op.drop_table("system_configs")
    op.drop_table("user_quotas")
    op.drop_table("usage")
    op.drop_table("provider_api_keys")
    op.drop_table("model_mappings")
    op.drop_table("models")
    op.drop_table("provider_endpoints")
    op.drop_table("api_keys")
    op.drop_table("global_models")
    op.drop_table("providers")
    op.drop_table("users")

    # Drop ENUM types
    op.execute("DROP TYPE IF EXISTS providerbillingtype")
    op.execute("DROP TYPE IF EXISTS userrole")
