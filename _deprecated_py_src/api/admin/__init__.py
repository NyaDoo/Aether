"""Admin API routers.

The admin surface remains Python-only host/control-plane scope. It is not part
of the compatibility frontdoor manifest that Rust is preparing to absorb.
"""

from __future__ import annotations

from fastapi import APIRouter
from starlette.routing import BaseRoute

from .adaptive import router as adaptive_router
from .api_keys import router as api_keys_router
from .billing import router as billing_router
from .endpoints import router as endpoints_router
from .models import router as models_router
from .modules import router as modules_router
from .monitoring import router as monitoring_router
from .payments import router as payments_router
from .pool import router as pool_router
from .provider_oauth import router as provider_oauth_router
from .provider_ops import router as provider_ops_router
from .provider_query import router as provider_query_router
from .provider_strategy import router as provider_strategy_router
from .providers import router as providers_router
from .security import router as security_router
from .stats import router as stats_router
from .system import router as system_router
from .usage import router as usage_router
from .users import router as users_router
from .video_tasks import router as video_tasks_router
from .wallets import router as wallets_router

_RUST_OWNED_ADMIN_ROUTE_SIGNATURES = frozenset(
    {
        ("GET", "/api/admin/modules/status"),
        ("GET", "/api/admin/modules/status/{module_name}"),
        ("PUT", "/api/admin/modules/status/{module_name}/enabled"),
        ("GET", "/api/admin/system/version"),
        ("GET", "/api/admin/system/check-update"),
        ("GET", "/api/admin/system/aws-regions"),
        ("GET", "/api/admin/system/stats"),
        ("GET", "/api/admin/system/settings"),
        ("GET", "/api/admin/system/config/export"),
        ("GET", "/api/admin/system/users/export"),
        ("POST", "/api/admin/system/config/import"),
        ("POST", "/api/admin/system/users/import"),
        ("POST", "/api/admin/system/smtp/test"),
        ("POST", "/api/admin/system/cleanup"),
        ("POST", "/api/admin/system/purge/config"),
        ("POST", "/api/admin/system/purge/users"),
        ("POST", "/api/admin/system/purge/usage"),
        ("POST", "/api/admin/system/purge/audit-logs"),
        ("POST", "/api/admin/system/purge/request-bodies"),
        ("POST", "/api/admin/system/purge/stats"),
        ("PUT", "/api/admin/system/settings"),
        ("GET", "/api/admin/system/configs"),
        ("GET", "/api/admin/system/configs/{key}"),
        ("PUT", "/api/admin/system/configs/{key}"),
        ("DELETE", "/api/admin/system/configs/{key}"),
        ("GET", "/api/admin/system/api-formats"),
        ("GET", "/api/admin/system/email/templates"),
        ("GET", "/api/admin/system/email/templates/{template_type}"),
        ("PUT", "/api/admin/system/email/templates/{template_type}"),
        ("POST", "/api/admin/system/email/templates/{template_type}/preview"),
        ("POST", "/api/admin/system/email/templates/{template_type}/reset"),
        ("GET", "/api/admin/providers/"),
        ("POST", "/api/admin/providers/"),
        ("PATCH", "/api/admin/providers/{provider_id}"),
        ("DELETE", "/api/admin/providers/{provider_id}"),
        ("GET", "/api/admin/providers/summary"),
        ("GET", "/api/admin/providers/{provider_id}/summary"),
        ("GET", "/api/admin/providers/{provider_id}/health-monitor"),
        ("GET", "/api/admin/providers/{provider_id}/mapping-preview"),
        ("GET", "/api/admin/providers/{provider_id}/delete-task/{task_id}"),
        ("GET", "/api/admin/providers/{provider_id}/pool-status"),
        ("POST", "/api/admin/providers/{provider_id}/pool/clear-cooldown/{key_id}"),
        ("POST", "/api/admin/providers/{provider_id}/pool/reset-cost/{key_id}"),
        ("GET", "/api/admin/providers/{provider_id}/models"),
        ("POST", "/api/admin/providers/{provider_id}/models"),
        ("GET", "/api/admin/providers/{provider_id}/models/{model_id}"),
        ("PATCH", "/api/admin/providers/{provider_id}/models/{model_id}"),
        ("DELETE", "/api/admin/providers/{provider_id}/models/{model_id}"),
        ("POST", "/api/admin/providers/{provider_id}/models/batch"),
        ("GET", "/api/admin/providers/{provider_id}/available-source-models"),
        ("POST", "/api/admin/providers/{provider_id}/assign-global-models"),
        ("POST", "/api/admin/providers/{provider_id}/import-from-upstream"),
        ("GET", "/api/admin/endpoints/providers/{provider_id}/endpoints"),
        ("POST", "/api/admin/endpoints/providers/{provider_id}/endpoints"),
        ("GET", "/api/admin/endpoints/defaults/{api_format}/body-rules"),
        ("GET", "/api/admin/endpoints/{endpoint_id}"),
        ("PUT", "/api/admin/endpoints/{endpoint_id}"),
        ("DELETE", "/api/admin/endpoints/{endpoint_id}"),
        ("PUT", "/api/admin/endpoints/keys/{key_id}"),
        ("GET", "/api/admin/endpoints/keys/grouped-by-format"),
        ("GET", "/api/admin/endpoints/keys/{key_id}/reveal"),
        ("GET", "/api/admin/endpoints/keys/{key_id}/export"),
        ("DELETE", "/api/admin/endpoints/keys/{key_id}"),
        ("POST", "/api/admin/endpoints/keys/batch-delete"),
        ("POST", "/api/admin/endpoints/keys/{key_id}/clear-oauth-invalid"),
        ("GET", "/api/admin/endpoints/providers/{provider_id}/keys"),
        ("POST", "/api/admin/endpoints/providers/{provider_id}/keys"),
        ("POST", "/api/admin/endpoints/providers/{provider_id}/refresh-quota"),
        ("GET", "/api/admin/endpoints/rpm/key/{key_id}"),
        ("DELETE", "/api/admin/endpoints/rpm/key/{key_id}"),
        ("GET", "/api/admin/endpoints/health/summary"),
        ("GET", "/api/admin/endpoints/health/status"),
        ("GET", "/api/admin/endpoints/health/api-formats"),
        ("GET", "/api/admin/endpoints/health/key/{key_id}"),
        ("PATCH", "/api/admin/endpoints/health/keys/{key_id}"),
        ("PATCH", "/api/admin/endpoints/health/keys"),
        ("GET", "/api/admin/provider-oauth/supported-types"),
        ("POST", "/api/admin/provider-oauth/keys/{key_id}/start"),
        ("POST", "/api/admin/provider-oauth/keys/{key_id}/complete"),
        ("POST", "/api/admin/provider-oauth/keys/{key_id}/refresh"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/start"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/complete"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/import-refresh-token"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/device-authorize"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/device-poll"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/batch-import"),
        ("POST", "/api/admin/provider-oauth/providers/{provider_id}/batch-import/tasks"),
        ("GET", "/api/admin/provider-oauth/providers/{provider_id}/batch-import/tasks/{task_id}"),
        ("GET", "/api/admin/adaptive/keys"),
        ("PATCH", "/api/admin/adaptive/keys/{key_id}/mode"),
        ("GET", "/api/admin/adaptive/keys/{key_id}/stats"),
        ("DELETE", "/api/admin/adaptive/keys/{key_id}/learning"),
        ("PATCH", "/api/admin/adaptive/keys/{key_id}/limit"),
        ("GET", "/api/admin/adaptive/summary"),
        ("GET", "/api/admin/provider-ops/architectures"),
        ("GET", "/api/admin/provider-ops/architectures/{architecture_id}"),
        ("GET", "/api/admin/provider-ops/providers/{provider_id}/status"),
        ("GET", "/api/admin/provider-ops/providers/{provider_id}/config"),
        ("PUT", "/api/admin/provider-ops/providers/{provider_id}/config"),
        ("DELETE", "/api/admin/provider-ops/providers/{provider_id}/config"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/connect"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/disconnect"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/verify"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/actions/{action_type}"),
        ("GET", "/api/admin/provider-ops/providers/{provider_id}/balance"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/balance"),
        ("POST", "/api/admin/provider-ops/providers/{provider_id}/checkin"),
        ("POST", "/api/admin/provider-ops/batch/balance"),
        ("GET", "/api/admin/billing/presets"),
        ("POST", "/api/admin/billing/presets/apply"),
        ("GET", "/api/admin/billing/rules"),
        ("GET", "/api/admin/billing/rules/{rule_id}"),
        ("POST", "/api/admin/billing/rules"),
        ("PUT", "/api/admin/billing/rules/{rule_id}"),
        ("GET", "/api/admin/billing/collectors"),
        ("GET", "/api/admin/billing/collectors/{collector_id}"),
        ("POST", "/api/admin/billing/collectors"),
        ("PUT", "/api/admin/billing/collectors/{collector_id}"),
        ("PUT", "/api/admin/provider-strategy/providers/{provider_id}/billing"),
        ("GET", "/api/admin/provider-strategy/providers/{provider_id}/stats"),
        ("GET", "/api/admin/provider-strategy/strategies"),
        ("DELETE", "/api/admin/provider-strategy/providers/{provider_id}/quota"),
        ("POST", "/api/admin/provider-query/models"),
        ("POST", "/api/admin/provider-query/test-model"),
        ("POST", "/api/admin/provider-query/test-model-failover"),
        ("GET", "/api/admin/payments/orders"),
        ("GET", "/api/admin/payments/orders/{order_id}"),
        ("POST", "/api/admin/payments/orders/{order_id}/expire"),
        ("POST", "/api/admin/payments/orders/{order_id}/credit"),
        ("POST", "/api/admin/payments/orders/{order_id}/fail"),
        ("GET", "/api/admin/payments/callbacks"),
        ("POST", "/api/admin/security/ip/blacklist"),
        ("DELETE", "/api/admin/security/ip/blacklist/{ip_address}"),
        ("GET", "/api/admin/security/ip/blacklist/stats"),
        ("POST", "/api/admin/security/ip/whitelist"),
        ("DELETE", "/api/admin/security/ip/whitelist/{ip_address}"),
        ("GET", "/api/admin/security/ip/whitelist"),
        ("GET", "/api/admin/stats/providers/quota-usage"),
        ("GET", "/api/admin/stats/comparison"),
        ("GET", "/api/admin/stats/errors/distribution"),
        ("GET", "/api/admin/stats/performance/percentiles"),
        ("GET", "/api/admin/stats/cost/forecast"),
        ("GET", "/api/admin/stats/cost/savings"),
        ("GET", "/api/admin/stats/leaderboard/api-keys"),
        ("GET", "/api/admin/stats/leaderboard/models"),
        ("GET", "/api/admin/stats/leaderboard/users"),
        ("GET", "/api/admin/stats/time-series"),
        ("GET", "/api/admin/monitoring/audit-logs"),
        ("GET", "/api/admin/monitoring/system-status"),
        ("GET", "/api/admin/monitoring/suspicious-activities"),
        ("GET", "/api/admin/monitoring/user-behavior/{user_id}"),
        ("GET", "/api/admin/monitoring/resilience-status"),
        ("GET", "/api/admin/monitoring/resilience/circuit-history"),
        ("DELETE", "/api/admin/monitoring/resilience/error-stats"),
        ("GET", "/api/admin/monitoring/trace/{request_id}"),
        ("GET", "/api/admin/monitoring/trace/stats/provider/{provider_id}"),
        ("GET", "/api/admin/monitoring/cache/stats"),
        ("GET", "/api/admin/monitoring/cache/affinity/{user_identifier}"),
        ("GET", "/api/admin/monitoring/cache/affinities"),
        ("DELETE", "/api/admin/monitoring/cache/users/{user_identifier}"),
        (
            "DELETE",
            "/api/admin/monitoring/cache/affinity/{affinity_key}/{endpoint_id}/{model_id}/{api_format}",
        ),
        ("DELETE", "/api/admin/monitoring/cache"),
        ("DELETE", "/api/admin/monitoring/cache/providers/{provider_id}"),
        ("GET", "/api/admin/monitoring/cache/config"),
        ("GET", "/api/admin/monitoring/cache/metrics"),
        ("GET", "/api/admin/monitoring/cache/model-mapping/stats"),
        ("DELETE", "/api/admin/monitoring/cache/model-mapping"),
        ("DELETE", "/api/admin/monitoring/cache/model-mapping/{model_name}"),
        (
            "DELETE",
            "/api/admin/monitoring/cache/model-mapping/provider/{provider_id}/{global_model_id}",
        ),
        ("GET", "/api/admin/monitoring/cache/redis-keys"),
        ("DELETE", "/api/admin/monitoring/cache/redis-keys/{category}"),
        ("GET", "/api/admin/usage/aggregation/stats"),
        ("GET", "/api/admin/usage/stats"),
        ("GET", "/api/admin/usage/heatmap"),
        ("GET", "/api/admin/usage/records"),
        ("GET", "/api/admin/usage/active"),
        ("GET", "/api/admin/usage/cache-affinity/hit-analysis"),
        ("GET", "/api/admin/usage/cache-affinity/interval-timeline"),
        ("GET", "/api/admin/usage/cache-affinity/ttl-analysis"),
        ("GET", "/api/admin/usage/{usage_id}/curl"),
        ("GET", "/api/admin/usage/{usage_id}"),
        ("POST", "/api/admin/usage/{usage_id}/replay"),
        ("GET", "/api/admin/video-tasks"),
        ("GET", "/api/admin/video-tasks/stats"),
        ("GET", "/api/admin/video-tasks/{task_id}"),
        ("POST", "/api/admin/video-tasks/{task_id}/cancel"),
        ("GET", "/api/admin/video-tasks/{task_id}/video"),
        ("GET", "/api/admin/wallets"),
        ("GET", "/api/admin/wallets/ledger"),
        ("GET", "/api/admin/wallets/refund-requests"),
        ("GET", "/api/admin/wallets/{wallet_id}"),
        ("GET", "/api/admin/wallets/{wallet_id}/transactions"),
        ("GET", "/api/admin/wallets/{wallet_id}/refunds"),
        ("POST", "/api/admin/wallets/{wallet_id}/adjust"),
        ("POST", "/api/admin/wallets/{wallet_id}/recharge"),
        ("POST", "/api/admin/wallets/{wallet_id}/refunds/{refund_id}/process"),
        ("POST", "/api/admin/wallets/{wallet_id}/refunds/{refund_id}/complete"),
        ("POST", "/api/admin/wallets/{wallet_id}/refunds/{refund_id}/fail"),
        ("GET", "/api/admin/api-keys"),
        ("POST", "/api/admin/api-keys"),
        ("GET", "/api/admin/api-keys/{key_id}"),
        ("PUT", "/api/admin/api-keys/{key_id}"),
        ("PATCH", "/api/admin/api-keys/{key_id}"),
        ("DELETE", "/api/admin/api-keys/{key_id}"),
        ("GET", "/api/admin/users"),
        ("POST", "/api/admin/users"),
        ("GET", "/api/admin/users/{user_id}"),
        ("PUT", "/api/admin/users/{user_id}"),
        ("DELETE", "/api/admin/users/{user_id}"),
        ("GET", "/api/admin/users/{user_id}/sessions"),
        ("DELETE", "/api/admin/users/{user_id}/sessions"),
        ("DELETE", "/api/admin/users/{user_id}/sessions/{session_id}"),
        ("GET", "/api/admin/users/{user_id}/api-keys"),
        ("POST", "/api/admin/users/{user_id}/api-keys"),
        ("DELETE", "/api/admin/users/{user_id}/api-keys/{key_id}"),
        ("PUT", "/api/admin/users/{user_id}/api-keys/{key_id}"),
        ("PATCH", "/api/admin/users/{user_id}/api-keys/{key_id}/lock"),
        ("GET", "/api/admin/users/{user_id}/api-keys/{key_id}/full-key"),
        ("GET", "/api/admin/pool/overview"),
        ("GET", "/api/admin/pool/scheduling-presets"),
        ("GET", "/api/admin/pool/{provider_id}/keys"),
        ("GET", "/api/admin/pool/{provider_id}/keys/batch-delete-task/{task_id}"),
        ("POST", "/api/admin/pool/{provider_id}/keys/batch-action"),
        ("POST", "/api/admin/pool/{provider_id}/keys/batch-import"),
        ("POST", "/api/admin/pool/{provider_id}/keys/cleanup-banned"),
        ("POST", "/api/admin/pool/{provider_id}/keys/resolve-selection"),
        ("GET", "/api/admin/proxy-nodes"),
        ("GET", "/api/admin/models/catalog"),
        ("GET", "/api/admin/models/external"),
        ("DELETE", "/api/admin/models/external/cache"),
        ("GET", "/api/admin/models/global"),
        ("POST", "/api/admin/models/global"),
        ("GET", "/api/admin/models/global/{global_model_id}"),
        ("PATCH", "/api/admin/models/global/{global_model_id}"),
        ("DELETE", "/api/admin/models/global/{global_model_id}"),
        ("POST", "/api/admin/models/global/batch-delete"),
        ("POST", "/api/admin/models/global/{global_model_id}/assign-to-providers"),
        ("GET", "/api/admin/models/global/{global_model_id}/providers"),
        ("GET", "/api/admin/models/global/{global_model_id}/routing"),
    }
)


def _route_is_rust_owned(route: BaseRoute) -> bool:
    path = getattr(route, "path", None)
    methods = getattr(route, "methods", None)
    if not isinstance(path, str) or not methods:
        return False
    return any(
        (method, path) in _RUST_OWNED_ADMIN_ROUTE_SIGNATURES
        for method in methods
        if method not in {"HEAD", "OPTIONS"}
    )


def _build_python_admin_router() -> APIRouter:
    """Admin/control-plane routes that still require the Python host."""
    admin_router = APIRouter()
    admin_router.include_router(system_router)
    admin_router.include_router(users_router)
    admin_router.include_router(providers_router)
    admin_router.include_router(api_keys_router)
    admin_router.include_router(billing_router)
    admin_router.include_router(usage_router)
    admin_router.include_router(monitoring_router)
    admin_router.include_router(payments_router)
    admin_router.include_router(endpoints_router)
    admin_router.include_router(provider_strategy_router)
    admin_router.include_router(provider_oauth_router)
    admin_router.include_router(adaptive_router)
    admin_router.include_router(models_router)
    admin_router.include_router(security_router)
    admin_router.include_router(stats_router)
    admin_router.include_router(provider_query_router)
    admin_router.include_router(modules_router)
    admin_router.include_router(pool_router)
    admin_router.include_router(provider_ops_router)
    admin_router.include_router(video_tasks_router)
    admin_router.include_router(wallets_router)
    admin_router.routes = [route for route in admin_router.routes if not _route_is_rust_owned(route)]
    return admin_router


# Admin/control-plane 在本轮 frontdoor cutover 后仍保留在 Python 宿主。
python_admin_router = _build_python_admin_router()
router = python_admin_router

# 注意：以下路由已迁移到模块系统，由 ModuleRegistry 动态注册
# - ldap_router: 当 LDAP_AVAILABLE=true 时注册
# - management_tokens_router: 当 MANAGEMENT_TOKENS_AVAILABLE=true 时注册
# - proxy_nodes_router: 当 PROXY_NODES_AVAILABLE=true 时注册

__all__ = ["python_admin_router", "router"]
