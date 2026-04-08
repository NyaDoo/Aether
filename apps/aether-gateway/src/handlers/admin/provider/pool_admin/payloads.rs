use crate::handlers::admin::provider::shared::support::{
    AdminProviderPoolConfig, AdminProviderPoolRuntimeState,
};
use aether_admin::provider::pool as admin_provider_pool_pure;
use aether_data_contracts::repository::provider_catalog::StoredProviderCatalogKey;

pub(super) fn build_admin_pool_key_payload(
    key: &StoredProviderCatalogKey,
    runtime: &AdminProviderPoolRuntimeState,
    pool_config: Option<AdminProviderPoolConfig>,
) -> serde_json::Value {
    admin_provider_pool_pure::build_admin_pool_key_payload(
        key,
        &admin_provider_pool_pure::AdminPoolKeyPayloadContext {
            cooldown_reason: runtime.cooldown_reason_by_key.get(&key.id).cloned(),
            cooldown_ttl_seconds: runtime
                .cooldown_reason_by_key
                .get(&key.id)
                .and_then(|_| runtime.cooldown_ttl_by_key.get(&key.id).copied()),
            cost_window_usage: runtime
                .cost_window_usage_by_key
                .get(&key.id)
                .copied()
                .unwrap_or(0),
            sticky_sessions: runtime
                .sticky_sessions_by_key
                .get(&key.id)
                .copied()
                .unwrap_or(0),
            lru_score: runtime.lru_score_by_key.get(&key.id).copied(),
            cost_limit: pool_config.and_then(|config| config.cost_limit_per_key_tokens),
        },
    )
}
