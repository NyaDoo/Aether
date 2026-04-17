use crate::handlers::admin::provider::shared::support::{
    AdminProviderPoolConfig, AdminProviderPoolSchedulingPreset, AdminProviderPoolUnschedulableRule,
};
use serde_json::{Map, Value};

fn json_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|raw| u64::try_from(raw).ok()))
}

fn parse_pool_scheduling_presets(
    raw_pool_advanced: &Map<String, Value>,
) -> Vec<AdminProviderPoolSchedulingPreset> {
    let Some(presets) = raw_pool_advanced
        .get("scheduling_presets")
        .and_then(Value::as_array)
    else {
        return raw_pool_advanced
            .get("lru_enabled")
            .and_then(Value::as_bool)
            .filter(|enabled| *enabled)
            .map(|_| {
                vec![AdminProviderPoolSchedulingPreset {
                    preset: "lru".to_string(),
                    enabled: true,
                    mode: None,
                }]
            })
            .unwrap_or_default();
    };

    let mut normalized = Vec::new();
    for item in presets {
        if let Some(preset) = item
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            normalized.push(AdminProviderPoolSchedulingPreset {
                preset: preset.to_ascii_lowercase(),
                enabled: true,
                mode: None,
            });
            continue;
        }

        let Some(object) = item.as_object() else {
            continue;
        };
        let Some(preset) = object
            .get("preset")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        normalized.push(AdminProviderPoolSchedulingPreset {
            preset: preset.to_ascii_lowercase(),
            enabled: object
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            mode: object
                .get("mode")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| value.to_ascii_lowercase()),
        });
    }

    if normalized.is_empty()
        && raw_pool_advanced
            .get("lru_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        normalized.push(AdminProviderPoolSchedulingPreset {
            preset: "lru".to_string(),
            enabled: true,
            mode: None,
        });
    }

    normalized
}

fn parse_pool_unschedulable_rules(
    raw_pool_advanced: &Map<String, Value>,
) -> Vec<AdminProviderPoolUnschedulableRule> {
    raw_pool_advanced
        .get("unschedulable_rules")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let object = item.as_object()?;
            let keyword = object
                .get("keyword")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            Some(AdminProviderPoolUnschedulableRule {
                keyword: keyword.to_string(),
                duration_minutes: object
                    .get("duration_minutes")
                    .and_then(json_u64)
                    .filter(|value| *value > 0)
                    .unwrap_or(5),
            })
        })
        .collect()
}

fn admin_provider_pool_lru_enabled(
    raw_pool_advanced: &Map<String, Value>,
    scheduling_presets: &[AdminProviderPoolSchedulingPreset],
) -> bool {
    if let Some(explicit) = raw_pool_advanced
        .get("lru_enabled")
        .and_then(Value::as_bool)
    {
        return explicit;
    }

    scheduling_presets
        .iter()
        .any(|item| item.enabled && item.preset.eq_ignore_ascii_case("lru"))
}

pub(crate) fn admin_provider_pool_config(
    provider: &aether_data_contracts::repository::provider_catalog::StoredProviderCatalogProvider,
) -> Option<AdminProviderPoolConfig> {
    admin_provider_pool_config_from_config_value(provider.config.as_ref())
}

pub(crate) fn admin_provider_pool_config_from_config_value(
    config: Option<&serde_json::Value>,
) -> Option<AdminProviderPoolConfig> {
    let raw_pool_advanced = config
        .and_then(Value::as_object)
        .and_then(|config| config.get("pool_advanced"))?;

    let Some(pool_advanced) = raw_pool_advanced.as_object() else {
        return Some(AdminProviderPoolConfig {
            scheduling_presets: Vec::new(),
            unschedulable_rules: Vec::new(),
            lru_enabled: false,
            skip_exhausted_accounts: false,
            sticky_session_ttl_seconds: 3600,
            latency_window_seconds: 3600,
            latency_sample_limit: 50,
            cost_window_seconds: 18_000,
            cost_limit_per_key_tokens: None,
            rate_limit_cooldown_seconds: 300,
            overload_cooldown_seconds: 30,
            health_policy_enabled: true,
            stream_timeout_threshold: 3,
            stream_timeout_window_seconds: 1800,
            stream_timeout_cooldown_seconds: 300,
        });
    };

    let scheduling_presets = parse_pool_scheduling_presets(pool_advanced);
    let unschedulable_rules = parse_pool_unschedulable_rules(pool_advanced);

    Some(AdminProviderPoolConfig {
        lru_enabled: admin_provider_pool_lru_enabled(pool_advanced, &scheduling_presets),
        scheduling_presets,
        unschedulable_rules,
        skip_exhausted_accounts: pool_advanced
            .get("skip_exhausted_accounts")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        sticky_session_ttl_seconds: pool_advanced
            .get("sticky_session_ttl_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(3600),
        latency_window_seconds: pool_advanced
            .get("latency_window_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(3600),
        latency_sample_limit: pool_advanced
            .get("latency_sample_limit")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(50),
        cost_window_seconds: pool_advanced
            .get("cost_window_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(18_000),
        cost_limit_per_key_tokens: pool_advanced
            .get("cost_limit_per_key_tokens")
            .and_then(json_u64),
        rate_limit_cooldown_seconds: pool_advanced
            .get("rate_limit_cooldown_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(300),
        overload_cooldown_seconds: pool_advanced
            .get("overload_cooldown_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(30),
        health_policy_enabled: pool_advanced
            .get("health_policy_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        stream_timeout_threshold: pool_advanced
            .get("stream_timeout_threshold")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(3),
        stream_timeout_window_seconds: pool_advanced
            .get("stream_timeout_window_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(1800),
        stream_timeout_cooldown_seconds: pool_advanced
            .get("stream_timeout_cooldown_seconds")
            .and_then(json_u64)
            .filter(|value| *value > 0)
            .unwrap_or(300),
    })
}

#[cfg(test)]
mod tests {
    use super::{admin_provider_pool_config, admin_provider_pool_config_from_config_value};
    use aether_data_contracts::repository::provider_catalog::StoredProviderCatalogProvider;
    use serde_json::json;

    fn sample_provider(config: serde_json::Value) -> StoredProviderCatalogProvider {
        StoredProviderCatalogProvider::new(
            "provider-1".to_string(),
            "provider-1".to_string(),
            Some("https://example.com".to_string()),
            "codex".to_string(),
        )
        .expect("provider should build")
        .with_transport_fields(
            true,
            false,
            false,
            None,
            None,
            None,
            None,
            None,
            Some(config),
        )
    }

    #[test]
    fn defaults_skip_exhausted_accounts_to_false() {
        let provider = sample_provider(json!({ "pool_advanced": {} }));
        let config = admin_provider_pool_config(&provider).expect("pool config should exist");

        assert!(!config.skip_exhausted_accounts);
    }

    #[test]
    fn parses_skip_exhausted_accounts_from_pool_advanced() {
        let provider = sample_provider(json!({
            "pool_advanced": {
                "skip_exhausted_accounts": true,
                "lru_enabled": true,
                "sticky_session_ttl_seconds": 600,
                "latency_window_seconds": 900,
                "latency_sample_limit": 75,
                "cost_window_seconds": 7200,
                "cost_limit_per_key_tokens": 12000,
                "rate_limit_cooldown_seconds": 420,
                "overload_cooldown_seconds": 45,
                "health_policy_enabled": false,
                "stream_timeout_threshold": 4,
                "stream_timeout_window_seconds": 900,
                "stream_timeout_cooldown_seconds": 180
            }
        }));
        let config = admin_provider_pool_config(&provider).expect("pool config should exist");

        assert!(config.skip_exhausted_accounts);
        assert!(config.lru_enabled);
        assert_eq!(config.sticky_session_ttl_seconds, 600);
        assert_eq!(config.latency_window_seconds, 900);
        assert_eq!(config.latency_sample_limit, 75);
        assert_eq!(config.cost_window_seconds, 7200);
        assert_eq!(config.cost_limit_per_key_tokens, Some(12_000));
        assert_eq!(config.rate_limit_cooldown_seconds, 420);
        assert_eq!(config.overload_cooldown_seconds, 45);
        assert!(!config.health_policy_enabled);
        assert_eq!(config.stream_timeout_threshold, 4);
        assert_eq!(config.stream_timeout_window_seconds, 900);
        assert_eq!(config.stream_timeout_cooldown_seconds, 180);
    }

    #[test]
    fn parses_pool_config_from_generic_config_value() {
        let config = admin_provider_pool_config_from_config_value(Some(&json!({
            "pool_advanced": {
                "scheduling_presets": [{"preset": "lru", "enabled": true}],
                "cost_limit_per_key_tokens": 4096
            }
        })))
        .expect("pool config should parse");

        assert!(config.lru_enabled);
        assert_eq!(config.scheduling_presets.len(), 1);
        assert_eq!(config.scheduling_presets[0].preset, "lru");
        assert_eq!(config.cost_limit_per_key_tokens, Some(4096));
    }

    #[test]
    fn parses_object_style_scheduling_presets_with_modes() {
        let config = admin_provider_pool_config_from_config_value(Some(&json!({
            "pool_advanced": {
                "scheduling_presets": [
                    {"preset": "cache_affinity", "enabled": false},
                    {"preset": "plus_first", "enabled": true, "mode": "plus_only"}
                ]
            }
        })))
        .expect("pool config should parse");

        assert!(!config.lru_enabled);
        assert_eq!(config.scheduling_presets.len(), 2);
        assert_eq!(config.scheduling_presets[0].preset, "cache_affinity");
        assert!(!config.scheduling_presets[0].enabled);
        assert_eq!(config.scheduling_presets[1].preset, "plus_first");
        assert_eq!(
            config.scheduling_presets[1].mode.as_deref(),
            Some("plus_only")
        );
    }

    #[test]
    fn parses_unschedulable_rules_from_pool_advanced() {
        let config = admin_provider_pool_config_from_config_value(Some(&json!({
            "pool_advanced": {
                "unschedulable_rules": [
                    {"keyword": "suspended", "duration_minutes": 15},
                    {"keyword": "review_required"}
                ]
            }
        })))
        .expect("pool config should parse");

        assert_eq!(config.unschedulable_rules.len(), 2);
        assert_eq!(config.unschedulable_rules[0].keyword, "suspended");
        assert_eq!(config.unschedulable_rules[0].duration_minutes, 15);
        assert_eq!(config.unschedulable_rules[1].keyword, "review_required");
        assert_eq!(config.unschedulable_rules[1].duration_minutes, 5);
    }
}
