use std::collections::BTreeSet;

use aether_contracts::{ExecutionPlan, ExecutionResult};
use regex::Regex;
use serde_json::{json, Value};
use tracing::debug;

use crate::provider_transport::GatewayProviderTransportSnapshot;
use crate::AppState;

fn local_candidate_index(report_context: Option<&serde_json::Value>) -> Option<u64> {
    report_context
        .and_then(serde_json::Value::as_object)
        .and_then(|context| context.get("candidate_index"))
        .and_then(serde_json::Value::as_u64)
}

fn should_failover_local_upstream_status(status_code: u16) -> bool {
    status_code >= 400
}

fn sync_plan_kind_disables_local_candidate_failover(plan_kind: &str) -> bool {
    matches!(
        plan_kind,
        "openai_video_delete_sync" | "openai_video_cancel_sync" | "gemini_video_cancel_sync"
    )
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct LocalFailoverPolicy {
    max_retries: Option<u64>,
    stop_status_codes: BTreeSet<u16>,
    continue_status_codes: BTreeSet<u16>,
    success_failover_patterns: Vec<LocalFailoverRegexRule>,
    error_stop_patterns: Vec<LocalFailoverRegexRule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalFailoverRegexRule {
    pattern: String,
    status_codes: BTreeSet<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalFailoverDecision {
    UseDefault,
    RetryNextCandidate,
    StopLocalFailover,
}

impl LocalFailoverDecision {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::UseDefault => "use_default",
            Self::RetryNextCandidate => "retry_next_candidate",
            Self::StopLocalFailover => "stop_local_failover",
        }
    }
}

pub(crate) async fn should_retry_next_local_candidate_sync(
    state: &AppState,
    plan: &ExecutionPlan,
    plan_kind: &str,
    report_context: Option<&serde_json::Value>,
    result: &ExecutionResult,
    response_text: Option<&str>,
) -> bool {
    if sync_plan_kind_disables_local_candidate_failover(plan_kind) {
        return false;
    }
    matches!(
        resolve_local_failover_decision(
            state,
            plan,
            report_context,
            result.status_code,
            response_text,
        )
        .await,
        LocalFailoverDecision::RetryNextCandidate
    )
}

pub(crate) async fn should_stop_local_candidate_failover_sync(
    state: &AppState,
    plan: &ExecutionPlan,
    plan_kind: &str,
    report_context: Option<&serde_json::Value>,
    result: &ExecutionResult,
    response_text: Option<&str>,
) -> bool {
    if sync_plan_kind_disables_local_candidate_failover(plan_kind) {
        return false;
    }
    matches!(
        resolve_local_failover_decision(
            state,
            plan,
            report_context,
            result.status_code,
            response_text,
        )
        .await,
        LocalFailoverDecision::StopLocalFailover
    )
}

pub(crate) fn should_fallback_to_control_sync(
    plan_kind: &str,
    result: &ExecutionResult,
    body_json: Option<&serde_json::Value>,
    has_body_bytes: bool,
    explicit_finalize: bool,
    mapped_error_finalize: bool,
) -> bool {
    if explicit_finalize
        && matches!(
            plan_kind,
            "openai_video_delete_sync" | "openai_video_cancel_sync" | "gemini_video_cancel_sync"
        )
    {
        return false;
    }

    if !matches!(
        plan_kind,
        "openai_video_create_sync"
            | "openai_video_remix_sync"
            | "gemini_video_create_sync"
            | "openai_chat_sync"
            | "openai_cli_sync"
            | "openai_compact_sync"
            | "claude_chat_sync"
            | "gemini_chat_sync"
            | "claude_cli_sync"
            | "gemini_cli_sync"
    ) {
        return false;
    }

    if explicit_finalize {
        return result.status_code < 400 && body_json.is_none() && !has_body_bytes;
    }

    if mapped_error_finalize {
        return false;
    }

    if result.status_code >= 400 {
        return true;
    }

    let Some(body_json) = body_json else {
        return true;
    };

    body_json.get("error").is_some()
}

pub(crate) fn should_finalize_sync_response(report_kind: Option<&str>) -> bool {
    report_kind.is_some_and(|kind| kind.ends_with("_finalize"))
}

pub(crate) fn resolve_core_sync_error_finalize_report_kind(
    plan_kind: &str,
    result: &ExecutionResult,
    body_json: Option<&serde_json::Value>,
) -> Option<String> {
    let has_embedded_error = body_json.is_some_and(|value| value.get("error").is_some());
    if result.status_code < 400 && !has_embedded_error {
        return None;
    }

    let report_kind = match plan_kind {
        "openai_chat_sync" => "openai_chat_sync_finalize",
        "openai_cli_sync" => "openai_cli_sync_finalize",
        "openai_compact_sync" => "openai_compact_sync_finalize",
        "claude_chat_sync" => "claude_chat_sync_finalize",
        "gemini_chat_sync" => "gemini_chat_sync_finalize",
        "claude_cli_sync" => "claude_cli_sync_finalize",
        "gemini_cli_sync" => "gemini_cli_sync_finalize",
        _ => return None,
    };

    Some(report_kind.to_string())
}

pub(crate) async fn should_retry_next_local_candidate_stream(
    state: &AppState,
    plan: &ExecutionPlan,
    _plan_kind: &str,
    report_context: Option<&serde_json::Value>,
    status_code: u16,
    response_text: Option<&str>,
) -> bool {
    matches!(
        resolve_local_candidate_failover_decision_stream(
            state,
            plan,
            report_context,
            status_code,
            response_text,
        )
        .await,
        LocalFailoverDecision::RetryNextCandidate
    )
}

pub(crate) async fn should_stop_local_candidate_failover_stream(
    state: &AppState,
    plan: &ExecutionPlan,
    _plan_kind: &str,
    report_context: Option<&serde_json::Value>,
    status_code: u16,
    response_text: Option<&str>,
) -> bool {
    matches!(
        resolve_local_candidate_failover_decision_stream(
            state,
            plan,
            report_context,
            status_code,
            response_text,
        )
        .await,
        LocalFailoverDecision::StopLocalFailover
    )
}

pub(crate) async fn resolve_local_candidate_failover_decision_stream(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
    status_code: u16,
    response_text: Option<&str>,
) -> LocalFailoverDecision {
    resolve_local_failover_decision(state, plan, report_context, status_code, response_text).await
}

pub(crate) fn local_failover_response_text(
    body_json: Option<&serde_json::Value>,
    body_bytes: &[u8],
    fallback_text: Option<&str>,
) -> Option<String> {
    if let Some(body_json) = body_json {
        return serde_json::to_string(body_json).ok();
    }
    if !body_bytes.is_empty() {
        return Some(String::from_utf8_lossy(body_bytes).into_owned());
    }
    fallback_text
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

async fn resolve_local_failover_decision(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
    status_code: u16,
    response_text: Option<&str>,
) -> LocalFailoverDecision {
    let Some(candidate_index) = local_candidate_index(report_context) else {
        return LocalFailoverDecision::UseDefault;
    };
    let policy = resolve_local_failover_policy(state, plan, report_context).await;
    let response_text = response_text
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if policy.stop_status_codes.contains(&status_code) {
        return LocalFailoverDecision::StopLocalFailover;
    }

    if status_code >= 400
        && response_text.is_some_and(|text| {
            policy
                .error_stop_patterns
                .iter()
                .any(|rule| local_failover_regex_rule_matches(rule, text, status_code))
        })
    {
        return LocalFailoverDecision::StopLocalFailover;
    }

    if policy
        .max_retries
        .is_some_and(|max_retries| candidate_index >= max_retries)
    {
        return LocalFailoverDecision::UseDefault;
    }

    if status_code == 200
        && response_text.is_some_and(|text| {
            policy
                .success_failover_patterns
                .iter()
                .any(|rule| local_failover_regex_rule_matches(rule, text, status_code))
        })
    {
        return LocalFailoverDecision::RetryNextCandidate;
    }

    if policy.continue_status_codes.contains(&status_code) {
        return LocalFailoverDecision::RetryNextCandidate;
    }

    if should_failover_local_upstream_status(status_code) {
        return LocalFailoverDecision::RetryNextCandidate;
    }

    LocalFailoverDecision::UseDefault
}

async fn resolve_local_failover_policy(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
) -> LocalFailoverPolicy {
    if let Some(policy) = local_failover_policy_from_report_context(report_context) {
        debug!(
            event_name = "local_failover_policy_loaded",
            log_type = "debug",
            request_id = %plan.request_id,
            provider_id = %plan.provider_id,
            endpoint_id = %plan.endpoint_id,
            key_id = %plan.key_id,
            source = "report_context",
            max_retries = ?policy.max_retries,
            stop_status_code_count = policy.stop_status_codes.len(),
            continue_status_code_count = policy.continue_status_codes.len(),
            success_failover_pattern_count = policy.success_failover_patterns.len(),
            error_stop_pattern_count = policy.error_stop_patterns.len(),
            "gateway loaded local failover policy from report context"
        );
        return policy;
    }

    let transport = match state
        .read_provider_transport_snapshot(&plan.provider_id, &plan.endpoint_id, &plan.key_id)
        .await
    {
        Ok(Some(transport)) => transport,
        Ok(None) | Err(_) => return LocalFailoverPolicy::default(),
    };
    let policy = local_failover_policy_from_transport(&transport);
    debug!(
        event_name = "local_failover_policy_loaded",
        log_type = "debug",
        request_id = %plan.request_id,
        provider_id = %plan.provider_id,
        endpoint_id = %plan.endpoint_id,
        key_id = %plan.key_id,
        source = "transport_snapshot",
        max_retries = ?policy.max_retries,
        stop_status_code_count = policy.stop_status_codes.len(),
        continue_status_code_count = policy.continue_status_codes.len(),
        success_failover_pattern_count = policy.success_failover_patterns.len(),
        error_stop_pattern_count = policy.error_stop_patterns.len(),
        "gateway loaded local failover policy from transport snapshot"
    );
    policy
}

fn local_failover_policy_from_transport(
    transport: &GatewayProviderTransportSnapshot,
) -> LocalFailoverPolicy {
    let rules = transport
        .provider
        .config
        .as_ref()
        .and_then(|config| config.get("failover_rules"))
        .and_then(serde_json::Value::as_object);
    let max_retries = rules
        .and_then(|value| value.get("max_retries"))
        .and_then(parse_u64_value)
        .or_else(|| {
            transport
                .endpoint
                .max_retries
                .and_then(|value| u64::try_from(value).ok())
        })
        .or_else(|| {
            transport
                .provider
                .max_retries
                .and_then(|value| u64::try_from(value).ok())
        });

    LocalFailoverPolicy {
        max_retries,
        stop_status_codes: rules
            .map(|value| {
                parse_status_code_set(
                    value,
                    &[
                        "stop_on_status_codes",
                        "early_stop_status_codes",
                        "non_retryable_status_codes",
                        "stop_status_codes",
                    ],
                )
            })
            .unwrap_or_default(),
        continue_status_codes: rules
            .map(|value| {
                parse_status_code_set(
                    value,
                    &[
                        "continue_on_status_codes",
                        "retryable_status_codes",
                        "retry_on_status_codes",
                        "continue_status_codes",
                    ],
                )
            })
            .unwrap_or_default(),
        success_failover_patterns: rules
            .map(|value| parse_regex_rules(value, "success_failover_patterns"))
            .unwrap_or_default(),
        error_stop_patterns: rules
            .map(|value| parse_regex_rules(value, "error_stop_patterns"))
            .unwrap_or_default(),
    }
}

fn local_failover_policy_from_report_context(
    report_context: Option<&Value>,
) -> Option<LocalFailoverPolicy> {
    let object = report_context
        .and_then(Value::as_object)?
        .get("local_failover_policy")?
        .as_object()?;

    Some(LocalFailoverPolicy {
        max_retries: object.get("max_retries").and_then(parse_u64_value),
        stop_status_codes: object
            .get("stop_status_codes")
            .map(parse_status_code_list)
            .unwrap_or_default(),
        continue_status_codes: object
            .get("continue_status_codes")
            .map(parse_status_code_list)
            .unwrap_or_default(),
        success_failover_patterns: parse_regex_rules(object, "success_failover_patterns"),
        error_stop_patterns: parse_regex_rules(object, "error_stop_patterns"),
    })
}

fn parse_status_code_list(value: &Value) -> BTreeSet<u16> {
    value
        .as_array()
        .into_iter()
        .flat_map(|values| values.iter())
        .filter_map(|value| parse_u64_value(value).and_then(|value| u16::try_from(value).ok()))
        .collect()
}

fn local_failover_policy_to_value(policy: &LocalFailoverPolicy) -> Value {
    json!({
        "max_retries": policy.max_retries,
        "stop_status_codes": policy.stop_status_codes.iter().copied().collect::<Vec<_>>(),
        "continue_status_codes": policy.continue_status_codes.iter().copied().collect::<Vec<_>>(),
        "success_failover_patterns": policy.success_failover_patterns.iter().map(local_failover_regex_rule_to_value).collect::<Vec<_>>(),
        "error_stop_patterns": policy.error_stop_patterns.iter().map(local_failover_regex_rule_to_value).collect::<Vec<_>>(),
    })
}

fn local_failover_regex_rule_to_value(rule: &LocalFailoverRegexRule) -> Value {
    json!({
        "pattern": rule.pattern,
        "status_codes": rule.status_codes.iter().copied().collect::<Vec<_>>(),
    })
}

pub(crate) fn append_local_failover_policy_to_value(
    value: Value,
    transport: &GatewayProviderTransportSnapshot,
) -> Value {
    let Value::Object(mut object) = value else {
        return value;
    };
    object.insert(
        "local_failover_policy".to_string(),
        local_failover_policy_to_value(&local_failover_policy_from_transport(transport)),
    );
    Value::Object(object)
}

fn parse_regex_rules(
    rules: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Vec<LocalFailoverRegexRule> {
    rules
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flat_map(|items| items.iter())
        .filter_map(parse_regex_rule)
        .collect()
}

fn parse_regex_rule(value: &serde_json::Value) -> Option<LocalFailoverRegexRule> {
    let object = value.as_object()?;
    let pattern = object
        .get("pattern")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    Some(LocalFailoverRegexRule {
        pattern: pattern.to_string(),
        status_codes: object
            .get("status_codes")
            .and_then(serde_json::Value::as_array)
            .into_iter()
            .flat_map(|values| values.iter())
            .filter_map(|value| parse_u64_value(value).and_then(|value| u16::try_from(value).ok()))
            .collect(),
    })
}

fn local_failover_regex_rule_matches(
    rule: &LocalFailoverRegexRule,
    response_text: &str,
    status_code: u16,
) -> bool {
    if !rule.status_codes.is_empty() && !rule.status_codes.contains(&status_code) {
        return false;
    }

    Regex::new(&rule.pattern)
        .ok()
        .is_some_and(|regex| regex.is_match(response_text))
}

fn parse_status_code_set(
    rules: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
) -> BTreeSet<u16> {
    keys.iter()
        .filter_map(|key| rules.get(*key))
        .filter_map(serde_json::Value::as_array)
        .flat_map(|values| values.iter())
        .filter_map(|value| parse_u64_value(value).and_then(|value| u16::try_from(value).ok()))
        .collect()
}

fn parse_u64_value(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
}

pub(crate) fn should_fallback_to_control_stream(
    plan_kind: &str,
    status_code: u16,
    mapped_error_finalize: bool,
) -> bool {
    if mapped_error_finalize {
        return false;
    }

    matches!(
        plan_kind,
        "openai_chat_stream"
            | "claude_chat_stream"
            | "gemini_chat_stream"
            | "openai_cli_stream"
            | "openai_compact_stream"
            | "claude_cli_stream"
            | "gemini_cli_stream"
    ) && status_code >= 400
}

pub(crate) fn resolve_core_stream_error_finalize_report_kind(
    plan_kind: &str,
    status_code: u16,
) -> Option<String> {
    if status_code < 400 {
        return None;
    }

    let report_kind = match plan_kind {
        "openai_chat_stream" => "openai_chat_sync_finalize",
        "claude_chat_stream" => "claude_chat_sync_finalize",
        "gemini_chat_stream" => "gemini_chat_sync_finalize",
        "openai_cli_stream" => "openai_cli_sync_finalize",
        "openai_compact_stream" => "openai_compact_sync_finalize",
        "claude_cli_stream" => "claude_cli_sync_finalize",
        "gemini_cli_stream" => "gemini_cli_sync_finalize",
        _ => return None,
    };

    Some(report_kind.to_string())
}

pub(crate) fn resolve_core_stream_direct_finalize_report_kind(plan_kind: &str) -> Option<String> {
    let report_kind = match plan_kind {
        "openai_chat_stream" => "openai_chat_sync_finalize",
        "claude_chat_stream" => "claude_chat_sync_finalize",
        "gemini_chat_stream" => "gemini_chat_sync_finalize",
        "openai_cli_stream" => "openai_cli_sync_finalize",
        "openai_compact_stream" => "openai_compact_sync_finalize",
        "claude_cli_stream" => "claude_cli_sync_finalize",
        "gemini_cli_stream" => "gemini_cli_sync_finalize",
        _ => return None,
    };

    Some(report_kind.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use aether_contracts::ExecutionResult;
    use aether_data::repository::provider_catalog::InMemoryProviderCatalogReadRepository;
    use aether_data_contracts::repository::provider_catalog::{
        StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
    };

    use super::{
        resolve_core_stream_error_finalize_report_kind,
        resolve_core_sync_error_finalize_report_kind, resolve_local_failover_policy,
        should_fallback_to_control_stream, should_fallback_to_control_sync,
        should_retry_next_local_candidate_stream, should_retry_next_local_candidate_sync,
        should_stop_local_candidate_failover_stream, should_stop_local_candidate_failover_sync,
        LocalFailoverPolicy, LocalFailoverRegexRule,
    };
    use crate::data::GatewayDataState;
    use crate::AppState;

    fn sample_plan() -> aether_contracts::ExecutionPlan {
        aether_contracts::ExecutionPlan {
            request_id: "req-1".to_string(),
            candidate_id: Some("cand-1".to_string()),
            provider_name: Some("provider-1".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            method: "POST".to_string(),
            url: "https://example.com/v1/chat/completions".to_string(),
            headers: Default::default(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: aether_contracts::RequestBody::from_json(serde_json::json!({"model":"gpt-5"})),
            stream: false,
            client_api_format: "openai:chat".to_string(),
            provider_api_format: "openai:chat".to_string(),
            model_name: Some("gpt-5".to_string()),
            proxy: None,
            tls_profile: None,
            timeouts: None,
        }
    }

    fn sample_provider(config: Option<serde_json::Value>) -> StoredProviderCatalogProvider {
        StoredProviderCatalogProvider::new(
            "provider-1".to_string(),
            "provider-1".to_string(),
            Some("https://provider.example".to_string()),
            "custom".to_string(),
        )
        .expect("provider should build")
        .with_transport_fields(true, false, false, None, Some(3), None, None, None, config)
    }

    fn sample_endpoint() -> StoredProviderCatalogEndpoint {
        StoredProviderCatalogEndpoint::new(
            "endpoint-1".to_string(),
            "provider-1".to_string(),
            "openai:chat".to_string(),
            Some("openai".to_string()),
            Some("chat".to_string()),
            true,
        )
        .expect("endpoint should build")
        .with_transport_fields(
            "https://api.provider.example".to_string(),
            None,
            None,
            Some(2),
            None,
            None,
            None,
            None,
        )
        .expect("endpoint transport should build")
    }

    fn sample_key() -> StoredProviderCatalogKey {
        StoredProviderCatalogKey::new(
            "key-1".to_string(),
            "provider-1".to_string(),
            "key-1".to_string(),
            "api_key".to_string(),
            None,
            true,
        )
        .expect("key should build")
        .with_transport_fields(
            Some(serde_json::json!(["openai:chat"])),
            "plain-upstream-key".to_string(),
            None,
            None,
            Some(serde_json::json!({"openai:chat": 1})),
            None,
            None,
            None,
            None,
        )
        .expect("key transport should build")
    }

    fn build_state_with_provider_config(config: Option<serde_json::Value>) -> AppState {
        let provider_catalog = InMemoryProviderCatalogReadRepository::seed(
            vec![sample_provider(config)],
            vec![sample_endpoint()],
            vec![sample_key()],
        );
        let data_state = GatewayDataState::with_provider_transport_reader_for_tests(
            std::sync::Arc::new(provider_catalog),
            "development-key",
        );
        AppState::new()
            .expect("state should build")
            .with_data_state_for_tests(data_state)
    }

    #[test]
    fn sync_failover_marks_chat_errors() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 502,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };

        assert!(should_fallback_to_control_sync(
            "openai_chat_sync",
            &result,
            None,
            false,
            false,
            false,
        ));
        assert_eq!(
            resolve_core_sync_error_finalize_report_kind("openai_chat_sync", &result, None),
            Some("openai_chat_sync_finalize".to_string())
        );
    }

    #[test]
    fn stream_failover_marks_chat_errors() {
        assert!(should_fallback_to_control_stream(
            "openai_chat_stream",
            502,
            false,
        ));
        assert_eq!(
            resolve_core_stream_error_finalize_report_kind("openai_chat_stream", 502),
            Some("openai_chat_sync_finalize".to_string())
        );
    }

    #[tokio::test]
    async fn sync_retry_next_candidate_requires_local_candidate_context() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 502,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                None,
            )
            .await
        );
        assert!(
            should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "claude_cli_sync",
                Some(&local_report_context),
                &result,
                None,
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                None,
                &result,
                None,
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "claude_chat_sync",
                None,
                &result,
                None,
            )
            .await
        );
    }

    #[tokio::test]
    async fn sync_retry_next_candidate_treats_rate_limit_as_retryable() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 429,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                None,
            )
            .await
        );
    }

    #[tokio::test]
    async fn sync_retry_next_candidate_treats_client_error_as_failover_by_default() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 401,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                Some("{\"error\":{\"message\":\"invalid auth token\"}}"),
            )
            .await
        );
    }

    #[tokio::test]
    async fn sync_retry_next_candidate_skips_video_follow_up_plan_kinds() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 404,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        for plan_kind in [
            "openai_video_delete_sync",
            "openai_video_cancel_sync",
            "gemini_video_cancel_sync",
        ] {
            assert!(
                !should_retry_next_local_candidate_sync(
                    &state,
                    &plan,
                    plan_kind,
                    Some(&local_report_context),
                    &result,
                    None,
                )
                .await,
                "{plan_kind} should not retry local failover candidates"
            );
            assert!(
                !should_stop_local_candidate_failover_sync(
                    &state,
                    &plan,
                    plan_kind,
                    Some(&local_report_context),
                    &result,
                    None,
                )
                .await,
                "{plan_kind} should not use local failover stop decisions"
            );
        }
    }

    #[tokio::test]
    async fn stream_retry_next_candidate_requires_local_candidate_context() {
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&local_report_context),
                502,
                None,
            )
            .await
        );
        assert!(
            should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "gemini_cli_stream",
                Some(&local_report_context),
                502,
                None,
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                None,
                502,
                None,
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "claude_chat_stream",
                None,
                502,
                None,
            )
            .await
        );
    }

    #[tokio::test]
    async fn stream_retry_next_candidate_treats_rate_limit_as_retryable() {
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&local_report_context),
                429,
                None,
            )
            .await
        );
    }

    #[tokio::test]
    async fn stream_retry_next_candidate_treats_client_error_as_failover_by_default() {
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(None);
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&local_report_context),
                403,
                Some("{\"error\":{\"message\":\"invalid auth token\"}}"),
            )
            .await
        );
    }

    #[test]
    fn resolve_local_failover_policy_reads_provider_rules() {
        let state = build_state_with_provider_config(Some(serde_json::json!({
            "failover_rules": {
                "max_retries": 1,
                "stop_on_status_codes": [503],
                "continue_on_status_codes": [409, 429]
            }
        })));
        let plan = sample_plan();
        let runtime = tokio::runtime::Runtime::new().expect("runtime should build");

        let policy = runtime.block_on(resolve_local_failover_policy(&state, &plan, None));
        assert_eq!(
            policy,
            LocalFailoverPolicy {
                max_retries: Some(1),
                stop_status_codes: [503].into_iter().collect(),
                continue_status_codes: [409, 429].into_iter().collect(),
                success_failover_patterns: Vec::new(),
                error_stop_patterns: Vec::new(),
            }
        );
    }

    #[tokio::test]
    async fn local_failover_policy_can_stop_retryable_statuses_and_continue_non_retryable_statuses()
    {
        let state = build_state_with_provider_config(Some(serde_json::json!({
            "failover_rules": {
                "max_retries": 2,
                "stop_on_status_codes": [503],
                "continue_on_status_codes": [409]
            }
        })));
        let plan = sample_plan();
        let first_candidate = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let third_candidate = serde_json::json!({
            "candidate_index": 2,
            "retry_index": 0,
        });

        assert!(
            !should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&first_candidate),
                503,
                None,
            )
            .await
        );
        assert!(
            should_stop_local_candidate_failover_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&first_candidate),
                503,
                None,
            )
            .await
        );
        assert!(
            should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&first_candidate),
                409,
                None,
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_stream(
                &state,
                &plan,
                "openai_chat_stream",
                Some(&third_candidate),
                429,
                None,
            )
            .await
        );
    }

    #[test]
    fn resolve_local_failover_policy_reads_regex_rules() {
        let state = build_state_with_provider_config(Some(serde_json::json!({
            "failover_rules": {
                "success_failover_patterns": [
                    {"pattern": "relay:.*格式错误"}
                ],
                "error_stop_patterns": [
                    {"pattern": "content_policy_violation", "status_codes": [400, 403]}
                ]
            }
        })));
        let plan = sample_plan();
        let runtime = tokio::runtime::Runtime::new().expect("runtime should build");

        let policy = runtime.block_on(resolve_local_failover_policy(&state, &plan, None));
        assert_eq!(
            policy.success_failover_patterns,
            vec![LocalFailoverRegexRule {
                pattern: "relay:.*格式错误".to_string(),
                status_codes: BTreeSet::new(),
            }]
        );
        assert_eq!(
            policy.error_stop_patterns,
            vec![LocalFailoverRegexRule {
                pattern: "content_policy_violation".to_string(),
                status_codes: [400, 403].into_iter().collect(),
            }]
        );
    }

    #[tokio::test]
    async fn success_failover_pattern_can_retry_sync_candidate() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 200,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(Some(serde_json::json!({
            "failover_rules": {
                "success_failover_patterns": [
                    {"pattern": "relay:.*格式错误"}
                ]
            }
        })));
        let plan = sample_plan();

        assert!(
            should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                Some("{\"error\":\"relay: 返回格式错误\"}"),
            )
            .await
        );
    }

    #[tokio::test]
    async fn error_stop_pattern_can_stop_sync_failover() {
        let result = ExecutionResult {
            request_id: "req-1".to_string(),
            candidate_id: None,
            status_code: 400,
            headers: Default::default(),
            body: None,
            telemetry: None,
            error: None,
        };
        let local_report_context = serde_json::json!({
            "candidate_index": 0,
            "retry_index": 0,
        });
        let state = build_state_with_provider_config(Some(serde_json::json!({
            "failover_rules": {
                "error_stop_patterns": [
                    {"pattern": "content_policy_violation", "status_codes": [400]}
                ]
            }
        })));
        let plan = sample_plan();

        assert!(
            should_stop_local_candidate_failover_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                Some("{\"error\":\"content_policy_violation\"}"),
            )
            .await
        );
        assert!(
            !should_retry_next_local_candidate_sync(
                &state,
                &plan,
                "openai_chat_sync",
                Some(&local_report_context),
                &result,
                Some("{\"error\":\"content_policy_violation\"}"),
            )
            .await
        );
    }
}
