use super::*;

const BIG_PROMPT: &str = "五段大 prompt 之一：这里模拟数据宝轮换的稳定大 prompt 正文。";
const AFFINITY_CACHE_KEY: &str = "databao:gpt-5:A:1";

fn hash_api_key(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn sha256_hex(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn auth_snapshot(api_key_id: &str, user_id: &str) -> StoredAuthApiKeySnapshot {
    StoredAuthApiKeySnapshot::new(
        user_id.to_string(),
        "alice".to_string(),
        Some("alice@example.com".to_string()),
        "user".to_string(),
        "local".to_string(),
        true,
        false,
        Some(serde_json::json!(["openai"])),
        Some(serde_json::json!(["openai:chat"])),
        Some(serde_json::json!(["gpt-5"])),
        api_key_id.to_string(),
        Some("default".to_string()),
        true,
        false,
        false,
        Some(60),
        Some(5),
        Some(4_102_444_800),
        Some(serde_json::json!(["openai"])),
        Some(serde_json::json!(["openai:chat"])),
        Some(serde_json::json!(["gpt-5"])),
    )
    .expect("auth snapshot should build")
}

fn auth_export_record(
    snapshot: &StoredAuthApiKeySnapshot,
    key_hash: String,
    feature_settings: Option<serde_json::Value>,
) -> aether_data::repository::auth::StoredAuthApiKeyExportRecord {
    aether_data::repository::auth::StoredAuthApiKeyExportRecord::new(
        snapshot.user_id.clone(),
        snapshot.api_key_id.clone(),
        key_hash,
        None,
        snapshot.api_key_name.clone(),
        snapshot
            .api_key_allowed_providers
            .as_ref()
            .map(|value| serde_json::json!(value)),
        snapshot
            .api_key_allowed_api_formats
            .as_ref()
            .map(|value| serde_json::json!(value)),
        snapshot
            .api_key_allowed_models
            .as_ref()
            .map(|value| serde_json::json!(value)),
        snapshot.api_key_rate_limit,
        snapshot.api_key_concurrent_limit,
        None,
        snapshot.api_key_is_active,
        snapshot
            .api_key_expires_at_unix_secs
            .map(|value| value as i64),
        false,
        0,
        0,
        0.0,
        snapshot.api_key_is_standalone,
    )
    .expect("auth api key export record should build")
    .with_feature_settings(feature_settings)
}

fn candidate_row(test_id: &str) -> StoredMinimalCandidateSelectionRow {
    StoredMinimalCandidateSelectionRow {
        provider_id: format!("provider-{test_id}"),
        provider_name: "openai".to_string(),
        provider_type: "custom".to_string(),
        provider_priority: 10,
        provider_is_active: true,
        endpoint_id: format!("endpoint-{test_id}"),
        endpoint_api_format: "openai:chat".to_string(),
        endpoint_api_family: Some("openai".to_string()),
        endpoint_kind: Some("chat".to_string()),
        endpoint_is_active: true,
        key_id: format!("key-{test_id}"),
        key_name: "prod".to_string(),
        key_auth_type: "api_key".to_string(),
        key_is_active: true,
        key_api_formats: Some(vec!["openai:chat".to_string()]),
        key_allowed_models: None,
        key_capabilities: None,
        key_internal_priority: 5,
        key_global_priority_by_format: Some(serde_json::json!({"openai:chat": 1})),
        model_id: format!("model-{test_id}"),
        global_model_id: format!("global-model-{test_id}"),
        global_model_name: "gpt-5".to_string(),
        global_model_mappings: None,
        global_model_supports_streaming: Some(true),
        model_provider_model_name: "gpt-5-upstream".to_string(),
        model_provider_model_mappings: Some(vec![StoredProviderModelMapping {
            name: "gpt-5-upstream".to_string(),
            priority: 1,
            api_formats: Some(vec!["openai:chat".to_string()]),
            endpoint_ids: Some(vec![format!("endpoint-{test_id}")]),
            operations: None,
        }]),
        model_supports_streaming: Some(true),
        model_is_active: true,
        model_is_available: true,
    }
}

fn provider(test_id: &str) -> StoredProviderCatalogProvider {
    StoredProviderCatalogProvider::new(
        format!("provider-{test_id}"),
        "openai".to_string(),
        Some("https://example.com".to_string()),
        "custom".to_string(),
    )
    .expect("provider should build")
    .with_transport_fields(true, false, true, None, Some(2), None, Some(20.0), None, None)
}

fn endpoint(test_id: &str, base_url: String) -> StoredProviderCatalogEndpoint {
    StoredProviderCatalogEndpoint::new(
        format!("endpoint-{test_id}"),
        format!("provider-{test_id}"),
        "openai:chat".to_string(),
        Some("openai".to_string()),
        Some("chat".to_string()),
        true,
    )
    .expect("endpoint should build")
    .with_transport_fields(base_url, None, None, Some(2), None, None, None, None)
    .expect("endpoint transport should build")
}

fn key(test_id: &str) -> StoredProviderCatalogKey {
    StoredProviderCatalogKey::new(
        format!("key-{test_id}"),
        format!("provider-{test_id}"),
        "prod".to_string(),
        "api_key".to_string(),
        None,
        true,
    )
    .expect("key should build")
    .with_transport_fields(
        Some(serde_json::json!(["openai:chat"])),
        encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, "sk-upstream-cache-affinity")
            .expect("api key should encrypt"),
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

fn cache_affinity_feature_settings() -> serde_json::Value {
    json!({
        "prompt_cache_affinity": {
            "enabled": true,
            "models": ["gpt-5"],
            "profiles": [
                {
                    "id": "A",
                    "version": 1,
                    "match_first_user_sha256": sha256_hex(BIG_PROMPT),
                    "cache_key": AFFINITY_CACHE_KEY
                }
            ]
        }
    })
}

async fn run_sync_cache_affinity_case(
    test_id: &str,
    feature_settings: Option<serde_json::Value>,
    request_body: serde_json::Value,
) -> serde_json::Value {
    let seen_provider_body = Arc::new(Mutex::new(None::<serde_json::Value>));
    let seen_provider_body_clone = Arc::clone(&seen_provider_body);
    let provider_app = Router::new().route(
        "/chat/completions",
        any(move |request: Request| {
            let seen_provider_body_inner = Arc::clone(&seen_provider_body_clone);
            async move {
                let (_parts, body) = request.into_parts();
                let raw_body = to_bytes(body, usize::MAX).await.expect("body should read");
                let payload: serde_json::Value =
                    serde_json::from_slice(&raw_body).expect("provider payload should parse");
                *seen_provider_body_inner.lock().expect("mutex should lock") = Some(payload);
                Json(json!({
                    "id": "chatcmpl-cache-affinity",
                    "object": "chat.completion",
                    "model": "gpt-5-upstream",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
                }))
            }
        }),
    );
    let (provider_url, provider_handle) = start_server(provider_app).await;
    let snapshot = auth_snapshot(&format!("api-key-{test_id}"), &format!("user-{test_id}"));
    let key_hash = hash_api_key(&format!("sk-client-{test_id}"));
    let auth_repository = Arc::new(
        InMemoryAuthApiKeySnapshotRepository::seed(vec![(
            Some(key_hash.clone()),
            snapshot.clone(),
        )])
        .with_export_records(vec![auth_export_record(
            &snapshot,
            key_hash,
            feature_settings,
        )]),
    );
    let candidate_selection_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            candidate_row(test_id),
        ]));
    let request_candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
    let provider_catalog_repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![provider(test_id)],
        vec![endpoint(test_id, provider_url)],
        vec![key(test_id)],
    ));
    let data_state = crate::data::GatewayDataState::with_auth_candidate_selection_provider_catalog_and_request_candidate_repository_for_tests(
        auth_repository,
        candidate_selection_repository,
        provider_catalog_repository,
        Arc::clone(&request_candidate_repository),
        DEVELOPMENT_ENCRYPTION_KEY,
    );
    let gateway_state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(data_state);
    let gateway = build_router_with_state(gateway_state);
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header(http::header::CONTENT_TYPE, "application/json")
        .header(
            http::header::AUTHORIZATION,
            format!("Bearer sk-client-{test_id}"),
        )
        .header(TRACE_ID_HEADER, format!("trace-{test_id}"))
        .body(request_body.to_string())
        .send()
        .await
        .expect("request should succeed");

    let status = response.status();
    let response_text = response.text().await.expect("body should read");
    assert_eq!(status, StatusCode::OK, "{response_text}");

    let seen = seen_provider_body
        .lock()
        .expect("mutex should lock")
        .clone()
        .expect("provider request should be captured");

    gateway_handle.abort();
    provider_handle.abort();

    seen
}

fn databao_style_request() -> serde_json::Value {
    json!({
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": BIG_PROMPT}
        ]
    })
}

large_stack_async_test!(
    ai_execute_sync_cache_affinity_injects_profile_key_for_matched_prompt,
    ai_execute_sync_cache_affinity_injects_profile_key_for_matched_prompt_impl
);

async fn ai_execute_sync_cache_affinity_injects_profile_key_for_matched_prompt_impl() {
    let seen = run_sync_cache_affinity_case(
        "cache-affinity-inject",
        Some(cache_affinity_feature_settings()),
        databao_style_request(),
    )
    .await;

    assert_eq!(seen["prompt_cache_key"], AFFINITY_CACHE_KEY);
    assert_eq!(seen["messages"][1]["content"], BIG_PROMPT);
}

large_stack_async_test!(
    ai_execute_sync_cache_affinity_keeps_client_prompt_cache_key,
    ai_execute_sync_cache_affinity_keeps_client_prompt_cache_key_impl
);

async fn ai_execute_sync_cache_affinity_keeps_client_prompt_cache_key_impl() {
    let mut request = databao_style_request();
    request["prompt_cache_key"] = json!("client-owned-key");
    let seen = run_sync_cache_affinity_case(
        "cache-affinity-client-key",
        Some(cache_affinity_feature_settings()),
        request,
    )
    .await;

    assert_eq!(seen["prompt_cache_key"], "client-owned-key");
}

large_stack_async_test!(
    ai_execute_sync_cache_affinity_unconfigured_is_passthrough,
    ai_execute_sync_cache_affinity_unconfigured_is_passthrough_impl
);

async fn ai_execute_sync_cache_affinity_unconfigured_is_passthrough_impl() {
    let seen = run_sync_cache_affinity_case(
        "cache-affinity-unconfigured",
        None,
        databao_style_request(),
    )
    .await;

    assert!(
        seen.get("prompt_cache_key").is_none(),
        "unconfigured request must stay untouched: {seen}"
    );
}

large_stack_async_test!(
    ai_execute_sync_cache_affinity_auto_mode_derives_key_without_profiles,
    ai_execute_sync_cache_affinity_auto_mode_derives_key_without_profiles_impl
);

async fn ai_execute_sync_cache_affinity_auto_mode_derives_key_without_profiles_impl() {
    let big_prompt = "生产轮换的新大提示词".repeat(500);
    let mut request = databao_style_request();
    request["messages"][1]["content"] = json!(big_prompt);
    let seen = run_sync_cache_affinity_case(
        "cache-affinity-auto",
        Some(json!({
            "prompt_cache_affinity": {
                "enabled": true,
                "models": ["gpt-5"],
                "auto": {"min_chars": 1000}
            }
        })),
        request,
    )
    .await;

    let key = seen["prompt_cache_key"]
        .as_str()
        .expect("auto mode should inject a prompt_cache_key");
    assert!(key.starts_with("aff:"), "unexpected key {key}");
}

large_stack_async_test!(
    ai_execute_sync_cache_affinity_unmatched_prompt_is_passthrough,
    ai_execute_sync_cache_affinity_unmatched_prompt_is_passthrough_impl
);

async fn ai_execute_sync_cache_affinity_unmatched_prompt_is_passthrough_impl() {
    let mut request = databao_style_request();
    request["messages"][1]["content"] = json!("完全不同的动态提问");
    let seen = run_sync_cache_affinity_case(
        "cache-affinity-unmatched",
        Some(cache_affinity_feature_settings()),
        request,
    )
    .await;

    assert!(
        seen.get("prompt_cache_key").is_none(),
        "unmatched prompt must stay untouched: {seen}"
    );
}
