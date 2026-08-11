use super::{
    hash_api_key, sample_models_candidate_row, unrestricted_models_snapshot,
    InMemoryAuthApiKeySnapshotRepository, InMemoryMinimalCandidateSelectionReadRepository,
    InMemoryVideoTaskRepository, StoredAuthApiKeySnapshot, UpsertVideoTask, VideoTaskLookupKey,
    VideoTaskReadRepository, VideoTaskStatus, VideoTaskWriteRepository, DEVELOPMENT_ENCRYPTION_KEY,
};
use crate::image_capabilities::openai_image_gateway_max_generation_count;
use crate::tests::video::{simple_video_provider_catalog_repository, video_auth_repository};
use crate::tests::{
    any, build_router_with_state, build_state_with_execution_runtime_override, json, start_server,
    to_bytes, AppState, Arc, Body, Json, Mutex, Request, Router, StatusCode, EXECUTION_PATH_HEADER,
    EXECUTION_PATH_LOCAL_AI_PUBLIC, EXECUTION_PATH_LOCAL_EXECUTION_RUNTIME_MISS,
};
use aether_data::repository::global_models::InMemoryGlobalModelReadRepository;
use aether_data::DataLayerError;
use aether_data_contracts::repository::candidate_selection::{
    MinimalCandidateSelectionReadRepository, StoredMinimalCandidateSelectionRow,
    StoredPoolKeyCandidateRowsByKeyIdsQuery, StoredPoolKeyCandidateRowsQuery,
    StoredRequestedModelCandidateRowsQuery,
};
use aether_data_contracts::repository::global_models::{
    StoredAdminGlobalModel, UpdateAdminGlobalModelRecord,
};
use async_trait::async_trait;
use axum::response::IntoResponse;
use std::collections::HashMap;
use std::future::pending;
use std::sync::atomic::{AtomicBool, Ordering};

fn codex_models_snapshot(api_key_id: &str, user_id: &str) -> StoredAuthApiKeySnapshot {
    StoredAuthApiKeySnapshot::new(
        user_id.to_string(),
        "alice".to_string(),
        Some("alice@example.com".to_string()),
        "user".to_string(),
        "local".to_string(),
        true,
        false,
        Some(json!(["codex"])),
        Some(json!(["openai:responses"])),
        Some(json!(["frontier-sol", "broken-luna"])),
        api_key_id.to_string(),
        Some("codex-models".to_string()),
        true,
        false,
        false,
        Some(10),
        Some(5),
        Some(4_102_444_800),
        Some(json!(["codex"])),
        Some(json!(["openai:responses"])),
        Some(json!(["frontier-sol", "broken-luna"])),
    )
    .expect("Codex models auth snapshot should build")
}

fn sample_codex_models_candidate_row(
    provider_id: &str,
    global_model_name: &str,
    source_model_name: &str,
) -> StoredMinimalCandidateSelectionRow {
    let mut row = sample_models_candidate_row(
        provider_id,
        "codex",
        "openai:responses",
        global_model_name,
        10,
    );
    row.provider_type = "codex".to_string();
    row.key_auth_type = "oauth".to_string();
    row.model_provider_model_name = source_model_name.to_string();
    row.model_provider_model_mappings = Some(vec![
        aether_data_contracts::repository::candidate_selection::StoredProviderModelMapping {
            name: source_model_name.to_string(),
            priority: 1,
            api_formats: Some(vec!["openai:responses".to_string()]),
            endpoint_ids: None,
            operations: None,
        },
    ]);
    row
}

fn complete_codex_model_card(source_model_name: &str) -> serde_json::Value {
    json!({
        "id": source_model_name,
        "api_formats": ["openai:responses"],
        "slug": source_model_name,
        "display_name": "GPT-5.6-Sol",
        "description": "Frontier coding model",
        "default_reasoning_level": "low",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Low"},
            {"effort": "medium", "description": "Medium"},
            {"effort": "high", "description": "High"},
            {"effort": "xhigh", "description": "XHigh"},
            {"effort": "max", "description": "Max"},
            {"effort": "ultra", "description": "Ultra"}
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 1,
        "availability_nux": null,
        "upgrade": null,
        "base_instructions": "Use the current Codex instructions.",
        "model_messages": null,
        "support_verbosity": true,
        "default_verbosity": "low",
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_parallel_tool_calls": true,
        "experimental_supported_tools": [],
        "minimal_client_version": "0.144.0",
        "future_capability": {"enabled": true}
    })
}

fn gemini_operation_status_label(status: VideoTaskStatus) -> &'static str {
    match status {
        VideoTaskStatus::Pending => "Pending",
        VideoTaskStatus::Submitted => "Submitted",
        VideoTaskStatus::Queued => "Queued",
        VideoTaskStatus::Processing => "Processing",
        VideoTaskStatus::Completed => "Completed",
        VideoTaskStatus::Failed => "Failed",
        VideoTaskStatus::Cancelled => "Cancelled",
        VideoTaskStatus::Expired => "Expired",
        VideoTaskStatus::Deleted => "Deleted",
    }
}

fn sample_gemini_video_task(
    id: &str,
    short_id: &str,
    user_id: &str,
    api_key_id: &str,
    external_task_id: &str,
    status: VideoTaskStatus,
) -> UpsertVideoTask {
    let completed = matches!(status, VideoTaskStatus::Completed);
    UpsertVideoTask {
        id: id.to_string(),
        short_id: Some(short_id.to_string()),
        request_id: format!("request-{id}"),
        user_id: Some(user_id.to_string()),
        api_key_id: Some(api_key_id.to_string()),
        username: Some(format!("user-{user_id}")),
        api_key_name: Some("video-key".to_string()),
        external_task_id: Some(external_task_id.to_string()),
        provider_id: Some("provider-gemini-video-local-1".to_string()),
        endpoint_id: Some("endpoint-gemini-video-local-1".to_string()),
        key_id: Some("key-gemini-video-local-1".to_string()),
        client_api_format: Some("gemini:video".to_string()),
        provider_api_format: Some("gemini:video".to_string()),
        format_converted: false,
        model: Some("veo-3".to_string()),
        prompt: Some("gemini video prompt".to_string()),
        original_request_body: Some(json!({"prompt": "gemini video prompt"})),
        duration_seconds: Some(8),
        resolution: Some("720p".to_string()),
        aspect_ratio: Some("16:9".to_string()),
        size: Some("720p".to_string()),
        status,
        progress_percent: if completed { 100 } else { 50 },
        progress_message: None,
        retry_count: 0,
        poll_interval_seconds: 10,
        next_poll_at_unix_secs: (!completed).then_some(124),
        poll_count: 0,
        max_poll_count: 360,
        created_at_unix_ms: 123,
        submitted_at_unix_secs: Some(123),
        completed_at_unix_secs: completed.then_some(124),
        updated_at_unix_secs: 124,
        error_code: None,
        error_message: None,
        video_url: None,
        request_metadata: Some(json!({
            "rust_local_snapshot": {
                "Gemini": {
                    "local_short_id": short_id,
                    "upstream_operation_name": external_task_id,
                    "user_id": user_id,
                    "api_key_id": api_key_id,
                    "model": "veo-3",
                    "status": gemini_operation_status_label(status),
                    "progress_percent": if completed { 100 } else { 50 },
                    "error_code": null,
                    "error_message": null,
                    "metadata": {},
                    "persistence": {
                        "request_id": format!("request-{id}"),
                        "username": format!("user-{user_id}"),
                        "api_key_name": "video-key",
                        "client_api_format": "gemini:video",
                        "provider_api_format": "gemini:video",
                        "original_request_body": {
                            "prompt": "gemini video prompt"
                        },
                        "format_converted": false
                    },
                    "transport": {
                        "upstream_base_url": "https://generativelanguage.googleapis.com",
                        "provider_name": "gemini-video",
                        "provider_id": "provider-gemini-video-local-1",
                        "endpoint_id": "endpoint-gemini-video-local-1",
                        "key_id": "key-gemini-video-local-1",
                        "headers": {
                            "x-goog-api-key": "sk-upstream-gemini-video",
                            "content-type": "application/json"
                        },
                        "content_type": "application/json",
                        "model_name": "veo-3-upstream",
                        "proxy": null,
                        "transport_profile": null,
                        "timeouts": null
                    }
                }
            }
        })),
    }
}

fn sample_doubao_video_task(
    id: &str,
    user_id: &str,
    api_key_id: &str,
    status: VideoTaskStatus,
) -> UpsertVideoTask {
    let mut task = sample_gemini_video_task(
        id,
        id,
        user_id,
        api_key_id,
        &format!("upstream-{id}"),
        status,
    );
    task.short_id = None;
    task.client_api_format = Some("doubao:video".to_string());
    task.provider_api_format = Some("doubao:video".to_string());
    task.model = Some("seedance-1-5-pro".to_string());
    task.prompt = Some("doubao video prompt".to_string());
    task.original_request_body = Some(json!({
        "model": "seedance-1-5-pro",
        "content": [{"type": "text", "text": "doubao video prompt"}]
    }));
    task.video_url = Some(format!("https://tos.example.invalid/{id}.mp4?X-Sig=test"));
    if status == VideoTaskStatus::Cancelled {
        let now_unix_secs = aether_video_tasks_core::current_unix_timestamp_secs();
        task.completed_at_unix_secs = Some(now_unix_secs.saturating_sub(60));
        task.updated_at_unix_secs = now_unix_secs.saturating_sub(60);
    }
    task.request_metadata = None;
    task
}

struct PendingMinimalCandidateSelectionReadRepository;

impl PendingMinimalCandidateSelectionReadRepository {
    async fn pending_rows(
        &self,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        pending::<Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError>>().await
    }
}

#[async_trait]
impl MinimalCandidateSelectionReadRepository for PendingMinimalCandidateSelectionReadRepository {
    async fn list_for_exact_api_format(
        &self,
        _api_format: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }

    async fn list_for_exact_api_format_and_global_model(
        &self,
        _api_format: &str,
        _global_model_name: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }

    async fn list_for_exact_api_format_and_requested_model(
        &self,
        _api_format: &str,
        _requested_model_name: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }

    async fn list_for_exact_api_format_and_requested_model_page(
        &self,
        _query: &StoredRequestedModelCandidateRowsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }

    async fn list_pool_key_rows_for_group(
        &self,
        _query: &StoredPoolKeyCandidateRowsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }

    async fn list_pool_key_rows_for_group_key_ids(
        &self,
        _query: &StoredPoolKeyCandidateRowsByKeyIdsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        self.pending_rows().await
    }
}

struct CachedToggleMinimalCandidateSelectionReadRepository {
    row: StoredMinimalCandidateSelectionRow,
    active: AtomicBool,
    cached_rows_by_api_format: Mutex<HashMap<String, Vec<StoredMinimalCandidateSelectionRow>>>,
}

impl CachedToggleMinimalCandidateSelectionReadRepository {
    fn new(row: StoredMinimalCandidateSelectionRow) -> Self {
        Self {
            row,
            active: AtomicBool::new(true),
            cached_rows_by_api_format: Mutex::new(HashMap::new()),
        }
    }

    fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::SeqCst);
    }

    fn rows_for_api_format(&self, api_format: &str) -> Vec<StoredMinimalCandidateSelectionRow> {
        let api_format = api_format.trim().to_string();
        let mut cached = self
            .cached_rows_by_api_format
            .lock()
            .expect("candidate row cache lock");
        if let Some(rows) = cached.get(&api_format) {
            return rows.clone();
        }

        let rows = if self.active.load(Ordering::SeqCst)
            && self
                .row
                .endpoint_api_format
                .eq_ignore_ascii_case(&api_format)
        {
            vec![self.row.clone()]
        } else {
            Vec::new()
        };
        cached.insert(api_format, rows.clone());
        rows
    }
}

#[async_trait]
impl MinimalCandidateSelectionReadRepository
    for CachedToggleMinimalCandidateSelectionReadRepository
{
    fn clear_local_cache(&self) {
        self.cached_rows_by_api_format
            .lock()
            .expect("candidate row cache lock")
            .clear();
    }

    async fn list_for_exact_api_format(
        &self,
        api_format: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(self.rows_for_api_format(api_format))
    }

    async fn list_for_exact_api_format_and_global_model(
        &self,
        api_format: &str,
        global_model_name: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(self
            .rows_for_api_format(api_format)
            .into_iter()
            .filter(|row| row.global_model_name == global_model_name)
            .collect())
    }

    async fn list_for_exact_api_format_and_requested_model(
        &self,
        api_format: &str,
        requested_model_name: &str,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(self
            .rows_for_api_format(api_format)
            .into_iter()
            .filter(|row| row.global_model_name == requested_model_name)
            .collect())
    }

    async fn list_for_exact_api_format_and_requested_model_page(
        &self,
        query: &StoredRequestedModelCandidateRowsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(self
            .rows_for_api_format(&query.api_format)
            .into_iter()
            .filter(|row| row.global_model_name == query.requested_model_name)
            .skip(query.offset as usize)
            .take(query.limit as usize)
            .collect())
    }

    async fn list_pool_key_rows_for_group(
        &self,
        _query: &StoredPoolKeyCandidateRowsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(Vec::new())
    }

    async fn list_pool_key_rows_for_group_key_ids(
        &self,
        _query: &StoredPoolKeyCandidateRowsByKeyIdsQuery,
    ) -> Result<Vec<StoredMinimalCandidateSelectionRow>, DataLayerError> {
        Ok(Vec::new())
    }
}

#[tokio::test]
async fn gateway_handles_public_openai_models_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Body::from("proxied"))
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-models")),
        unrestricted_models_snapshot("key-1", "user-1"),
    )]));
    let candidate_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            sample_models_candidate_row("provider-openai", "openai", "openai:chat", "gpt-5", 10),
            sample_models_candidate_row("provider-openai", "openai", "openai:chat", "gpt-4.1", 10),
        ]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-openai-models")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["object"], "list");
    assert_eq!(payload["data"][0]["id"], "gpt-4.1");
    assert_eq!(payload["data"][1]["id"], "gpt-5");
    assert_eq!(payload["data"][0]["owned_by"], "aether");
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_serves_codex_model_cards_for_versioned_models_requests() {
    let codex_row =
        sample_codex_models_candidate_row("provider-codex-models", "frontier-sol", "gpt-5.6-sol");
    let incomplete_codex_row = sample_codex_models_candidate_row(
        "provider-codex-incomplete",
        "broken-luna",
        "gpt-5.6-luna",
    );
    let candidate_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            codex_row.clone(),
            incomplete_codex_row.clone(),
            sample_models_candidate_row(
                "provider-openai-responses",
                "openai",
                "openai:responses",
                "custom-responses-model",
                20,
            ),
        ]));
    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![
        (
            Some(hash_api_key("sk-codex-models")),
            codex_models_snapshot("key-codex-models", "user-codex-models"),
        ),
        (
            Some(hash_api_key("sk-standard-models")),
            unrestricted_models_snapshot("key-standard-models", "user-standard-models"),
        ),
    ]));
    let state = AppState::new()
        .expect("gateway should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                candidate_repository,
                auth_repository,
            ),
        );
    state
        .runtime_kv_setex(
            &format!(
                "upstream_models:{}:{}",
                codex_row.provider_id, codex_row.key_id
            ),
            &serde_json::to_string(&vec![complete_codex_model_card("gpt-5.6-sol")])
                .expect("model cache should serialize"),
            60,
        )
        .await
        .expect("model cache should seed");
    state
        .runtime_kv_setex(
            &format!(
                "upstream_models:{}:{}",
                incomplete_codex_row.provider_id, incomplete_codex_row.key_id
            ),
            &serde_json::to_string(&vec![json!({
                "id": "gpt-5.6-luna",
                "slug": "gpt-5.6-luna",
                "display_name": "GPT-5.6-Luna"
            })])
            .expect("incomplete model cache should serialize"),
            60,
        )
        .await
        .expect("incomplete model cache should seed");

    let gateway = build_router_with_state(state);
    let (gateway_url, gateway_handle) = start_server(gateway).await;
    let client = reqwest::Client::new();

    let codex_response = client
        .get(format!("{gateway_url}/v1/models?client_version=0.144.1"))
        .header("authorization", "Bearer sk-codex-models")
        .send()
        .await
        .expect("Codex models request should succeed");
    assert_eq!(codex_response.status(), StatusCode::OK);
    let codex_payload: serde_json::Value = codex_response
        .json()
        .await
        .expect("Codex models body should parse");
    assert_eq!(codex_payload["models"].as_array().map(Vec::len), Some(1));
    assert_eq!(codex_payload["models"][0]["slug"], "frontier-sol");
    assert_eq!(
        codex_payload["models"][0]["supported_reasoning_levels"][5]["effort"],
        "ultra"
    );
    assert_eq!(
        codex_payload["models"][0]["future_capability"],
        json!({"enabled": true})
    );
    assert!(codex_payload["models"][0].get("id").is_none());
    assert!(codex_payload["models"][0].get("api_formats").is_none());
    assert!(codex_payload.get("object").is_none());

    let standard_response = client
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-standard-models")
        .send()
        .await
        .expect("standard models request should succeed");
    assert_eq!(standard_response.status(), StatusCode::OK);
    let standard_payload: serde_json::Value = standard_response
        .json()
        .await
        .expect("standard models body should parse");
    assert_eq!(standard_payload["object"], "list");
    assert!(standard_payload["data"].is_array());
    assert!(standard_payload.get("models").is_none());

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_openai_models_list_drops_disabled_global_model_after_cache_invalidation() {
    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-models-cache")),
        unrestricted_models_snapshot("key-models-cache", "user-models-cache"),
    )]));
    let row = sample_models_candidate_row(
        "provider-openai-cache",
        "openai",
        "openai:chat",
        "gpt-5",
        10,
    );
    let global_model_id = row.global_model_id.clone();
    let candidate_repository = Arc::new(CachedToggleMinimalCandidateSelectionReadRepository::new(
        row.clone(),
    ));
    let global_model_repository = Arc::new(
        InMemoryGlobalModelReadRepository::seed(Vec::new()).with_admin_global_models(vec![
            StoredAdminGlobalModel::new(
                global_model_id.clone(),
                row.global_model_name.clone(),
                "GPT 5".to_string(),
                true,
                None,
                None,
                None,
                None,
                0,
                1,
                0,
                Some(1_711_000_000),
                Some(1_711_000_000),
            )
            .expect("global model should build"),
        ]),
    );
    let state = AppState::new()
        .expect("gateway should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                candidate_repository.clone(),
                auth_repository,
            )
            .with_global_model_repository_for_tests(global_model_repository),
        );
    let gateway = build_router_with_state(state.clone());
    let (gateway_url, gateway_handle) = start_server(gateway).await;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-openai-models-cache")
        .send()
        .await
        .expect("initial models request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["data"][0]["id"], "gpt-5");

    candidate_repository.set_active(false);
    let disabled_global_model = UpdateAdminGlobalModelRecord::new(
        global_model_id,
        "GPT 5".to_string(),
        false,
        None,
        None,
        None,
        None,
    )
    .expect("global model update record should build");
    state
        .update_admin_global_model(&disabled_global_model)
        .await
        .expect("global model update should succeed")
        .expect("global model should update");

    let response = client
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-openai-models-cache")
        .send()
        .await
        .expect("models request after disable should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(
        payload["data"]
            .as_array()
            .expect("data should be an array")
            .len(),
        0
    );

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_returns_empty_openai_models_when_candidate_rows_stall() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Body::from("proxied"))
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-models-stalled")),
        unrestricted_models_snapshot("key-stalled", "user-stalled"),
    )]));
    let candidate_repository = Arc::new(PendingMinimalCandidateSelectionReadRepository);

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .expect("client should build")
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-openai-models-stalled")
        .send()
        .await
        .expect("request should return before client timeout");

    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["object"], "list");
    assert_eq!(
        payload["data"]
            .as_array()
            .expect("data should be an array")
            .len(),
        0
    );
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_returns_not_found_for_openai_model_detail_when_candidate_rows_stall() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Body::from("proxied"))
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-model-detail-stalled")),
        unrestricted_models_snapshot("key-detail-stalled", "user-detail-stalled"),
    )]));
    let candidate_repository = Arc::new(PendingMinimalCandidateSelectionReadRepository);

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
        .expect("client should build")
        .get(format!("{gateway_url}/v1/models/gpt-stalled"))
        .header("authorization", "Bearer sk-openai-model-detail-stalled")
        .send()
        .await
        .expect("request should return before client timeout");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["error"]["code"], "model_not_found");
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_handles_public_openai_models_with_cross_format_candidates_without_hitting_fallback_probe(
) {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Body::from("proxied"))
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-models-cross-format")),
        unrestricted_models_snapshot("key-1", "user-1"),
    )]));
    let candidate_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            sample_models_candidate_row(
                "provider-claude",
                "claude",
                "claude:messages",
                "claude-3-7-sonnet",
                10,
            ),
        ]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let client = reqwest::Client::new();
    let list_response = client
        .get(format!("{gateway_url}/v1/models"))
        .header("authorization", "Bearer sk-openai-models-cross-format")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(list_response.status(), StatusCode::OK);
    let list_payload: serde_json::Value =
        list_response.json().await.expect("json body should parse");
    assert_eq!(list_payload["object"], "list");
    assert_eq!(list_payload["data"][0]["id"], "claude-3-7-sonnet");
    assert_eq!(list_payload["data"][0]["owned_by"], "aether");

    let detail_response = client
        .get(format!("{gateway_url}/v1/models/claude-3-7-sonnet"))
        .header("authorization", "Bearer sk-openai-models-cross-format")
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(detail_response.status(), StatusCode::OK);
    let detail_payload: serde_json::Value = detail_response
        .json()
        .await
        .expect("json body should parse");
    assert_eq!(detail_payload["id"], "claude-3-7-sonnet");
    assert_eq!(detail_payload["owned_by"], "aether");

    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_handles_public_claude_models_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-claude-models")),
        unrestricted_models_snapshot("key-claude", "user-claude"),
    )]));
    let candidate_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            sample_models_candidate_row(
                "provider-claude",
                "claude",
                "claude:messages",
                "claude-3-7-sonnet",
                10,
            ),
            sample_models_candidate_row(
                "provider-claude",
                "claude",
                "claude:messages",
                "claude-3-5-haiku",
                10,
            ),
        ]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!("{gateway_url}/v1/models?limit=1"))
        .header("x-api-key", "sk-claude-models")
        .header("anthropic-version", "2023-06-01")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["data"][0]["id"], "claude-3-5-haiku");
    assert_eq!(payload["first_id"], "claude-3-5-haiku");
    assert_eq!(payload["last_id"], "claude-3-5-haiku");
    assert_eq!(payload["has_more"], true);
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_handles_public_gemini_models_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-gemini-models")),
        unrestricted_models_snapshot("key-gemini", "user-gemini"),
    )]));
    let candidate_repository =
        Arc::new(InMemoryMinimalCandidateSelectionReadRepository::seed(vec![
            sample_models_candidate_row(
                "provider-gemini",
                "gemini",
                "gemini:generate_content",
                "gemini-2.5-flash",
                10,
            ),
            sample_models_candidate_row(
                "provider-gemini",
                "gemini",
                "gemini:generate_content",
                "gemini-2.5-pro",
                10,
            ),
        ]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_minimal_candidate_selection_and_auth_for_tests(
                    candidate_repository,
                    auth_repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!(
            "{gateway_url}/v1beta/models?pageSize=1&key=sk-gemini-models"
        ))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["models"][0]["name"], "models/gemini-2.5-flash");
    assert_eq!(payload["nextPageToken"], "1");
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_handles_antigravity_v1internal_control_plane_without_proxying() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(AppState::new().expect("gateway should build"));
    let (gateway_url, gateway_handle) = start_server(gateway).await;
    let client = reqwest::Client::new();

    let user_settings = json!({
        "preferredModelId": "gemini-3.1-flash-lite",
        "theme": "dark"
    });
    let requests = vec![
        (
            "/v1internal:loadCodeAssist",
            json!({"metadata": {"ideType": "ANTIGRAVITY_CLI"}}),
        ),
        (
            "/v1internal:fetchAvailableModels",
            json!({"project": "aether-antigravity-local"}),
        ),
        (
            "/v1internal:retrieveUserQuotaSummary",
            json!({"project": "aether-antigravity-local"}),
        ),
        (
            "/v1internal:fetchUserInfo",
            json!({"project": "aether-antigravity-local"}),
        ),
        (
            "/v1internal:fetchAdminControls",
            json!({"project": "aether-antigravity-local"}),
        ),
        ("/v1internal:listExperiments", json!({})),
        (
            "/v1internal:recordCodeAssistMetrics",
            json!({
                "project": "aether-antigravity-local",
                "requestId": "opaque-request-id",
                "metrics": []
            }),
        ),
        (
            "/v1internal:writeTrajectoryAcls",
            json!({"trajectoryId": "trajectory-ant-123"}),
        ),
        (
            "/v1internal:setUserSettings",
            json!({"userSettings": user_settings.clone()}),
        ),
    ];

    for (path, request_body) in requests {
        let response = client
            .post(format!("{gateway_url}{path}"))
            .header("authorization", "Bearer ant-access-token")
            .header("user-agent", "antigravity/cli/1.0.2 linux/arm64")
            .json(&request_body)
            .send()
            .await
            .expect("request should succeed");

        assert_eq!(response.status(), StatusCode::OK, "path {path}");
        assert_eq!(
            response
                .headers()
                .get(EXECUTION_PATH_HEADER)
                .and_then(|value| value.to_str().ok()),
            Some(EXECUTION_PATH_LOCAL_AI_PUBLIC),
            "path {path}"
        );
        let payload: serde_json::Value = response.json().await.expect("json body should parse");

        match path {
            "/v1internal:loadCodeAssist" => {
                assert_eq!(
                    payload["cloudaicompanionProject"],
                    "aether-antigravity-local"
                );
                assert_eq!(payload["currentTier"]["id"], "free-tier");
                assert_eq!(payload["currentTier"]["name"], "Antigravity");
                assert_eq!(payload["paidTier"]["id"], "g1-pro-tier");
                assert_eq!(payload["gcpManaged"], false);
                assert_eq!(payload["allowedTiers"][0]["id"], "free-tier");
                assert_eq!(payload["allowedTiers"][0]["isDefault"], true);
                assert_eq!(payload["allowedTiers"][1]["id"], "standard-tier");
                assert_eq!(
                    payload["upgradeSubscriptionUri"],
                    "https://codeassist.google.com/upgrade"
                );
            }
            "/v1internal:fetchAvailableModels" => {
                assert_eq!(payload["defaultAgentModelId"], "gemini-3.5-flash-low");
                assert_eq!(
                    payload["tieredModelIds"]["flash"],
                    json!(["gemini-3-flash-agent"])
                );
                assert_eq!(
                    payload["tieredModelIds"]["pro"],
                    json!(["gemini-3.1-pro-low"])
                );
                assert_eq!(
                    payload["models"]["gemini-3-flash-agent"]["displayName"],
                    "Gemini 3.5 Flash (High)"
                );
                assert_eq!(
                    payload["models"]["gemini-3.5-flash-low"]["displayName"],
                    "Gemini 3.5 Flash (Medium)"
                );
                assert_eq!(
                    payload["models"]["gemini-3.5-flash-extra-low"]["displayName"],
                    "Gemini 3.5 Flash (Low)"
                );
                assert_eq!(
                    payload["models"]["gemini-pro-agent"]["displayName"],
                    "Gemini 3.1 Pro (High)"
                );
                assert_eq!(
                    payload["models"]["claude-opus-4-6-thinking"]["displayName"],
                    "Claude Opus 4.6 (Thinking)"
                );
                assert_eq!(
                    payload["models"]["gpt-oss-120b-medium"]["displayName"],
                    "GPT-OSS 120B (Medium)"
                );
                assert_eq!(
                    payload["models"]["gemini-pro-agent"]["model"],
                    "MODEL_PLACEHOLDER_M16"
                );
                assert_eq!(
                    payload["models"]["gemini-3.1-pro-high"]["model"],
                    "MODEL_PLACEHOLDER_M37"
                );
                assert_eq!(
                    payload["models"]["gemini-3.5-flash-extra-low"]["model"],
                    "MODEL_PLACEHOLDER_M187"
                );
                assert_eq!(
                    payload["models"]["claude-sonnet-4-6"]["apiProvider"],
                    "API_PROVIDER_ANTHROPIC_VERTEX"
                );
                assert_eq!(
                    payload["models"]["gpt-oss-120b-medium"]["apiProvider"],
                    "API_PROVIDER_OPENAI_VERTEX"
                );
                assert_eq!(
                    payload["models"]["gemini-3.5-flash-low"]["apiProvider"],
                    "API_PROVIDER_GOOGLE_GEMINI"
                );
                assert_eq!(
                    payload["models"]["gemini-2.5-flash-lite"]["model"],
                    "MODEL_GOOGLE_GEMINI_2_5_FLASH_LITE"
                );
                assert_eq!(
                    payload["agentModelSorts"][0]["groups"][0]["modelIds"],
                    json!([
                        "gemini-3.5-flash-low",
                        "gemini-3-flash-agent",
                        "gemini-3.5-flash-extra-low",
                        "gemini-3.1-pro-low",
                        "gemini-pro-agent",
                        "claude-sonnet-4-6",
                        "claude-opus-4-6-thinking",
                        "gpt-oss-120b-medium"
                    ])
                );
                assert_eq!(
                    payload["deprecatedModelIds"]["gemini-3.1-pro-high"]["newModelId"],
                    "gemini-pro-agent"
                );
                assert_eq!(payload["commandModelIds"], json!(["gemini-3-flash"]));
                assert_eq!(
                    payload["imageGenerationModelIds"],
                    json!(["gemini-3.1-flash-image"])
                );
                assert_eq!(payload["tabModelIds"], json!(["chat_20706", "chat_23310"]));
                assert_eq!(payload["mqueryModelIds"], json!(["gemini-3.1-flash-lite"]));
                assert_eq!(
                    payload["webSearchModelIds"],
                    json!(["gemini-3.1-flash-lite"])
                );
                assert_eq!(
                    payload["commitMessageModelIds"],
                    json!(["gemini-3.1-flash-lite"])
                );
            }
            "/v1internal:fetchUserInfo" => {
                assert_eq!(payload["regionCode"], "US");
                assert_eq!(
                    payload["userSettings"]["preferredModelId"],
                    "gemini-3.5-flash-low"
                );
            }
            "/v1internal:retrieveUserQuotaSummary" => {
                assert_eq!(payload["description"], "");
                assert_eq!(payload["groups"], json!([]));
            }
            "/v1internal:fetchAdminControls" => {
                assert_eq!(payload, json!({}));
            }
            "/v1internal:listExperiments" => {
                assert_eq!(payload["experimentIds"], json!([]));
                assert_eq!(payload["flags"], json!([]));
            }
            "/v1internal:recordCodeAssistMetrics" => {
                assert_eq!(payload, json!({}));
            }
            "/v1internal:writeTrajectoryAcls" => {
                assert_eq!(payload, json!({}));
            }
            "/v1internal:setUserSettings" => {
                assert_eq!(payload["userSettings"], user_settings);
            }
            other => panic!("unexpected path {other}"),
        }
    }

    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_does_not_locally_reject_image_model_name_on_chat_completions() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-chat-image-model")),
        unrestricted_models_snapshot(
            "key-openai-chat-image-model",
            "user-openai-chat-image-model",
        ),
    )]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_auth_api_key_data_reader_for_tests(auth_repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/chat/completions"))
        .header("authorization", "Bearer sk-openai-chat-image-model")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(
            serde_json::to_vec(&json!({
                "model": "gpt-image-2",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .expect("request body should encode"),
        )
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_EXECUTION_RUNTIME_MISS)
    );
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_rejects_image_request_above_gateway_limit_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-image-n")),
        unrestricted_models_snapshot("key-openai-image-n", "user-openai-image-n"),
    )]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_auth_api_key_data_reader_for_tests(auth_repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/generations"))
        .header("authorization", "Bearer sk-openai-image-n")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(
            serde_json::to_vec(&json!({
                "model": "grok-imagine-image-lite",
                "prompt": "draw",
                "n": openai_image_gateway_max_generation_count() + 1,
                "response_format": "b64_json"
            }))
            .expect("request body should encode"),
        )
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_AI_PUBLIC)
    );
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(
        payload["detail"],
        format!(
            "当前图片反代仅支持 n=1..{}",
            openai_image_gateway_max_generation_count()
        )
    );
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_does_not_mount_image_variation_route_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-openai-image-variation")),
        unrestricted_models_snapshot("key-openai-image-variation", "user-openai-image-variation"),
    )]));

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_auth_api_key_data_reader_for_tests(auth_repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .post(format!("{gateway_url}/v1/images/variations"))
        .header("authorization", "Bearer sk-openai-image-variation")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(
            serde_json::to_vec(&json!({
                "model": "dall-e-2",
                "response_format": "url"
            }))
            .expect("request body should encode"),
        )
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_handles_gemini_operation_detail_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-gemini-operation-detail")),
        unrestricted_models_snapshot(
            "key-gemini-operation-detail",
            "user-gemini-operation-detail",
        ),
    )]));
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_gemini_video_task(
            "task-gemini-operation-detail",
            "opshort123",
            "user-gemini-operation-detail",
            "key-gemini-operation-detail",
            "operations/ext-op-123",
            VideoTaskStatus::Completed,
        ))
        .await
        .expect("upsert should succeed");

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_auth_and_video_task_repository_for_tests(
                    auth_repository,
                    repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!(
            "{gateway_url}/v1beta/operations/opshort123?key=sk-gemini-operation-detail"
        ))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_AI_PUBLIC)
    );
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["name"], "models/veo-3/operations/opshort123");
    assert_eq!(payload["done"], true);
    assert_eq!(
        payload["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"],
        "/v1beta/files/aev_opshort123:download?alt=media"
    );
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_lists_gemini_operations_without_hitting_fallback_probe() {
    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-gemini-operation-list")),
        unrestricted_models_snapshot("key-gemini-operation-list", "user-gemini-operation-list"),
    )]));
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_gemini_video_task(
            "task-gemini-operation-list-1",
            "opshort-list-1",
            "user-gemini-operation-list",
            "key-gemini-operation-list",
            "operations/ext-list-1",
            VideoTaskStatus::Completed,
        ))
        .await
        .expect("upsert should succeed");
    repository
        .upsert(sample_gemini_video_task(
            "task-gemini-operation-list-2",
            "opshort-list-2",
            "user-gemini-operation-list",
            "key-gemini-operation-list",
            "operations/ext-list-2",
            VideoTaskStatus::Processing,
        ))
        .await
        .expect("upsert should succeed");
    repository
        .upsert(sample_gemini_video_task(
            "task-gemini-operation-list-other",
            "opshort-list-other",
            "user-other",
            "key-other",
            "operations/ext-list-other",
            VideoTaskStatus::Completed,
        ))
        .await
        .expect("upsert should succeed");

    let (_unused_fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_auth_and_video_task_repository_for_tests(
                    auth_repository,
                    repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!(
            "{gateway_url}/v1beta/operations?key=sk-gemini-operation-list"
        ))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_AI_PUBLIC)
    );
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    let operations = payload["operations"]
        .as_array()
        .expect("operations should be an array");
    assert_eq!(operations.len(), 2);
    let operation_names = operations
        .iter()
        .map(|value| {
            value["name"]
                .as_str()
                .expect("operation name should be a string")
                .to_string()
        })
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        operation_names,
        std::collections::BTreeSet::from([
            "models/veo-3/operations/opshort-list-1".to_string(),
            "models/veo-3/operations/opshort-list-2".to_string(),
        ])
    );
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    fallback_probe_handle.abort();
}

#[tokio::test]
async fn gateway_doubao_task_list_total_counts_all_matching_pages() {
    let auth_repository = Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key("sk-doubao-task-list")),
        unrestricted_models_snapshot("key-doubao-task-list", "user-doubao-task-list"),
    )]));
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    for (id, status) in [
        ("task-doubao-list-1", VideoTaskStatus::Completed),
        ("task-doubao-list-2", VideoTaskStatus::Processing),
        ("task-doubao-list-3", VideoTaskStatus::Cancelled),
        ("task-doubao-list-pending", VideoTaskStatus::Pending),
        ("task-doubao-list-submitted", VideoTaskStatus::Submitted),
        ("task-doubao-list-queued", VideoTaskStatus::Queued),
        ("task-doubao-list-failed", VideoTaskStatus::Failed),
        ("task-doubao-list-expired", VideoTaskStatus::Expired),
    ] {
        repository
            .upsert(sample_doubao_video_task(
                id,
                "user-doubao-task-list",
                "key-doubao-task-list",
                status,
            ))
            .await
            .expect("upsert should succeed");
    }
    let mut expired_cancelled = sample_doubao_video_task(
        "task-doubao-list-cancelled-expired",
        "user-doubao-task-list",
        "key-doubao-task-list",
        VideoTaskStatus::Cancelled,
    );
    let expired_at = aether_video_tasks_core::current_unix_timestamp_secs()
        .saturating_sub(aether_video_tasks_core::DOUBAO_CANCELLED_TASK_RETENTION_SECONDS);
    expired_cancelled.completed_at_unix_secs = Some(expired_at);
    expired_cancelled.updated_at_unix_secs = expired_at;
    repository
        .upsert(expired_cancelled)
        .await
        .expect("expired cancelled task should seed");
    repository
        .upsert(sample_doubao_video_task(
            "task-doubao-list-other-user",
            "user-other",
            "key-other",
            VideoTaskStatus::Completed,
        ))
        .await
        .expect("upsert should succeed");
    let mut openai_client_task = sample_doubao_video_task(
        "task-openai-via-doubao-list-hidden",
        "user-doubao-task-list",
        "key-doubao-task-list",
        VideoTaskStatus::Completed,
    );
    openai_client_task.client_api_format = Some("openai:video".to_string());
    openai_client_task.format_converted = true;
    repository
        .upsert(openai_client_task)
        .await
        .expect("cross-format task should seed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_auth_and_video_task_repository_for_tests(
                    auth_repository,
                    repository,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .get(format!(
            "{gateway_url}/v3/contents/generations/tasks?page_size=1&page_num=2"
        ))
        .bearer_auth("sk-doubao-task-list")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_AI_PUBLIC)
    );
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["items"].as_array().map(Vec::len), Some(1));
    assert_eq!(payload["total"], json!(8));

    for (filter, expected_total, expected_status) in [
        ("queued", 3, "queued"),
        ("failed", 2, "failed"),
        ("cancelled", 1, "cancelled"),
    ] {
        let response = reqwest::Client::new()
            .get(format!(
                "{gateway_url}/v3/contents/generations/tasks?page_size=10&filter.status={filter}"
            ))
            .bearer_auth("sk-doubao-task-list")
            .send()
            .await
            .expect("status-filtered request should succeed");
        assert_eq!(response.status(), StatusCode::OK);
        let payload: serde_json::Value = response.json().await.expect("json body should parse");
        assert_eq!(payload["total"], json!(expected_total));
        let items = payload["items"].as_array().expect("items array");
        assert_eq!(items.len(), expected_total);
        assert!(items.iter().all(|item| item["status"] == expected_status));
        assert!(
            items.iter().all(|item| item.get("content").is_none()),
            "{filter} list leaked a signed content URL"
        );
    }

    let response = reqwest::Client::new()
        .get(format!(
            "{gateway_url}/v3/contents/generations/tasks?filter.task_ids=task-openai-via-doubao-list-hidden"
        ))
        .bearer_auth("sk-doubao-task-list")
        .send()
        .await
        .expect("task-id filtered request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let payload: serde_json::Value = response.json().await.expect("json body should parse");
    assert_eq!(payload["items"], json!([]));
    assert_eq!(payload["total"], json!(0));

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_cancels_gemini_operation_without_hitting_fallback_probe() {
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SeenExecutionRuntimeSyncRequest {
        method: String,
        url: String,
        api_key: String,
    }

    let fallback_probe_hits = Arc::new(Mutex::new(0usize));
    let fallback_probe_hits_clone = Arc::clone(&fallback_probe_hits);
    let fallback_probe = Router::new().route(
        "/{*path}",
        any(move |_request: Request| {
            let fallback_probe_hits_inner = Arc::clone(&fallback_probe_hits_clone);
            async move {
                *fallback_probe_hits_inner.lock().expect("mutex should lock") += 1;
                (StatusCode::OK, Json(json!({"proxied": true}))).into_response()
            }
        }),
    );

    let seen_execution_runtime = Arc::new(Mutex::new(None::<SeenExecutionRuntimeSyncRequest>));
    let seen_execution_runtime_clone = Arc::clone(&seen_execution_runtime);
    let execution_runtime = Router::new().route(
        "/v1/execute/sync",
        any(move |request: Request| {
            let seen_execution_runtime_inner = Arc::clone(&seen_execution_runtime_clone);
            async move {
                let (_parts, body) = request.into_parts();
                let raw_body = to_bytes(body, usize::MAX).await.expect("body should read");
                let payload: serde_json::Value = serde_json::from_slice(&raw_body)
                    .expect("execution runtime payload should parse");
                *seen_execution_runtime_inner
                    .lock()
                    .expect("mutex should lock") = Some(SeenExecutionRuntimeSyncRequest {
                    method: payload
                        .get("method")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    url: payload
                        .get("url")
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    api_key: payload
                        .get("headers")
                        .and_then(|value| value.get("x-goog-api-key"))
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                });
                Json(json!({
                    "request_id": "trace-gemini-operation-cancel",
                    "status_code": 200,
                    "headers": {
                        "content-type": "application/json"
                    },
                    "body": {
                        "json_body": {}
                    },
                    "telemetry": {
                        "elapsed_ms": 12
                    }
                }))
            }
        }),
    );

    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_gemini_video_task(
            "task-gemini-operation-cancel",
            "opshort-cancel",
            "user-gemini-operation-cancel",
            "key-gemini-operation-cancel",
            "operations/ext-op-123",
            VideoTaskStatus::Submitted,
        ))
        .await
        .expect("upsert should succeed");

    let (fallback_probe_url, fallback_probe_handle) = start_server(fallback_probe).await;
    let (execution_runtime_url, execution_runtime_handle) = start_server(execution_runtime).await;
    let provider_catalog_repository = simple_video_provider_catalog_repository(
        "provider-gemini-video-local-1",
        "endpoint-gemini-video-local-1",
        "key-gemini-video-local-1",
        "gemini",
        "gemini:video",
        "https://generativelanguage.googleapis.com",
        "sk-upstream-gemini-video",
    );
    let gateway = build_router_with_state(
        build_state_with_execution_runtime_override(execution_runtime_url)
            .with_video_task_truth_source_mode(crate::tests::VideoTaskTruthSourceMode::RustAuthoritative)
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_video_task_repository_and_provider_transport_for_tests(
                    Arc::clone(&repository),
                    provider_catalog_repository,
                    DEVELOPMENT_ENCRYPTION_KEY,
                )
                .with_auth_api_key_reader(video_auth_repository(
                    "sk-gemini-operation-cancel",
                    "key-gemini-operation-cancel",
                    "user-gemini-operation-cancel",
                    "gemini",
                    "gemini:video",
                    "veo-3",
                )),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let response = reqwest::Client::new()
        .post(format!(
            "{gateway_url}/v1beta/operations/opshort-cancel:cancel"
        ))
        .header("x-goog-api-key", "sk-gemini-operation-cancel")
        .header(http::header::CONTENT_TYPE, "application/json")
        .body("{}")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(EXECUTION_PATH_HEADER)
            .and_then(|value| value.to_str().ok()),
        Some(EXECUTION_PATH_LOCAL_AI_PUBLIC)
    );
    assert_eq!(
        response
            .json::<serde_json::Value>()
            .await
            .expect("json body should parse"),
        json!({})
    );

    let seen_execution_runtime_request = seen_execution_runtime
        .lock()
        .expect("mutex should lock")
        .clone()
        .expect("execution runtime sync should be captured");
    assert_eq!(seen_execution_runtime_request.method, "POST");
    assert_eq!(
        seen_execution_runtime_request.url,
        "https://generativelanguage.googleapis.com/v1beta/models/veo-3/operations/ext-op-123:cancel"
    );
    assert_eq!(
        seen_execution_runtime_request.api_key,
        "sk-upstream-gemini-video"
    );

    let stored = repository
        .find(VideoTaskLookupKey::Id("task-gemini-operation-cancel"))
        .await
        .expect("task lookup should succeed")
        .expect("task should exist");
    assert_eq!(stored.status, VideoTaskStatus::Cancelled);
    assert_eq!(*fallback_probe_hits.lock().expect("mutex should lock"), 0);

    gateway_handle.abort();
    execution_runtime_handle.abort();
    fallback_probe_handle.abort();
}
