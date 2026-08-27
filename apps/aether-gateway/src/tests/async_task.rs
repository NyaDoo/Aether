use aether_crypto::{encrypt_python_fernet_plaintext, DEVELOPMENT_ENCRYPTION_KEY};
use aether_data::repository::provider_catalog::InMemoryProviderCatalogReadRepository;
use aether_data::repository::usage::InMemoryUsageReadRepository;
use aether_data::repository::video_tasks::InMemoryVideoTaskRepository;
use aether_data_contracts::repository::provider_catalog::{
    StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
};
use aether_data_contracts::repository::usage::{UsageAuditListQuery, UsageReadRepository};
use aether_data_contracts::repository::video_tasks::{
    UpsertVideoTask, VideoTaskStatus, VideoTaskWriteRepository,
};
use aether_usage_runtime::UsageRuntimeConfig;

use super::{
    any, build_router_with_state, build_state_with_execution_runtime_override, json,
    signed_internal_admin_headers, start_server, to_bytes, AppState, Arc, Body, Bytes, HeaderValue,
    Json, Mutex, Request, Response, Router, StatusCode, VideoTaskTruthSourceMode,
};

fn sample_video_task(
    id: &str,
    status: VideoTaskStatus,
    created_at_unix_ms: u64,
    model: &str,
    user_id: &str,
    client_api_format: &str,
) -> UpsertVideoTask {
    UpsertVideoTask {
        id: id.to_string(),
        short_id: Some(format!("short-{id}")),
        request_id: format!("request-{id}"),
        user_id: Some(user_id.to_string()),
        api_key_id: Some(format!("api-key-{id}")),
        username: Some(format!("user-{user_id}")),
        api_key_name: Some("primary".to_string()),
        external_task_id: Some(format!("ext-{id}")),
        provider_id: Some("provider-1".to_string()),
        endpoint_id: Some("endpoint-1".to_string()),
        key_id: Some("provider-key-1".to_string()),
        client_api_format: Some(client_api_format.to_string()),
        provider_api_format: Some(client_api_format.to_string()),
        format_converted: false,
        model: Some(model.to_string()),
        prompt: Some(format!("prompt-{id}")),
        original_request_body: Some(json!({"prompt": format!("prompt-{id}")})),
        duration_seconds: Some(4),
        resolution: Some("720p".to_string()),
        aspect_ratio: Some("16:9".to_string()),
        size: Some("1280x720".to_string()),
        status,
        progress_percent: if matches!(status, VideoTaskStatus::Completed) {
            100
        } else {
            50
        },
        progress_message: Some("ok".to_string()),
        retry_count: 0,
        poll_interval_seconds: 10,
        next_poll_at_unix_secs: if status.is_active() {
            Some(created_at_unix_ms.saturating_add(10))
        } else {
            None
        },
        poll_count: 0,
        max_poll_count: 360,
        created_at_unix_ms,
        submitted_at_unix_secs: Some(created_at_unix_ms),
        completed_at_unix_secs: if matches!(status, VideoTaskStatus::Completed) {
            Some(created_at_unix_ms.saturating_add(30))
        } else {
            None
        },
        updated_at_unix_secs: created_at_unix_ms.saturating_add(5),
        error_code: None,
        error_message: None,
        video_url: None,
        request_metadata: None,
    }
}

fn openai_video_provider_catalog(base_url: &str) -> Arc<InMemoryProviderCatalogReadRepository> {
    Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![StoredProviderCatalogProvider::new(
            "provider-1".to_string(),
            "OpenAI Video".to_string(),
            Some("https://openai.com".to_string()),
            "openai".to_string(),
        )
        .expect("provider should build")
        .with_transport_fields(
            true,
            false,
            false,
            None,
            Some(2),
            None,
            Some(20.0),
            None,
            None,
        )],
        vec![StoredProviderCatalogEndpoint::new(
            "endpoint-1".to_string(),
            "provider-1".to_string(),
            "openai:video".to_string(),
            Some("openai".to_string()),
            Some("video".to_string()),
            true,
        )
        .expect("endpoint should build")
        .with_transport_fields(
            base_url.to_string(),
            None,
            None,
            Some(2),
            None,
            None,
            None,
            None,
        )
        .expect("endpoint transport should build")],
        vec![StoredProviderCatalogKey::new(
            "provider-key-1".to_string(),
            "provider-1".to_string(),
            "primary".to_string(),
            "bearer".to_string(),
            None,
            true,
        )
        .expect("key should build")
        .with_transport_fields(
            Some(json!(["openai:video"])),
            encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, "sk-upstream-openai-video")
                .expect("api key should encrypt"),
            None,
            None,
            Some(json!({"openai:video": 1})),
            None,
            None,
            None,
            None,
        )
        .expect("key transport should build")],
    ))
}

#[tokio::test]
async fn gateway_lists_video_tasks_via_internal_async_task_endpoint() {
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_video_task(
            "task-1",
            VideoTaskStatus::Completed,
            100,
            "sora-2",
            "user-1",
            "openai:video",
        ))
        .await
        .expect("upsert should succeed");
    repository
        .upsert(sample_video_task(
            "task-2",
            VideoTaskStatus::Processing,
            200,
            "sora-2",
            "user-1",
            "openai:video",
        ))
        .await
        .expect("upsert should succeed");
    repository
        .upsert(sample_video_task(
            "task-3",
            VideoTaskStatus::Completed,
            300,
            "veo-3",
            "user-2",
            "gemini:video",
        ))
        .await
        .expect("upsert should succeed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_data_repository_for_tests(repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks?status=completed&page=1&page_size=1";
    let response = reqwest::Client::new()
        .get(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::GET, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("body should parse");
    assert_eq!(body["total"], 2);
    assert_eq!(body["page"], 1);
    assert_eq!(body["page_size"], 1);
    assert_eq!(body["pages"], 2);
    assert_eq!(body["items"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["items"][0]["id"], "task-3");

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_exposes_video_task_stats_via_internal_async_task_endpoint() {
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    let now_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time should be monotonic")
        .as_secs();
    repository
        .upsert(sample_video_task(
            "task-1",
            VideoTaskStatus::Completed,
            now_unix_secs,
            "sora-2",
            "user-1",
            "openai:video",
        ))
        .await
        .expect("upsert should succeed");
    repository
        .upsert(sample_video_task(
            "task-2",
            VideoTaskStatus::Processing,
            now_unix_secs.saturating_sub(86_400),
            "veo-3",
            "user-2",
            "gemini:video",
        ))
        .await
        .expect("upsert should succeed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_data_repository_for_tests(repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/stats";
    let response = reqwest::Client::new()
        .get(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::GET, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("body should parse");
    assert_eq!(body["total"], 2);
    assert_eq!(body["by_status"]["completed"], 1);
    assert_eq!(body["by_status"]["processing"], 1);
    assert_eq!(body["by_model"]["sora-2"], 1);
    assert_eq!(body["by_model"]["veo-3"], 1);
    assert_eq!(body["today_count"], 1);
    assert_eq!(body["processing_count"], 1);

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_reads_video_task_detail_via_internal_async_task_endpoint() {
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_video_task(
            "task-1",
            VideoTaskStatus::Completed,
            100,
            "sora-2",
            "user-1",
            "openai:video",
        ))
        .await
        .expect("upsert should succeed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_data_repository_for_tests(repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/task-1";
    let response = reqwest::Client::new()
        .get(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::GET, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::OK);
    let body: serde_json::Value = response.json().await.expect("body should parse");
    assert_eq!(body["id"], "task-1");
    assert_eq!(body["request_id"], "request-task-1");
    assert_eq!(body["model"], "sora-2");
    assert_eq!(body["status"], "Completed");

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_redirects_direct_video_task_video_from_internal_async_task_endpoint() {
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    let mut task = sample_video_task(
        "task-redirect",
        VideoTaskStatus::Completed,
        100,
        "sora-2",
        "user-1",
        "openai:video",
    );
    task.video_url = Some("https://cdn.example.com/video-task-redirect.mp4".to_string());
    repository
        .upsert(task)
        .await
        .expect("upsert should succeed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_data_repository_for_tests(repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("client should build");
    let path = "/_gateway/async-tasks/video-tasks/task-redirect/video";
    let response = client
        .get(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::GET, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response
            .headers()
            .get(http::header::LOCATION)
            .and_then(|value| value.to_str().ok()),
        Some("https://cdn.example.com/video-task-redirect.mp4")
    );

    gateway_handle.abort();
}

#[tokio::test]
async fn gateway_proxies_gemini_video_task_video_from_internal_async_task_endpoint() {
    let seen_api_key = Arc::new(Mutex::new(None::<String>));
    let seen_api_key_clone = Arc::clone(&seen_api_key);

    let upstream = Router::new().route(
        "/generativelanguage.googleapis.com/v1beta/files/video-task-123:download",
        any(move |request: Request| {
            let seen_api_key_inner = Arc::clone(&seen_api_key_clone);
            async move {
                *seen_api_key_inner.lock().expect("mutex should lock") = request
                    .headers()
                    .get("x-goog-api-key")
                    .and_then(|value| value.to_str().ok())
                    .map(ToOwned::to_owned);
                let mut response = Response::builder()
                    .status(StatusCode::OK)
                    .body(Body::from("proxied-video-bytes"))
                    .expect("response should build");
                response.headers_mut().insert(
                    http::header::CONTENT_TYPE,
                    HeaderValue::from_static("video/mp4"),
                );
                response
            }
        }),
    );
    let (upstream_url, upstream_handle) = start_server(upstream).await;

    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    let mut task = sample_video_task(
        "task-proxy",
        VideoTaskStatus::Completed,
        100,
        "veo-3",
        "user-1",
        "gemini:video",
    );
    task.provider_id = Some("provider-gemini-video-1".to_string());
    task.endpoint_id = Some("endpoint-gemini-video-1".to_string());
    task.key_id = Some("key-gemini-video-1".to_string());
    let task_video_url = format!(
        "{upstream_url}/generativelanguage.googleapis.com/v1beta/files/video-task-123:download"
    );
    task.video_url = Some(task_video_url.clone());
    repository
        .upsert(task)
        .await
        .expect("upsert should succeed");

    let provider_catalog_repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![StoredProviderCatalogProvider::new(
            "provider-gemini-video-1".to_string(),
            "gemini".to_string(),
            Some("https://ai.google.dev".to_string()),
            "custom".to_string(),
        )
        .expect("provider should build")
        .with_transport_fields(
            true,
            false,
            false,
            None,
            Some(2),
            None,
            Some(20.0),
            None,
            None,
        )],
        vec![StoredProviderCatalogEndpoint::new(
            "endpoint-gemini-video-1".to_string(),
            "provider-gemini-video-1".to_string(),
            "gemini:video".to_string(),
            Some("gemini".to_string()),
            Some("video".to_string()),
            true,
        )
        .expect("endpoint should build")
        .with_transport_fields(
            upstream_url.clone(),
            None,
            None,
            Some(2),
            None,
            None,
            None,
            None,
        )
        .expect("endpoint transport should build")],
        vec![StoredProviderCatalogKey::new(
            "key-gemini-video-1".to_string(),
            "provider-gemini-video-1".to_string(),
            "primary".to_string(),
            "api_key".to_string(),
            None,
            true,
        )
        .expect("key should build")
        .with_transport_fields(
            Some(json!(["gemini:video"])),
            encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, "gemini-upstream-secret")
                .expect("api key should encrypt"),
            None,
            None,
            Some(json!({"gemini:video": 1})),
            None,
            None,
            None,
            None,
        )
        .expect("key transport should build")],
    ));

    let gateway = build_router_with_state(
        AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_video_task_repository_and_provider_transport_for_tests(
                repository,
                provider_catalog_repository,
                DEVELOPMENT_ENCRYPTION_KEY,
            ),
        ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/task-proxy/video";
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("client should build");
    let response = client
        .get(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::GET, path))
        .send()
        .await
        .expect("request should succeed");

    // The test server is HTTP on a loopback origin.  The production path only
    // proxies Gemini media from the exact HTTPS Google origin, so this
    // deliberately falls back to a redirect and must never attach the API key.
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response
            .headers()
            .get(http::header::LOCATION)
            .and_then(|value| value.to_str().ok()),
        Some(task_video_url.as_str())
    );
    assert_eq!(
        seen_api_key.lock().expect("mutex should lock").as_deref(),
        None
    );

    gateway_handle.abort();
    upstream_handle.abort();
}

#[tokio::test]
async fn gateway_cancels_openai_video_task_via_internal_async_task_endpoint() {
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SeenExecutionRuntimeSyncRequest {
        method: String,
        url: String,
        authorization: String,
    }

    let seen_execution_runtime = Arc::new(Mutex::new(None::<SeenExecutionRuntimeSyncRequest>));
    let seen_execution_runtime_clone = Arc::clone(&seen_execution_runtime);
    let execution_runtime = Router::new().route(
        "/v1/execute/sync",
        any(move |request: Request| {
            let seen_execution_runtime_inner = Arc::clone(&seen_execution_runtime_clone);
            async move {
                let (_parts, body) = request.into_parts();
                let raw_body = to_bytes(body, usize::MAX).await.expect("body should read");
                let payload: serde_json::Value =
                    serde_json::from_slice(&raw_body).expect("payload should parse");
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
                    authorization: payload
                        .get("headers")
                        .and_then(|value| value.get("authorization"))
                        .and_then(|value| value.as_str())
                        .unwrap_or_default()
                        .to_string(),
                });
                Json(json!({
                    "request_id": "cancel-openai-task-1",
                    "status_code": 204,
                    "headers": {},
                    "telemetry": {
                        "elapsed_ms": 11
                    }
                }))
            }
        }),
    );

    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    let mut task = sample_video_task(
        "task-openai-cancel",
        VideoTaskStatus::Processing,
        100,
        "sora-2",
        "user-1",
        "openai:video",
    );
    task.external_task_id = Some("ext-video-task-123".to_string());
    task.request_metadata = Some(json!({
        "rust_local_snapshot": {
            "OpenAi": {
                "local_task_id": "task-openai-cancel",
                "upstream_task_id": "ext-video-task-123",
                "created_at_unix_ms": 100,
                "user_id": "user-1",
                "api_key_id": "api-key-task-openai-cancel",
                "model": "sora-2",
                "prompt": "prompt-task-openai-cancel",
                "size": "1280x720",
                "seconds": "4",
                "remixed_from_video_id": null,
                "status": "Processing",
                "progress_percent": 50,
                "completed_at_unix_secs": null,
                "expires_at_unix_secs": null,
                "error_code": null,
                "error_message": null,
                "video_url": null,
                "persistence": {
                    "request_id": "request-task-openai-cancel",
                    "username": "user-user-1",
                    "api_key_name": "primary",
                    "client_api_format": "openai:video",
                    "provider_api_format": "openai:video",
                    "original_request_body": {
                        "prompt": "prompt-task-openai-cancel"
                    },
                    "format_converted": false
                },
                "transport": {
                    "upstream_base_url": "https://api.openai.example/v1",
                    "provider_name": "openai-video",
                    "provider_id": "provider-1",
                    "endpoint_id": "endpoint-1",
                    "key_id": "provider-key-1",
                    "headers": {
                        "authorization": "Bearer sk-upstream-openai-video",
                        "content-type": "application/json"
                    },
                    "content_type": "application/json",
                    "model_name": "sora-2-upstream",
                    "proxy": null,
                    "transport_profile": null,
                    "timeouts": null
                }
            }
        }
    }));
    repository
        .upsert(task)
        .await
        .expect("upsert should succeed");

    let (execution_runtime_url, execution_runtime_handle) = start_server(execution_runtime).await;
    let provider_catalog_repository =
        openai_video_provider_catalog("https://api.openai.example/v1");
    let gateway = build_router_with_state(
        build_state_with_execution_runtime_override(execution_runtime_url)
            .with_video_task_truth_source_mode(VideoTaskTruthSourceMode::RustAuthoritative)
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_video_task_repository_and_provider_transport_for_tests(
                    Arc::clone(&repository),
                    provider_catalog_repository,
                    DEVELOPMENT_ENCRYPTION_KEY,
                ),
            ),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/task-openai-cancel/cancel";
    let response = reqwest::Client::new()
        .post(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::POST, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .json::<serde_json::Value>()
            .await
            .expect("body should parse"),
        json!({
            "id": "task-openai-cancel",
            "status": "cancelled",
            "message": "Task cancelled successfully",
        })
    );

    let seen_execution_runtime_request = seen_execution_runtime
        .lock()
        .expect("mutex should lock")
        .clone()
        .expect("execution runtime should be called");
    assert_eq!(seen_execution_runtime_request.method, "DELETE");
    assert_eq!(
        seen_execution_runtime_request.url,
        "https://api.openai.example/v1/videos/ext-video-task-123"
    );
    assert_eq!(
        seen_execution_runtime_request.authorization,
        "Bearer sk-upstream-openai-video"
    );

    let detail_path = "/_gateway/async-tasks/video-tasks/task-openai-cancel";
    let detail = reqwest::Client::new()
        .get(format!("{gateway_url}{detail_path}"))
        .headers(signed_internal_admin_headers(
            http::Method::GET,
            detail_path,
        ))
        .send()
        .await
        .expect("detail request should succeed");
    assert_eq!(detail.status(), StatusCode::OK);
    let detail_json: serde_json::Value = detail.json().await.expect("detail should parse");
    assert_eq!(detail_json["status"], "Cancelled");
    assert!(detail_json["next_poll_at_unix_secs"].is_null());
    assert_eq!(
        detail_json["request_metadata"]["rust_local_snapshot"]["OpenAi"]["status"],
        "Cancelled"
    );

    gateway_handle.abort();
    execution_runtime_handle.abort();
}

#[tokio::test]
async fn gateway_cancels_openai_video_task_via_internal_async_task_endpoint_without_execution_runtime_override(
) {
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SeenUpstreamDeleteRequest {
        method: String,
        path: String,
        authorization: String,
    }

    let seen_upstream = Arc::new(Mutex::new(None::<SeenUpstreamDeleteRequest>));
    let seen_upstream_clone = Arc::clone(&seen_upstream);
    let upstream = Router::new().route(
        "/v1/videos/ext-video-task-123",
        any(move |request: Request| {
            let seen_upstream_inner = Arc::clone(&seen_upstream_clone);
            async move {
                *seen_upstream_inner.lock().expect("mutex should lock") =
                    Some(SeenUpstreamDeleteRequest {
                        method: request.method().as_str().to_string(),
                        path: request.uri().path().to_string(),
                        authorization: request
                            .headers()
                            .get(http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok())
                            .unwrap_or_default()
                            .to_string(),
                    });
                StatusCode::NO_CONTENT
            }
        }),
    );

    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
    let mut task = sample_video_task(
        "task-openai-cancel-direct",
        VideoTaskStatus::Processing,
        100,
        "sora-2",
        "user-1",
        "openai:video",
    );
    task.external_task_id = Some("ext-video-task-123".to_string());
    let (upstream_url, upstream_handle) = start_server(upstream).await;
    let upstream_api_root = format!("{upstream_url}/v1");
    task.request_metadata = Some(json!({
        "rust_local_snapshot": {
            "OpenAi": {
                "local_task_id": "task-openai-cancel-direct",
                "upstream_task_id": "ext-video-task-123",
                "created_at_unix_ms": 100,
                "user_id": "user-1",
                "api_key_id": "api-key-task-openai-cancel",
                "model": "sora-2",
                "prompt": "prompt-task-openai-cancel",
                "size": "1280x720",
                "seconds": "4",
                "remixed_from_video_id": null,
                "status": "Processing",
                "progress_percent": 50,
                "completed_at_unix_secs": null,
                "expires_at_unix_secs": null,
                "error_code": null,
                "error_message": null,
                "video_url": null,
                "persistence": {
                    "request_id": "request-task-openai-cancel-direct",
                    "username": "user-user-1",
                    "api_key_name": "primary",
                    "client_api_format": "openai:video",
                    "provider_api_format": "openai:video",
                    "original_request_body": {
                        "prompt": "prompt-task-openai-cancel"
                    },
                    "format_converted": false
                },
                "transport": {
                    "upstream_base_url": upstream_api_root,
                    "provider_name": "openai-video",
                    "provider_id": "provider-1",
                    "endpoint_id": "endpoint-1",
                    "key_id": "provider-key-1",
                    "headers": {
                        "authorization": "Bearer sk-upstream-openai-video",
                        "content-type": "application/json"
                    },
                    "content_type": "application/json",
                    "model_name": "sora-2-upstream",
                    "proxy": null,
                    "transport_profile": null,
                    "timeouts": null
                }
            }
        }
    }));
    repository
        .upsert(task)
        .await
        .expect("upsert should succeed");

    let provider_catalog_repository = openai_video_provider_catalog(&upstream_api_root);
    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_truth_source_mode(VideoTaskTruthSourceMode::RustAuthoritative)
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_video_task_provider_transport_and_usage_repository_for_tests(
                    Arc::clone(&repository),
                    provider_catalog_repository,
                    Arc::clone(&usage_repository),
                    DEVELOPMENT_ENCRYPTION_KEY,
                ),
            )
            .with_usage_runtime_for_tests(UsageRuntimeConfig {
                enabled: true,
                enqueue_retry_initial_backoff_ms: 1,
                enqueue_retry_max_backoff_ms: 1,
                ..UsageRuntimeConfig::default()
            }),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/task-openai-cancel-direct/cancel";
    let response = reqwest::Client::new()
        .post(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::POST, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(
        response
            .json::<serde_json::Value>()
            .await
            .expect("body should parse"),
        json!({
            "id": "task-openai-cancel-direct",
            "status": "cancelled",
            "message": "Task cancelled successfully",
        })
    );

    assert_eq!(
        seen_upstream.lock().expect("mutex should lock").clone(),
        Some(SeenUpstreamDeleteRequest {
            method: "DELETE".to_string(),
            path: "/v1/videos/ext-video-task-123".to_string(),
            authorization: "Bearer sk-upstream-openai-video".to_string(),
        })
    );

    let detail_path = "/_gateway/async-tasks/video-tasks/task-openai-cancel-direct";
    let detail = reqwest::Client::new()
        .get(format!("{gateway_url}{detail_path}"))
        .headers(signed_internal_admin_headers(
            http::Method::GET,
            detail_path,
        ))
        .send()
        .await
        .expect("detail request should succeed");
    assert_eq!(detail.status(), StatusCode::OK);
    let detail_json: serde_json::Value = detail.json().await.expect("detail should parse");
    assert_eq!(detail_json["status"], "Cancelled");
    assert!(detail_json["next_poll_at_unix_secs"].is_null());
    assert_eq!(
        detail_json["request_metadata"]["rust_local_snapshot"]["OpenAi"]["status"],
        "Cancelled"
    );

    let cancel_usage = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let rows = usage_repository
                .list_usage_audits(&UsageAuditListQuery {
                    limit: Some(10),
                    newest_first: true,
                    ..UsageAuditListQuery::default()
                })
                .await
                .expect("usage list should succeed");
            if let Some(usage) = rows
                .into_iter()
                .find(|usage| usage.operation() == Some("video.cancel"))
            {
                break usage;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("independent cancel usage should become durable");
    assert!(cancel_usage.request_id.starts_with("async-task-cancel-"));
    assert_ne!(cancel_usage.request_id, "request-task-openai-cancel-direct");
    assert_eq!(cancel_usage.status, "completed");
    assert_eq!(cancel_usage.billing_status, "void");
    assert_eq!(cancel_usage.request_type.as_deref(), Some("video"));
    assert_eq!(cancel_usage.operation(), Some("video.cancel"));
    assert_eq!(
        cancel_usage
            .request_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("parent_request_id"))
            .and_then(serde_json::Value::as_str),
        Some("request-task-openai-cancel-direct")
    );
    assert_eq!(
        cancel_usage
            .request_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("usage_scope"))
            .and_then(serde_json::Value::as_str),
        Some("video_upstream_control_action")
    );
    assert_eq!(cancel_usage.client_response_headers, None);
    assert_eq!(cancel_usage.client_response_body, None);

    let original_usage = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if let Some(usage) = usage_repository
                .find_by_request_id("request-task-openai-cancel-direct")
                .await
                .expect("original usage lookup should succeed")
            {
                break usage;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("original generation usage should become terminal");
    assert_eq!(original_usage.status, "cancelled");
    assert_eq!(original_usage.billing_status, "void");
    assert_eq!(original_usage.operation(), Some("video.create"));

    gateway_handle.abort();
    upstream_handle.abort();
}

#[tokio::test]
async fn gateway_rejects_terminal_video_task_cancel_via_internal_async_task_endpoint() {
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    repository
        .upsert(sample_video_task(
            "task-cancelled-already",
            VideoTaskStatus::Completed,
            100,
            "sora-2",
            "user-1",
            "openai:video",
        ))
        .await
        .expect("upsert should succeed");

    let gateway = build_router_with_state(
        AppState::new()
            .expect("gateway state should build")
            .with_video_task_data_repository_for_tests(repository),
    );
    let (gateway_url, gateway_handle) = start_server(gateway).await;

    let path = "/_gateway/async-tasks/video-tasks/task-cancelled-already/cancel";
    let response = reqwest::Client::new()
        .post(format!("{gateway_url}{path}"))
        .headers(signed_internal_admin_headers(http::Method::POST, path))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .json::<serde_json::Value>()
            .await
            .expect("body should parse"),
        json!({
            "error": {
                "message": "Cannot cancel task with status: completed",
            }
        })
    );

    gateway_handle.abort();
}
