use std::sync::{Arc, Mutex};

use aether_crypto::{encrypt_python_fernet_plaintext, DEVELOPMENT_ENCRYPTION_KEY};
use aether_data::repository::candidates::InMemoryRequestCandidateRepository;
use aether_data::repository::provider_catalog::InMemoryProviderCatalogReadRepository;
use aether_data::repository::usage::InMemoryUsageReadRepository;
use aether_data_contracts::repository::candidates::{
    RequestCandidateReadRepository, RequestCandidateStatus, RequestCandidateWriteRepository,
    UpsertRequestCandidateRecord,
};
use aether_data_contracts::repository::provider_catalog::StoredProviderCatalogProvider;
use aether_data_contracts::repository::usage::StoredRequestUsageAudit;
use axum::body::Body;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use super::ProviderCheckinRunSummary;
use crate::AppState;

async fn start_server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = crate::test_support::bind_loopback_listener()
        .await
        .expect("listener should bind");
    let addr = listener.local_addr().expect("local addr should resolve");
    let handle = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .expect("server should run");
    });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn gateway_background_request_candidate_cleanup_deletes_expired_entries_in_batches() {
    fn now_unix_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    async fn seed_candidate(
        repository: &InMemoryRequestCandidateRepository,
        id: &str,
        created_at_unix_ms: u64,
    ) {
        repository
            .upsert(UpsertRequestCandidateRecord {
                id: id.to_string(),
                request_id: format!("req-{id}"),
                user_id: Some("user-1".to_string()),
                api_key_id: Some("api-key-1".to_string()),
                username: Some("alice".to_string()),
                api_key_name: Some("default".to_string()),
                candidate_index: 0,
                retry_index: 0,
                provider_id: Some("provider-1".to_string()),
                endpoint_id: Some("endpoint-1".to_string()),
                key_id: Some("key-1".to_string()),
                status: RequestCandidateStatus::Success,
                skip_reason: None,
                is_cached: Some(false),
                status_code: Some(200),
                error_type: None,
                error_message: None,
                latency_ms: Some(10),
                concurrent_requests: Some(1),
                extra_data: None,
                required_capabilities: None,
                created_at_unix_ms: Some(created_at_unix_ms),
                started_at_unix_ms: Some(created_at_unix_ms),
                finished_at_unix_ms: Some(created_at_unix_ms.saturating_add(1)),
            })
            .await
            .expect("candidate should seed");
    }

    let repository = Arc::new(InMemoryRequestCandidateRepository::default());
    seed_candidate(&repository, "cand-expired-1", 1_000).await;
    seed_candidate(&repository, "cand-expired-2", 2_000).await;
    seed_candidate(&repository, "cand-active", now_unix_ms()).await;

    let data_state = crate::data::GatewayDataState::with_request_candidate_repository_for_tests(
        Arc::clone(&repository),
    )
    .with_system_config_values_for_tests([
        ("enable_auto_cleanup".to_string(), json!(true)),
        ("cleanup_batch_size".to_string(), json!(1)),
        (
            "request_candidates_cleanup_batch_size".to_string(),
            json!(1),
        ),
        ("request_candidates_retention_days".to_string(), json!(30)),
    ]);

    let gateway_state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(data_state);
    let background_tasks = gateway_state.spawn_background_tasks();
    assert!(!background_tasks.is_empty(), "cleanup worker should spawn");

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
    loop {
        let rows = repository
            .list_recent(10)
            .await
            .expect("list recent should succeed");
        if rows.len() == 1 {
            assert_eq!(rows[0].id, "cand-active");
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "cleanup worker did not delete expired request candidates within 500ms"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }

    background_tasks.shutdown().await;
}

#[tokio::test]
async fn gateway_startup_terminal_sweep_after_restart_reconciles_exact_durable_usage() {
    fn terminal_usage(
        request_id: &str,
        candidate_id: Option<&str>,
        candidate_index: Option<u64>,
        status: &str,
        now_unix_secs: i64,
    ) -> StoredRequestUsageAudit {
        let mut usage = StoredRequestUsageAudit::new(
            format!("usage-{request_id}"),
            request_id.to_string(),
            Some("user-1".to_string()),
            Some("api-key-1".to_string()),
            None,
            None,
            "provider-1".to_string(),
            "model-1".to_string(),
            None,
            Some("provider-1".to_string()),
            Some("endpoint-1".to_string()),
            Some("key-1".to_string()),
            Some("chat".to_string()),
            Some("openai:chat".to_string()),
            Some("openai".to_string()),
            Some("chat".to_string()),
            Some("openai:chat".to_string()),
            Some("openai".to_string()),
            Some("chat".to_string()),
            false,
            false,
            3,
            4,
            7,
            0.01,
            0.01,
            Some(match status {
                "completed" => 200,
                "cancelled" => 499,
                _ => 502,
            }),
            (status != "completed").then(|| format!("{status} request")),
            (status != "completed").then(|| status.to_string()),
            Some(25),
            Some(5),
            status.to_string(),
            if status == "completed" {
                "settled".to_string()
            } else {
                "void".to_string()
            },
            now_unix_secs.saturating_mul(1_000),
            now_unix_secs,
            Some(now_unix_secs),
        )
        .expect("terminal usage should build");
        usage.candidate_id = candidate_id.map(str::to_string);
        usage.candidate_index = candidate_index;
        usage
    }

    fn active_candidate(
        id: &str,
        request_id: &str,
        candidate_index: u32,
        created_at_unix_ms: u64,
    ) -> UpsertRequestCandidateRecord {
        UpsertRequestCandidateRecord {
            id: id.to_string(),
            request_id: request_id.to_string(),
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            username: None,
            api_key_name: None,
            candidate_index,
            retry_index: 0,
            provider_id: Some("provider-1".to_string()),
            endpoint_id: Some("endpoint-1".to_string()),
            key_id: Some("key-1".to_string()),
            status: RequestCandidateStatus::Streaming,
            skip_reason: None,
            is_cached: Some(false),
            status_code: Some(200),
            error_type: None,
            error_message: None,
            latency_ms: None,
            concurrent_requests: None,
            extra_data: None,
            required_capabilities: None,
            created_at_unix_ms: Some(created_at_unix_ms),
            started_at_unix_ms: Some(created_at_unix_ms),
            finished_at_unix_ms: None,
        }
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let now_unix_ms = now.as_millis() as u64;
    let now_unix_secs = now.as_secs() as i64;
    let candidates = Arc::new(InMemoryRequestCandidateRepository::default());
    for record in [
        active_candidate("cand-completed", "req-completed", 0, now_unix_ms),
        active_candidate("cand-failed", "req-failed", 1, now_unix_ms + 1),
        active_candidate("cand-cancelled", "req-cancelled", 2, now_unix_ms + 2),
        active_candidate("cand-mismatch", "req-mismatch", 3, now_unix_ms + 3),
        active_candidate(
            "cand-metadata-only",
            "req-metadata-only",
            4,
            now_unix_ms + 4,
        ),
    ] {
        candidates
            .upsert(record)
            .await
            .expect("candidate should seed");
    }
    let usages = Arc::new(InMemoryUsageReadRepository::seed([
        terminal_usage(
            "req-completed",
            Some("cand-completed"),
            Some(0),
            "completed",
            now_unix_secs,
        ),
        terminal_usage(
            "req-failed",
            Some("cand-failed"),
            Some(1),
            "failed",
            now_unix_secs,
        ),
        terminal_usage(
            "req-cancelled",
            Some("cand-cancelled"),
            Some(2),
            "cancelled",
            now_unix_secs,
        ),
        terminal_usage(
            "req-mismatch",
            Some("different-candidate"),
            Some(3),
            "completed",
            now_unix_secs,
        ),
        {
            let mut usage =
                terminal_usage("req-metadata-only", None, None, "completed", now_unix_secs);
            usage.request_metadata = Some(json!({
                "candidate_id": "cand-metadata-only",
                "candidate_index": 4,
            }));
            usage
        },
    ]));
    let data_state =
        crate::data::GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
            Arc::clone(&candidates),
            usages,
        )
        .with_system_config_values_for_tests([
            (
                "request_candidate_terminal_reconciliation_batch_size".to_string(),
                json!(2),
            ),
            (
                "request_candidate_terminal_reconciliation_batches_per_run".to_string(),
                json!(4),
            ),
        ]);
    let state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(data_state);

    // Starting a fresh worker models a process restart after usage committed
    // but before the candidate half of the terminal handoff was persisted.
    // The batch size of two also forces the startup sweep through two keyset
    // cursor pages; repeating page one would leave failed/cancelled active.
    let background_tasks = state.spawn_background_tasks();
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
    loop {
        let completed = candidates
            .list_by_request_id("req-completed")
            .await
            .expect("completed candidate should read");
        let failed = candidates
            .list_by_request_id("req-failed")
            .await
            .expect("failed candidate should read");
        let cancelled = candidates
            .list_by_request_id("req-cancelled")
            .await
            .expect("cancelled candidate should read");
        if completed[0].status == RequestCandidateStatus::Success
            && failed[0].status == RequestCandidateStatus::Failed
            && cancelled[0].status == RequestCandidateStatus::Cancelled
        {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "startup candidate reconciliation did not complete within two seconds"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let mismatch = candidates
        .list_by_request_id("req-mismatch")
        .await
        .expect("mismatched candidate should read");
    assert_eq!(mismatch[0].status, RequestCandidateStatus::Streaming);
    let metadata_only = candidates
        .list_by_request_id("req-metadata-only")
        .await
        .expect("metadata-only candidate should read");
    assert_eq!(metadata_only[0].status, RequestCandidateStatus::Streaming);
    background_tasks.shutdown().await;
}

#[tokio::test]
async fn gateway_provider_checkin_runs_local_query_balance_for_configured_provider() {
    fn sample_provider(provider_id: &str, ops_url: &str) -> StoredProviderCatalogProvider {
        StoredProviderCatalogProvider::new(
            provider_id.to_string(),
            "openai".to_string(),
            Some("https://example.com".to_string()),
            "custom".to_string(),
        )
        .expect("provider should build")
        .with_routing_fields(10)
        .with_transport_fields(
            true,
            false,
            true,
            None,
            None,
            None,
            None,
            None,
            Some(json!({
                "provider_ops": {
                    "architecture_id": "generic_api",
                    "base_url": ops_url,
                    "connector": {
                        "auth_type": "api_key",
                        "config": {
                            "auth_method": "bearer"
                        },
                        "credentials": {
                            "api_key": encrypt_python_fernet_plaintext(
                                DEVELOPMENT_ENCRYPTION_KEY,
                                "live-secret-api-key",
                            ).expect("api key should encrypt"),
                        }
                    }
                }
            })),
        )
    }

    let checkin_hits = Arc::new(Mutex::new(0usize));
    let checkin_hits_clone = Arc::clone(&checkin_hits);
    let balance_hits = Arc::new(Mutex::new(0usize));
    let balance_hits_clone = Arc::clone(&balance_hits);
    let ops = Router::new()
        .route(
            "/api/user/checkin",
            post(move |headers: axum::http::HeaderMap| {
                let checkin_hits_inner = Arc::clone(&checkin_hits_clone);
                async move {
                    *checkin_hits_inner.lock().expect("mutex should lock") += 1;
                    assert_eq!(
                        headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer live-secret-api-key")
                    );
                    (
                        StatusCode::OK,
                        Json(json!({
                            "success": true,
                            "message": "签到成功",
                        })),
                    )
                }
            }),
        )
        .route(
            "/api/user/balance",
            get(move |headers: axum::http::HeaderMap| {
                let balance_hits_inner = Arc::clone(&balance_hits_clone);
                async move {
                    *balance_hits_inner.lock().expect("mutex should lock") += 1;
                    assert_eq!(
                        headers
                            .get(axum::http::header::AUTHORIZATION)
                            .and_then(|value| value.to_str().ok()),
                        Some("Bearer live-secret-api-key")
                    );
                    (
                        StatusCode::OK,
                        Json(json!({
                            "success": true,
                            "data": {
                                "quota": 5000000,
                                "used_quota": 1000000
                            }
                        })),
                    )
                }
            }),
        );

    let (ops_url, ops_handle) = start_server(ops).await;
    let repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![sample_provider("provider-openai", &ops_url)],
        vec![],
        vec![],
    ));
    let gateway_state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_provider_catalog_repository_for_tests(repository)
                .with_system_config_values_for_tests([
                    ("enable_provider_checkin".to_string(), json!(true)),
                    ("provider_checkin_time".to_string(), json!("01:05")),
                ])
                .with_encryption_key_for_tests(DEVELOPMENT_ENCRYPTION_KEY),
        );

    let summary = crate::maintenance::perform_provider_checkin_once(&gateway_state)
        .await
        .expect("provider checkin should succeed");

    assert_eq!(
        summary,
        ProviderCheckinRunSummary {
            attempted: 1,
            succeeded: 1,
            failed: 0,
            skipped: 0,
        }
    );
    assert_eq!(*checkin_hits.lock().expect("mutex should lock"), 1);
    assert_eq!(*balance_hits.lock().expect("mutex should lock"), 1);

    ops_handle.abort();
}

#[tokio::test]
async fn gateway_provider_checkin_counts_anyrouter_auto_signin_as_success() {
    fn sample_provider(provider_id: &str, ops_url: &str) -> StoredProviderCatalogProvider {
        StoredProviderCatalogProvider::new(
            provider_id.to_string(),
            "openai".to_string(),
            Some("https://example.com".to_string()),
            "custom".to_string(),
        )
        .expect("provider should build")
        .with_routing_fields(10)
        .with_transport_fields(
            true,
            false,
            true,
            None,
            None,
            None,
            None,
            None,
            Some(json!({
                "provider_ops": {
                    "architecture_id": "anyrouter",
                    "base_url": ops_url,
                    "connector": {
                        "auth_type": "cookie",
                        "config": {},
                        "credentials": {
                            "session_cookie": encrypt_python_fernet_plaintext(
                                DEVELOPMENT_ENCRYPTION_KEY,
                                "session=MTIzfGVIaDRlQUpwWkFOcGJuU3F1d0RfVkhsNWVYa0lkWE5sY201aGJXVUdjM1J5YVc1bkRCQUFCV0ZzYVdObHxzaWc",
                            ).expect("session cookie should encrypt"),
                        }
                    }
                }
            })),
        )
    }

    let sign_in_hits = Arc::new(Mutex::new(0usize));
    let sign_in_hits_clone = Arc::clone(&sign_in_hits);
    let balance_hits = Arc::new(Mutex::new(0usize));
    let balance_hits_clone = Arc::clone(&balance_hits);
    let ops = Router::new()
        .route(
            "/",
            get(|| async move {
                (
                    StatusCode::OK,
                    Body::from(
                        "<html><script>var arg1 = '0123456789abcdef0123456789abcdef01234567';</script></html>",
                    ),
                )
            }),
        )
        .route(
            "/api/user/sign_in",
            post(move |headers: axum::http::HeaderMap| {
                let sign_in_hits_inner = Arc::clone(&sign_in_hits_clone);
                async move {
                    *sign_in_hits_inner.lock().expect("mutex should lock") += 1;
                    let cookie = headers
                        .get(axum::http::header::COOKIE)
                        .and_then(|value| value.to_str().ok())
                        .expect("cookie header should exist");
                    assert!(cookie.contains("acw_sc__v2="));
                    assert!(cookie.contains(
                        "session=MTIzfGVIaDRlQUpwWkFOcGJuU3F1d0RfVkhsNWVYa0lkWE5sY201aGJXVUdjM1J5YVc1bkRCQUFCV0ZzYVdObHxzaWc"
                    ));
                    assert_eq!(
                        headers
                            .get("New-Api-User")
                            .and_then(|value| value.to_str().ok()),
                        Some("42")
                    );
                    (
                        StatusCode::OK,
                        Json(json!({
                            "success": true,
                            "message": "Anyrouter 签到成功",
                        })),
                    )
                }
            }),
        )
        .route(
            "/api/user/self",
            get(move |headers: axum::http::HeaderMap| {
                let balance_hits_inner = Arc::clone(&balance_hits_clone);
                async move {
                    *balance_hits_inner.lock().expect("mutex should lock") += 1;
                    let cookie = headers
                        .get(axum::http::header::COOKIE)
                        .and_then(|value| value.to_str().ok())
                        .expect("cookie header should exist");
                    assert!(cookie.contains("acw_sc__v2="));
                    assert_eq!(
                        headers
                            .get("New-Api-User")
                            .and_then(|value| value.to_str().ok()),
                        Some("42")
                    );
                    (
                        StatusCode::OK,
                        Json(json!({
                            "success": true,
                            "data": {
                                "quota": 2500000,
                                "used_quota": 500000
                            }
                        })),
                    )
                }
            }),
        );

    let (ops_url, ops_handle) = start_server(ops).await;
    let repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![sample_provider("provider-anyrouter", &ops_url)],
        vec![],
        vec![],
    ));
    let gateway_state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_provider_catalog_repository_for_tests(repository)
                .with_system_config_values_for_tests([
                    ("enable_provider_checkin".to_string(), json!(true)),
                    ("provider_checkin_time".to_string(), json!("01:05")),
                ])
                .with_encryption_key_for_tests(DEVELOPMENT_ENCRYPTION_KEY),
        );

    let summary = crate::maintenance::perform_provider_checkin_once(&gateway_state)
        .await
        .expect("provider checkin should succeed");

    assert_eq!(
        summary,
        ProviderCheckinRunSummary {
            attempted: 1,
            succeeded: 1,
            failed: 0,
            skipped: 0,
        }
    );
    assert_eq!(*sign_in_hits.lock().expect("mutex should lock"), 1);
    assert_eq!(*balance_hits.lock().expect("mutex should lock"), 1);

    ops_handle.abort();
}

#[tokio::test]
async fn gateway_provider_checkin_skips_when_disabled_via_system_config() {
    let repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![],
        vec![],
        vec![],
    ));
    let gateway_state = AppState::new()
        .expect("gateway state should build")
        .with_data_state_for_tests(
            crate::data::GatewayDataState::with_provider_catalog_repository_for_tests(repository)
                .with_system_config_values_for_tests([(
                    "enable_provider_checkin".to_string(),
                    json!(false),
                )]),
        );

    let summary = crate::maintenance::perform_provider_checkin_once(&gateway_state)
        .await
        .expect("provider checkin should short-circuit");

    assert_eq!(
        summary,
        ProviderCheckinRunSummary {
            attempted: 0,
            succeeded: 0,
            failed: 0,
            skipped: 0,
        }
    );
}
