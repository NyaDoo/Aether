use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use aether_contracts::{ExecutionPlan, ExecutionResult, ExecutionTelemetry, ResponseBody};
use aether_crypto::{encrypt_python_fernet_plaintext, DEVELOPMENT_ENCRYPTION_KEY};
use aether_data::repository::provider_catalog::InMemoryProviderCatalogReadRepository;
use aether_data_contracts::repository::provider_catalog::{
    StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
};
use base64::Engine as _;
use http::StatusCode;
use serde_json::{json, Value};

use super::super::super::{
    build_router_with_state, sample_endpoint, sample_key, sample_provider, start_server, AppState,
};
use crate::constants::{
    GATEWAY_HEADER, TRUSTED_ADMIN_SESSION_ID_HEADER, TRUSTED_ADMIN_USER_ID_HEADER,
    TRUSTED_ADMIN_USER_ROLE_HEADER,
};
use crate::data::GatewayDataState;

const ASSET_LIBRARY_TEST_STACK_BYTES: usize = 16 * 1024 * 1024;

fn run_asset_library_test<F, Fut>(test_name: &'static str, make_future: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + 'static,
{
    let handle = std::thread::Builder::new()
        .name(test_name.to_string())
        .stack_size(ASSET_LIBRARY_TEST_STACK_BYTES)
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime should build");
            runtime.block_on(make_future());
        })
        .expect("asset library test thread should spawn");

    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

fn asset_key(id: &str, provider_id: &str, secret: &str) -> StoredProviderCatalogKey {
    let mut key = sample_key(id, provider_id, "doubao:asset_library", secret);
    key.auth_type = "bearer".to_string();
    key.capabilities = Some(json!({"ark_asset_library": true}));
    key
}

fn asset_endpoint(id: &str, provider_id: &str, base_url: &str) -> StoredProviderCatalogEndpoint {
    let mut endpoint = sample_endpoint(id, provider_id, "doubao:asset_library", base_url);
    endpoint.endpoint_kind = Some("asset_library".to_string());
    endpoint
}

fn k23_asset_endpoint(id: &str, provider_id: &str) -> StoredProviderCatalogEndpoint {
    let mut endpoint = asset_endpoint(id, provider_id, "https://ai.k23.cn");
    endpoint.custom_path = Some("/seedance/assets/".to_string());
    endpoint
}

fn api_key_asset_key(id: &str, provider_id: &str, secret: &str) -> StoredProviderCatalogKey {
    let mut key = sample_key(id, provider_id, "doubao:asset_library", secret);
    key.auth_type = "api_key".to_string();
    key.capabilities = Some(json!({"ark_asset_library": true}));
    key.upstream_metadata = Some(json!({"api_key_header": "api-key"}));
    key
}

fn aksk_asset_key(id: &str, provider_id: &str) -> StoredProviderCatalogKey {
    let mut key = sample_key(
        id,
        provider_id,
        "doubao:asset_library",
        "unused-placeholder",
    );
    key.auth_type = "volc_aksk".to_string();
    key.capabilities = Some(json!({"ark_asset_library": true}));
    key.encrypted_auth_config = Some(
        encrypt_python_fernet_plaintext(
            DEVELOPMENT_ENCRYPTION_KEY,
            r#"{"access_key_id":"AKLTK23EXAMPLE","secret_access_key":"k23-secret-example","region":"cn-beijing","service":"ark"}"#,
        )
        .expect("AK/SK auth config should encrypt"),
    );
    key
}

fn assert_k23_probe_contract(plan: &ExecutionPlan) {
    assert_eq!(plan.method, "POST");
    assert_eq!(
        plan.url,
        "https://ai.k23.cn/seedance/assets/?Action=ListAssetGroups&Version=2024-01-01"
    );
    assert_eq!(
        plan.headers.get("content-type").map(String::as_str),
        Some("application/json")
    );
    let body = base64::engine::general_purpose::STANDARD
        .decode(
            plan.body
                .body_bytes_b64
                .as_deref()
                .expect("request body should be encoded"),
        )
        .expect("request body should decode");
    let body: Value = serde_json::from_slice(&body).expect("request body should be json");
    assert_eq!(
        body,
        json!({
            "PageNumber": 1,
            "PageSize": 1,
            "Filter": {"GroupType": "AIGC"},
        })
    );
}

fn state_with_catalog_and_runtime<F>(
    providers: Vec<StoredProviderCatalogProvider>,
    endpoints: Vec<StoredProviderCatalogEndpoint>,
    keys: Vec<StoredProviderCatalogKey>,
    execute: F,
) -> AppState
where
    F: Fn(&ExecutionPlan) -> Result<ExecutionResult, crate::GatewayError> + Send + Sync + 'static,
{
    let repository = Arc::new(InMemoryProviderCatalogReadRepository::seed(
        providers, endpoints, keys,
    ));
    AppState::new()
        .expect("gateway should build")
        .with_execution_runtime_sync_override_for_tests(execute)
        .with_data_state_for_tests(
            GatewayDataState::with_provider_catalog_reader_for_tests(repository)
                .with_encryption_key_for_tests(DEVELOPMENT_ENCRYPTION_KEY),
        )
}

async fn post_asset_test(gateway_url: &str, key_id: &str, endpoint_id: &str) -> reqwest::Response {
    post_asset_test_payload(gateway_url, key_id, json!({"endpoint_id": endpoint_id})).await
}

async fn post_asset_test_payload(
    gateway_url: &str,
    key_id: &str,
    payload: Value,
) -> reqwest::Response {
    reqwest::Client::new()
        .post(format!(
            "{gateway_url}/api/admin/endpoints/keys/{key_id}/asset-library/test"
        ))
        .header(GATEWAY_HEADER, "rust-phase3b")
        .header(TRUSTED_ADMIN_USER_ID_HEADER, "admin-user-123")
        .header(TRUSTED_ADMIN_USER_ROLE_HEADER, "admin")
        .header(TRUSTED_ADMIN_SESSION_ID_HEADER, "session-123")
        .json(&payload)
        .send()
        .await
        .expect("request should succeed")
}

#[test]
fn asset_library_connection_test_uses_exact_saved_key_and_endpoint() {
    run_asset_library_test("asset-library-connection-test-exact-transport", || async {
        let plans = Arc::new(Mutex::new(Vec::<ExecutionPlan>::new()));
        let captured_plans = Arc::clone(&plans);
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-ark", "ark relay", 10)],
            vec![
                k23_asset_endpoint("endpoint-selected", "provider-ark"),
                asset_endpoint(
                    "endpoint-unused",
                    "provider-ark",
                    "https://unused.example.com/",
                ),
            ],
            vec![
                asset_key("key-selected", "provider-ark", "selected-secret"),
                asset_key("key-unused", "provider-ark", "unused-secret"),
            ],
            move |plan| {
                captured_plans
                    .lock()
                    .expect("plans should lock")
                    .push(plan.clone());
                Ok(ExecutionResult {
                    request_id: plan.request_id.clone(),
                    candidate_id: plan.candidate_id.clone(),
                    status_code: 200,
                    headers: BTreeMap::new(),
                    response_observation: None,
                    body: Some(ResponseBody {
                        json_body: Some(json!({
                            "ResponseMetadata": {
                                "RequestId": "20260328000000000000000000000000",
                                "Action": "ListAssetGroups",
                                "Version": "2024-01-01",
                                "Service": "ark",
                                "Region": "cn-beijing"
                            },
                            "Result": {
                                "TotalCount": 5,
                                "Items": [{
                                    "Id": "group-20260328123456-abcde",
                                    "Name": "产品宣传素材组",
                                    "Title": "产品宣传素材组",
                                    "Description": "存放产品宣传相关的参考图片",
                                    "GroupType": "AIGC",
                                    "ProjectName": "default",
                                    "CreateTime": "2026-03-28T12:34:56Z",
                                    "UpdateTime": "2026-03-28T12:34:56Z"
                                }],
                                "PageNumber": 1,
                                "PageSize": 10
                            },
                        })),
                        body_bytes_b64: None,
                    }),
                    telemetry: Some(ExecutionTelemetry {
                        ttfb_ms: Some(8),
                        elapsed_ms: Some(18),
                        upstream_bytes: Some(64),
                    }),
                    error: None,
                })
            },
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-selected", "endpoint-selected").await;
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["action"], "ListAssetGroups");
        assert_eq!(payload["provider_id"], "provider-ark");
        assert_eq!(payload["endpoint_id"], "endpoint-selected");
        assert_eq!(payload["key_id"], "key-selected");
        assert_eq!(payload["status_code"], 200);
        assert_eq!(payload["latency_ms"], 18);
        assert_eq!(payload["request_id"], "20260328000000000000000000000000");
        assert_eq!(payload["total"], 5);
        assert_eq!(payload["error_code"], Value::Null);
        assert_eq!(payload["error_message"], Value::Null);

        let plans = plans.lock().expect("plans should lock");
        assert_eq!(plans.len(), 1);
        let plan = &plans[0];
        assert_eq!(plan.provider_id, "provider-ark");
        assert_eq!(plan.endpoint_id, "endpoint-selected");
        assert_eq!(plan.key_id, "key-selected");
        assert_eq!(plan.client_api_format, "doubao:asset_library");
        assert_eq!(plan.provider_api_format, "doubao:asset_library");
        assert_k23_probe_contract(plan);
        assert_eq!(
            plan.headers.get("authorization").map(String::as_str),
            Some("Bearer selected-secret")
        );

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_preserves_api_key_relay_auth() {
    run_asset_library_test("asset-library-connection-test-api-key", || async {
        let captured_plan = Arc::new(Mutex::new(None::<ExecutionPlan>));
        let captured_plan_for_runtime = Arc::clone(&captured_plan);
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-k23", "k23 relay", 10)],
            vec![k23_asset_endpoint("endpoint-k23", "provider-k23")],
            vec![api_key_asset_key("key-k23", "provider-k23", "k23-api-key")],
            move |plan| {
                *captured_plan_for_runtime
                    .lock()
                    .expect("captured plan should lock") = Some(plan.clone());
                Ok(ExecutionResult {
                    request_id: plan.request_id.clone(),
                    candidate_id: plan.candidate_id.clone(),
                    status_code: 200,
                    headers: BTreeMap::new(),
                    response_observation: None,
                    body: Some(ResponseBody {
                        json_body: Some(json!({
                            "ResponseMetadata": {"RequestId": "k23-api-key-request"},
                            "Result": {"TotalCount": 0, "Items": []}
                        })),
                        body_bytes_b64: None,
                    }),
                    telemetry: None,
                    error: None,
                })
            },
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-k23", "endpoint-k23").await;
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["total"], 0);

        let captured_plan = captured_plan.lock().expect("captured plan should lock");
        let plan = captured_plan
            .as_ref()
            .expect("execution plan should be captured");
        assert_k23_probe_contract(plan);
        assert_eq!(
            plan.headers.get("api-key").map(String::as_str),
            Some("k23-api-key")
        );
        assert!(!plan.headers.contains_key("authorization"));

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_preserves_volc_aksk_auth() {
    run_asset_library_test("asset-library-connection-test-aksk", || async {
        let captured_plan = Arc::new(Mutex::new(None::<ExecutionPlan>));
        let captured_plan_for_runtime = Arc::clone(&captured_plan);
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-ark", "Volcengine Ark", 10)],
            vec![k23_asset_endpoint("endpoint-ark", "provider-ark")],
            vec![aksk_asset_key("key-ark", "provider-ark")],
            move |plan| {
                *captured_plan_for_runtime
                    .lock()
                    .expect("captured plan should lock") = Some(plan.clone());
                Ok(ExecutionResult {
                    request_id: plan.request_id.clone(),
                    candidate_id: plan.candidate_id.clone(),
                    status_code: 200,
                    headers: BTreeMap::new(),
                    response_observation: None,
                    body: Some(ResponseBody {
                        json_body: Some(json!({
                            "ResponseMetadata": {"RequestId": "ark-aksk-request"},
                            "Result": {"TotalCount": "2", "Items": []}
                        })),
                        body_bytes_b64: None,
                    }),
                    telemetry: None,
                    error: None,
                })
            },
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-ark", "endpoint-ark").await;
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["total"], 2);

        let captured_plan = captured_plan.lock().expect("captured plan should lock");
        let plan = captured_plan
            .as_ref()
            .expect("execution plan should be captured");
        assert_k23_probe_contract(plan);
        let authorization = plan
            .headers
            .get("authorization")
            .expect("AK/SK request should be signed");
        assert!(authorization.starts_with("HMAC-SHA256 Credential=AKLTK23EXAMPLE/"));
        assert!(authorization.contains("/cn-beijing/ark/request"));
        assert!(plan.headers.contains_key("x-date"));
        assert!(plan.headers.contains_key("x-content-sha256"));
        assert_eq!(
            plan.headers.get("host").map(String::as_str),
            Some("ai.k23.cn")
        );

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_accepts_items_with_null_total() {
    run_asset_library_test("asset-library-connection-test-items-null-total", || async {
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-ark", "ark relay", 10)],
            vec![asset_endpoint(
                "endpoint-ark",
                "provider-ark",
                "https://assets.example.com/",
            )],
            vec![asset_key("key-ark", "provider-ark", "provider-secret")],
            |plan| {
                Ok(ExecutionResult {
                    request_id: plan.request_id.clone(),
                    candidate_id: plan.candidate_id.clone(),
                    status_code: 200,
                    headers: BTreeMap::new(),
                    response_observation: None,
                    body: Some(ResponseBody {
                        json_body: Some(json!({
                            "ResponseMetadata": {"RequestId": "k23-request"},
                            "result": {"Total": null, "Items": []},
                        })),
                        body_bytes_b64: None,
                    }),
                    telemetry: Some(ExecutionTelemetry {
                        ttfb_ms: Some(4),
                        elapsed_ms: Some(9),
                        upstream_bytes: Some(48),
                    }),
                    error: None,
                })
            },
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-ark", "endpoint-ark").await;
        assert_eq!(response.status(), StatusCode::OK);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["success"], true, "payload: {payload}");
        assert_eq!(payload["request_id"], "k23-request");
        assert_eq!(payload["total"], Value::Null);

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_rejects_2xx_non_protocol_json() {
    run_asset_library_test("asset-library-connection-test-invalid-success", || async {
        let invalid_bodies = [
            json!({}),
            json!([]),
            json!({"message": "ok"}),
            json!({"Result": {}}),
            json!({"Result": {"Items": {}}}),
        ];

        for upstream_body in invalid_bodies {
            let state = state_with_catalog_and_runtime(
                vec![sample_provider("provider-ark", "ark relay", 10)],
                vec![asset_endpoint(
                    "endpoint-ark",
                    "provider-ark",
                    "https://assets.example.com/",
                )],
                vec![asset_key("key-ark", "provider-ark", "provider-secret")],
                move |plan| {
                    Ok(ExecutionResult {
                        request_id: plan.request_id.clone(),
                        candidate_id: plan.candidate_id.clone(),
                        status_code: 200,
                        headers: BTreeMap::new(),
                        response_observation: None,
                        body: Some(ResponseBody {
                            json_body: Some(upstream_body.clone()),
                            body_bytes_b64: None,
                        }),
                        telemetry: Some(ExecutionTelemetry {
                            ttfb_ms: Some(3),
                            elapsed_ms: Some(7),
                            upstream_bytes: Some(24),
                        }),
                        error: None,
                    })
                },
            );
            let gateway = build_router_with_state(state);
            let (gateway_url, gateway_handle) = start_server(gateway).await;

            let response = post_asset_test(&gateway_url, "key-ark", "endpoint-ark").await;
            assert_eq!(response.status(), StatusCode::OK);
            let payload: Value = response.json().await.expect("response should be json");
            assert_eq!(payload["success"], false, "payload: {payload}");
            assert_eq!(payload["error_code"], "InvalidUpstreamResponse");
            assert_eq!(
                payload["error_message"],
                "素材库上游返回的 ListAssetGroups 响应结构无效"
            );

            gateway_handle.abort();
        }
    });
}

#[test]
fn asset_library_connection_test_requires_endpoint_id() {
    run_asset_library_test(
        "asset-library-connection-test-required-endpoint",
        || async {
            let state = state_with_catalog_and_runtime(
                vec![sample_provider("provider-ark", "ark relay", 10)],
                vec![asset_endpoint(
                    "endpoint-ark",
                    "provider-ark",
                    "https://assets.example.com/",
                )],
                vec![asset_key("key-ark", "provider-ark", "provider-secret")],
                |_| panic!("missing endpoint_id must be rejected before execution"),
            );
            let gateway = build_router_with_state(state);
            let (gateway_url, gateway_handle) = start_server(gateway).await;

            let response = post_asset_test_payload(&gateway_url, "key-ark", json!({})).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let payload: Value = response.json().await.expect("response should be json");
            assert_eq!(
                payload["detail"],
                "请求体必须是合法的 JSON 对象，且 endpoint_id 为必填字符串字段"
            );

            gateway_handle.abort();
        },
    );
}

#[test]
fn asset_library_connection_test_reports_redirect_without_leaking_response_details() {
    run_asset_library_test("asset-library-connection-test-redirect", || async {
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-ark", "ark relay", 10)],
            vec![asset_endpoint(
                "endpoint-ark",
                "provider-ark",
                "https://assets.example.com/no-trailing-slash",
            )],
            vec![asset_key("key-ark", "provider-ark", "redirect-secret")],
            |plan| {
                Ok(ExecutionResult {
                    request_id: plan.request_id.clone(),
                    candidate_id: plan.candidate_id.clone(),
                    status_code: 307,
                    headers: BTreeMap::from([(
                        "location".to_string(),
                        "https://private.example.com/redirected".to_string(),
                    )]),
                    response_observation: None,
                    body: None,
                    telemetry: Some(ExecutionTelemetry {
                        ttfb_ms: Some(9),
                        elapsed_ms: Some(11),
                        upstream_bytes: Some(0),
                    }),
                    error: None,
                })
            },
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-ark", "endpoint-ark").await;
        assert_eq!(response.status(), StatusCode::OK);
        let response_body = response.text().await.expect("response text should read");
        assert!(!response_body.contains("redirect-secret"));
        assert!(!response_body.contains("private.example.com"));
        let payload: Value = serde_json::from_str(&response_body).expect("response should be json");
        assert_eq!(payload["success"], false);
        assert_eq!(payload["status_code"], 307, "payload: {payload}");
        assert_eq!(payload["latency_ms"], 11);
        assert_eq!(payload["error_code"], "UpstreamRedirect");
        assert!(payload["error_message"]
            .as_str()
            .is_some_and(|message| message.contains("Base URL") && message.contains("/")));

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_rejects_cross_provider_endpoint_without_fallback() {
    run_asset_library_test("asset-library-connection-test-cross-provider", || async {
        let state = state_with_catalog_and_runtime(
            vec![
                sample_provider("provider-a", "provider a", 10),
                sample_provider("provider-b", "provider b", 20),
            ],
            vec![
                asset_endpoint(
                    "endpoint-valid-a",
                    "provider-a",
                    "https://provider-a.example.com/",
                ),
                asset_endpoint(
                    "endpoint-requested-b",
                    "provider-b",
                    "https://provider-b.example.com/",
                ),
            ],
            vec![asset_key("key-a", "provider-a", "provider-a-secret")],
            |_| panic!("cross-provider request must not execute or fall back"),
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-a", "endpoint-requested-b").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["detail"], "Endpoint 与 Key 不属于同一个 Provider");

        gateway_handle.abort();
    });
}

#[test]
fn asset_library_connection_test_requires_explicit_capability() {
    run_asset_library_test("asset-library-connection-test-capability", || async {
        let mut key = asset_key("key-ark", "provider-ark", "provider-secret");
        key.capabilities = Some(json!({"ark_asset_library": false}));
        let state = state_with_catalog_and_runtime(
            vec![sample_provider("provider-ark", "ark relay", 10)],
            vec![asset_endpoint(
                "endpoint-ark",
                "provider-ark",
                "https://assets.example.com/",
            )],
            vec![key],
            |_| panic!("missing capability must be rejected before execution"),
        );
        let gateway = build_router_with_state(state);
        let (gateway_url, gateway_handle) = start_server(gateway).await;

        let response = post_asset_test(&gateway_url, "key-ark", "endpoint-ark").await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let payload: Value = response.json().await.expect("response should be json");
        assert_eq!(payload["detail"], "Key 缺少 ark_asset_library 能力");

        gateway_handle.abort();
    });
}
