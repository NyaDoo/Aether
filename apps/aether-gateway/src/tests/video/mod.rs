use std::sync::{Arc, Mutex};

use aether_crypto::{encrypt_python_fernet_plaintext, DEVELOPMENT_ENCRYPTION_KEY};
use aether_data::repository::auth::{
    InMemoryAuthApiKeySnapshotRepository, StoredAuthApiKeySnapshot,
};
use aether_data::repository::provider_catalog::InMemoryProviderCatalogReadRepository;
use aether_data::repository::video_tasks::InMemoryVideoTaskRepository;
use aether_data_contracts::repository::provider_catalog::{
    StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
};
use aether_data_contracts::repository::video_tasks::{
    UpsertVideoTask, VideoTaskLookupKey, VideoTaskReadRepository, VideoTaskWriteRepository,
};
use axum::body::{to_bytes, Body, Bytes};
use axum::response::Response;
use axum::routing::any;
use axum::{extract::Request, Json, Router};
use http::header::{HeaderName, HeaderValue};
use http::StatusCode;
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::constants::{
    CONTROL_EXECUTED_HEADER, CONTROL_EXECUTE_FALLBACK_HEADER, EXECUTION_PATH_HEADER,
    TRACE_ID_HEADER,
};

use super::{
    build_router, build_router_with_state, build_state_with_execution_runtime_override,
    start_server, AppState, VideoTaskTruthSourceMode,
};

fn hash_api_key(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub(super) fn video_auth_repository(
    raw_api_key: &str,
    api_key_id: &str,
    user_id: &str,
    provider: &str,
    api_format: &str,
    model: &str,
) -> Arc<InMemoryAuthApiKeySnapshotRepository> {
    let snapshot = StoredAuthApiKeySnapshot::new(
        user_id.to_string(),
        "video-user".to_string(),
        Some("video@example.com".to_string()),
        "user".to_string(),
        "local".to_string(),
        true,
        false,
        Some(json!([provider])),
        Some(json!([api_format])),
        Some(json!([model])),
        api_key_id.to_string(),
        Some("video-key".to_string()),
        true,
        false,
        false,
        Some(60),
        Some(5),
        Some(4_102_444_800),
        Some(json!([provider])),
        Some(json!([api_format])),
        Some(json!([model])),
    )
    .expect("video auth snapshot should build");

    Arc::new(InMemoryAuthApiKeySnapshotRepository::seed(vec![(
        Some(hash_api_key(raw_api_key)),
        snapshot,
    )]))
}

pub(super) fn simple_video_provider_catalog_repository(
    provider_id: &str,
    endpoint_id: &str,
    key_id: &str,
    family: &str,
    api_format: &str,
    base_url: &str,
    upstream_key: &str,
) -> Arc<InMemoryProviderCatalogReadRepository> {
    let provider = StoredProviderCatalogProvider::new(
        provider_id.to_string(),
        family.to_string(),
        Some(base_url.to_string()),
        "custom".to_string(),
    )
    .expect("video provider should build");
    let endpoint = StoredProviderCatalogEndpoint::new(
        endpoint_id.to_string(),
        provider_id.to_string(),
        api_format.to_string(),
        Some(family.to_string()),
        Some("video".to_string()),
        true,
    )
    .expect("video endpoint should build")
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
    .expect("video endpoint transport should build");
    let key = StoredProviderCatalogKey::new(
        key_id.to_string(),
        provider_id.to_string(),
        "prod".to_string(),
        "api_key".to_string(),
        None,
        true,
    )
    .expect("video key should build")
    .with_transport_fields(
        Some(json!([api_format])),
        encrypt_python_fernet_plaintext(DEVELOPMENT_ENCRYPTION_KEY, upstream_key)
            .expect("video key should encrypt"),
        None,
        None,
        Some(json!({ (api_format): 1 })),
        None,
        None,
        None,
        None,
    )
    .expect("video key transport should build");
    Arc::new(InMemoryProviderCatalogReadRepository::seed(
        vec![provider],
        vec![endpoint],
        vec![key],
    ))
}

mod data_read;
mod gemini_sync_create;
mod gemini_sync_task;
mod openai_sync_create;
mod openai_sync_task;
mod registry_poller;
mod routing;
mod stream;
