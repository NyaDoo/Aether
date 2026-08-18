use std::sync::Arc;

use aether_data::repository::asset_library::InMemoryAssetLibraryRepository;
use aether_data_contracts::repository::asset_library::{UpsertAssetGroupRecord, UpsertAssetRecord};
use aether_provider_transport::snapshot::{
    GatewayProviderTransportEndpoint, GatewayProviderTransportKey,
    GatewayProviderTransportProvider, GatewayProviderTransportSnapshot,
};
use serde_json::{json, Value};

use crate::data::GatewayDataState;
use crate::material_assets::project_video_asset_references;
use crate::AppState;

const USER_ID: &str = "user-asset-owner";
const PROVIDER_ID: &str = "provider-ark";
const ASSET_ENDPOINT_ID: &str = "endpoint-ark-assets";
const ASSET_KEY_ID: &str = "key-ark-aksk";
const VIDEO_ENDPOINT_ID: &str = "endpoint-seedance-video";
const VIDEO_KEY_ID: &str = "key-seedance-bearer";

struct ProjectionFixture {
    state: AppState,
    transport: GatewayProviderTransportSnapshot,
}

impl ProjectionFixture {
    async fn new() -> Self {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(GatewayDataState::with_asset_library_repository_for_tests(
                repository,
            ));

        Self {
            state,
            transport: test_transport(),
        }
    }

    async fn insert_asset(&self, asset_id: &str, user_id: &str, status: &str) {
        self.insert_asset_with_binding(
            asset_id,
            user_id,
            status,
            PROVIDER_ID,
            ASSET_ENDPOINT_ID,
            ASSET_KEY_ID,
        )
        .await;
    }

    async fn insert_asset_with_binding(
        &self,
        asset_id: &str,
        user_id: &str,
        status: &str,
        provider_id: &str,
        endpoint_id: &str,
        key_id: &str,
    ) {
        let writer = self
            .state
            .data
            .asset_library_write_repository()
            .expect("asset library writer should be configured");
        let group_id = format!("group-{asset_id}");
        writer
            .upsert_group(UpsertAssetGroupRecord {
                id: group_id.clone(),
                upstream_group_id: Some(format!("upstream-{group_id}")),
                user_id: user_id.to_string(),
                api_key_id: None,
                provider_id: provider_id.to_string(),
                endpoint_id: endpoint_id.to_string(),
                key_id: key_id.to_string(),
                group_type: "character".to_string(),
                name: format!("Group for {asset_id}"),
                description: None,
                status: "Active".to_string(),
                created_at_unix_secs: 1,
                updated_at_unix_secs: 1,
                deleted_at_unix_secs: None,
            })
            .await
            .expect("asset group should be inserted");
        writer
            .upsert_asset(UpsertAssetRecord {
                id: asset_id.to_string(),
                upstream_asset_id: Some(format!("upstream-{asset_id}")),
                group_id,
                user_id: user_id.to_string(),
                api_key_id: None,
                asset_type: "image".to_string(),
                name: format!("Asset {asset_id}"),
                status: status.to_string(),
                error_code: None,
                error_message: None,
                moderation: None,
                last_inference_at_unix_secs: None,
                source_url_fingerprint: None,
                provider_url: None,
                provider_url_expires_at_unix_secs: None,
                sanitized_metadata: None,
                is_deleted: false,
                deleted_at_unix_secs: None,
                created_at_unix_secs: 1,
                updated_at_unix_secs: 1,
            })
            .await
            .expect("asset should be inserted");
    }
}

#[tokio::test]
async fn video_asset_projection_replaces_nested_owned_active_asset_reference() {
    let fixture = ProjectionFixture::new().await;
    fixture
        .insert_asset("asset-success", USER_ID, "Active")
        .await;
    let body = json!({
        "model": "Doubao-Seedance-2.0",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "asset://asset-success"},
                "role": "reference_image"
            },
            {"type": "text", "text": "Keep this prompt unchanged"}
        ]
    });

    let projected =
        project_video_asset_references(&fixture.state, USER_ID, &fixture.transport, &body)
            .await
            .expect("owned Active asset should be projected");

    assert_eq!(
        projected["content"][0]["image_url"]["url"],
        "asset://upstream-asset-success"
    );
    assert_eq!(
        projected["content"][1]["text"],
        "Keep this prompt unchanged"
    );
    assert_eq!(projected["model"], "Doubao-Seedance-2.0");
}

#[tokio::test]
async fn video_asset_projection_rejects_asset_owned_by_another_user() {
    let fixture = ProjectionFixture::new().await;
    fixture
        .insert_asset("asset-other-owner", "user-other", "Active")
        .await;

    let error = project_video_asset_references(
        &fixture.state,
        USER_ID,
        &fixture.transport,
        &body_for("asset-other-owner"),
    )
    .await
    .expect_err("cross-user asset reference must fail closed");

    assert!(error.contains("不存在或不属于当前用户"), "{error}");
}

#[tokio::test]
async fn video_asset_projection_rejects_non_active_asset() {
    let fixture = ProjectionFixture::new().await;
    fixture
        .insert_asset("asset-processing", USER_ID, "Processing")
        .await;

    let error = project_video_asset_references(
        &fixture.state,
        USER_ID,
        &fixture.transport,
        &body_for("asset-processing"),
    )
    .await
    .expect_err("non-Active asset reference must fail closed");

    assert!(error.contains("必须为 Active"), "{error}");
}

#[tokio::test]
async fn video_asset_projection_rejects_provider_mismatch() {
    let fixture = ProjectionFixture::new().await;
    fixture
        .insert_asset_with_binding(
            "asset-provider-mismatch",
            USER_ID,
            "Active",
            "provider-other",
            ASSET_ENDPOINT_ID,
            ASSET_KEY_ID,
        )
        .await;

    let error = project_video_asset_references(
        &fixture.state,
        USER_ID,
        &fixture.transport,
        &body_for("asset-provider-mismatch"),
    )
    .await
    .expect_err("provider binding must match");

    assert!(error.contains("Provider 不一致"), "{error}");
}

#[tokio::test]
async fn video_asset_projection_preserves_optional_project_name() {
    let fixture = ProjectionFixture::new().await;
    fixture
        .insert_asset("asset-project-override", USER_ID, "Active")
        .await;
    let mut body = body_for("asset-project-override");
    body["ProjectName"] = json!("project-overridden-by-request");

    let projected =
        project_video_asset_references(&fixture.state, USER_ID, &fixture.transport, &body)
            .await
            .expect("ProjectName is an upstream request field, not an Aether ownership binding");

    assert_eq!(projected["ProjectName"], "project-overridden-by-request");
    assert_eq!(
        projected["content"][0]["image_url"]["url"],
        "asset://upstream-asset-project-override"
    );
}

#[tokio::test]
async fn video_asset_projection_ignores_references_outside_content() {
    let fixture = ProjectionFixture::new().await;
    let body = json!({
        "model": "asset://asset-outside-content",
        "content": [{"type": "text", "text": "ordinary prompt"}]
    });

    let projected =
        project_video_asset_references(&fixture.state, USER_ID, &fixture.transport, &body)
            .await
            .expect("only content is part of the material asset reference protocol");

    assert_eq!(projected, body);
}

fn body_for(asset_id: &str) -> Value {
    json!({
        "model": "Doubao-Seedance-2.0",
        "content": [{
            "type": "image_url",
            "image_url": {"url": format!("asset://{asset_id}")},
            "role": "reference_image"
        }]
    })
}

fn test_transport() -> GatewayProviderTransportSnapshot {
    GatewayProviderTransportSnapshot {
        provider: GatewayProviderTransportProvider {
            id: PROVIDER_ID.to_string(),
            name: "Volcengine Ark".to_string(),
            provider_type: "volcengine".to_string(),
            website: None,
            is_active: true,
            keep_priority_on_conversion: false,
            enable_format_conversion: true,
            concurrent_limit: None,
            max_retries: None,
            proxy: None,
            request_timeout_secs: None,
            stream_first_byte_timeout_secs: None,
            config: None,
        },
        endpoint: GatewayProviderTransportEndpoint {
            id: VIDEO_ENDPOINT_ID.to_string(),
            provider_id: PROVIDER_ID.to_string(),
            api_format: "doubao:video".to_string(),
            api_family: Some("video".to_string()),
            endpoint_kind: Some("video_generation".to_string()),
            is_active: true,
            base_url: "https://ark.cn-beijing.volces.com".to_string(),
            header_rules: None,
            body_rules: None,
            max_retries: None,
            custom_path: None,
            config: None,
            format_acceptance_config: None,
            proxy: None,
        },
        key: GatewayProviderTransportKey {
            id: VIDEO_KEY_ID.to_string(),
            provider_id: PROVIDER_ID.to_string(),
            name: "Seedance relay Bearer".to_string(),
            auth_type: "bearer".to_string(),
            is_active: true,
            api_formats: Some(vec!["doubao:video".to_string()]),
            auth_type_by_format: None,
            allow_auth_channel_mismatch_formats: None,
            allowed_models: None,
            capabilities: None,
            rate_multipliers: None,
            global_priority_by_format: None,
            expires_at_unix_secs: None,
            proxy: None,
            fingerprint: None,
            upstream_metadata: None,
            decrypted_api_key: "relay-secret".to_string(),
            decrypted_auth_config: None,
        },
    }
}
