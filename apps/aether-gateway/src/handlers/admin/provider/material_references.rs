use crate::data::{AssetProviderReference, AssetProviderReferenceCounts};
use crate::handlers::admin::request::AdminAppState;
use crate::GatewayError;
use axum::{
    body::Body,
    http,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub(crate) async fn count_material_references(
    state: &AdminAppState<'_>,
    reference: AssetProviderReference<'_>,
) -> Result<AssetProviderReferenceCounts, GatewayError> {
    state
        .as_ref()
        .data
        .count_asset_provider_references(reference)
        .await
        .map_err(|error| GatewayError::Internal(error.to_string()))
}

pub(crate) fn material_reference_conflict_message(
    resource_kind: &str,
    resource_id: &str,
    counts: AssetProviderReferenceCounts,
) -> String {
    format!(
        "{resource_kind} {resource_id} 仍被素材库引用（素材组 {} 个，真人验证会话 {} 个），请保留该资源以维持素材审计与后续访问",
        counts.asset_groups, counts.visual_validation_sessions
    )
}

pub(crate) fn material_reference_conflict_response(
    resource_kind: &str,
    resource_id: &str,
    counts: AssetProviderReferenceCounts,
) -> Response<Body> {
    (
        http::StatusCode::CONFLICT,
        Json(json!({
            "detail": material_reference_conflict_message(resource_kind, resource_id, counts),
            "code": "asset_library_reference_conflict",
            "references": {
                "asset_groups": counts.asset_groups,
                "visual_validation_sessions": counts.visual_validation_sessions,
                "total": counts.total(),
            }
        })),
    )
        .into_response()
}
