use crate::gateway::{
    AppState, GatewayControlDecision, GatewayControlSyncDecisionResponse, GatewayError,
};

use super::family::{
    maybe_build_stream_via_standard_family_payload, maybe_build_sync_via_standard_family_payload,
};
pub(crate) use crate::gateway::ai_pipeline::conversion::request::normalize_claude_request_to_openai_chat_request;

pub(crate) mod chat;
pub(crate) mod cli;

pub(crate) async fn maybe_build_sync_local_claude_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    maybe_build_sync_via_standard_family_payload(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            chat::resolve_sync_spec(plan_kind).or_else(|| cli::resolve_sync_spec(plan_kind))
        },
    )
    .await
}

pub(crate) async fn maybe_build_stream_local_claude_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    maybe_build_stream_via_standard_family_payload(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            chat::resolve_stream_spec(plan_kind).or_else(|| cli::resolve_stream_spec(plan_kind))
        },
    )
    .await
}
