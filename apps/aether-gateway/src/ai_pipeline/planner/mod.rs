use crate::gateway::{AppState, GatewayControlDecision, GatewayError};

pub(crate) mod candidate_affinity;
pub(crate) mod common;
mod decision;
pub(crate) mod passthrough;
pub(crate) mod plan_builders;
pub(crate) mod specialized;
pub(crate) mod standard;

pub(crate) use self::candidate_affinity::prefer_local_tunnel_owner_candidates;
pub(crate) use self::common::{
    parse_direct_request_body, CLAUDE_CHAT_STREAM_PLAN_KIND, CLAUDE_CHAT_SYNC_PLAN_KIND,
    CLAUDE_CLI_STREAM_PLAN_KIND, CLAUDE_CLI_SYNC_PLAN_KIND, EXECUTION_RUNTIME_STREAM_ACTION,
    EXECUTION_RUNTIME_STREAM_DECISION_ACTION, EXECUTION_RUNTIME_SYNC_ACTION,
    EXECUTION_RUNTIME_SYNC_DECISION_ACTION, GEMINI_CHAT_STREAM_PLAN_KIND,
    GEMINI_CHAT_SYNC_PLAN_KIND, GEMINI_CLI_STREAM_PLAN_KIND, GEMINI_CLI_SYNC_PLAN_KIND,
    GEMINI_FILES_DELETE_PLAN_KIND, GEMINI_FILES_DOWNLOAD_PLAN_KIND, GEMINI_FILES_GET_PLAN_KIND,
    GEMINI_FILES_LIST_PLAN_KIND, GEMINI_FILES_UPLOAD_PLAN_KIND, GEMINI_VIDEO_CANCEL_SYNC_PLAN_KIND,
    GEMINI_VIDEO_CREATE_SYNC_PLAN_KIND, OPENAI_CHAT_STREAM_PLAN_KIND, OPENAI_CHAT_SYNC_PLAN_KIND,
    OPENAI_CLI_STREAM_PLAN_KIND, OPENAI_CLI_SYNC_PLAN_KIND, OPENAI_COMPACT_STREAM_PLAN_KIND,
    OPENAI_COMPACT_SYNC_PLAN_KIND, OPENAI_VIDEO_CANCEL_SYNC_PLAN_KIND,
    OPENAI_VIDEO_CONTENT_PLAN_KIND, OPENAI_VIDEO_CREATE_SYNC_PLAN_KIND,
    OPENAI_VIDEO_DELETE_SYNC_PLAN_KIND, OPENAI_VIDEO_REMIX_SYNC_PLAN_KIND,
};
pub(crate) use crate::gateway::ai_pipeline::contracts::{
    build_gateway_plan_request, generic_decision_missing_exact_provider_request,
    GatewayControlPlanRequest, GatewayControlPlanResponse, GatewayControlSyncDecisionResponse,
};
pub(crate) use crate::gateway::ai_pipeline::conversion::request::{
    convert_openai_chat_request_to_claude_request, convert_openai_chat_request_to_gemini_request,
    convert_openai_chat_request_to_openai_cli_request, extract_openai_text_content,
    normalize_openai_cli_request_to_openai_chat_request, parse_openai_tool_result_content,
};
pub(crate) use crate::gateway::scheduler::{
    is_matching_stream_request,
    resolve_execution_runtime_stream_plan_kind as resolve_stream_plan_kind,
    resolve_execution_runtime_sync_plan_kind as resolve_sync_plan_kind,
};
pub(crate) use passthrough::{
    maybe_build_stream_local_same_format_provider_decision_payload,
    maybe_build_sync_local_same_format_provider_decision_payload,
};
pub(crate) use specialized::{
    maybe_build_stream_local_gemini_files_decision_payload,
    maybe_build_sync_local_gemini_files_decision_payload,
    maybe_build_sync_local_video_decision_payload,
};
pub(crate) use standard::{
    copy_request_number_field, copy_request_number_field_as,
    map_openai_reasoning_effort_to_claude_output, map_openai_reasoning_effort_to_gemini_budget,
    maybe_build_stream_local_decision_payload,
    maybe_build_stream_local_openai_cli_decision_payload,
    maybe_build_stream_local_standard_decision_payload, maybe_build_sync_local_decision_payload,
    maybe_build_sync_local_openai_cli_decision_payload,
    maybe_build_sync_local_standard_decision_payload, parse_openai_stop_sequences,
    resolve_openai_chat_max_tokens, value_as_u64,
};

pub(crate) async fn maybe_build_sync_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    decision::maybe_build_sync_decision_payload(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        body_base64,
        body_is_empty,
    )
    .await
}

pub(crate) async fn maybe_build_stream_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    decision::maybe_build_stream_decision_payload(state, parts, trace_id, decision, body_json).await
}

pub(crate) async fn maybe_build_sync_plan_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
) -> Result<Option<GatewayControlPlanResponse>, GatewayError> {
    decision::maybe_build_sync_plan_payload_impl(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        body_base64,
        body_is_empty,
    )
    .await
}

pub(crate) async fn maybe_build_stream_plan_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Result<Option<GatewayControlPlanResponse>, GatewayError> {
    decision::maybe_build_stream_plan_payload_impl(state, parts, trace_id, decision, body_json)
        .await
}
