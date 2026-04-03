use axum::body::{Body, Bytes};
use axum::http::Response;

use crate::gateway::{AppState, GatewayControlDecision, GatewayError};

pub(crate) mod adapters;
pub(crate) mod provider_types;

pub(crate) async fn maybe_execute_sync_request(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &Bytes,
    trace_id: &str,
    decision: Option<&GatewayControlDecision>,
) -> Result<Option<Response<Body>>, GatewayError> {
    super::super::execution_runtime::maybe_execute_via_execution_runtime_sync(
        state, parts, body_bytes, trace_id, decision,
    )
    .await
}

pub(crate) async fn maybe_execute_stream_request(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &Bytes,
    trace_id: &str,
    decision: Option<&GatewayControlDecision>,
) -> Result<Option<Response<Body>>, GatewayError> {
    super::super::execution_runtime::maybe_execute_via_execution_runtime_stream(
        state, parts, body_bytes, trace_id, decision,
    )
    .await
}

pub(crate) use super::super::execution_runtime::{
    execute_execution_runtime_stream, execute_execution_runtime_sync,
    execute_execution_runtime_sync_plan,
};
pub(crate) use super::adaptation::private_envelope::{
    maybe_build_provider_private_stream_normalizer,
    maybe_normalize_provider_private_sync_report_payload,
    normalize_provider_private_report_context, normalize_provider_private_response_value,
    provider_private_response_allows_sync_finalize, transform_provider_private_stream_line,
};
pub(crate) use adapters::antigravity::{
    build_antigravity_safe_v1internal_request, build_antigravity_static_identity_headers,
    build_antigravity_v1internal_url, classify_local_antigravity_request_support,
    resolve_local_antigravity_request_auth, AntigravityEnvelopeRequestType,
    AntigravityRequestAuthSupport, AntigravityRequestEnvelopeSupport,
    AntigravityRequestSideSupport, AntigravityRequestUrlAction,
};
pub(crate) use adapters::claude::{
    build_claude_messages_url, build_passthrough_headers_with_auth, build_passthrough_path_url,
    resolve_local_standard_auth, supports_local_standard_transport_with_network,
};
pub(crate) use adapters::claude_code::{
    build_claude_code_messages_url, build_claude_code_passthrough_headers,
    sanitize_claude_code_request_body, supports_local_claude_code_transport_with_network,
};
pub(crate) use adapters::gemini::{
    build_gemini_content_url, build_gemini_files_passthrough_url,
    build_gemini_video_predict_long_running_url, resolve_local_gemini_auth,
};
#[cfg(test)]
pub(crate) use adapters::generic_oauth::GenericOAuthRefreshAdapter;
pub(crate) use adapters::kiro::KiroOAuthRefreshAdapter;
pub(crate) use adapters::kiro::{
    build_kiro_generate_assistant_response_url, build_kiro_provider_headers,
    build_kiro_provider_request_body, supports_local_kiro_request_transport_with_network,
    KiroAuthConfig, KiroRequestAuth, KiroToClaudeCliStreamState, KIRO_ENVELOPE_NAME,
};
pub(crate) use adapters::openai::{
    build_openai_chat_url, build_openai_cli_url, build_openai_passthrough_headers,
    resolve_local_openai_chat_auth, supports_local_openai_chat_transport,
};
pub(crate) use adapters::vertex::{
    build_vertex_api_key_gemini_content_url, resolve_local_vertex_api_key_query_auth,
    supports_local_vertex_api_key_gemini_transport_with_network,
};
pub(crate) use provider_types::{
    fixed_provider_template, is_codex_cli_backend_url, provider_type_admin_oauth_template,
    provider_type_enables_format_conversion_by_default, provider_type_is_fixed,
    provider_type_is_fixed_for_admin_oauth, provider_type_supports_local_openai_chat_transport,
    provider_type_supports_local_same_format_transport, provider_type_supports_model_fetch,
    ProviderOAuthTemplate, ADMIN_PROVIDER_OAUTH_TEMPLATE_TYPES,
};
