use std::collections::BTreeMap;

use aether_contracts::{ExecutionPlan, ExecutionTimeouts, ProxySnapshot};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::gateway::headers::collect_control_headers;
use crate::gateway::{AppState, GatewayControlAuthContext, GatewayControlDecision, GatewayError};

#[derive(Debug, Serialize)]
pub(crate) struct GatewayControlPlanRequest {
    pub(crate) trace_id: String,
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) query_string: Option<String>,
    pub(crate) headers: BTreeMap<String, String>,
    pub(crate) body_json: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) body_base64: Option<String>,
    pub(crate) auth_context: Option<GatewayControlAuthContext>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct GatewayControlPlanResponse {
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) plan_kind: Option<String>,
    #[serde(default)]
    pub(crate) plan: Option<ExecutionPlan>,
    #[serde(default)]
    pub(crate) report_kind: Option<String>,
    #[serde(default)]
    pub(crate) report_context: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) auth_context: Option<GatewayControlAuthContext>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct GatewayControlSyncDecisionResponse {
    pub(crate) action: String,
    #[serde(default)]
    pub(crate) decision_kind: Option<String>,
    #[serde(default)]
    pub(crate) execution_strategy: Option<String>,
    #[serde(default)]
    pub(crate) conversion_mode: Option<String>,
    #[serde(default)]
    pub(crate) request_id: Option<String>,
    #[serde(default)]
    pub(crate) candidate_id: Option<String>,
    #[serde(default)]
    pub(crate) provider_name: Option<String>,
    #[serde(default)]
    pub(crate) provider_id: Option<String>,
    #[serde(default)]
    pub(crate) endpoint_id: Option<String>,
    #[serde(default)]
    pub(crate) key_id: Option<String>,
    #[serde(default)]
    pub(crate) upstream_base_url: Option<String>,
    #[serde(default)]
    pub(crate) upstream_url: Option<String>,
    #[serde(default)]
    pub(crate) provider_request_method: Option<String>,
    #[serde(default)]
    pub(crate) auth_header: Option<String>,
    #[serde(default)]
    pub(crate) auth_value: Option<String>,
    #[serde(default)]
    pub(crate) provider_api_format: Option<String>,
    #[serde(default)]
    pub(crate) client_api_format: Option<String>,
    #[serde(default)]
    pub(crate) provider_contract: Option<String>,
    #[serde(default)]
    pub(crate) client_contract: Option<String>,
    #[serde(default)]
    pub(crate) model_name: Option<String>,
    #[serde(default)]
    pub(crate) mapped_model: Option<String>,
    #[serde(default)]
    pub(crate) prompt_cache_key: Option<String>,
    #[serde(default)]
    pub(crate) extra_headers: BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) provider_request_headers: BTreeMap<String, String>,
    #[serde(default)]
    pub(crate) provider_request_body: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) provider_request_body_base64: Option<String>,
    #[serde(default)]
    pub(crate) content_type: Option<String>,
    #[serde(default)]
    pub(crate) proxy: Option<ProxySnapshot>,
    #[serde(default)]
    pub(crate) tls_profile: Option<String>,
    #[serde(default)]
    pub(crate) timeouts: Option<ExecutionTimeouts>,
    #[serde(default)]
    pub(crate) upstream_is_stream: bool,
    #[serde(default)]
    pub(crate) report_kind: Option<String>,
    #[serde(default)]
    pub(crate) report_context: Option<serde_json::Value>,
    #[serde(default)]
    pub(crate) auth_context: Option<GatewayControlAuthContext>,
}

fn decision_has_exact_provider_request(payload: &GatewayControlSyncDecisionResponse) -> bool {
    !payload.provider_request_headers.is_empty()
        && (payload.provider_request_body.is_some()
            || payload
                .provider_request_body_base64
                .as_ref()
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false))
}

pub(crate) fn generic_decision_missing_exact_provider_request(
    payload: &GatewayControlSyncDecisionResponse,
) -> bool {
    if decision_has_exact_provider_request(payload) {
        return false;
    }

    warn!(
        decision_kind = payload.decision_kind.as_deref().unwrap_or_default(),
        provider_api_format = payload.provider_api_format.as_deref().unwrap_or_default(),
        client_api_format = payload.client_api_format.as_deref().unwrap_or_default(),
        "gateway generic decision missing exact provider request; falling back to plan"
    );
    true
}

pub(crate) async fn build_gateway_plan_request(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: serde_json::Value,
    body_base64: Option<String>,
) -> Result<GatewayControlPlanRequest, GatewayError> {
    let auth_context = crate::gateway::resolve_execution_runtime_auth_context(
        state,
        decision,
        &parts.headers,
        &parts.uri,
        trace_id,
    )
    .await?;

    Ok(GatewayControlPlanRequest {
        trace_id: trace_id.to_string(),
        method: parts.method.to_string(),
        path: parts.uri.path().to_string(),
        query_string: parts.uri.query().map(ToOwned::to_owned),
        headers: collect_control_headers(&parts.headers),
        body_json,
        body_base64,
        auth_context,
    })
}
