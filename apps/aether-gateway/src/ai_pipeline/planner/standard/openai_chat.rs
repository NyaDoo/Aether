use serde_json::{Map, Value};
use tracing::warn;

use crate::gateway::ai_pipeline::planner::{
    OPENAI_CHAT_STREAM_PLAN_KIND, OPENAI_CHAT_SYNC_PLAN_KIND,
};
use crate::gateway::{
    AppState, GatewayControlDecision, GatewayControlSyncDecisionResponse, GatewayError,
    LocalExecutionRuntimeMissDiagnostic,
};

use crate::gateway::ai_pipeline::planner::standard::{
    convert_openai_chat_request_to_claude_request, convert_openai_chat_request_to_gemini_request,
    convert_openai_chat_request_to_openai_cli_request, extract_openai_text_content,
    parse_openai_tool_result_content,
};

#[path = "openai/chat/decision.rs"]
mod decision;
#[path = "openai/chat/plans.rs"]
mod plans;

use self::decision::{
    materialize_local_openai_chat_candidate_attempts,
    maybe_build_local_openai_chat_decision_payload_for_candidate, LocalOpenAiChatDecisionInput,
};
use self::plans::{
    build_local_openai_chat_miss_diagnostic, build_local_openai_chat_stream_plan_and_reports,
    build_local_openai_chat_sync_plan_and_reports, current_unix_secs,
    list_local_openai_chat_candidates, resolve_local_openai_chat_decision_input,
    set_local_openai_chat_miss_diagnostic,
};

pub(crate) async fn build_local_openai_chat_sync_plan_and_reports_for_kind(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<
    Vec<crate::gateway::ai_pipeline::planner::plan_builders::LocalSyncPlanAndReport>,
    GatewayError,
> {
    build_local_openai_chat_sync_plan_and_reports(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await
}

pub(crate) async fn build_local_openai_chat_stream_plan_and_reports_for_kind(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<
    Vec<crate::gateway::ai_pipeline::planner::plan_builders::LocalStreamPlanAndReport>,
    GatewayError,
> {
    build_local_openai_chat_stream_plan_and_reports(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await
}

pub(crate) fn set_local_openai_chat_execution_exhausted_diagnostic(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    body_json: &serde_json::Value,
    plan_count: usize,
) {
    state.set_local_execution_runtime_miss_diagnostic(
        trace_id,
        LocalExecutionRuntimeMissDiagnostic {
            candidate_count: Some(plan_count),
            ..build_local_openai_chat_miss_diagnostic(
                decision,
                plan_kind,
                body_json.get("model").and_then(|value| value.as_str()),
                "execution_runtime_candidates_exhausted",
            )
        },
    );
}

pub(crate) async fn maybe_build_sync_local_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    if plan_kind != OPENAI_CHAT_SYNC_PLAN_KIND {
        return Ok(None);
    }

    let Some(input) = resolve_local_openai_chat_decision_input(
        state, trace_id, decision, body_json, plan_kind, false,
    )
    .await
    else {
        return Ok(None);
    };

    let candidates = match list_local_openai_chat_candidates(state, &input, false).await {
        Ok(candidates) => candidates,
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local openai chat sync decision scheduler selection failed"
            );
            return Ok(None);
        }
    };

    let attempts =
        materialize_local_openai_chat_candidate_attempts(state, trace_id, &input, candidates).await;

    for attempt in attempts {
        if let Some(payload) = maybe_build_local_openai_chat_decision_payload_for_candidate(
            state,
            parts,
            trace_id,
            body_json,
            &input,
            attempt,
            OPENAI_CHAT_SYNC_PLAN_KIND,
            "openai_chat_sync_success",
            false,
        )
        .await
        {
            return Ok(Some(payload));
        }
    }

    Ok(None)
}

pub(crate) async fn maybe_build_stream_local_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    if plan_kind != OPENAI_CHAT_STREAM_PLAN_KIND {
        return Ok(None);
    }

    let Some(input) = resolve_local_openai_chat_decision_input(
        state, trace_id, decision, body_json, plan_kind, false,
    )
    .await
    else {
        return Ok(None);
    };

    let candidates = match list_local_openai_chat_candidates(state, &input, true).await {
        Ok(candidates) => candidates,
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local openai chat stream decision scheduler selection failed"
            );
            return Ok(None);
        }
    };

    let attempts =
        materialize_local_openai_chat_candidate_attempts(state, trace_id, &input, candidates).await;

    for attempt in attempts {
        if let Some(payload) = maybe_build_local_openai_chat_decision_payload_for_candidate(
            state,
            parts,
            trace_id,
            body_json,
            &input,
            attempt,
            OPENAI_CHAT_STREAM_PLAN_KIND,
            "openai_chat_stream_success",
            true,
        )
        .await
        {
            return Ok(Some(payload));
        }
    }

    Ok(None)
}
pub(crate) fn parse_openai_stop_sequences(stop: Option<&Value>) -> Option<Vec<Value>> {
    match stop {
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Some(vec![Value::String(value.clone())])
        }
        Some(Value::Array(values)) => Some(
            values
                .iter()
                .filter_map(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| Value::String(value.to_string()))
                .collect::<Vec<_>>(),
        )
        .filter(|values| !values.is_empty()),
        _ => None,
    }
}

pub(crate) fn resolve_openai_chat_max_tokens(request: &Map<String, Value>) -> u64 {
    request
        .get("max_completion_tokens")
        .and_then(value_as_u64)
        .or_else(|| request.get("max_tokens").and_then(value_as_u64))
        .unwrap_or(4096)
}

pub(crate) fn value_as_u64(value: &Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
}

pub(crate) fn copy_request_number_field(
    request: &Map<String, Value>,
    target: &mut Map<String, Value>,
    key: &str,
) {
    copy_request_number_field_as(request, target, key, key);
}

pub(crate) fn copy_request_number_field_as(
    request: &Map<String, Value>,
    target: &mut Map<String, Value>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = request.get(source_key).cloned() {
        if value.is_number() {
            target.insert(target_key.to_string(), value);
        }
    }
}

pub(crate) fn map_openai_reasoning_effort_to_claude_output(value: &str) -> Option<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some("low"),
        "medium" => Some("medium"),
        "high" | "xhigh" => Some("high"),
        _ => None,
    }
}

pub(crate) fn map_openai_reasoning_effort_to_gemini_budget(value: &str) -> Option<u64> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some(1024),
        "medium" => Some(4096),
        "high" => Some(8192),
        "xhigh" => Some(16_384),
        _ => None,
    }
}
