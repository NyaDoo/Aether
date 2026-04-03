use std::collections::BTreeMap;

use axum::body::Body;
use axum::http::Response;
use serde_json::Value;

use crate::gateway::ai_pipeline::contracts::core_success_background_report_kind;
pub(crate) use crate::gateway::ai_pipeline::runtime::{
    normalize_provider_private_response_value as unwrap_local_finalize_response_value,
    provider_private_response_allows_sync_finalize as local_finalize_allows_envelope,
};
use crate::gateway::api::response::build_client_response_from_parts;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) struct LocalCoreSyncFinalizeOutcome {
    pub(crate) response: Response<Body>,
    pub(crate) background_report: Option<GatewaySyncReportRequest>,
}

pub(crate) fn build_local_success_outcome(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
    body_json: Value,
) -> Result<LocalCoreSyncFinalizeOutcome, GatewayError> {
    let headers = payload.headers.clone();
    let background_report =
        map_local_finalize_to_success_report(payload, body_json.clone(), headers.clone());
    build_local_success_outcome_with_report(
        trace_id,
        decision,
        payload.status_code,
        body_json,
        headers,
        background_report,
    )
}

pub(crate) fn build_local_success_outcome_with_report(
    trace_id: &str,
    decision: &GatewayControlDecision,
    status_code: u16,
    body_json: Value,
    mut headers: BTreeMap<String, String>,
    background_report: Option<GatewaySyncReportRequest>,
) -> Result<LocalCoreSyncFinalizeOutcome, GatewayError> {
    headers.remove("content-encoding");
    headers.remove("content-length");
    headers.insert("content-type".to_string(), "application/json".to_string());
    let body_bytes =
        serde_json::to_vec(&body_json).map_err(|err| GatewayError::Internal(err.to_string()))?;
    headers.insert("content-length".to_string(), body_bytes.len().to_string());
    let response = build_client_response_from_parts(
        status_code,
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?;
    Ok(LocalCoreSyncFinalizeOutcome {
        response,
        background_report,
    })
}

pub(crate) fn build_local_success_outcome_with_conversion_report(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
    client_body_json: Value,
    provider_body_json: Value,
) -> Result<LocalCoreSyncFinalizeOutcome, GatewayError> {
    let Some(report_kind) =
        map_local_finalize_kind_to_success_report_kind(payload.report_kind.as_str())
    else {
        return build_local_success_outcome_with_report(
            trace_id,
            decision,
            payload.status_code,
            client_body_json,
            payload.headers.clone(),
            None,
        );
    };

    let report_payload = GatewaySyncReportRequest {
        trace_id: payload.trace_id.clone(),
        report_kind: report_kind.to_string(),
        report_context: payload.report_context.clone(),
        status_code: payload.status_code,
        headers: payload.headers.clone(),
        body_json: Some(provider_body_json),
        client_body_json: Some(client_body_json.clone()),
        body_base64: None,
        telemetry: payload.telemetry.clone(),
    };

    build_local_success_outcome_with_report(
        trace_id,
        decision,
        payload.status_code,
        client_body_json,
        payload.headers.clone(),
        Some(report_payload),
    )
}

fn map_local_finalize_to_success_report(
    payload: &GatewaySyncReportRequest,
    body_json: Value,
    headers: BTreeMap<String, String>,
) -> Option<GatewaySyncReportRequest> {
    let report_kind = map_local_finalize_kind_to_success_report_kind(payload.report_kind.as_str())?;

    Some(GatewaySyncReportRequest {
        trace_id: payload.trace_id.clone(),
        report_kind: report_kind.to_string(),
        report_context: payload.report_context.clone(),
        status_code: payload.status_code,
        headers,
        body_json: Some(body_json),
        client_body_json: None,
        body_base64: None,
        telemetry: payload.telemetry.clone(),
    })
}

fn map_local_finalize_kind_to_success_report_kind(report_kind: &str) -> Option<&'static str> {
    core_success_background_report_kind(report_kind)
}

pub(crate) fn canonicalize_tool_arguments(value: Option<Value>) -> String {
    match value {
        Some(Value::String(text)) => text,
        Some(other) => serde_json::to_string(&other).unwrap_or_else(|_| "null".to_string()),
        None => "{}".to_string(),
    }
}

pub(crate) fn build_generated_tool_call_id(index: usize) -> String {
    format!("call_auto_{index}")
}

pub(crate) fn parse_stream_json_events(body: &[u8]) -> Option<Vec<Value>> {
    let text = std::str::from_utf8(body).ok()?;
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some(Vec::new());
    }

    if trimmed.starts_with('[') {
        let array_value: Value = serde_json::from_str(trimmed).ok()?;
        let array = array_value.as_array()?;
        return Some(
            array
                .iter()
                .filter(|value| value.is_object())
                .cloned()
                .collect(),
        );
    }

    let mut events = Vec::new();
    let mut current_event_type: Option<String> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim_matches('\r').trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(event_name) = line.strip_prefix("event:") {
            current_event_type = Some(event_name.trim().to_string());
            continue;
        }
        let data_line = if let Some(rest) = line.strip_prefix("data:") {
            rest.trim()
        } else {
            line
        };
        if data_line.is_empty() || data_line == "[DONE]" {
            continue;
        }

        let mut event: Value = serde_json::from_str(data_line).ok()?;
        if let Some(event_object) = event.as_object_mut() {
            if !event_object.contains_key("type") {
                if let Some(event_name) = current_event_type.take() {
                    event_object.insert("type".to_string(), Value::String(event_name));
                }
            }
        }
        events.push(event);
        current_event_type = None;
    }

    Some(events)
}
