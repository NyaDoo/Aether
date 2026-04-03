use std::collections::BTreeMap;

use aether_contracts::ExecutionPlan;
use axum::body::Body;
use axum::http::Response;
use serde_json::json;

use crate::gateway::api::response::build_client_response_from_parts;
use crate::gateway::video_tasks::{LocalVideoTaskSnapshot, VideoTaskSyncReportMode};
use crate::gateway::VideoTaskService;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) struct LocalVideoSyncSuccessOutcome {
    pub(crate) response: Response<Body>,
    pub(crate) report_payload: GatewaySyncReportRequest,
    pub(crate) report_mode: VideoTaskSyncReportMode,
    pub(crate) local_task_snapshot: Option<LocalVideoTaskSnapshot>,
}

fn cloned_report_context_object(
    payload: &GatewaySyncReportRequest,
) -> serde_json::Map<String, serde_json::Value> {
    payload
        .report_context
        .clone()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn build_local_video_success_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Result<Response<Body>, GatewayError> {
    let body_bytes =
        serde_json::to_vec(body_json).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let mut headers = BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("content-length".to_string(), body_bytes.len().to_string());
    build_client_response_from_parts(
        http::StatusCode::OK.as_u16(),
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )
}

pub(crate) fn maybe_build_local_video_success_outcome(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
    video_tasks: &VideoTaskService,
    plan: &ExecutionPlan,
) -> Result<Option<LocalVideoSyncSuccessOutcome>, GatewayError> {
    if payload.status_code >= 400 {
        return Ok(None);
    }

    let provider_body = match payload
        .body_json
        .as_ref()
        .and_then(serde_json::Value::as_object)
    {
        Some(value) => value,
        None => return Ok(None),
    };
    let mut report_context = cloned_report_context_object(payload);
    let Some(plan) = video_tasks.prepare_sync_success(
        payload.report_kind.as_str(),
        provider_body,
        &report_context,
        plan,
    ) else {
        return Ok(None);
    };
    plan.apply_to_report_context(&mut report_context);
    let client_body_json = plan.client_body_json();

    let response = build_local_video_success_response(trace_id, decision, &client_body_json)?;
    let report_payload = GatewaySyncReportRequest {
        trace_id: payload.trace_id.clone(),
        report_kind: plan.success_report_kind().to_string(),
        report_context: Some(serde_json::Value::Object(report_context)),
        status_code: payload.status_code,
        headers: payload.headers.clone(),
        body_json: payload.body_json.clone(),
        client_body_json: Some(client_body_json),
        body_base64: None,
        telemetry: payload.telemetry.clone(),
    };

    Ok(Some(LocalVideoSyncSuccessOutcome {
        response,
        report_payload,
        report_mode: plan.report_mode(),
        local_task_snapshot: matches!(plan.report_mode(), VideoTaskSyncReportMode::Background)
            .then(|| plan.to_snapshot()),
    }))
}

pub(crate) fn maybe_build_local_sync_finalize_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<Response<Body>>, GatewayError> {
    let (status_code, body_json) = match payload.report_kind.as_str() {
        "openai_video_delete_sync_finalize" => {
            if payload.status_code >= 400 && payload.status_code != 404 {
                return Ok(None);
            }
            let Some(task_id) = payload
                .report_context
                .as_ref()
                .and_then(|value| value.get("task_id"))
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            else {
                return Ok(None);
            };
            (
                http::StatusCode::OK,
                json!({
                    "id": task_id,
                    "object": "video",
                    "deleted": true,
                }),
            )
        }
        "openai_video_cancel_sync_finalize" | "gemini_video_cancel_sync_finalize" => {
            if payload.status_code >= 400 {
                return Ok(None);
            }
            (http::StatusCode::OK, json!({}))
        }
        _ => return Ok(None),
    };

    let body_bytes =
        serde_json::to_vec(&body_json).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let mut headers = BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("content-length".to_string(), body_bytes.len().to_string());

    Ok(Some(build_client_response_from_parts(
        status_code.as_u16(),
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?))
}

pub(crate) fn maybe_build_local_video_error_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<Response<Body>>, GatewayError> {
    if !matches!(
        payload.report_kind.as_str(),
        "openai_video_create_sync_finalize"
            | "openai_video_remix_sync_finalize"
            | "gemini_video_create_sync_finalize"
            | "openai_video_delete_sync_finalize"
            | "openai_video_cancel_sync_finalize"
            | "gemini_video_cancel_sync_finalize"
    ) {
        return Ok(None);
    }

    if payload.status_code < 400 {
        return Ok(None);
    }

    let response_body = payload.body_json.clone().unwrap_or_else(|| json!({}));
    let body_bytes = serde_json::to_vec(&response_body)
        .map_err(|err| GatewayError::Internal(err.to_string()))?;

    let mut response_headers = payload.headers.clone();
    response_headers.remove("content-encoding");
    response_headers.remove("content-length");
    response_headers.insert("content-type".to_string(), "application/json".to_string());
    response_headers.insert("content-length".to_string(), body_bytes.len().to_string());

    Ok(Some(build_client_response_from_parts(
        payload.status_code,
        &response_headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?))
}

pub(crate) fn resolve_local_sync_success_background_report_kind(
    report_kind: &str,
) -> Option<String> {
    let mapped = match report_kind {
        "openai_video_delete_sync_finalize" => "openai_video_delete_sync_success",
        "openai_video_cancel_sync_finalize" => "openai_video_cancel_sync_success",
        "gemini_video_cancel_sync_finalize" => "gemini_video_cancel_sync_success",
        _ => return None,
    };
    Some(mapped.to_string())
}

pub(crate) fn resolve_local_sync_error_background_report_kind(report_kind: &str) -> Option<String> {
    let mapped = match report_kind {
        "openai_video_create_sync_finalize" => "openai_video_create_sync_error",
        "openai_video_remix_sync_finalize" => "openai_video_remix_sync_error",
        "gemini_video_create_sync_finalize" => "gemini_video_create_sync_error",
        "openai_video_delete_sync_finalize" => "openai_video_delete_sync_error",
        "openai_video_cancel_sync_finalize" => "openai_video_cancel_sync_error",
        "gemini_video_cancel_sync_finalize" => "gemini_video_cancel_sync_error",
        _ => return None,
    };
    Some(mapped.to_string())
}
