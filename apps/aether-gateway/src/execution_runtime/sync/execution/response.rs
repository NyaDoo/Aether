use std::collections::BTreeMap;

use aether_contracts::ExecutionPlan;
use axum::body::Body;
use axum::http::header::HeaderValue;
use axum::http::{HeaderMap, Response};
use serde_json::json;

use crate::api::response::{
    build_client_response_from_parts, build_client_response_from_parts_with_mutator,
};
use crate::async_task::VideoTaskService;
use crate::control::GatewayControlDecision;
use crate::video_tasks::{
    build_local_sync_finalize_read_response, LocalVideoTaskSnapshot, VideoTaskSyncReportMode,
};
pub(crate) use crate::video_tasks::{
    resolve_local_sync_error_background_report_kind,
    resolve_local_sync_success_background_report_kind,
};
use crate::{usage::GatewaySyncReportRequest, GatewayError};

pub(crate) enum LocalVideoSyncSuccessBuild {
    Handled(LocalVideoSyncSuccessOutcome),
    NotHandled(GatewaySyncReportRequest),
}

pub(crate) struct LocalVideoSyncSuccessOutcome {
    pub(crate) response: Response<Body>,
    pub(crate) report_payload: GatewaySyncReportRequest,
    pub(crate) original_report_context: Option<serde_json::Value>,
    pub(crate) report_mode: VideoTaskSyncReportMode,
    pub(crate) local_task_snapshot: Option<LocalVideoTaskSnapshot>,
}

fn cloned_report_context_object(
    payload: &GatewaySyncReportRequest,
) -> serde_json::Map<String, serde_json::Value> {
    payload
        .report_context
        .as_ref()
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn is_openai_video_client_with_doubao_provider(payload: &GatewaySyncReportRequest) -> bool {
    let Some(report_context) = payload
        .report_context
        .as_ref()
        .and_then(serde_json::Value::as_object)
    else {
        return false;
    };
    let Some(client_api_format) = report_context
        .get("client_api_format")
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };
    let Some(provider_api_format) = report_context
        .get("provider_api_format")
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };

    crate::ai_serving::normalize_api_format_alias(client_api_format) == "openai:video"
        && crate::ai_serving::normalize_api_format_alias(provider_api_format) == "doubao:video"
}

fn provider_error_field<'a>(
    body_json: &'a serde_json::Value,
    field_name: &str,
) -> Option<&'a serde_json::Value> {
    [
        body_json.get("error"),
        body_json.pointer("/ResponseMetadata/Error"),
        Some(body_json),
    ]
    .into_iter()
    .flatten()
    .filter_map(serde_json::Value::as_object)
    .find_map(|object| {
        object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(field_name))
            .map(|(_, value)| value)
    })
}

fn provider_error_code(body_json: &serde_json::Value) -> Option<String> {
    let code = provider_error_field(body_json, "code")?;
    match code {
        serde_json::Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
        serde_json::Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn openai_video_error_body(body_json: &serde_json::Value) -> serde_json::Value {
    let message = provider_error_field(body_json, "message")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Video generation request failed");
    let code = provider_error_code(body_json);

    json!({
        "error": {
            "message": message,
            "type": "video_generation_error",
            "param": null,
            "code": code,
        }
    })
}

fn build_local_video_success_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Result<(Response<Body>, serde_json::Value), GatewayError> {
    let body_bytes =
        serde_json::to_vec(body_json).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let mut headers = BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("content-length".to_string(), body_bytes.len().to_string());
    let response = build_client_response_from_parts(
        http::StatusCode::OK.as_u16(),
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?;
    let captured_headers = capture_response_headers(response.headers());
    Ok((response, captured_headers))
}

fn capture_response_headers(headers: &HeaderMap) -> serde_json::Value {
    serde_json::Value::Object(
        headers
            .iter()
            .filter_map(|(name, value)| {
                value.to_str().ok().map(|value| {
                    (
                        name.as_str().to_string(),
                        serde_json::Value::String(value.to_string()),
                    )
                })
            })
            .collect(),
    )
}

pub(crate) fn maybe_build_local_video_success_outcome(
    trace_id: &str,
    decision: &GatewayControlDecision,
    mut payload: GatewaySyncReportRequest,
    video_tasks: &VideoTaskService,
    plan: &ExecutionPlan,
) -> Result<LocalVideoSyncSuccessBuild, GatewayError> {
    if payload.status_code >= 400 {
        return Ok(LocalVideoSyncSuccessBuild::NotHandled(payload));
    }

    let mut report_context = cloned_report_context_object(&payload);
    let prepared_plan = {
        let provider_body = match payload
            .body_json
            .as_ref()
            .and_then(serde_json::Value::as_object)
        {
            Some(value) => value,
            None => return Ok(LocalVideoSyncSuccessBuild::NotHandled(payload)),
        };
        video_tasks.prepare_sync_success(
            payload.report_kind.as_str(),
            provider_body,
            &report_context,
            plan,
        )
    };
    let Some(plan) = prepared_plan else {
        return Ok(LocalVideoSyncSuccessBuild::NotHandled(payload));
    };
    plan.apply_to_report_context(&mut report_context);
    let client_body_json = plan.client_body_json();

    let (response, client_response_headers) =
        build_local_video_success_response(trace_id, decision, &client_body_json)?;
    report_context.insert(
        "client_response_headers".to_string(),
        client_response_headers,
    );
    let original_report_context = payload.report_context.take();
    payload.report_kind = plan.success_report_kind().to_string();
    payload.report_context = Some(serde_json::Value::Object(report_context));
    payload.client_body_json = Some(client_body_json);

    Ok(LocalVideoSyncSuccessBuild::Handled(
        LocalVideoSyncSuccessOutcome {
            response,
            report_payload: payload,
            original_report_context,
            report_mode: plan.report_mode(),
            local_task_snapshot: matches!(plan.report_mode(), VideoTaskSyncReportMode::Background)
                .then(|| plan.to_snapshot()),
        },
    ))
}

pub(crate) fn maybe_build_local_sync_finalize_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &mut GatewaySyncReportRequest,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(read_response) = build_local_sync_finalize_read_response(
        payload.report_kind.as_str(),
        payload.status_code,
        payload.report_context.as_ref(),
    ) else {
        return Ok(None);
    };

    let body_bytes = serde_json::to_vec(&read_response.body_json)
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    let mut headers = BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("content-length".to_string(), body_bytes.len().to_string());

    let response = build_client_response_from_parts(
        read_response.status_code,
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?;
    let client_response_headers = capture_response_headers(response.headers());
    let mut report_context = cloned_report_context_object(payload);
    report_context.insert(
        "client_response_headers".to_string(),
        client_response_headers,
    );
    payload.report_context = Some(serde_json::Value::Object(report_context));
    payload.client_body_json = Some(read_response.body_json);

    Ok(Some(response))
}

pub(crate) fn maybe_build_local_video_error_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<Response<Body>>, GatewayError> {
    if resolve_local_sync_error_background_report_kind(payload.report_kind.as_str()).is_none() {
        return Ok(None);
    }

    if payload.status_code < 400 {
        return Ok(None);
    }

    let empty_body = json!({});
    let upstream_body = payload.body_json.as_ref().unwrap_or(&empty_body);
    let projected_body = is_openai_video_client_with_doubao_provider(payload)
        .then(|| openai_video_error_body(upstream_body));
    let response_body = projected_body.as_ref().unwrap_or(upstream_body);
    let body_bytes =
        serde_json::to_vec(response_body).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let body_len = body_bytes.len().to_string();

    Ok(Some(build_client_response_from_parts_with_mutator(
        payload.status_code,
        &payload.headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
        |headers| {
            headers.remove(http::header::CONTENT_ENCODING);
            headers.remove(http::header::CONTENT_LENGTH);
            headers.insert(
                http::header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            headers.insert(
                http::header::CONTENT_LENGTH,
                HeaderValue::from_str(body_len.as_str())
                    .map_err(|err| GatewayError::Internal(err.to_string()))?,
            );
            Ok(())
        },
    )?))
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::to_bytes;
    use serde_json::json;

    #[tokio::test]
    async fn local_video_success_response_returns_the_headers_written_to_the_client() {
        let decision = GatewayControlDecision::synthetic(
            "/v1/videos",
            Some("ai_public".to_string()),
            Some("openai".to_string()),
            Some("video".to_string()),
            Some("openai:video".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let body = json!({"id": "video-task-1", "status": "queued"});

        let (response, captured_headers) =
            build_local_video_success_response("trace-success", &decision, &body)
                .expect("local video success response should build");

        assert_eq!(
            captured_headers["content-type"],
            serde_json::Value::String("application/json".to_string())
        );
        assert_eq!(
            captured_headers["content-length"],
            serde_json::Value::String(
                serde_json::to_vec(&body)
                    .expect("body should serialize")
                    .len()
                    .to_string()
            )
        );
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            captured_headers["content-type"].as_str()
        );
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok()),
            captured_headers["content-length"].as_str()
        );
    }

    #[tokio::test]
    async fn local_video_finalize_records_projected_client_body_and_headers() {
        let decision = GatewayControlDecision::synthetic(
            "/v1/videos/task-1/cancel",
            Some("ai_public".to_string()),
            Some("openai".to_string()),
            Some("video".to_string()),
            Some("openai:video".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let mut payload = GatewaySyncReportRequest {
            trace_id: "trace-cancel".to_string(),
            report_kind: "openai_video_cancel_sync_finalize".to_string(),
            report_context: Some(json!({"task_id": "task-1"})),
            status_code: 200,
            headers: BTreeMap::from([(
                "x-provider-request-id".to_string(),
                "upstream-1".to_string(),
            )]),
            body_json: Some(json!({"provider_status": "cancelled"})),
            client_body_json: None,
            body_base64: None,
            telemetry: None,
        };

        let response =
            maybe_build_local_sync_finalize_response("trace-cancel", &decision, &mut payload)
                .expect("cancel response should build")
                .expect("cancel finalize should be handled");

        assert_eq!(payload.client_body_json, Some(json!({})));
        let captured_headers = payload
            .report_context
            .as_ref()
            .and_then(|context| context.get("client_response_headers"))
            .expect("client response headers should be captured");
        assert_eq!(
            captured_headers["content-type"].as_str(),
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
        );
        assert_eq!(
            captured_headers["content-length"].as_str(),
            response
                .headers()
                .get(http::header::CONTENT_LENGTH)
                .and_then(|value| value.to_str().ok())
        );
        assert_eq!(
            payload.body_json,
            Some(json!({"provider_status": "cancelled"}))
        );
    }

    #[tokio::test]
    async fn local_video_error_response_rewrites_headers_without_mutating_payload() {
        let decision = GatewayControlDecision::synthetic(
            "/v1/videos",
            Some("ai_public".to_string()),
            Some("openai".to_string()),
            Some("video".to_string()),
            Some("openai:video".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let payload = GatewaySyncReportRequest {
            trace_id: "trace-payload".to_string(),
            report_kind: "openai_video_create_sync_finalize".to_string(),
            report_context: Some(json!({
                "request_id": "req_123",
            })),
            status_code: http::StatusCode::BAD_GATEWAY.as_u16(),
            headers: BTreeMap::from([
                ("content-encoding".to_string(), "gzip".to_string()),
                ("content-length".to_string(), "999".to_string()),
                ("x-upstream-id".to_string(), "video-123".to_string()),
            ]),
            body_json: Some(json!({
                "error": {
                    "type": "video_backend_error",
                    "message": "backend failed",
                }
            })),
            client_body_json: None,
            body_base64: None,
            telemetry: None,
        };

        let response =
            maybe_build_local_video_error_response("trace-response", &decision, &payload)
                .expect("video error response should build")
                .expect("video error response should match local video error kinds");

        assert_eq!(response.status(), http::StatusCode::BAD_GATEWAY);
        assert_eq!(
            response
                .headers()
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(response.headers().get(http::header::CONTENT_ENCODING), None);
        assert_eq!(
            response
                .headers()
                .get("x-upstream-id")
                .and_then(|value| value.to_str().ok()),
            Some("video-123")
        );
        assert_eq!(
            payload.headers.get("content-encoding").map(String::as_str),
            Some("gzip")
        );
        assert_eq!(
            payload.headers.get("content-length").map(String::as_str),
            Some("999")
        );

        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should read");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("response body should parse"),
            payload
                .body_json
                .clone()
                .expect("payload body should exist")
        );
    }

    #[tokio::test]
    async fn cross_format_video_error_is_projected_to_openai_envelope() {
        let decision = GatewayControlDecision::synthetic(
            "/v1/videos",
            Some("ai_public".to_string()),
            Some("openai".to_string()),
            Some("video".to_string()),
            Some("openai:video".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let payload = GatewaySyncReportRequest {
            trace_id: "trace-cross-error".to_string(),
            report_kind: "openai_video_create_sync_finalize".to_string(),
            report_context: Some(json!({
                "request_id": "req_cross_error",
                "client_api_format": "openai:video",
                "provider_api_format": "doubao:video",
            })),
            status_code: http::StatusCode::BAD_GATEWAY.as_u16(),
            headers: BTreeMap::new(),
            body_json: Some(json!({
                "error": {
                    "code": "InputImageSensitiveContentDetected",
                    "message": "blocked by provider policy",
                    "provider_private_detail": "must not leak",
                },
                "request_id": "ark-request-id",
            })),
            client_body_json: None,
            body_base64: None,
            telemetry: None,
        };
        let original_body = payload.body_json.clone();

        let response = maybe_build_local_video_error_response(
            "trace-cross-error-response",
            &decision,
            &payload,
        )
        .expect("cross-format video error should build")
        .expect("cross-format video error should match local video error kinds");
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should read");

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("response body should parse"),
            json!({
                "error": {
                    "message": "blocked by provider policy",
                    "type": "video_generation_error",
                    "param": null,
                    "code": "InputImageSensitiveContentDetected",
                }
            })
        );
        assert_eq!(payload.body_json, original_body);
    }

    #[tokio::test]
    async fn native_doubao_video_error_keeps_provider_body_unchanged() {
        let decision = GatewayControlDecision::synthetic(
            "/api/v3/contents/generations/tasks",
            Some("ai_public".to_string()),
            Some("doubao".to_string()),
            Some("video".to_string()),
            Some("doubao:video".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let payload = GatewaySyncReportRequest {
            trace_id: "trace-doubao-error".to_string(),
            report_kind: "doubao_video_create_sync_finalize".to_string(),
            report_context: Some(json!({
                "request_id": "req_doubao_error",
                "client_api_format": "doubao:video",
                "provider_api_format": "doubao:video",
            })),
            status_code: http::StatusCode::BAD_REQUEST.as_u16(),
            headers: BTreeMap::new(),
            body_json: Some(json!({
                "ResponseMetadata": {
                    "Error": {
                        "Code": "InvalidParameter",
                        "Message": "native Ark error",
                    }
                },
                "request_id": "ark-request-id",
            })),
            client_body_json: None,
            body_base64: None,
            telemetry: None,
        };
        let original_body = payload.body_json.clone();

        let response = maybe_build_local_video_error_response(
            "trace-doubao-error-response",
            &decision,
            &payload,
        )
        .expect("native Doubao video error should build")
        .expect("native Doubao video error should match local video error kinds");
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should read");

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&body).expect("response body should parse"),
            original_body.expect("payload body should exist")
        );
    }
}
