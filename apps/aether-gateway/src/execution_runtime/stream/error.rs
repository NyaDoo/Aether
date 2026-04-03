use std::collections::BTreeMap;

use aether_contracts::{StreamFrame, StreamFramePayload};
use axum::body::Body;
use axum::http::Response;
use base64::Engine as _;
use futures_util::StreamExt;
use serde_json::json;
use tokio_util::codec::{FramedRead, LinesCodec};
use tracing::warn;

use crate::gateway::api::response::build_client_response_from_parts;
use crate::gateway::execution_runtime::ndjson::decode_stream_frame_ndjson;
use crate::gateway::execution_runtime::submission::{has_nested_error, strip_utf8_bom_and_ws};
use crate::gateway::{GatewayControlDecision, GatewayError};
use crate::gateway::{
    GEMINI_FILES_DOWNLOAD_PLAN_KIND, MAX_ERROR_BODY_BYTES, MAX_STREAM_PREFETCH_FRAMES,
    OPENAI_VIDEO_CONTENT_PLAN_KIND,
};

#[derive(Debug)]
pub(super) enum StreamPrefetchInspection {
    NeedMore,
    NonError,
    EmbeddedError(serde_json::Value),
}

pub(super) fn decode_stream_error_body(
    headers: &BTreeMap<String, String>,
    error_body: &[u8],
) -> (Option<serde_json::Value>, Option<String>) {
    if error_body.is_empty() {
        return (None, None);
    }

    let content_type = headers
        .get("content-type")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    let looks_json = content_type.contains("json") || content_type.ends_with("+json");
    if looks_json {
        if let Ok(json_body) = serde_json::from_slice::<serde_json::Value>(error_body) {
            return (Some(json_body), None);
        }
    }

    (
        None,
        Some(base64::engine::general_purpose::STANDARD.encode(error_body)),
    )
}

pub(super) fn inspect_prefetched_stream_body(
    headers: &BTreeMap<String, String>,
    body: &[u8],
) -> StreamPrefetchInspection {
    if body.is_empty() {
        return StreamPrefetchInspection::NeedMore;
    }

    let stripped = strip_utf8_bom_and_ws(body);
    let content_type = headers
        .get("content-type")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    let looks_json = content_type.contains("json") || content_type.ends_with("+json");
    if looks_json || stripped.starts_with(b"{") || stripped.starts_with(b"[") {
        if let Ok(json_body) = serde_json::from_slice::<serde_json::Value>(stripped) {
            return if has_nested_error(&json_body) {
                StreamPrefetchInspection::EmbeddedError(json_body)
            } else {
                StreamPrefetchInspection::NonError
            };
        }
    }

    let text = String::from_utf8_lossy(body);
    let mut saw_meaningful_line = false;
    for line in text.lines().take(MAX_STREAM_PREFETCH_FRAMES) {
        let line = line.trim_matches('\r').trim();
        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            continue;
        }

        let data_line = line.strip_prefix("data: ").unwrap_or(line).trim();
        if data_line.is_empty() {
            continue;
        }
        if data_line == "[DONE]" {
            return StreamPrefetchInspection::NonError;
        }

        saw_meaningful_line = true;
        match serde_json::from_str::<serde_json::Value>(data_line) {
            Ok(json_body) => {
                return if has_nested_error(&json_body) {
                    StreamPrefetchInspection::EmbeddedError(json_body)
                } else {
                    StreamPrefetchInspection::NonError
                };
            }
            Err(_) => {
                if data_line.ends_with('}') || data_line.ends_with(']') {
                    return StreamPrefetchInspection::NonError;
                }
            }
        }
    }

    if saw_meaningful_line {
        StreamPrefetchInspection::NonError
    } else {
        StreamPrefetchInspection::NeedMore
    }
}

pub(super) async fn collect_error_body<R>(
    lines: &mut FramedRead<R, LinesCodec>,
) -> Result<Vec<u8>, GatewayError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    let mut body = Vec::new();
    while let Some(frame) = read_next_frame(lines).await? {
        match frame.payload {
            StreamFramePayload::Data { chunk_b64, text } => {
                let chunk = if let Some(chunk_b64) = chunk_b64 {
                    base64::engine::general_purpose::STANDARD
                        .decode(chunk_b64)
                        .map_err(|err| GatewayError::Internal(err.to_string()))?
                } else {
                    text.unwrap_or_default().into_bytes()
                };
                body.extend_from_slice(&chunk);
                if body.len() >= MAX_ERROR_BODY_BYTES {
                    body.truncate(MAX_ERROR_BODY_BYTES);
                    break;
                }
            }
            StreamFramePayload::Telemetry { .. } => {}
            StreamFramePayload::Eof { .. } => break,
            StreamFramePayload::Error { error } => {
                warn!(error = %error.message, "execution runtime stream emitted error frame while collecting error body");
                break;
            }
            StreamFramePayload::Headers { .. } => {}
        }
    }
    Ok(body)
}

pub(super) async fn read_next_frame<R>(
    lines: &mut FramedRead<R, LinesCodec>,
) -> Result<Option<StreamFrame>, GatewayError>
where
    R: tokio::io::AsyncRead + Unpin,
{
    while let Some(line) = lines.next().await {
        let line = line.map_err(|err| GatewayError::Internal(err.to_string()))?;
        if line.trim().is_empty() {
            continue;
        }
        let frame = decode_stream_frame_ndjson(line.as_bytes())?;
        return Ok(Some(frame));
    }
    Ok(None)
}

pub(super) fn build_execution_runtime_error_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    status_code: u16,
    headers: BTreeMap<String, String>,
    error_body: Vec<u8>,
) -> Result<Response<Body>, GatewayError> {
    let content_type = headers
        .get("content-type")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();

    if plan_kind == GEMINI_FILES_DOWNLOAD_PLAN_KIND && !content_type.starts_with("application/json")
    {
        let wrapped = serde_json::to_vec(&json!({
            "error": String::from_utf8_lossy(&error_body).to_string(),
        }))
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
        let wrapped_headers =
            BTreeMap::from([("content-type".to_string(), "application/json".to_string())]);
        return build_client_response_from_parts(
            status_code,
            &wrapped_headers,
            Body::from(wrapped),
            trace_id,
            Some(decision),
        );
    }

    if plan_kind == OPENAI_VIDEO_CONTENT_PLAN_KIND && !content_type.starts_with("application/json")
    {
        let wrapped = serde_json::to_vec(&json!({
            "error": {
                "type": "upstream_error",
                "message": "Video not available",
            }
        }))
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
        let wrapped_headers =
            BTreeMap::from([("content-type".to_string(), "application/json".to_string())]);
        return build_client_response_from_parts(
            status_code,
            &wrapped_headers,
            Body::from(wrapped),
            trace_id,
            Some(decision),
        );
    }

    build_client_response_from_parts(
        status_code,
        &headers,
        Body::from(error_body),
        trace_id,
        Some(decision),
    )
}
