use axum::{
    body::Body,
    http,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

pub(crate) fn build_proxy_error_response(
    status: http::StatusCode,
    error_type: &str,
    message: impl Into<String>,
    details: Option<serde_json::Value>,
) -> Response<Body> {
    let message = message.into();
    let mut error = serde_json::Map::new();
    error.insert("type".to_string(), json!(error_type));
    error.insert("message".to_string(), json!(message));
    if let Some(details) = details {
        error.insert("details".to_string(), details);
    }
    (
        status,
        Json(json!({ "error": serde_json::Value::Object(error) })),
    )
        .into_response()
}
