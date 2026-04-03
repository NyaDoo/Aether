use serde_json::{Map, Value};

#[cfg(test)]
use crate::gateway::ai_pipeline::contracts::core_success_background_report_kind as contract_core_success_background_report_kind;
use crate::gateway::ai_pipeline::contracts::{
    core_error_background_report_kind as contract_core_error_background_report_kind,
    core_error_default_client_api_format as contract_core_error_default_client_api_format,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LocalCoreSyncErrorKind {
    InvalidRequest,
    Authentication,
    PermissionDenied,
    NotFound,
    RateLimit,
    ContextLengthExceeded,
    Overloaded,
    ServerError,
}

pub(crate) fn is_core_error_finalize_kind(report_kind: &str) -> bool {
    core_error_default_client_api_format(report_kind).is_some()
}

pub(crate) fn core_error_default_client_api_format(report_kind: &str) -> Option<&'static str> {
    contract_core_error_default_client_api_format(report_kind)
}

pub(crate) fn core_error_background_report_kind(report_kind: &str) -> Option<&'static str> {
    contract_core_error_background_report_kind(report_kind)
}

#[cfg(test)]
pub(crate) fn core_success_background_report_kind(report_kind: &str) -> Option<&'static str> {
    contract_core_success_background_report_kind(report_kind)
}

pub(crate) fn build_core_error_body_for_client_format(
    client_api_format: &str,
    message: &str,
    code: Option<&str>,
    kind: LocalCoreSyncErrorKind,
) -> Option<Value> {
    let mut error_object = Map::new();
    error_object.insert("message".to_string(), Value::String(message.to_string()));

    match client_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:chat" | "openai:cli" | "openai:compact" => {
            error_object.insert(
                "type".to_string(),
                Value::String(map_local_sync_error_kind_to_openai_type(kind).to_string()),
            );
            if let Some(code) = code.filter(|value| !value.is_empty()) {
                error_object.insert("code".to_string(), Value::String(code.to_string()));
            }
            Some(Value::Object(Map::from_iter([(
                "error".to_string(),
                Value::Object(error_object),
            )])))
        }
        "claude:chat" | "claude:cli" => {
            error_object.insert(
                "type".to_string(),
                Value::String(map_local_sync_error_kind_to_claude_type(kind).to_string()),
            );
            if let Some(code) = code.filter(|value| !value.is_empty()) {
                error_object.insert("code".to_string(), Value::String(code.to_string()));
            }
            Some(Value::Object(Map::from_iter([
                ("type".to_string(), Value::String("error".to_string())),
                ("error".to_string(), Value::Object(error_object)),
            ])))
        }
        "gemini:chat" | "gemini:cli" => Some(Value::Object(Map::from_iter([(
            "error".to_string(),
            Value::Object(Map::from_iter([
                (
                    "code".to_string(),
                    Value::from(map_local_sync_error_kind_to_gemini_code(kind)),
                ),
                ("message".to_string(), Value::String(message.to_string())),
                (
                    "status".to_string(),
                    Value::String(map_local_sync_error_kind_to_gemini_status(kind).to_string()),
                ),
            ])),
        )]))),
        _ => None,
    }
}

fn map_local_sync_error_kind_to_openai_type(kind: LocalCoreSyncErrorKind) -> &'static str {
    match kind {
        LocalCoreSyncErrorKind::InvalidRequest => "invalid_request_error",
        LocalCoreSyncErrorKind::Authentication => "authentication_error",
        LocalCoreSyncErrorKind::PermissionDenied => "permission_error",
        LocalCoreSyncErrorKind::NotFound => "not_found_error",
        LocalCoreSyncErrorKind::RateLimit => "rate_limit_error",
        LocalCoreSyncErrorKind::ContextLengthExceeded => "context_length_exceeded",
        LocalCoreSyncErrorKind::Overloaded | LocalCoreSyncErrorKind::ServerError => "server_error",
    }
}

fn map_local_sync_error_kind_to_claude_type(kind: LocalCoreSyncErrorKind) -> &'static str {
    match kind {
        LocalCoreSyncErrorKind::InvalidRequest | LocalCoreSyncErrorKind::ContextLengthExceeded => {
            "invalid_request_error"
        }
        LocalCoreSyncErrorKind::Authentication => "authentication_error",
        LocalCoreSyncErrorKind::PermissionDenied => "permission_error",
        LocalCoreSyncErrorKind::NotFound => "not_found_error",
        LocalCoreSyncErrorKind::RateLimit => "rate_limit_error",
        LocalCoreSyncErrorKind::Overloaded | LocalCoreSyncErrorKind::ServerError => "api_error",
    }
}

fn map_local_sync_error_kind_to_gemini_code(kind: LocalCoreSyncErrorKind) -> u16 {
    match kind {
        LocalCoreSyncErrorKind::InvalidRequest | LocalCoreSyncErrorKind::ContextLengthExceeded => {
            400
        }
        LocalCoreSyncErrorKind::Authentication => 401,
        LocalCoreSyncErrorKind::PermissionDenied => 403,
        LocalCoreSyncErrorKind::NotFound => 404,
        LocalCoreSyncErrorKind::RateLimit => 429,
        LocalCoreSyncErrorKind::Overloaded => 503,
        LocalCoreSyncErrorKind::ServerError => 500,
    }
}

fn map_local_sync_error_kind_to_gemini_status(kind: LocalCoreSyncErrorKind) -> &'static str {
    match kind {
        LocalCoreSyncErrorKind::InvalidRequest | LocalCoreSyncErrorKind::ContextLengthExceeded => {
            "INVALID_ARGUMENT"
        }
        LocalCoreSyncErrorKind::Authentication => "UNAUTHENTICATED",
        LocalCoreSyncErrorKind::PermissionDenied => "PERMISSION_DENIED",
        LocalCoreSyncErrorKind::NotFound => "NOT_FOUND",
        LocalCoreSyncErrorKind::RateLimit => "RESOURCE_EXHAUSTED",
        LocalCoreSyncErrorKind::Overloaded => "UNAVAILABLE",
        LocalCoreSyncErrorKind::ServerError => "INTERNAL",
    }
}
