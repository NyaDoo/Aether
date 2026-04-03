use std::collections::BTreeMap;

use axum::body::Body;
use axum::http::Response;
use base64::Engine as _;
use serde_json::{json, Map, Value};

use crate::gateway::ai_pipeline::conversion::{
    sync_chat_response_conversion_kind, sync_cli_response_conversion_kind,
};
use crate::gateway::api::response::build_client_response_from_parts;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) use crate::gateway::ai_pipeline::finalize::common::{
    build_generated_tool_call_id, build_local_success_outcome,
    build_local_success_outcome_with_conversion_report, canonicalize_tool_arguments,
    local_finalize_allows_envelope, parse_stream_json_events, unwrap_local_finalize_response_value,
    LocalCoreSyncFinalizeOutcome,
};
pub(crate) use crate::gateway::ai_pipeline::finalize::standard::{
    aggregate_claude_stream_sync_response, aggregate_gemini_stream_sync_response,
    aggregate_openai_chat_stream_sync_response, aggregate_openai_cli_stream_sync_response,
    aggregate_standard_chat_stream_sync_response, aggregate_standard_cli_stream_sync_response,
    convert_claude_chat_response_to_openai_chat, convert_claude_cli_response_to_openai_cli,
    convert_gemini_chat_response_to_openai_chat, convert_gemini_cli_response_to_openai_cli,
    convert_openai_chat_response_to_claude_chat, convert_openai_chat_response_to_gemini_chat,
    convert_openai_chat_response_to_openai_cli, convert_openai_cli_response_to_openai_chat,
    convert_standard_chat_response, convert_standard_cli_response,
    maybe_build_local_claude_cli_stream_sync_response,
    maybe_build_local_claude_stream_sync_response, maybe_build_local_claude_sync_response,
    maybe_build_local_gemini_cli_stream_sync_response,
    maybe_build_local_gemini_stream_sync_response, maybe_build_local_gemini_sync_response,
    maybe_build_local_openai_chat_cross_format_stream_sync_response,
    maybe_build_local_openai_chat_cross_format_sync_response,
    maybe_build_local_openai_chat_stream_sync_response,
    maybe_build_local_openai_chat_sync_response,
    maybe_build_local_openai_cli_cross_format_stream_sync_response,
    maybe_build_local_openai_cli_cross_format_sync_response,
    maybe_build_local_openai_cli_stream_sync_response,
};

pub(crate) fn maybe_build_local_core_sync_finalize_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    let Some(normalized_payload) =
        crate::gateway::ai_pipeline::adaptation::private_envelope::maybe_normalize_provider_private_sync_report_payload(payload)?
    else {
        return Ok(None);
    };
    let payload = &normalized_payload;
    if let Some(response) =
        maybe_build_local_openai_chat_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_openai_chat_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) = maybe_build_local_openai_chat_cross_format_stream_sync_response(
        trace_id, decision, payload,
    )? {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_openai_cli_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_openai_cli_cross_format_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_claude_cli_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_gemini_cli_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_claude_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) = maybe_build_local_claude_sync_response(trace_id, decision, payload)? {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_gemini_stream_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) = maybe_build_local_gemini_sync_response(trace_id, decision, payload)? {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_openai_chat_cross_format_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_openai_cli_cross_format_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) = maybe_build_local_standard_chat_cross_format_stream_sync_response(
        trace_id, decision, payload,
    )? {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_standard_chat_cross_format_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    if let Some(response) = maybe_build_local_standard_cli_cross_format_stream_sync_response(
        trace_id, decision, payload,
    )? {
        return Ok(Some(response));
    }
    if let Some(response) =
        maybe_build_local_standard_cli_cross_format_sync_response(trace_id, decision, payload)?
    {
        return Ok(Some(response));
    }
    Ok(None)
}

fn maybe_build_local_standard_chat_cross_format_stream_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    if !matches!(
        payload.report_kind.as_str(),
        "openai_chat_sync_finalize" | "claude_chat_sync_finalize" | "gemini_chat_sync_finalize"
    ) || payload.status_code >= 400
    {
        return Ok(None);
    }

    let Some(report_context) = payload.report_context.as_ref() else {
        return Ok(None);
    };
    let provider_api_format = report_context
        .get("provider_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let client_api_format = report_context
        .get("client_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !local_finalize_allows_envelope(report_context)
        || sync_chat_response_conversion_kind(&provider_api_format, &client_api_format).is_none()
    {
        return Ok(None);
    }
    let Some(body_base64) = payload.body_base64.as_deref() else {
        return Ok(None);
    };
    let body_bytes = base64::engine::general_purpose::STANDARD
        .decode(body_base64)
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    let Some(aggregated) =
        aggregate_standard_chat_stream_sync_response(&body_bytes, &provider_api_format)
    else {
        return Ok(None);
    };
    let Some(aggregated) = unwrap_local_finalize_response_value(aggregated, report_context)? else {
        return Ok(None);
    };
    let Some(converted) = convert_standard_chat_response(
        &aggregated,
        &provider_api_format,
        &client_api_format,
        report_context,
    ) else {
        return Ok(None);
    };

    Ok(Some(build_local_success_outcome_with_conversion_report(
        trace_id, decision, payload, converted, aggregated,
    )?))
}

fn maybe_build_local_standard_chat_cross_format_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    if !matches!(
        payload.report_kind.as_str(),
        "openai_chat_sync_finalize" | "claude_chat_sync_finalize" | "gemini_chat_sync_finalize"
    ) || payload.status_code >= 400
    {
        return Ok(None);
    }

    let Some(report_context) = payload.report_context.as_ref() else {
        return Ok(None);
    };
    let provider_api_format = report_context
        .get("provider_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let client_api_format = report_context
        .get("client_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !local_finalize_allows_envelope(report_context)
        || sync_chat_response_conversion_kind(&provider_api_format, &client_api_format).is_none()
    {
        return Ok(None);
    }
    let Some(body_json) = payload.body_json.as_ref() else {
        return Ok(None);
    };
    let Some(body_json) = unwrap_local_finalize_response_value(body_json.clone(), report_context)?
    else {
        return Ok(None);
    };
    let Some(converted) = convert_standard_chat_response(
        &body_json,
        &provider_api_format,
        &client_api_format,
        report_context,
    ) else {
        return Ok(None);
    };

    Ok(Some(build_local_success_outcome_with_conversion_report(
        trace_id, decision, payload, converted, body_json,
    )?))
}

fn maybe_build_local_standard_cli_cross_format_stream_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    if !matches!(
        payload.report_kind.as_str(),
        "openai_cli_sync_finalize"
            | "openai_compact_sync_finalize"
            | "claude_cli_sync_finalize"
            | "gemini_cli_sync_finalize"
    ) || payload.status_code >= 400
    {
        return Ok(None);
    }

    let Some(report_context) = payload.report_context.as_ref() else {
        return Ok(None);
    };
    let provider_api_format = report_context
        .get("provider_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let client_api_format = report_context
        .get("client_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !local_finalize_allows_envelope(report_context)
        || sync_cli_response_conversion_kind(&provider_api_format, &client_api_format).is_none()
    {
        return Ok(None);
    }
    let Some(body_base64) = payload.body_base64.as_deref() else {
        return Ok(None);
    };
    let body_bytes = base64::engine::general_purpose::STANDARD
        .decode(body_base64)
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    let Some(aggregated) =
        aggregate_standard_cli_stream_sync_response(&body_bytes, &provider_api_format)
    else {
        return Ok(None);
    };
    let Some(aggregated) = unwrap_local_finalize_response_value(aggregated, report_context)? else {
        return Ok(None);
    };
    let Some(converted) = convert_standard_cli_response(
        &aggregated,
        &provider_api_format,
        &client_api_format,
        report_context,
    ) else {
        return Ok(None);
    };

    Ok(Some(build_local_success_outcome_with_conversion_report(
        trace_id, decision, payload, converted, aggregated,
    )?))
}

fn maybe_build_local_standard_cli_cross_format_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    if !matches!(
        payload.report_kind.as_str(),
        "openai_cli_sync_finalize"
            | "openai_compact_sync_finalize"
            | "claude_cli_sync_finalize"
            | "gemini_cli_sync_finalize"
    ) || payload.status_code >= 400
    {
        return Ok(None);
    }

    let Some(report_context) = payload.report_context.as_ref() else {
        return Ok(None);
    };
    let provider_api_format = report_context
        .get("provider_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let client_api_format = report_context
        .get("client_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !local_finalize_allows_envelope(report_context)
        || sync_cli_response_conversion_kind(&provider_api_format, &client_api_format).is_none()
    {
        return Ok(None);
    }
    let Some(body_json) = payload.body_json.as_ref() else {
        return Ok(None);
    };
    let Some(body_json) = unwrap_local_finalize_response_value(body_json.clone(), report_context)?
    else {
        return Ok(None);
    };
    let Some(converted) = convert_standard_cli_response(
        &body_json,
        &provider_api_format,
        &client_api_format,
        report_context,
    ) else {
        return Ok(None);
    };

    Ok(Some(build_local_success_outcome_with_conversion_report(
        trace_id, decision, payload, converted, body_json,
    )?))
}

#[cfg(test)]
#[path = "../tests_sync.rs"]
mod tests;
