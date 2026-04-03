use axum::body::Body;
use axum::http::Response;

use crate::gateway::ai_pipeline::planner::passthrough::provider::plans::{
    build_local_stream_plan_and_reports as build_same_format_stream_plan_and_reports,
    build_local_sync_plan_and_reports as build_same_format_sync_plan_and_reports,
    resolve_stream_spec as resolve_same_format_stream_spec,
    resolve_sync_spec as resolve_same_format_sync_spec,
};
use crate::gateway::ai_pipeline::planner::standard::family::{
    build_local_stream_plan_and_reports as build_standard_stream_plan_and_reports,
    build_local_sync_plan_and_reports as build_standard_sync_plan_and_reports, LocalStandardSpec,
};
use crate::gateway::ai_pipeline::planner::standard::{claude, gemini, openai_chat, openai_cli};
use crate::gateway::ai_pipeline::planner::{
    parse_direct_request_body, GatewayControlSyncDecisionResponse,
    EXECUTION_RUNTIME_STREAM_DECISION_ACTION, EXECUTION_RUNTIME_SYNC_DECISION_ACTION,
};
use crate::gateway::ai_pipeline::{planner, runtime};
use crate::gateway::executor::candidate_loop::{
    execute_stream_plan_and_reports, execute_sync_plan_and_reports,
};
use crate::gateway::intent;
use crate::gateway::{AppState, GatewayControlDecision, GatewayError};

pub(crate) async fn maybe_execute_sync_local_path(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &axum::body::Bytes,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Result<Option<Response<Body>>, GatewayError> {
    intent::maybe_execute_via_sync_intent_path(state, parts, body_bytes, trace_id, decision).await
}

pub(crate) async fn maybe_execute_stream_local_path(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &axum::body::Bytes,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Result<Option<Response<Body>>, GatewayError> {
    intent::maybe_execute_via_stream_intent_path(state, parts, body_bytes, trace_id, decision).await
}

pub(crate) async fn maybe_execute_sync_via_local_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports = openai_chat::build_local_openai_chat_sync_plan_and_reports_for_kind(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    let plan_count = plan_and_reports.len();
    if let Some(response) = execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await?
    {
        return Ok(Some(response));
    }

    openai_chat::set_local_openai_chat_execution_exhausted_diagnostic(
        state, trace_id, decision, plan_kind, body_json, plan_count,
    );
    Ok(None)
}

pub(crate) async fn maybe_execute_stream_via_local_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports = openai_chat::build_local_openai_chat_stream_plan_and_reports_for_kind(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    let plan_count = plan_and_reports.len();
    if let Some(response) =
        execute_stream_plan_and_reports(state, trace_id, decision, plan_kind, plan_and_reports)
            .await?
    {
        return Ok(Some(response));
    }

    openai_chat::set_local_openai_chat_execution_exhausted_diagnostic(
        state, trace_id, decision, plan_kind, body_json, plan_count,
    );
    Ok(None)
}

pub(crate) async fn maybe_execute_sync_via_local_openai_cli_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports = openai_cli::build_local_openai_cli_sync_plan_and_reports_for_kind(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await
}

pub(crate) async fn maybe_execute_stream_via_local_openai_cli_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports = openai_cli::build_local_openai_cli_stream_plan_and_reports_for_kind(
        state, parts, trace_id, decision, body_json, plan_kind,
    )
    .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_stream_plan_and_reports(state, trace_id, decision, plan_kind, plan_and_reports).await
}

pub(crate) async fn maybe_execute_sync_via_standard_family_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
    resolve_sync_spec: fn(&str) -> Option<LocalStandardSpec>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(spec) = resolve_sync_spec(plan_kind) else {
        return Ok(None);
    };

    let plan_and_reports =
        build_standard_sync_plan_and_reports(state, parts, trace_id, decision, body_json, spec)
            .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await
}

pub(crate) async fn maybe_execute_stream_via_standard_family_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
    resolve_stream_spec: fn(&str) -> Option<LocalStandardSpec>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(spec) = resolve_stream_spec(plan_kind) else {
        return Ok(None);
    };

    let plan_and_reports =
        build_standard_stream_plan_and_reports(state, parts, trace_id, decision, body_json, spec)
            .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_stream_plan_and_reports(state, trace_id, decision, plan_kind, plan_and_reports).await
}

pub(crate) async fn maybe_execute_sync_via_local_standard_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    if let Some(response) = maybe_execute_sync_via_standard_family_decision(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            claude::chat::resolve_sync_spec(plan_kind)
                .or_else(|| claude::cli::resolve_sync_spec(plan_kind))
        },
    )
    .await?
    {
        return Ok(Some(response));
    }

    maybe_execute_sync_via_standard_family_decision(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            gemini::chat::resolve_sync_spec(plan_kind)
                .or_else(|| gemini::cli::resolve_sync_spec(plan_kind))
        },
    )
    .await
}

pub(crate) async fn maybe_execute_stream_via_local_standard_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    if let Some(response) = maybe_execute_stream_via_standard_family_decision(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            claude::chat::resolve_stream_spec(plan_kind)
                .or_else(|| claude::cli::resolve_stream_spec(plan_kind))
        },
    )
    .await?
    {
        return Ok(Some(response));
    }

    maybe_execute_stream_via_standard_family_decision(
        state,
        parts,
        trace_id,
        decision,
        body_json,
        plan_kind,
        |plan_kind| {
            gemini::chat::resolve_stream_spec(plan_kind)
                .or_else(|| gemini::cli::resolve_stream_spec(plan_kind))
        },
    )
    .await
}

pub(crate) async fn maybe_execute_sync_via_local_same_format_provider_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(spec) = resolve_same_format_sync_spec(plan_kind) else {
        return Ok(None);
    };

    let plan_and_reports =
        build_same_format_sync_plan_and_reports(state, parts, trace_id, decision, body_json, spec)
            .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await
}

pub(crate) async fn maybe_execute_stream_via_local_same_format_provider_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(spec) = resolve_same_format_stream_spec(plan_kind) else {
        return Ok(None);
    };

    let plan_and_reports = build_same_format_stream_plan_and_reports(
        state, parts, trace_id, decision, body_json, spec,
    )
    .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_stream_plan_and_reports(state, trace_id, decision, plan_kind, plan_and_reports).await
}

pub(crate) async fn maybe_execute_sync_via_local_gemini_files_decision(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports =
        planner::specialized::files::build_local_gemini_files_sync_plan_and_reports_for_kind(
            state,
            parts,
            body_json,
            body_base64,
            body_is_empty,
            trace_id,
            decision,
            plan_kind,
        )
        .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await
}

pub(crate) async fn maybe_execute_stream_via_local_gemini_files_decision(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports =
        planner::specialized::files::build_local_gemini_files_stream_plan_and_reports_for_kind(
            state, parts, trace_id, decision, plan_kind,
        )
        .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_stream_plan_and_reports(state, trace_id, decision, plan_kind, plan_and_reports).await
}

pub(crate) async fn maybe_execute_sync_via_local_video_decision(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    let plan_and_reports =
        planner::specialized::video::build_local_video_sync_plan_and_reports_for_kind(
            state, parts, body_json, trace_id, decision, plan_kind,
        )
        .await?;
    if plan_and_reports.is_empty() {
        return Ok(None);
    }

    execute_sync_plan_and_reports(
        state,
        parts,
        trace_id,
        decision,
        plan_kind,
        plan_and_reports,
    )
    .await
}

pub(crate) async fn maybe_execute_sync_request(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &axum::body::Bytes,
    trace_id: &str,
    decision: Option<&GatewayControlDecision>,
) -> Result<Option<Response<Body>>, GatewayError> {
    runtime::maybe_execute_sync_request(state, parts, body_bytes, trace_id, decision).await
}

pub(crate) async fn maybe_execute_stream_request(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &axum::body::Bytes,
    trace_id: &str,
    decision: Option<&GatewayControlDecision>,
) -> Result<Option<Response<Body>>, GatewayError> {
    runtime::maybe_execute_stream_request(state, parts, body_bytes, trace_id, decision).await
}

pub(crate) fn planner_decision_action(action: &str) -> bool {
    matches!(
        action,
        EXECUTION_RUNTIME_SYNC_DECISION_ACTION | EXECUTION_RUNTIME_STREAM_DECISION_ACTION
    )
}

pub(crate) fn parse_local_request_body(
    parts: &http::request::Parts,
    body_bytes: &axum::body::Bytes,
) -> Option<(serde_json::Value, Option<String>)> {
    parse_direct_request_body(parts, body_bytes)
}

pub(crate) fn decision_payload_is_direct_execution(
    payload: &GatewayControlSyncDecisionResponse,
) -> bool {
    planner_decision_action(payload.action.as_str())
}
