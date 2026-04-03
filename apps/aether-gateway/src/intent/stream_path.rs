use axum::body::{Body, Bytes};
use axum::http::Response;
use std::collections::BTreeMap;

use crate::gateway::ai_pipeline::planner::{
    maybe_build_stream_decision_payload, EXECUTION_RUNTIME_STREAM_DECISION_ACTION,
    OPENAI_VIDEO_CONTENT_PLAN_KIND,
};
use crate::gateway::ai_pipeline::planner::{
    maybe_build_stream_local_decision_payload,
    maybe_build_stream_local_gemini_files_decision_payload,
    maybe_build_stream_local_openai_cli_decision_payload,
    maybe_build_stream_local_same_format_provider_decision_payload,
};
use crate::gateway::api::response::build_client_response_from_parts;
use crate::gateway::executor::maybe_execute_stream_via_local_standard_decision;
use crate::gateway::executor::{
    maybe_execute_stream_via_local_decision, maybe_execute_stream_via_local_gemini_files_decision,
    maybe_execute_stream_via_local_openai_cli_decision,
    maybe_execute_stream_via_local_same_format_provider_decision, parse_local_request_body,
};
use crate::gateway::scheduler::{
    is_matching_stream_request, resolve_execution_runtime_stream_plan_kind,
    supports_stream_scheduler_decision_kind,
};
use crate::gateway::{
    execute_execution_runtime_stream, AppState, GatewayControlDecision,
    GatewayControlSyncDecisionResponse, GatewayError, GatewayFallbackReason,
};

pub(crate) async fn maybe_build_stream_decision_payload_via_local_path(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    maybe_build_stream_decision_payload(state, parts, trace_id, decision, body_json).await
}

pub(crate) async fn maybe_execute_via_stream_decision_path(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: &Bytes,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(plan_kind) = resolve_execution_runtime_stream_plan_kind(parts, decision) else {
        return Ok(None);
    };

    let Some((body_json, body_base64)) = parse_local_request_body(parts, body_bytes) else {
        return Ok(None);
    };

    if !is_matching_stream_request(plan_kind, parts, &body_json) {
        return Ok(None);
    }

    let bypass_cache_key =
        super::build_intent_plan_bypass_cache_key(plan_kind, parts, body_bytes, decision);
    if super::should_skip_intent_plan(state, &bypass_cache_key) {
        return Ok(None);
    }

    if let Some(response) =
        maybe_execute_local_video_task_content_stream(state, parts, trace_id, decision, plan_kind)
            .await?
    {
        return Ok(Some(response));
    }

    if supports_stream_scheduler_decision_kind(plan_kind) {
        if let Some(response) = maybe_execute_stream_via_local_decision(
            state, parts, trace_id, decision, &body_json, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }

        if let Some(response) = maybe_execute_stream_via_local_openai_cli_decision(
            state, parts, trace_id, decision, &body_json, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }

        if let Some(response) = maybe_execute_stream_via_local_standard_decision(
            state, parts, trace_id, decision, &body_json, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }

        if let Some(response) = maybe_execute_stream_via_local_same_format_provider_decision(
            state, parts, trace_id, decision, &body_json, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }

        if let Some(response) = maybe_execute_stream_via_local_gemini_files_decision(
            state, parts, trace_id, decision, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }

        if let Some(response) = super::maybe_execute_stream_via_remote_decision(
            state, parts, trace_id, decision, &body_json, plan_kind,
        )
        .await?
        {
            return Ok(Some(response));
        }
    }

    super::maybe_execute_stream_via_plan_fallback(
        state,
        parts,
        trace_id,
        decision,
        &body_json,
        body_base64,
        plan_kind,
        bypass_cache_key,
        if supports_stream_scheduler_decision_kind(plan_kind) {
            GatewayFallbackReason::RemoteDecisionMiss
        } else {
            GatewayFallbackReason::SchedulerDecisionUnsupported
        },
    )
    .await
}

async fn maybe_execute_local_video_task_content_stream(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<Response<Body>>, GatewayError> {
    if plan_kind != OPENAI_VIDEO_CONTENT_PLAN_KIND
        || decision.route_family.as_deref() != Some("openai")
    {
        return Ok(None);
    }

    let _ = state
        .hydrate_video_task_for_route(decision.route_family.as_deref(), parts.uri.path())
        .await?;

    if let Some(task_id) =
        crate::gateway::video_tasks::extract_openai_task_id_from_content_path(parts.uri.path())
    {
        let refresh_path = format!("/v1/videos/{task_id}");
        if let Some(refresh_plan) = state.video_tasks.prepare_read_refresh_sync_plan(
            Some("openai"),
            &refresh_path,
            trace_id,
        ) {
            state.execute_video_task_refresh_plan(&refresh_plan).await?;
        }
    }

    let Some(action) = state.video_tasks.prepare_openai_content_stream_action(
        parts.uri.path(),
        parts.uri.query(),
        trace_id,
    ) else {
        return Ok(None);
    };

    match action {
        crate::gateway::video_tasks::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } => Ok(Some(build_json_response(
            trace_id,
            decision,
            status_code,
            &body_json,
        )?)),
        crate::gateway::video_tasks::LocalVideoTaskContentAction::StreamPlan(plan) => {
            execute_execution_runtime_stream(state, plan, trace_id, decision, plan_kind, None, None)
                .await
        }
    }
}

fn build_json_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    status_code: u16,
    body_json: &serde_json::Value,
) -> Result<Response<Body>, GatewayError> {
    let body_bytes =
        serde_json::to_vec(body_json).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let mut headers = BTreeMap::new();
    headers.insert("content-type".to_string(), "application/json".to_string());
    headers.insert("content-length".to_string(), body_bytes.len().to_string());
    build_client_response_from_parts(
        status_code,
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )
}

async fn maybe_build_local_video_task_content_stream_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    if plan_kind != OPENAI_VIDEO_CONTENT_PLAN_KIND
        || decision.route_family.as_deref() != Some("openai")
    {
        return Ok(None);
    }

    let _ = state
        .hydrate_video_task_for_route(decision.route_family.as_deref(), parts.uri.path())
        .await?;

    let Some(action) = state.video_tasks.prepare_openai_content_stream_action(
        parts.uri.path(),
        parts.uri.query(),
        trace_id,
    ) else {
        return Ok(None);
    };

    let crate::gateway::video_tasks::LocalVideoTaskContentAction::StreamPlan(plan) = action else {
        return Ok(None);
    };
    let provider_contract = plan.provider_api_format.clone();
    let client_contract = plan.client_api_format.clone();
    let execution_strategy = if plan.provider_api_format == plan.client_api_format {
        crate::gateway::ExecutionStrategy::LocalSameFormat
    } else {
        crate::gateway::ExecutionStrategy::LocalCrossFormat
    };
    let conversion_mode = if plan.provider_api_format == plan.client_api_format {
        crate::gateway::ConversionMode::None
    } else {
        crate::gateway::ConversionMode::Bidirectional
    };

    Ok(Some(GatewayControlSyncDecisionResponse {
        action: EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string(),
        decision_kind: Some(plan_kind.to_string()),
        execution_strategy: Some(execution_strategy.as_str().to_string()),
        conversion_mode: Some(conversion_mode.as_str().to_string()),
        request_id: Some(plan.request_id),
        candidate_id: plan.candidate_id,
        provider_name: plan.provider_name,
        provider_id: Some(plan.provider_id),
        endpoint_id: Some(plan.endpoint_id),
        key_id: Some(plan.key_id),
        upstream_base_url: None,
        upstream_url: Some(plan.url),
        provider_request_method: Some(plan.method),
        auth_header: None,
        auth_value: None,
        provider_api_format: Some(plan.provider_api_format),
        client_api_format: Some(plan.client_api_format),
        provider_contract: Some(provider_contract),
        client_contract: Some(client_contract),
        model_name: plan.model_name,
        mapped_model: None,
        prompt_cache_key: None,
        extra_headers: BTreeMap::new(),
        provider_request_headers: plan.headers,
        provider_request_body: None,
        provider_request_body_base64: None,
        content_type: plan.content_type,
        proxy: plan.proxy,
        tls_profile: plan.tls_profile,
        timeouts: plan.timeouts,
        upstream_is_stream: true,
        report_kind: None,
        report_context: None,
        auth_context: decision.auth_context.clone(),
    }))
}
