use std::collections::BTreeMap;

use aether_data::repository::candidates::{RequestCandidateStatus, UpsertRequestCandidateRecord};
use serde_json::json;
use tracing::warn;
use uuid::Uuid;

use crate::gateway::ai_pipeline::planner::plan_builders::{
    build_passthrough_stream_plan_from_decision, build_passthrough_sync_plan_from_decision,
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::prefer_local_tunnel_owner_candidates;
use crate::gateway::ai_pipeline::planner::{
    EXECUTION_RUNTIME_STREAM_DECISION_ACTION, EXECUTION_RUNTIME_SYNC_DECISION_ACTION,
    GEMINI_FILES_DELETE_PLAN_KIND, GEMINI_FILES_DOWNLOAD_PLAN_KIND, GEMINI_FILES_GET_PLAN_KIND,
    GEMINI_FILES_LIST_PLAN_KIND, GEMINI_FILES_UPLOAD_PLAN_KIND,
};
use crate::gateway::headers::collect_control_headers;
use crate::gateway::provider_transport::{
    apply_local_body_rules, apply_local_header_rules, build_gemini_files_passthrough_url,
    build_passthrough_headers_with_auth, resolve_local_gemini_auth,
    resolve_transport_execution_timeouts, resolve_transport_proxy_snapshot_with_tunnel_affinity,
    resolve_transport_tls_profile, supports_local_gemini_transport_with_network,
};
use crate::gateway::scheduler::{
    current_unix_secs, list_selectable_candidates_for_required_capability_without_requested_model,
    record_local_request_candidate_status, GatewayMinimalCandidateSelectionCandidate,
};
use crate::gateway::{
    AppState, GatewayControlDecision, GatewayControlSyncDecisionResponse, GatewayError,
};

const GEMINI_FILES_CANDIDATE_API_FORMAT: &str = "gemini:chat";
const GEMINI_FILES_CLIENT_API_FORMAT: &str = "gemini:files";
const GEMINI_FILES_REQUIRED_CAPABILITY: &str = "gemini_files";

#[derive(Debug, Clone, Copy)]
struct LocalGeminiFilesSpec {
    decision_kind: &'static str,
    report_kind: Option<&'static str>,
    require_streaming: bool,
}

#[derive(Debug, Clone)]
struct LocalGeminiFilesDecisionInput {
    auth_context: crate::gateway::GatewayControlAuthContext,
    auth_snapshot: crate::gateway::gateway_data::StoredGatewayAuthApiKeySnapshot,
}

#[derive(Debug, Clone)]
struct LocalGeminiFilesCandidateAttempt {
    candidate: GatewayMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    candidate_id: String,
}

pub(crate) async fn build_local_gemini_files_sync_plan_and_reports_for_kind(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Vec<LocalSyncPlanAndReport>, GatewayError> {
    let Some(spec) = resolve_sync_spec(plan_kind) else {
        return Ok(Vec::new());
    };

    build_local_sync_plan_and_reports(
        state,
        parts,
        body_json,
        body_base64,
        body_is_empty,
        trace_id,
        decision,
        spec,
    )
    .await
}

pub(crate) async fn build_local_gemini_files_stream_plan_and_reports_for_kind(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Vec<LocalStreamPlanAndReport>, GatewayError> {
    let Some(spec) = resolve_stream_spec(plan_kind) else {
        return Ok(Vec::new());
    };

    build_local_stream_plan_and_reports(state, parts, trace_id, decision, spec).await
}

pub(crate) async fn maybe_build_sync_local_gemini_files_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    let Some(spec) = resolve_sync_spec(plan_kind) else {
        return Ok(None);
    };

    let Some(input) = resolve_local_gemini_files_decision_input(state, trace_id, decision).await
    else {
        return Ok(None);
    };

    let attempts =
        materialize_local_gemini_files_candidate_attempts(state, trace_id, &input).await?;

    for attempt in attempts {
        if let Some(payload) = maybe_build_local_gemini_files_decision_payload_for_candidate(
            state,
            parts,
            body_json,
            body_base64,
            body_is_empty,
            trace_id,
            &input,
            attempt,
            spec,
        )
        .await
        {
            return Ok(Some(payload));
        }
    }

    Ok(None)
}

pub(crate) async fn maybe_build_stream_local_gemini_files_decision_payload(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
) -> Result<Option<GatewayControlSyncDecisionResponse>, GatewayError> {
    let Some(spec) = resolve_stream_spec(plan_kind) else {
        return Ok(None);
    };

    let Some(input) = resolve_local_gemini_files_decision_input(state, trace_id, decision).await
    else {
        return Ok(None);
    };

    let attempts =
        materialize_local_gemini_files_candidate_attempts(state, trace_id, &input).await?;

    let empty_body_json = serde_json::Value::Null;
    for attempt in attempts {
        if let Some(payload) = maybe_build_local_gemini_files_decision_payload_for_candidate(
            state,
            parts,
            &empty_body_json,
            None,
            true,
            trace_id,
            &input,
            attempt,
            spec,
        )
        .await
        {
            return Ok(Some(payload));
        }
    }

    Ok(None)
}

fn resolve_sync_spec(plan_kind: &str) -> Option<LocalGeminiFilesSpec> {
    match plan_kind {
        GEMINI_FILES_UPLOAD_PLAN_KIND => Some(LocalGeminiFilesSpec {
            decision_kind: GEMINI_FILES_UPLOAD_PLAN_KIND,
            report_kind: Some("gemini_files_store_mapping"),
            require_streaming: false,
        }),
        GEMINI_FILES_LIST_PLAN_KIND => Some(LocalGeminiFilesSpec {
            decision_kind: GEMINI_FILES_LIST_PLAN_KIND,
            report_kind: Some("gemini_files_store_mapping"),
            require_streaming: false,
        }),
        GEMINI_FILES_GET_PLAN_KIND => Some(LocalGeminiFilesSpec {
            decision_kind: GEMINI_FILES_GET_PLAN_KIND,
            report_kind: Some("gemini_files_store_mapping"),
            require_streaming: false,
        }),
        GEMINI_FILES_DELETE_PLAN_KIND => Some(LocalGeminiFilesSpec {
            decision_kind: GEMINI_FILES_DELETE_PLAN_KIND,
            report_kind: Some("gemini_files_delete_mapping"),
            require_streaming: false,
        }),
        _ => None,
    }
}

fn resolve_stream_spec(plan_kind: &str) -> Option<LocalGeminiFilesSpec> {
    match plan_kind {
        GEMINI_FILES_DOWNLOAD_PLAN_KIND => Some(LocalGeminiFilesSpec {
            decision_kind: GEMINI_FILES_DOWNLOAD_PLAN_KIND,
            report_kind: None,
            require_streaming: true,
        }),
        _ => None,
    }
}

async fn build_local_sync_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
    trace_id: &str,
    decision: &GatewayControlDecision,
    spec: LocalGeminiFilesSpec,
) -> Result<Vec<LocalSyncPlanAndReport>, GatewayError> {
    let Some(input) = resolve_local_gemini_files_decision_input(state, trace_id, decision).await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_gemini_files_candidate_attempts(state, trace_id, &input).await?;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_gemini_files_decision_payload_for_candidate(
            state,
            parts,
            body_json,
            body_base64,
            body_is_empty,
            trace_id,
            &input,
            attempt,
            spec,
        )
        .await
        else {
            continue;
        };

        match build_passthrough_sync_plan_from_decision(parts, payload) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    decision_kind = spec.decision_kind,
                    error = ?err,
                    "gateway local gemini files sync decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}

async fn build_local_stream_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    spec: LocalGeminiFilesSpec,
) -> Result<Vec<LocalStreamPlanAndReport>, GatewayError> {
    let Some(input) = resolve_local_gemini_files_decision_input(state, trace_id, decision).await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_gemini_files_candidate_attempts(state, trace_id, &input).await?;

    let mut plans = Vec::new();
    let empty_body_json = serde_json::Value::Null;
    for attempt in attempts {
        let Some(payload) = maybe_build_local_gemini_files_decision_payload_for_candidate(
            state,
            parts,
            &empty_body_json,
            None,
            true,
            trace_id,
            &input,
            attempt,
            spec,
        )
        .await
        else {
            continue;
        };

        match build_passthrough_stream_plan_from_decision(parts, payload) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    decision_kind = spec.decision_kind,
                    error = ?err,
                    "gateway local gemini files stream decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}

async fn resolve_local_gemini_files_decision_input(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Option<LocalGeminiFilesDecisionInput> {
    let Some(auth_context) = decision.auth_context.clone().filter(|auth_context| {
        !auth_context.user_id.trim().is_empty() && !auth_context.api_key_id.trim().is_empty()
    }) else {
        return None;
    };

    let auth_snapshot = match state
        .read_auth_api_key_snapshot(
            &auth_context.user_id,
            &auth_context.api_key_id,
            current_unix_secs(),
        )
        .await
    {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => return None,
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local gemini files decision auth snapshot read failed"
            );
            return None;
        }
    };

    Some(LocalGeminiFilesDecisionInput {
        auth_context,
        auth_snapshot,
    })
}

async fn materialize_local_gemini_files_candidate_attempts(
    state: &AppState,
    trace_id: &str,
    input: &LocalGeminiFilesDecisionInput,
) -> Result<Vec<LocalGeminiFilesCandidateAttempt>, GatewayError> {
    let candidates = list_selectable_candidates_for_required_capability_without_requested_model(
        state,
        GEMINI_FILES_CANDIDATE_API_FORMAT,
        GEMINI_FILES_REQUIRED_CAPABILITY,
        false,
        Some(&input.auth_snapshot),
        current_unix_secs(),
    )
    .await?;
    let candidates = prefer_local_tunnel_owner_candidates(state, candidates).await;

    let created_at_unix_secs = current_unix_secs();
    let mut attempts = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        let generated_candidate_id = Uuid::new_v4().to_string();
        let extra_data = json!({
            "provider_api_format": GEMINI_FILES_CLIENT_API_FORMAT,
            "client_api_format": GEMINI_FILES_CLIENT_API_FORMAT,
            "candidate_api_format": GEMINI_FILES_CANDIDATE_API_FORMAT,
            "global_model_id": candidate.global_model_id.clone(),
            "global_model_name": candidate.global_model_name.clone(),
            "model_id": candidate.model_id.clone(),
            "selected_provider_model_name": candidate.selected_provider_model_name.clone(),
            "mapping_matched_model": candidate.mapping_matched_model.clone(),
            "provider_name": candidate.provider_name.clone(),
            "key_name": candidate.key_name.clone(),
        });

        let candidate_id = match state
            .upsert_request_candidate(UpsertRequestCandidateRecord {
                id: generated_candidate_id.clone(),
                request_id: trace_id.to_string(),
                user_id: Some(input.auth_context.user_id.clone()),
                api_key_id: Some(input.auth_context.api_key_id.clone()),
                username: None,
                api_key_name: None,
                candidate_index: candidate_index as u32,
                retry_index: 0,
                provider_id: Some(candidate.provider_id.clone()),
                endpoint_id: Some(candidate.endpoint_id.clone()),
                key_id: Some(candidate.key_id.clone()),
                status: RequestCandidateStatus::Available,
                skip_reason: None,
                is_cached: Some(false),
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                concurrent_requests: None,
                extra_data: Some(extra_data),
                required_capabilities: candidate.key_capabilities.clone(),
                created_at_unix_secs: Some(created_at_unix_secs),
                started_at_unix_secs: None,
                finished_at_unix_secs: None,
            })
            .await
        {
            Ok(Some(stored)) => stored.id,
            Ok(None) => generated_candidate_id.clone(),
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    error = ?err,
                    "gateway local gemini files request candidate upsert failed"
                );
                generated_candidate_id.clone()
            }
        };

        attempts.push(LocalGeminiFilesCandidateAttempt {
            candidate,
            candidate_index: candidate_index as u32,
            candidate_id,
        });
    }

    Ok(attempts)
}

async fn maybe_build_local_gemini_files_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    body_json: &serde_json::Value,
    body_base64: Option<&str>,
    body_is_empty: bool,
    trace_id: &str,
    input: &LocalGeminiFilesDecisionInput,
    attempt: LocalGeminiFilesCandidateAttempt,
    spec: LocalGeminiFilesSpec,
) -> Option<GatewayControlSyncDecisionResponse> {
    let LocalGeminiFilesCandidateAttempt {
        candidate,
        candidate_index,
        candidate_id,
    } = attempt;

    let transport = match state
        .read_provider_transport_snapshot(
            &candidate.provider_id,
            &candidate.endpoint_id,
            &candidate.key_id,
        )
        .await
    {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            mark_skipped_local_gemini_files_candidate(
                state,
                input,
                trace_id,
                &candidate,
                candidate_index,
                &candidate_id,
                "transport_snapshot_missing",
            )
            .await;
            return None;
        }
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local gemini files provider transport read failed"
            );
            mark_skipped_local_gemini_files_candidate(
                state,
                input,
                trace_id,
                &candidate,
                candidate_index,
                &candidate_id,
                "transport_snapshot_read_failed",
            )
            .await;
            return None;
        }
    };

    if !supports_local_gemini_transport_with_network(&transport, GEMINI_FILES_CANDIDATE_API_FORMAT)
    {
        mark_skipped_local_gemini_files_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "transport_unsupported",
        )
        .await;
        return None;
    }

    let Some((auth_header, auth_value)) = resolve_local_gemini_auth(&transport) else {
        mark_skipped_local_gemini_files_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "transport_auth_unavailable",
        )
        .await;
        return None;
    };

    let custom_path = transport
        .endpoint
        .custom_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let passthrough_path = custom_path.unwrap_or(parts.uri.path());
    let upstream_url =
        if spec.decision_kind == GEMINI_FILES_UPLOAD_PLAN_KIND || custom_path.is_some() {
            build_gemini_files_passthrough_url(
                &transport.endpoint.base_url,
                passthrough_path,
                parts.uri.query(),
            )
        } else {
            build_gemini_files_passthrough_url(
                &transport.endpoint.base_url,
                passthrough_path,
                parts.uri.query(),
            )
        };
    let Some(upstream_url) = upstream_url else {
        mark_skipped_local_gemini_files_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "upstream_url_missing",
        )
        .await;
        return None;
    };

    let mut provider_request_body = if spec.decision_kind == GEMINI_FILES_UPLOAD_PLAN_KIND
        && !body_is_empty
        && body_base64.is_none()
    {
        Some(body_json.clone())
    } else {
        None
    };
    let provider_request_body_base64 = if spec.decision_kind == GEMINI_FILES_UPLOAD_PLAN_KIND {
        body_base64
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
    } else {
        None
    };
    let original_request_body = if let Some(body_bytes_b64) = provider_request_body_base64.clone() {
        json!({"body_bytes_b64": body_bytes_b64})
    } else if !body_is_empty {
        body_json.clone()
    } else {
        serde_json::Value::Null
    };
    if provider_request_body_base64.is_some() && transport.endpoint.body_rules.is_some() {
        mark_skipped_local_gemini_files_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "transport_body_rules_unsupported_for_binary_upload",
        )
        .await;
        return None;
    }
    if let Some(body) = provider_request_body.as_mut() {
        if !apply_local_body_rules(
            body,
            transport.endpoint.body_rules.as_ref(),
            Some(body_json),
        ) {
            mark_skipped_local_gemini_files_candidate(
                state,
                input,
                trace_id,
                &candidate,
                candidate_index,
                &candidate_id,
                "transport_body_rules_apply_failed",
            )
            .await;
            return None;
        }
    }
    let mut provider_request_headers = build_passthrough_headers_with_auth(
        &parts.headers,
        &auth_header,
        &auth_value,
        &BTreeMap::new(),
    );
    if !apply_local_header_rules(
        &mut provider_request_headers,
        transport.endpoint.header_rules.as_ref(),
        &[&auth_header, "content-type"],
        provider_request_body
            .as_ref()
            .unwrap_or(&original_request_body),
        Some(&original_request_body),
    ) {
        mark_skipped_local_gemini_files_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "transport_header_rules_apply_failed",
        )
        .await;
        return None;
    }
    let file_name = parts
        .uri
        .path()
        .trim_start_matches("/v1beta/")
        .trim()
        .to_string();
    let proxy = resolve_transport_proxy_snapshot_with_tunnel_affinity(state, &transport).await;
    let tls_profile = resolve_transport_tls_profile(&transport);

    Some(GatewayControlSyncDecisionResponse {
        action: if spec.require_streaming {
            EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string()
        } else {
            EXECUTION_RUNTIME_SYNC_DECISION_ACTION.to_string()
        },
        decision_kind: Some(spec.decision_kind.to_string()),
        execution_strategy: Some(
            crate::gateway::ExecutionStrategy::LocalSameFormat
                .as_str()
                .to_string(),
        ),
        conversion_mode: Some(crate::gateway::ConversionMode::None.as_str().to_string()),
        request_id: Some(trace_id.to_string()),
        candidate_id: Some(candidate_id.clone()),
        provider_name: Some(transport.provider.name.clone()),
        provider_id: Some(candidate.provider_id.clone()),
        endpoint_id: Some(candidate.endpoint_id.clone()),
        key_id: Some(candidate.key_id.clone()),
        upstream_base_url: Some(transport.endpoint.base_url.clone()),
        upstream_url: Some(upstream_url),
        provider_request_method: Some(parts.method.to_string()),
        auth_header: Some(auth_header),
        auth_value: Some(auth_value),
        provider_api_format: Some(GEMINI_FILES_CLIENT_API_FORMAT.to_string()),
        client_api_format: Some(GEMINI_FILES_CLIENT_API_FORMAT.to_string()),
        provider_contract: Some(GEMINI_FILES_CLIENT_API_FORMAT.to_string()),
        client_contract: Some(GEMINI_FILES_CLIENT_API_FORMAT.to_string()),
        model_name: Some("gemini-files".to_string()),
        mapped_model: Some(candidate.selected_provider_model_name.clone()),
        prompt_cache_key: None,
        extra_headers: BTreeMap::new(),
        provider_request_headers,
        provider_request_body,
        provider_request_body_base64,
        content_type: parts
            .headers
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned),
        proxy,
        tls_profile,
        timeouts: resolve_transport_execution_timeouts(&transport),
        upstream_is_stream: spec.require_streaming,
        report_kind: spec.report_kind.map(ToOwned::to_owned),
        report_context: Some(json!({
            "user_id": input.auth_context.user_id,
            "api_key_id": input.auth_context.api_key_id,
            "request_id": trace_id,
            "candidate_id": candidate_id,
            "candidate_index": candidate_index,
            "retry_index": 0,
            "model": "gemini-files",
            "provider_name": transport.provider.name,
            "provider_id": candidate.provider_id,
            "endpoint_id": candidate.endpoint_id,
            "key_id": candidate.key_id,
            "file_key_id": candidate.key_id,
            "file_name": file_name,
            "provider_api_format": GEMINI_FILES_CLIENT_API_FORMAT,
            "client_api_format": GEMINI_FILES_CLIENT_API_FORMAT,
            "original_headers": collect_control_headers(&parts.headers),
            "original_request_body": original_request_body,
            "has_envelope": false,
            "needs_conversion": false,
        })),
        auth_context: Some(input.auth_context.clone()),
    })
}

async fn mark_skipped_local_gemini_files_candidate(
    state: &AppState,
    input: &LocalGeminiFilesDecisionInput,
    trace_id: &str,
    candidate: &GatewayMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    candidate_id: &str,
    skip_reason: &'static str,
) {
    if let Err(err) = state
        .upsert_request_candidate(UpsertRequestCandidateRecord {
            id: candidate_id.to_string(),
            request_id: trace_id.to_string(),
            user_id: Some(input.auth_context.user_id.clone()),
            api_key_id: Some(input.auth_context.api_key_id.clone()),
            username: None,
            api_key_name: None,
            candidate_index,
            retry_index: 0,
            provider_id: Some(candidate.provider_id.clone()),
            endpoint_id: Some(candidate.endpoint_id.clone()),
            key_id: Some(candidate.key_id.clone()),
            status: RequestCandidateStatus::Skipped,
            skip_reason: Some(skip_reason.to_string()),
            is_cached: Some(false),
            status_code: None,
            error_type: None,
            error_message: None,
            latency_ms: None,
            concurrent_requests: None,
            extra_data: None,
            required_capabilities: candidate.key_capabilities.clone(),
            created_at_unix_secs: None,
            started_at_unix_secs: None,
            finished_at_unix_secs: Some(current_unix_secs()),
        })
        .await
    {
        warn!(
            trace_id = %trace_id,
            candidate_id = %candidate_id,
            skip_reason,
            error = ?err,
            "gateway local gemini files failed to persist skipped candidate"
        );
    }
}

async fn mark_unused_local_files_candidates<T>(state: &AppState, remaining: Vec<T>)
where
    T: LocalGeminiFilesPlanAndReport,
{
    for plan_and_report in remaining {
        record_local_request_candidate_status(
            state,
            plan_and_report.plan(),
            plan_and_report.report_context(),
            RequestCandidateStatus::Unused,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await;
    }
}

trait LocalGeminiFilesPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan;

    fn report_context(&self) -> Option<&serde_json::Value>;
}

impl LocalGeminiFilesPlanAndReport for LocalSyncPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}

impl LocalGeminiFilesPlanAndReport for LocalStreamPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}
