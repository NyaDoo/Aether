use std::collections::BTreeMap;

use aether_data::repository::candidates::{RequestCandidateStatus, UpsertRequestCandidateRecord};
use serde_json::json;
use tracing::warn;
use uuid::Uuid;

use crate::gateway::ai_pipeline::conversion::{
    request_conversion_direct_auth, request_conversion_kind, request_conversion_transport_supported,
};
use crate::gateway::ai_pipeline::planner::plan_builders::{
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::prefer_local_tunnel_owner_candidates;
use crate::gateway::ai_pipeline::planner::{
    EXECUTION_RUNTIME_STREAM_DECISION_ACTION, EXECUTION_RUNTIME_SYNC_DECISION_ACTION,
    OPENAI_CHAT_STREAM_PLAN_KIND,
};
use crate::gateway::headers::collect_control_headers;
use crate::gateway::provider_transport::{
    apply_local_header_rules, build_openai_passthrough_headers, ensure_upstream_auth_header,
    resolve_local_openai_chat_auth, resolve_transport_execution_timeouts,
    resolve_transport_proxy_snapshot_with_tunnel_affinity, resolve_transport_tls_profile,
    supports_local_openai_chat_transport, LocalResolvedOAuthRequestAuth,
};
use crate::gateway::scheduler::{
    record_local_request_candidate_status, GatewayMinimalCandidateSelectionCandidate,
};
use crate::gateway::{
    append_execution_contract_fields_to_value, AppState, ConversionMode, ExecutionStrategy,
    GatewayControlSyncDecisionResponse,
};

use super::plans::current_unix_secs;
use crate::gateway::ai_pipeline::planner::standard::{
    build_cross_format_openai_chat_request_body, build_cross_format_openai_chat_upstream_url,
    build_local_openai_chat_request_body, build_local_openai_chat_upstream_url,
};

#[derive(Debug, Clone)]
pub(super) struct LocalOpenAiChatDecisionInput {
    pub(super) auth_context: crate::gateway::GatewayControlAuthContext,
    pub(super) requested_model: String,
    pub(super) auth_snapshot: crate::gateway::gateway_data::StoredGatewayAuthApiKeySnapshot,
}

#[derive(Debug, Clone)]
pub(super) struct LocalOpenAiChatCandidateAttempt {
    pub(super) candidate: GatewayMinimalCandidateSelectionCandidate,
    pub(super) candidate_index: u32,
    pub(super) candidate_id: String,
}

pub(super) async fn maybe_build_local_openai_chat_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    body_json: &serde_json::Value,
    input: &LocalOpenAiChatDecisionInput,
    attempt: LocalOpenAiChatCandidateAttempt,
    decision_kind: &str,
    report_kind: &str,
    upstream_is_stream: bool,
) -> Option<GatewayControlSyncDecisionResponse> {
    let LocalOpenAiChatCandidateAttempt {
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
            mark_skipped_local_openai_chat_candidate(
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
                "gateway local openai chat decision provider transport read failed"
            );
            mark_skipped_local_openai_chat_candidate(
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

    let provider_api_format = transport.endpoint.api_format.trim().to_ascii_lowercase();
    match provider_api_format.as_str() {
        "openai:chat" => {
            build_same_format_local_openai_chat_decision_payload_for_candidate(
                state,
                parts,
                trace_id,
                body_json,
                input,
                &candidate,
                candidate_index,
                &candidate_id,
                decision_kind,
                report_kind,
                upstream_is_stream,
                &transport,
            )
            .await
        }
        "claude:chat" | "gemini:chat" | "openai:cli" | "openai:compact" => {
            build_cross_format_local_openai_chat_decision_payload_for_candidate(
                state,
                parts,
                trace_id,
                body_json,
                input,
                &candidate,
                candidate_index,
                &candidate_id,
                decision_kind,
                upstream_is_stream,
                &transport,
                provider_api_format.as_str(),
            )
            .await
        }
        _ => {
            mark_skipped_local_openai_chat_candidate(
                state,
                input,
                trace_id,
                &candidate,
                candidate_index,
                &candidate_id,
                "transport_unsupported",
            )
            .await;
            None
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn build_same_format_local_openai_chat_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    body_json: &serde_json::Value,
    input: &LocalOpenAiChatDecisionInput,
    candidate: &GatewayMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    candidate_id: &str,
    decision_kind: &str,
    report_kind: &str,
    upstream_is_stream: bool,
    transport: &crate::gateway::provider_transport::GatewayProviderTransportSnapshot,
) -> Option<GatewayControlSyncDecisionResponse> {
    if !supports_local_openai_chat_transport(transport) {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_unsupported",
        )
        .await;
        return None;
    }

    let oauth_auth = if resolve_local_openai_chat_auth(transport).is_none() {
        match state.resolve_local_oauth_request_auth(transport).await {
            Ok(Some(LocalResolvedOAuthRequestAuth::Header { name, value })) => Some((name, value)),
            Ok(Some(LocalResolvedOAuthRequestAuth::Kiro(_))) => None,
            Ok(None) => None,
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    provider_type = %transport.provider.provider_type,
                    error = ?err,
                    "gateway local openai chat oauth auth resolution failed"
                );
                None
            }
        }
    } else {
        None
    };

    let Some((auth_header, auth_value)) = resolve_local_openai_chat_auth(transport).or(oauth_auth)
    else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_auth_unavailable",
        )
        .await;
        return None;
    };
    let mapped_model = candidate.selected_provider_model_name.trim().to_string();
    if mapped_model.is_empty() {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "mapped_model_missing",
        )
        .await;
        return None;
    }

    let Some(provider_request_body) = build_local_openai_chat_request_body(
        body_json,
        &mapped_model,
        upstream_is_stream,
        transport.endpoint.body_rules.as_ref(),
    ) else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "provider_request_body_missing",
        )
        .await;
        return None;
    };

    let Some(upstream_url) = build_local_openai_chat_upstream_url(parts, transport) else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "upstream_url_missing",
        )
        .await;
        return None;
    };

    let mut provider_request_headers = build_openai_passthrough_headers(
        &parts.headers,
        &auth_header,
        &auth_value,
        &BTreeMap::new(),
        Some("application/json"),
    );
    if !apply_local_header_rules(
        &mut provider_request_headers,
        transport.endpoint.header_rules.as_ref(),
        &[&auth_header, "content-type"],
        &provider_request_body,
        Some(body_json),
    ) {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_header_rules_apply_failed",
        )
        .await;
        return None;
    }
    ensure_upstream_auth_header(&mut provider_request_headers, &auth_header, &auth_value);
    if upstream_is_stream {
        provider_request_headers
            .entry("accept".to_string())
            .or_insert_with(|| "text/event-stream".to_string());
    }
    let proxy = resolve_transport_proxy_snapshot_with_tunnel_affinity(state, transport).await;
    let tls_profile = resolve_transport_tls_profile(transport);
    let prompt_cache_key = provider_request_body
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    Some(GatewayControlSyncDecisionResponse {
        action: if upstream_is_stream {
            EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string()
        } else {
            EXECUTION_RUNTIME_SYNC_DECISION_ACTION.to_string()
        },
        decision_kind: Some(decision_kind.to_string()),
        execution_strategy: Some(ExecutionStrategy::LocalSameFormat.as_str().to_string()),
        conversion_mode: Some(ConversionMode::None.as_str().to_string()),
        request_id: Some(trace_id.to_string()),
        candidate_id: Some(candidate_id.to_string()),
        provider_name: Some(transport.provider.name.clone()),
        provider_id: Some(candidate.provider_id.clone()),
        endpoint_id: Some(candidate.endpoint_id.clone()),
        key_id: Some(candidate.key_id.clone()),
        upstream_base_url: Some(transport.endpoint.base_url.clone()),
        upstream_url: Some(upstream_url.clone()),
        provider_request_method: None,
        auth_header: Some(auth_header),
        auth_value: Some(auth_value),
        provider_api_format: Some("openai:chat".to_string()),
        client_api_format: Some("openai:chat".to_string()),
        provider_contract: Some("openai:chat".to_string()),
        client_contract: Some("openai:chat".to_string()),
        model_name: Some(input.requested_model.clone()),
        mapped_model: Some(mapped_model.clone()),
        prompt_cache_key,
        extra_headers: BTreeMap::new(),
        provider_request_headers: provider_request_headers.clone(),
        provider_request_body: Some(provider_request_body.clone()),
        provider_request_body_base64: None,
        content_type: Some("application/json".to_string()),
        proxy,
        tls_profile,
        timeouts: resolve_transport_execution_timeouts(transport),
        upstream_is_stream,
        report_kind: Some(report_kind.to_string()),
        report_context: Some(append_execution_contract_fields_to_value(
            json!({
                "user_id": input.auth_context.user_id,
                "api_key_id": input.auth_context.api_key_id,
                "request_id": trace_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "retry_index": 0,
                "model": input.requested_model,
                "provider_name": transport.provider.name,
                "provider_id": candidate.provider_id,
                "endpoint_id": candidate.endpoint_id,
                "key_id": candidate.key_id,
                "provider_api_format": "openai:chat",
                "client_api_format": "openai:chat",
                "mapped_model": mapped_model,
                "upstream_url": upstream_url,
                "provider_request_method": serde_json::Value::Null,
                "provider_request_headers": provider_request_headers,
                "provider_request_body": provider_request_body,
                "original_headers": collect_control_headers(&parts.headers),
                "original_request_body": body_json,
                "has_envelope": false,
                "needs_conversion": false,
            }),
            ExecutionStrategy::LocalSameFormat,
            ConversionMode::None,
            "openai:chat",
            "openai:chat",
        )),
        auth_context: Some(input.auth_context.clone()),
    })
}

#[allow(clippy::too_many_arguments)]
async fn build_cross_format_local_openai_chat_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    body_json: &serde_json::Value,
    input: &LocalOpenAiChatDecisionInput,
    candidate: &GatewayMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    candidate_id: &str,
    decision_kind: &str,
    upstream_is_stream: bool,
    transport: &crate::gateway::provider_transport::GatewayProviderTransportSnapshot,
    provider_api_format: &str,
) -> Option<GatewayControlSyncDecisionResponse> {
    let provider_api_format = provider_api_format.trim().to_ascii_lowercase();
    let Some(conversion_kind) =
        request_conversion_kind("openai:chat", provider_api_format.as_str())
    else {
        return None;
    };
    let transport_supported = request_conversion_transport_supported(transport, conversion_kind);
    if !transport_supported {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_unsupported",
        )
        .await;
        return None;
    }

    let resolve_auth = request_conversion_direct_auth(transport, conversion_kind);
    let oauth_auth = if resolve_auth.is_none() {
        match state.resolve_local_oauth_request_auth(transport).await {
            Ok(Some(LocalResolvedOAuthRequestAuth::Header { name, value })) => Some((name, value)),
            Ok(Some(LocalResolvedOAuthRequestAuth::Kiro(_))) => None,
            Ok(None) => None,
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    provider_type = %transport.provider.provider_type,
                    provider_api_format = %provider_api_format,
                    error = ?err,
                    "gateway local openai chat cross-format oauth auth resolution failed"
                );
                None
            }
        }
    } else {
        None
    };

    let Some((auth_header, auth_value)) = resolve_auth.or(oauth_auth) else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_auth_unavailable",
        )
        .await;
        return None;
    };

    let mapped_model = candidate.selected_provider_model_name.trim().to_string();
    if mapped_model.is_empty() {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "mapped_model_missing",
        )
        .await;
        return None;
    }

    let Some(provider_request_body) = build_cross_format_openai_chat_request_body(
        body_json,
        &mapped_model,
        provider_api_format.as_str(),
        upstream_is_stream,
        transport.endpoint.body_rules.as_ref(),
    ) else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "provider_request_body_missing",
        )
        .await;
        return None;
    };

    let Some(upstream_url) = build_cross_format_openai_chat_upstream_url(
        parts,
        transport,
        &mapped_model,
        provider_api_format.as_str(),
        upstream_is_stream,
    ) else {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "upstream_url_missing",
        )
        .await;
        return None;
    };

    let mut provider_request_headers = build_openai_passthrough_headers(
        &parts.headers,
        &auth_header,
        &auth_value,
        &BTreeMap::new(),
        Some("application/json"),
    );
    if !apply_local_header_rules(
        &mut provider_request_headers,
        transport.endpoint.header_rules.as_ref(),
        &[&auth_header, "content-type"],
        &provider_request_body,
        Some(body_json),
    ) {
        mark_skipped_local_openai_chat_candidate(
            state,
            input,
            trace_id,
            candidate,
            candidate_index,
            candidate_id,
            "transport_header_rules_apply_failed",
        )
        .await;
        return None;
    }
    ensure_upstream_auth_header(&mut provider_request_headers, &auth_header, &auth_value);
    if upstream_is_stream {
        provider_request_headers
            .entry("accept".to_string())
            .or_insert_with(|| "text/event-stream".to_string());
    }

    let report_kind = if decision_kind == OPENAI_CHAT_STREAM_PLAN_KIND {
        "openai_chat_stream_success"
    } else {
        "openai_chat_sync_finalize"
    };
    let proxy = resolve_transport_proxy_snapshot_with_tunnel_affinity(state, transport).await;
    let tls_profile = resolve_transport_tls_profile(transport);
    let prompt_cache_key = provider_request_body
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);

    Some(GatewayControlSyncDecisionResponse {
        action: if upstream_is_stream {
            EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string()
        } else {
            EXECUTION_RUNTIME_SYNC_DECISION_ACTION.to_string()
        },
        decision_kind: Some(decision_kind.to_string()),
        execution_strategy: Some(ExecutionStrategy::LocalCrossFormat.as_str().to_string()),
        conversion_mode: Some(ConversionMode::Bidirectional.as_str().to_string()),
        request_id: Some(trace_id.to_string()),
        candidate_id: Some(candidate_id.to_string()),
        provider_name: Some(transport.provider.name.clone()),
        provider_id: Some(candidate.provider_id.clone()),
        endpoint_id: Some(candidate.endpoint_id.clone()),
        key_id: Some(candidate.key_id.clone()),
        upstream_base_url: Some(transport.endpoint.base_url.clone()),
        upstream_url: Some(upstream_url.clone()),
        provider_request_method: None,
        auth_header: Some(auth_header),
        auth_value: Some(auth_value),
        provider_api_format: Some(provider_api_format.clone()),
        client_api_format: Some("openai:chat".to_string()),
        provider_contract: Some(provider_api_format.clone()),
        client_contract: Some("openai:chat".to_string()),
        model_name: Some(input.requested_model.clone()),
        mapped_model: Some(mapped_model.clone()),
        prompt_cache_key,
        extra_headers: BTreeMap::new(),
        provider_request_headers: provider_request_headers.clone(),
        provider_request_body: Some(provider_request_body.clone()),
        provider_request_body_base64: None,
        content_type: Some("application/json".to_string()),
        proxy,
        tls_profile,
        timeouts: resolve_transport_execution_timeouts(transport),
        upstream_is_stream,
        report_kind: Some(report_kind.to_string()),
        report_context: Some(append_execution_contract_fields_to_value(
            json!({
                "user_id": input.auth_context.user_id,
                "api_key_id": input.auth_context.api_key_id,
                "request_id": trace_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "retry_index": 0,
                "model": input.requested_model,
                "provider_name": transport.provider.name,
                "provider_id": candidate.provider_id,
                "endpoint_id": candidate.endpoint_id,
                "key_id": candidate.key_id,
                "provider_api_format": provider_api_format,
                "client_api_format": "openai:chat",
                "mapped_model": mapped_model,
                "upstream_url": upstream_url,
                "provider_request_method": serde_json::Value::Null,
                "provider_request_headers": provider_request_headers,
                "provider_request_body": provider_request_body,
                "original_headers": collect_control_headers(&parts.headers),
                "original_request_body": body_json,
                "has_envelope": false,
                "needs_conversion": true,
            }),
            ExecutionStrategy::LocalCrossFormat,
            ConversionMode::Bidirectional,
            "openai:chat",
            provider_api_format.as_str(),
        )),
        auth_context: Some(input.auth_context.clone()),
    })
}

pub(super) async fn mark_skipped_local_openai_chat_candidate(
    state: &AppState,
    input: &LocalOpenAiChatDecisionInput,
    trace_id: &str,
    candidate: &GatewayMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    candidate_id: &str,
    skip_reason: &'static str,
) {
    state.mutate_local_execution_runtime_miss_diagnostic(trace_id, |diagnostic| {
        *diagnostic
            .skip_reasons
            .entry(skip_reason.to_string())
            .or_insert(0) += 1;
        *diagnostic.skipped_candidate_count.get_or_insert(0) += 1;
    });
    let terminal_unix_secs = current_unix_secs();
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
            finished_at_unix_secs: Some(terminal_unix_secs),
        })
        .await
    {
        warn!(
            trace_id = %trace_id,
            candidate_id = %candidate_id,
            skip_reason,
            error = ?err,
            "gateway local openai chat decision failed to persist skipped candidate"
        );
    }
}

pub(super) async fn materialize_local_openai_chat_candidate_attempts(
    state: &AppState,
    trace_id: &str,
    input: &LocalOpenAiChatDecisionInput,
    candidates: Vec<GatewayMinimalCandidateSelectionCandidate>,
) -> Vec<LocalOpenAiChatCandidateAttempt> {
    let candidates = prefer_local_tunnel_owner_candidates(state, candidates).await;
    let created_at_unix_secs = current_unix_secs();
    let mut attempts = Vec::with_capacity(candidates.len());

    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        let generated_candidate_id = Uuid::new_v4().to_string();
        let provider_api_format = candidate.endpoint_api_format.trim().to_ascii_lowercase();
        let (execution_strategy, conversion_mode) = if provider_api_format == "openai:chat" {
            (ExecutionStrategy::LocalSameFormat, ConversionMode::None)
        } else {
            (
                ExecutionStrategy::LocalCrossFormat,
                ConversionMode::Bidirectional,
            )
        };
        let extra_data = append_execution_contract_fields_to_value(
            json!({
                "provider_api_format": provider_api_format,
                "client_api_format": "openai:chat",
                "global_model_id": candidate.global_model_id.clone(),
                "global_model_name": candidate.global_model_name.clone(),
                "model_id": candidate.model_id.clone(),
                "selected_provider_model_name": candidate.selected_provider_model_name.clone(),
                "mapping_matched_model": candidate.mapping_matched_model.clone(),
                "provider_name": candidate.provider_name.clone(),
                "key_name": candidate.key_name.clone(),
            }),
            execution_strategy,
            conversion_mode,
            "openai:chat",
            candidate.endpoint_api_format.trim(),
        );

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
                    "gateway local openai chat decision request candidate upsert failed"
                );
                generated_candidate_id.clone()
            }
        };

        attempts.push(LocalOpenAiChatCandidateAttempt {
            candidate,
            candidate_index: candidate_index as u32,
            candidate_id,
        });
    }

    attempts
}

pub(super) async fn mark_unused_local_openai_chat_candidates<T>(state: &AppState, remaining: Vec<T>)
where
    T: LocalOpenAiChatPlanAndReport,
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

pub(super) trait LocalOpenAiChatPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan;

    fn report_context(&self) -> Option<&serde_json::Value>;
}

impl LocalOpenAiChatPlanAndReport for LocalSyncPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}

impl LocalOpenAiChatPlanAndReport for LocalStreamPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}
