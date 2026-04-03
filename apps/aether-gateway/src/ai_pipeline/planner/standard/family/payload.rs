use std::collections::BTreeMap;

use aether_data::repository::candidates::{RequestCandidateStatus, UpsertRequestCandidateRecord};
use serde_json::json;
use tracing::warn;

use crate::gateway::headers::collect_control_headers;
use crate::gateway::provider_transport::{
    apply_local_header_rules, build_openai_passthrough_headers, ensure_upstream_auth_header,
    resolve_transport_execution_timeouts, resolve_transport_proxy_snapshot_with_tunnel_affinity,
    resolve_transport_tls_profile, LocalResolvedOAuthRequestAuth,
};
use crate::gateway::scheduler::current_unix_secs;
use crate::gateway::{
    append_execution_contract_fields_to_value, AppState, ConversionMode, ExecutionStrategy,
    GatewayControlSyncDecisionResponse, EXECUTION_RUNTIME_STREAM_DECISION_ACTION,
    EXECUTION_RUNTIME_SYNC_DECISION_ACTION,
};

use super::types::{LocalStandardCandidateAttempt, LocalStandardDecisionInput, LocalStandardSpec};

pub(super) async fn maybe_build_local_standard_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    body_json: &serde_json::Value,
    input: &LocalStandardDecisionInput,
    attempt: LocalStandardCandidateAttempt,
    spec: LocalStandardSpec,
) -> Option<GatewayControlSyncDecisionResponse> {
    let LocalStandardCandidateAttempt {
        candidate,
        candidate_index,
        candidate_id,
    } = attempt;
    let provider_api_format = candidate.endpoint_api_format.trim().to_ascii_lowercase();
    let Some(conversion_kind) = crate::gateway::ai_pipeline::conversion::request_conversion_kind(
        spec.api_format,
        provider_api_format.as_str(),
    ) else {
        if provider_api_format == spec.api_format {
            return None;
        }
        return None;
    };

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
            mark_skipped_local_standard_candidate(
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
                api_format = spec.api_format,
                error = ?err,
                "gateway local standard decision provider transport read failed"
            );
            mark_skipped_local_standard_candidate(
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

    if !crate::gateway::ai_pipeline::conversion::request_conversion_transport_supported(
        &transport,
        conversion_kind,
    ) {
        mark_skipped_local_standard_candidate(
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

    let resolved_auth = crate::gateway::ai_pipeline::conversion::request_conversion_direct_auth(
        &transport,
        conversion_kind,
    );
    let oauth_auth = if resolved_auth.is_none() {
        match state.resolve_local_oauth_request_auth(&transport).await {
            Ok(Some(LocalResolvedOAuthRequestAuth::Header { name, value })) => Some((name, value)),
            Ok(Some(LocalResolvedOAuthRequestAuth::Kiro(_))) => None,
            Ok(None) => None,
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    api_format = spec.api_format,
                    provider_type = %transport.provider.provider_type,
                    error = ?err,
                    "gateway local standard oauth auth resolution failed"
                );
                None
            }
        }
    } else {
        None
    };

    let Some((auth_header, auth_value)) = resolved_auth.or(oauth_auth) else {
        mark_skipped_local_standard_candidate(
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

    let mapped_model = candidate.selected_provider_model_name.trim().to_string();
    if mapped_model.is_empty() {
        mark_skipped_local_standard_candidate(
            state,
            input,
            trace_id,
            &candidate,
            candidate_index,
            &candidate_id,
            "mapped_model_missing",
        )
        .await;
        return None;
    }

    let provider_request_body =
        match crate::gateway::ai_pipeline::planner::standard::build_standard_request_body(
            body_json,
            spec.api_format,
            &mapped_model,
            provider_api_format.as_str(),
            parts.uri.path(),
            spec.require_streaming,
            transport.endpoint.body_rules.as_ref(),
        ) {
            Some(body) => body,
            None => {
                mark_skipped_local_standard_candidate(
                    state,
                    input,
                    trace_id,
                    &candidate,
                    candidate_index,
                    &candidate_id,
                    "provider_request_body_missing",
                )
                .await;
                return None;
            }
        };

    let upstream_url =
        match crate::gateway::ai_pipeline::planner::standard::build_standard_upstream_url(
            parts,
            &transport,
            &mapped_model,
            provider_api_format.as_str(),
            spec.require_streaming,
        ) {
            Some(url) => url,
            None => {
                mark_skipped_local_standard_candidate(
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
            }
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
        mark_skipped_local_standard_candidate(
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
    ensure_upstream_auth_header(&mut provider_request_headers, &auth_header, &auth_value);
    if spec.require_streaming {
        provider_request_headers
            .entry("accept".to_string())
            .or_insert_with(|| "text/event-stream".to_string());
    }

    Some(GatewayControlSyncDecisionResponse {
        action: if spec.require_streaming {
            EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string()
        } else {
            EXECUTION_RUNTIME_SYNC_DECISION_ACTION.to_string()
        },
        decision_kind: Some(spec.decision_kind.to_string()),
        execution_strategy: Some(ExecutionStrategy::LocalCrossFormat.as_str().to_string()),
        conversion_mode: Some(ConversionMode::Bidirectional.as_str().to_string()),
        request_id: Some(trace_id.to_string()),
        candidate_id: Some(candidate_id.clone()),
        provider_name: Some(candidate.provider_name.clone()),
        provider_id: Some(candidate.provider_id.clone()),
        endpoint_id: Some(candidate.endpoint_id.clone()),
        key_id: Some(candidate.key_id.clone()),
        upstream_base_url: Some(transport.endpoint.base_url.clone()),
        upstream_url: Some(upstream_url.clone()),
        provider_request_method: None,
        auth_header: Some(auth_header),
        auth_value: Some(auth_value),
        provider_api_format: Some(provider_api_format.clone()),
        client_api_format: Some(spec.api_format.to_string()),
        provider_contract: Some(provider_api_format.clone()),
        client_contract: Some(spec.api_format.to_string()),
        model_name: Some(input.requested_model.clone()),
        mapped_model: Some(mapped_model.clone()),
        prompt_cache_key: None,
        extra_headers: BTreeMap::new(),
        provider_request_headers: provider_request_headers.clone(),
        provider_request_body: Some(provider_request_body.clone()),
        provider_request_body_base64: None,
        content_type: Some("application/json".to_string()),
        proxy: resolve_transport_proxy_snapshot_with_tunnel_affinity(state, &transport).await,
        tls_profile: resolve_transport_tls_profile(&transport),
        timeouts: resolve_transport_execution_timeouts(&transport),
        upstream_is_stream: spec.require_streaming,
        report_kind: Some(spec.report_kind.to_string()),
        report_context: Some(append_execution_contract_fields_to_value(
            json!({
                "user_id": input.auth_context.user_id,
                "api_key_id": input.auth_context.api_key_id,
                "request_id": trace_id,
                "candidate_id": candidate_id,
                "candidate_index": candidate_index,
                "retry_index": 0,
                "model": input.requested_model,
                "provider_name": candidate.provider_name,
                "provider_id": candidate.provider_id,
                "endpoint_id": candidate.endpoint_id,
                "key_id": candidate.key_id,
                "provider_api_format": provider_api_format,
                "client_api_format": spec.api_format,
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
            spec.api_format,
            candidate.endpoint_api_format.as_str(),
        )),
        auth_context: Some(input.auth_context.clone()),
    })
}

pub(super) async fn mark_skipped_local_standard_candidate(
    state: &AppState,
    input: &LocalStandardDecisionInput,
    trace_id: &str,
    candidate: &crate::gateway::scheduler::GatewayMinimalCandidateSelectionCandidate,
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
            "gateway local standard decision failed to persist skipped candidate"
        );
    }
}
