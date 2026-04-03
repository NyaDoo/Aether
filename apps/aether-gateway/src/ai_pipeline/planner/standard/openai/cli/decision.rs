use std::collections::{BTreeMap, BTreeSet};

use aether_data::repository::candidates::{RequestCandidateStatus, UpsertRequestCandidateRecord};
use serde_json::{json, Value};
use tracing::warn;
use uuid::Uuid;

use crate::gateway::ai_pipeline::conversion::{
    request_conversion_direct_auth, request_conversion_kind, request_conversion_transport_supported,
};
use crate::gateway::ai_pipeline::planner::plan_builders::{
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::prefer_local_tunnel_owner_candidates;
use crate::gateway::ai_pipeline::planner::standard::{
    build_cross_format_openai_cli_request_body, build_cross_format_openai_cli_upstream_url,
    build_local_openai_cli_request_body, build_local_openai_cli_upstream_url,
};
use crate::gateway::ai_pipeline::planner::{
    EXECUTION_RUNTIME_STREAM_DECISION_ACTION, EXECUTION_RUNTIME_SYNC_DECISION_ACTION,
};
use crate::gateway::headers::collect_control_headers;
use crate::gateway::provider_transport::{
    apply_local_body_rules, apply_local_header_rules, build_antigravity_safe_v1internal_request,
    build_antigravity_static_identity_headers, build_openai_passthrough_headers,
    classify_local_antigravity_request_support, ensure_upstream_auth_header,
    resolve_local_gemini_auth, resolve_local_standard_auth, resolve_transport_execution_timeouts,
    resolve_transport_proxy_snapshot_with_tunnel_affinity, resolve_transport_tls_profile,
    supports_local_standard_transport_with_network, AntigravityEnvelopeRequestType,
    AntigravityRequestEnvelopeSupport, AntigravityRequestSideSupport,
    LocalResolvedOAuthRequestAuth,
};
use crate::gateway::scheduler::{
    current_unix_secs, list_selectable_candidates, record_local_request_candidate_status,
    GatewayMinimalCandidateSelectionCandidate,
};
use crate::gateway::{
    append_execution_contract_fields_to_value, AppState, ConversionMode, ExecutionStrategy,
    GatewayControlDecision, GatewayControlSyncDecisionResponse, GatewayError,
};

const ANTIGRAVITY_ENVELOPE_NAME: &str = "antigravity:v1internal";

#[derive(Debug, Clone, Copy)]
pub(super) struct LocalOpenAiCliSpec {
    pub(super) api_format: &'static str,
    pub(super) decision_kind: &'static str,
    pub(super) report_kind: &'static str,
    pub(super) compact: bool,
    pub(super) require_streaming: bool,
}

#[derive(Debug, Clone)]
pub(super) struct LocalOpenAiCliDecisionInput {
    pub(super) auth_context: crate::gateway::GatewayControlAuthContext,
    pub(super) requested_model: String,
    pub(super) auth_snapshot: crate::gateway::gateway_data::StoredGatewayAuthApiKeySnapshot,
}

#[derive(Debug, Clone)]
pub(super) struct LocalOpenAiCliCandidateAttempt {
    pub(super) candidate: GatewayMinimalCandidateSelectionCandidate,
    pub(super) candidate_index: u32,
    pub(super) candidate_id: String,
}

pub(super) async fn resolve_local_openai_cli_decision_input(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
) -> Option<LocalOpenAiCliDecisionInput> {
    let Some(auth_context) = decision.auth_context.clone().filter(|auth_context| {
        !auth_context.user_id.trim().is_empty() && !auth_context.api_key_id.trim().is_empty()
    }) else {
        return None;
    };

    let requested_model = body_json
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;

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
                "gateway local openai cli decision auth snapshot read failed"
            );
            return None;
        }
    };

    Some(LocalOpenAiCliDecisionInput {
        auth_context,
        requested_model,
        auth_snapshot,
    })
}

pub(super) async fn materialize_local_openai_cli_candidate_attempts(
    state: &AppState,
    trace_id: &str,
    input: &LocalOpenAiCliDecisionInput,
    spec: LocalOpenAiCliSpec,
) -> Result<Vec<LocalOpenAiCliCandidateAttempt>, GatewayError> {
    let mut seen_candidates = BTreeSet::new();
    let mut candidates = Vec::new();
    for candidate_api_format in candidate_api_formats_for_spec(spec) {
        let auth_snapshot = if *candidate_api_format == spec.api_format {
            Some(&input.auth_snapshot)
        } else {
            None
        };
        let mut selected_candidates = list_selectable_candidates(
            state,
            candidate_api_format,
            &input.requested_model,
            spec.require_streaming,
            auth_snapshot,
            current_unix_secs(),
        )
        .await?;
        if auth_snapshot.is_none() {
            selected_candidates.retain(|candidate| {
                auth_snapshot_allows_cross_format_openai_cli_candidate(
                    &input.auth_snapshot,
                    &input.requested_model,
                    candidate,
                )
            });
        }
        for candidate in selected_candidates {
            let candidate_key = format!(
                "{}:{}:{}:{}:{}:{}",
                candidate.provider_id,
                candidate.endpoint_id,
                candidate.key_id,
                candidate.model_id,
                candidate.selected_provider_model_name,
                candidate.endpoint_api_format,
            );
            if seen_candidates.insert(candidate_key) {
                candidates.push(candidate);
            }
        }
    }
    let candidates = prefer_local_tunnel_owner_candidates(state, candidates).await;

    let created_at_unix_secs = current_unix_secs();
    let mut attempts = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        let generated_candidate_id = Uuid::new_v4().to_string();
        let provider_api_format = candidate.endpoint_api_format.trim().to_ascii_lowercase();
        let execution_strategy =
            if provider_api_format == spec.api_format.trim().to_ascii_lowercase() {
                ExecutionStrategy::LocalSameFormat
            } else {
                ExecutionStrategy::LocalCrossFormat
            };
        let conversion_mode =
            if request_conversion_kind(spec.api_format, provider_api_format.as_str()).is_some() {
                ConversionMode::Bidirectional
            } else {
                ConversionMode::None
            };
        let extra_data = append_execution_contract_fields_to_value(
            json!({
                "provider_api_format": provider_api_format,
                "client_api_format": spec.api_format,
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
            spec.api_format,
            candidate.endpoint_api_format.as_str(),
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
                    api_format = spec.api_format,
                    error = ?err,
                    "gateway local openai cli decision request candidate upsert failed"
                );
                generated_candidate_id.clone()
            }
        };

        attempts.push(LocalOpenAiCliCandidateAttempt {
            candidate,
            candidate_index: candidate_index as u32,
            candidate_id,
        });
    }

    Ok(attempts)
}

fn auth_snapshot_allows_cross_format_openai_cli_candidate(
    auth_snapshot: &crate::gateway::gateway_data::StoredGatewayAuthApiKeySnapshot,
    requested_model: &str,
    candidate: &GatewayMinimalCandidateSelectionCandidate,
) -> bool {
    if let Some(allowed_providers) = auth_snapshot.effective_allowed_providers() {
        let provider_allowed = allowed_providers.iter().any(|value| {
            value
                .trim()
                .eq_ignore_ascii_case(candidate.provider_id.trim())
                || value
                    .trim()
                    .eq_ignore_ascii_case(candidate.provider_name.trim())
        });
        if !provider_allowed {
            return false;
        }
    }

    if let Some(allowed_models) = auth_snapshot.effective_allowed_models() {
        let model_allowed = allowed_models
            .iter()
            .any(|value| value == requested_model || value == &candidate.global_model_name);
        if !model_allowed {
            return false;
        }
    }

    true
}

pub(super) async fn maybe_build_local_openai_cli_decision_payload_for_candidate(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    body_json: &serde_json::Value,
    input: &LocalOpenAiCliDecisionInput,
    attempt: LocalOpenAiCliCandidateAttempt,
    spec: LocalOpenAiCliSpec,
) -> Option<GatewayControlSyncDecisionResponse> {
    let LocalOpenAiCliCandidateAttempt {
        candidate,
        candidate_index,
        candidate_id,
    } = attempt;
    let provider_api_format = candidate.endpoint_api_format.trim().to_ascii_lowercase();

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
            mark_skipped_local_openai_cli_candidate(
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
                "gateway local openai cli decision provider transport read failed"
            );
            mark_skipped_local_openai_cli_candidate(
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
    let is_antigravity = transport
        .provider
        .provider_type
        .trim()
        .eq_ignore_ascii_case("antigravity");

    let same_format = provider_api_format == spec.api_format.trim().to_ascii_lowercase();
    let conversion_kind = request_conversion_kind(spec.api_format, provider_api_format.as_str());
    let transport_supported = if same_format {
        supports_local_standard_transport_with_network(&transport, provider_api_format.as_str())
    } else {
        match conversion_kind {
            Some(_) if is_antigravity && provider_api_format == "gemini:cli" => true,
            Some(kind) => request_conversion_transport_supported(&transport, kind),
            None => false,
        }
    };
    if !transport_supported {
        mark_skipped_local_openai_cli_candidate(
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

    let resolved_auth = if same_format {
        match provider_api_format.as_str() {
            "gemini:cli" => resolve_local_gemini_auth(&transport),
            "claude:cli" | "openai:cli" | "openai:compact" => {
                resolve_local_standard_auth(&transport)
            }
            _ => None,
        }
    } else {
        conversion_kind.and_then(|kind| request_conversion_direct_auth(&transport, kind))
    };
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
                    "gateway local openai cli oauth auth resolution failed"
                );
                None
            }
        }
    } else {
        None
    };

    let Some((auth_header, auth_value)) = resolved_auth.or(oauth_auth) else {
        mark_skipped_local_openai_cli_candidate(
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
        mark_skipped_local_openai_cli_candidate(
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

    let needs_bidirectional_conversion = !same_format && conversion_kind.is_some();
    let Some(base_provider_request_body) = (if needs_bidirectional_conversion {
        build_cross_format_openai_cli_request_body(
            body_json,
            &mapped_model,
            spec.api_format,
            provider_api_format.as_str(),
            spec.require_streaming,
            transport.endpoint.body_rules.as_ref(),
        )
    } else {
        build_local_openai_cli_request_body(
            body_json,
            &mapped_model,
            spec.require_streaming,
            transport.endpoint.body_rules.as_ref(),
        )
    }) else {
        mark_skipped_local_openai_cli_candidate(
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
    };
    let antigravity_auth = if is_antigravity {
        match classify_local_antigravity_request_support(
            &transport,
            &base_provider_request_body,
            AntigravityEnvelopeRequestType::Agent,
        ) {
            AntigravityRequestSideSupport::Supported(spec) => Some(spec.auth),
            AntigravityRequestSideSupport::Unsupported(_) => {
                mark_skipped_local_openai_cli_candidate(
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
        }
    } else {
        None
    };
    let provider_request_body = if let Some(antigravity_auth) = antigravity_auth.as_ref() {
        match build_antigravity_safe_v1internal_request(
            antigravity_auth,
            trace_id,
            &mapped_model,
            &base_provider_request_body,
            AntigravityEnvelopeRequestType::Agent,
        ) {
            AntigravityRequestEnvelopeSupport::Supported(envelope) => envelope,
            AntigravityRequestEnvelopeSupport::Unsupported(_) => {
                mark_skipped_local_openai_cli_candidate(
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
        }
    } else {
        base_provider_request_body
    };
    let upstream_is_stream = spec.require_streaming || is_antigravity;

    let Some(upstream_url) = (if needs_bidirectional_conversion {
        build_cross_format_openai_cli_upstream_url(
            parts,
            &transport,
            &mapped_model,
            spec.api_format,
            provider_api_format.as_str(),
            upstream_is_stream,
        )
    } else {
        build_local_openai_cli_upstream_url(
            parts,
            &transport,
            provider_api_format.as_str() == "openai:compact",
        )
    }) else {
        mark_skipped_local_openai_cli_candidate(
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
    let mut provider_request_headers = build_openai_passthrough_headers(
        &parts.headers,
        &auth_header,
        &auth_value,
        &antigravity_auth
            .as_ref()
            .map(build_antigravity_static_identity_headers)
            .unwrap_or_default(),
        Some("application/json"),
    );
    if !apply_local_header_rules(
        &mut provider_request_headers,
        transport.endpoint.header_rules.as_ref(),
        &[&auth_header, "content-type"],
        &provider_request_body,
        Some(body_json),
    ) {
        mark_skipped_local_openai_cli_candidate(
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
    if upstream_is_stream {
        provider_request_headers
            .entry("accept".to_string())
            .or_insert_with(|| "text/event-stream".to_string());
    }
    let prompt_cache_key = provider_request_body
        .get("prompt_cache_key")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let proxy = resolve_transport_proxy_snapshot_with_tunnel_affinity(state, &transport).await;
    let tls_profile = resolve_transport_tls_profile(&transport);
    let execution_strategy = if provider_api_format == spec.api_format {
        ExecutionStrategy::LocalSameFormat
    } else {
        ExecutionStrategy::LocalCrossFormat
    };
    let conversion_mode = if needs_bidirectional_conversion {
        ConversionMode::Bidirectional
    } else {
        ConversionMode::None
    };
    Some(GatewayControlSyncDecisionResponse {
        action: if spec.require_streaming {
            EXECUTION_RUNTIME_STREAM_DECISION_ACTION.to_string()
        } else {
            EXECUTION_RUNTIME_SYNC_DECISION_ACTION.to_string()
        },
        decision_kind: Some(spec.decision_kind.to_string()),
        execution_strategy: Some(execution_strategy.as_str().to_string()),
        conversion_mode: Some(conversion_mode.as_str().to_string()),
        request_id: Some(trace_id.to_string()),
        candidate_id: Some(candidate_id.clone()),
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
        client_api_format: Some(spec.api_format.to_string()),
        provider_contract: Some(provider_api_format.clone()),
        client_contract: Some(spec.api_format.to_string()),
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
        timeouts: resolve_transport_execution_timeouts(&transport),
        upstream_is_stream,
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
                "provider_name": transport.provider.name,
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
                "has_envelope": is_antigravity,
                "envelope_name": if is_antigravity {
                    Some(ANTIGRAVITY_ENVELOPE_NAME)
                } else {
                    None
                },
                "needs_conversion": needs_bidirectional_conversion,
            }),
            execution_strategy,
            conversion_mode,
            spec.api_format,
            candidate.endpoint_api_format.as_str(),
        )),
        auth_context: Some(input.auth_context.clone()),
    })
}

fn candidate_api_formats_for_spec(spec: LocalOpenAiCliSpec) -> &'static [&'static str] {
    match spec.api_format {
        "openai:compact" => &["openai:compact", "openai:cli", "claude:cli", "gemini:cli"],
        "openai:cli" => &["openai:cli", "claude:cli", "gemini:cli"],
        _ => &[],
    }
}

async fn mark_skipped_local_openai_cli_candidate(
    state: &AppState,
    input: &LocalOpenAiCliDecisionInput,
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
            "gateway local openai cli decision failed to persist skipped candidate"
        );
    }
}

pub(super) async fn mark_unused_local_openai_cli_candidates<T>(state: &AppState, remaining: Vec<T>)
where
    T: LocalOpenAiCliPlanAndReport,
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

pub(super) trait LocalOpenAiCliPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan;

    fn report_context(&self) -> Option<&serde_json::Value>;
}

impl LocalOpenAiCliPlanAndReport for LocalSyncPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}

impl LocalOpenAiCliPlanAndReport for LocalStreamPlanAndReport {
    fn plan(&self) -> &aether_contracts::ExecutionPlan {
        &self.plan
    }

    fn report_context(&self) -> Option<&serde_json::Value> {
        self.report_context.as_ref()
    }
}
