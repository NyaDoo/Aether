use aether_data::repository::candidates::{RequestCandidateStatus, UpsertRequestCandidateRecord};
use serde_json::json;
use tracing::warn;
use uuid::Uuid;

use crate::gateway::scheduler::{current_unix_secs, list_selectable_candidates};
use crate::gateway::{
    append_execution_contract_fields_to_value, AppState, ConversionMode, ExecutionStrategy,
    GatewayControlDecision, GatewayError,
};

use super::types::{
    LocalSameFormatProviderCandidateAttempt, LocalSameFormatProviderDecisionInput,
    LocalSameFormatProviderFamily, LocalSameFormatProviderSpec,
};

pub(crate) async fn resolve_local_same_format_provider_decision_input(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    spec: LocalSameFormatProviderSpec,
) -> Option<LocalSameFormatProviderDecisionInput> {
    let Some(auth_context) = decision.auth_context.clone().filter(|auth_context| {
        !auth_context.user_id.trim().is_empty() && !auth_context.api_key_id.trim().is_empty()
    }) else {
        return None;
    };

    let requested_model = match spec.family {
        LocalSameFormatProviderFamily::Standard => body_json
            .get("model")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)?,
        LocalSameFormatProviderFamily::Gemini => {
            super::super::request::extract_gemini_model_from_path(parts.uri.path())?
        }
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
                api_format = spec.api_format,
                error = ?err,
                "gateway local same-format decision auth snapshot read failed"
            );
            return None;
        }
    };

    Some(LocalSameFormatProviderDecisionInput {
        auth_context,
        requested_model,
        auth_snapshot,
    })
}

pub(crate) async fn materialize_local_same_format_provider_candidate_attempts(
    state: &AppState,
    trace_id: &str,
    input: &LocalSameFormatProviderDecisionInput,
    spec: LocalSameFormatProviderSpec,
) -> Result<Vec<LocalSameFormatProviderCandidateAttempt>, GatewayError> {
    let candidates = list_selectable_candidates(
        state,
        spec.api_format,
        &input.requested_model,
        spec.require_streaming,
        Some(&input.auth_snapshot),
        current_unix_secs(),
    )
    .await?;
    let candidates = crate::gateway::ai_pipeline::planner::prefer_local_tunnel_owner_candidates(
        state, candidates,
    )
    .await;

    let created_at_unix_secs = current_unix_secs();
    let mut attempts = Vec::with_capacity(candidates.len());
    for (candidate_index, candidate) in candidates.into_iter().enumerate() {
        let generated_candidate_id = Uuid::new_v4().to_string();
        let extra_data = append_execution_contract_fields_to_value(
            json!({
                "provider_api_format": spec.api_format,
                "client_api_format": spec.api_format,
                "global_model_id": candidate.global_model_id.clone(),
                "global_model_name": candidate.global_model_name.clone(),
                "model_id": candidate.model_id.clone(),
                "selected_provider_model_name": candidate.selected_provider_model_name.clone(),
                "mapping_matched_model": candidate.mapping_matched_model.clone(),
                "provider_name": candidate.provider_name.clone(),
                "key_name": candidate.key_name.clone(),
            }),
            ExecutionStrategy::LocalSameFormat,
            ConversionMode::None,
            spec.api_format,
            spec.api_format,
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
                    "gateway local same-format decision request candidate upsert failed"
                );
                generated_candidate_id.clone()
            }
        };

        attempts.push(LocalSameFormatProviderCandidateAttempt {
            candidate,
            candidate_index: candidate_index as u32,
            candidate_id,
        });
    }

    Ok(attempts)
}
