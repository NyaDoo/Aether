use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::warn;

use super::{
    materialize_local_openai_chat_candidate_attempts,
    maybe_build_local_openai_chat_decision_payload_for_candidate, AppState, GatewayControlDecision,
    GatewayError, LocalExecutionRuntimeMissDiagnostic, LocalOpenAiChatDecisionInput,
};
use crate::gateway::ai_pipeline::planner::plan_builders::{
    build_openai_chat_stream_plan_from_decision, build_openai_chat_sync_plan_from_decision,
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::prefer_local_tunnel_owner_candidates;
use crate::gateway::ai_pipeline::planner::{
    OPENAI_CHAT_STREAM_PLAN_KIND, OPENAI_CHAT_SYNC_PLAN_KIND,
};
use crate::gateway::scheduler::{
    list_selectable_candidates, GatewayMinimalCandidateSelectionCandidate,
};

pub(super) fn build_local_openai_chat_miss_diagnostic(
    decision: &GatewayControlDecision,
    plan_kind: &str,
    requested_model: Option<&str>,
    reason: &str,
) -> LocalExecutionRuntimeMissDiagnostic {
    LocalExecutionRuntimeMissDiagnostic {
        reason: reason.to_string(),
        route_family: decision.route_family.clone(),
        route_kind: decision.route_kind.clone(),
        public_path: Some(decision.public_path.clone()),
        plan_kind: Some(plan_kind.to_string()),
        requested_model: requested_model.map(ToOwned::to_owned),
        candidate_count: None,
        skipped_candidate_count: None,
        skip_reasons: BTreeMap::new(),
    }
}

pub(super) fn set_local_openai_chat_miss_diagnostic(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    requested_model: Option<&str>,
    reason: &str,
) {
    state.set_local_execution_runtime_miss_diagnostic(
        trace_id,
        build_local_openai_chat_miss_diagnostic(decision, plan_kind, requested_model, reason),
    );
}

pub(super) async fn build_local_openai_chat_sync_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Vec<LocalSyncPlanAndReport>, GatewayError> {
    if plan_kind != OPENAI_CHAT_SYNC_PLAN_KIND {
        return Ok(Vec::new());
    }

    let Some(input) = resolve_local_openai_chat_decision_input(
        state, trace_id, decision, body_json, plan_kind, true,
    )
    .await
    else {
        return Ok(Vec::new());
    };

    let candidates = match list_local_openai_chat_candidates(state, &input, false).await {
        Ok(candidates) => candidates,
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local openai chat sync decision scheduler selection failed"
            );
            set_local_openai_chat_miss_diagnostic(
                state,
                trace_id,
                decision,
                plan_kind,
                Some(input.requested_model.as_str()),
                "scheduler_selection_failed",
            );
            return Ok(Vec::new());
        }
    };
    if candidates.is_empty() {
        state.set_local_execution_runtime_miss_diagnostic(
            trace_id,
            LocalExecutionRuntimeMissDiagnostic {
                candidate_count: Some(0),
                ..build_local_openai_chat_miss_diagnostic(
                    decision,
                    plan_kind,
                    Some(input.requested_model.as_str()),
                    "candidate_list_empty",
                )
            },
        );
        return Ok(Vec::new());
    }
    state.set_local_execution_runtime_miss_diagnostic(
        trace_id,
        LocalExecutionRuntimeMissDiagnostic {
            candidate_count: Some(candidates.len()),
            ..build_local_openai_chat_miss_diagnostic(
                decision,
                plan_kind,
                Some(input.requested_model.as_str()),
                "candidate_evaluation_incomplete",
            )
        },
    );

    let attempts =
        materialize_local_openai_chat_candidate_attempts(state, trace_id, &input, candidates).await;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_openai_chat_decision_payload_for_candidate(
            state,
            parts,
            trace_id,
            body_json,
            &input,
            attempt,
            OPENAI_CHAT_SYNC_PLAN_KIND,
            "openai_chat_sync_success",
            false,
        )
        .await
        else {
            continue;
        };

        match build_openai_chat_sync_plan_from_decision(parts, body_json, payload) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    error = ?err,
                    "gateway local openai chat sync decision plan build failed"
                );
            }
        }
    }

    state.mutate_local_execution_runtime_miss_diagnostic(trace_id, |diagnostic| {
        let candidate_count = diagnostic.candidate_count.unwrap_or(0);
        let skipped_candidate_count = diagnostic.skipped_candidate_count.unwrap_or(0);
        diagnostic.reason = if candidate_count > 0 && skipped_candidate_count >= candidate_count {
            "all_candidates_skipped".to_string()
        } else {
            "no_local_sync_plans".to_string()
        };
    });

    Ok(plans)
}

pub(super) async fn build_local_openai_chat_stream_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
) -> Result<Vec<LocalStreamPlanAndReport>, GatewayError> {
    if plan_kind != OPENAI_CHAT_STREAM_PLAN_KIND {
        return Ok(Vec::new());
    }

    let Some(input) = resolve_local_openai_chat_decision_input(
        state, trace_id, decision, body_json, plan_kind, true,
    )
    .await
    else {
        return Ok(Vec::new());
    };

    let candidates = match list_local_openai_chat_candidates(state, &input, true).await {
        Ok(candidates) => candidates,
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local openai chat stream decision scheduler selection failed"
            );
            set_local_openai_chat_miss_diagnostic(
                state,
                trace_id,
                decision,
                plan_kind,
                Some(input.requested_model.as_str()),
                "scheduler_selection_failed",
            );
            return Ok(Vec::new());
        }
    };
    if candidates.is_empty() {
        state.set_local_execution_runtime_miss_diagnostic(
            trace_id,
            LocalExecutionRuntimeMissDiagnostic {
                candidate_count: Some(0),
                ..build_local_openai_chat_miss_diagnostic(
                    decision,
                    plan_kind,
                    Some(input.requested_model.as_str()),
                    "candidate_list_empty",
                )
            },
        );
        return Ok(Vec::new());
    }
    state.set_local_execution_runtime_miss_diagnostic(
        trace_id,
        LocalExecutionRuntimeMissDiagnostic {
            candidate_count: Some(candidates.len()),
            ..build_local_openai_chat_miss_diagnostic(
                decision,
                plan_kind,
                Some(input.requested_model.as_str()),
                "candidate_evaluation_incomplete",
            )
        },
    );

    let attempts =
        materialize_local_openai_chat_candidate_attempts(state, trace_id, &input, candidates).await;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_openai_chat_decision_payload_for_candidate(
            state,
            parts,
            trace_id,
            body_json,
            &input,
            attempt,
            OPENAI_CHAT_STREAM_PLAN_KIND,
            "openai_chat_stream_success",
            true,
        )
        .await
        else {
            continue;
        };

        match build_openai_chat_stream_plan_from_decision(parts, body_json, payload) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    error = ?err,
                    "gateway local openai chat stream decision plan build failed"
                );
            }
        }
    }

    state.mutate_local_execution_runtime_miss_diagnostic(trace_id, |diagnostic| {
        let candidate_count = diagnostic.candidate_count.unwrap_or(0);
        let skipped_candidate_count = diagnostic.skipped_candidate_count.unwrap_or(0);
        diagnostic.reason = if candidate_count > 0 && skipped_candidate_count >= candidate_count {
            "all_candidates_skipped".to_string()
        } else {
            "no_local_stream_plans".to_string()
        };
    });

    Ok(plans)
}

pub(super) fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) async fn list_local_openai_chat_candidates(
    state: &AppState,
    input: &LocalOpenAiChatDecisionInput,
    require_streaming: bool,
) -> Result<Vec<GatewayMinimalCandidateSelectionCandidate>, GatewayError> {
    let now_unix_secs = current_unix_secs();
    let mut combined = Vec::new();
    let mut seen = BTreeSet::new();

    let api_formats = if require_streaming {
        vec!["openai:chat", "claude:chat", "gemini:chat", "openai:cli"]
    } else {
        vec![
            "openai:chat",
            "claude:chat",
            "gemini:chat",
            "openai:cli",
            "openai:compact",
        ]
    };

    for api_format in api_formats {
        let auth_snapshot = if api_format == "openai:chat" {
            Some(&input.auth_snapshot)
        } else {
            None
        };
        let mut candidates = list_selectable_candidates(
            state,
            api_format,
            &input.requested_model,
            require_streaming,
            auth_snapshot,
            now_unix_secs,
        )
        .await?;
        if api_format != "openai:chat" {
            candidates.retain(|candidate| {
                auth_snapshot_allows_cross_format_openai_chat_candidate(
                    &input.auth_snapshot,
                    &input.requested_model,
                    candidate,
                )
            });
        }
        for candidate in candidates {
            let candidate_key = format!(
                "{}:{}:{}:{}:{}",
                candidate.provider_id,
                candidate.endpoint_id,
                candidate.key_id,
                candidate.model_id,
                candidate.selected_provider_model_name,
            );
            if seen.insert(candidate_key) {
                combined.push(candidate);
            }
        }
    }

    Ok(combined)
}

fn auth_snapshot_allows_cross_format_openai_chat_candidate(
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

pub(super) async fn resolve_local_openai_chat_decision_input(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    plan_kind: &str,
    record_miss_diagnostic: bool,
) -> Option<LocalOpenAiChatDecisionInput> {
    let Some(auth_context) = decision.auth_context.clone().filter(|auth_context| {
        !auth_context.user_id.trim().is_empty() && !auth_context.api_key_id.trim().is_empty()
    }) else {
        warn!(
            trace_id = %trace_id,
            route_class = ?decision.route_class,
            route_family = ?decision.route_family,
            route_kind = ?decision.route_kind,
            "gateway local openai chat decision skipped: missing_auth_context"
        );
        if record_miss_diagnostic {
            set_local_openai_chat_miss_diagnostic(
                state,
                trace_id,
                decision,
                plan_kind,
                body_json.get("model").and_then(|value| value.as_str()),
                "missing_auth_context",
            );
        }
        return None;
    };

    let Some(requested_model) = body_json
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    else {
        warn!(
            trace_id = %trace_id,
            "gateway local openai chat decision skipped: missing_requested_model"
        );
        if record_miss_diagnostic {
            set_local_openai_chat_miss_diagnostic(
                state,
                trace_id,
                decision,
                plan_kind,
                None,
                "missing_requested_model",
            );
        }
        return None;
    };

    let now_unix_secs = current_unix_secs();
    let auth_snapshot = match state
        .read_auth_api_key_snapshot(
            &auth_context.user_id,
            &auth_context.api_key_id,
            now_unix_secs,
        )
        .await
    {
        Ok(Some(snapshot)) => snapshot,
        Ok(None) => {
            warn!(
                trace_id = %trace_id,
                user_id = %auth_context.user_id,
                api_key_id = %auth_context.api_key_id,
                "gateway local openai chat decision skipped: auth_snapshot_missing"
            );
            if record_miss_diagnostic {
                set_local_openai_chat_miss_diagnostic(
                    state,
                    trace_id,
                    decision,
                    plan_kind,
                    Some(requested_model.as_str()),
                    "auth_snapshot_missing",
                );
            }
            return None;
        }
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                error = ?err,
                "gateway local openai chat decision auth snapshot read failed"
            );
            if record_miss_diagnostic {
                set_local_openai_chat_miss_diagnostic(
                    state,
                    trace_id,
                    decision,
                    plan_kind,
                    Some(requested_model.as_str()),
                    "auth_snapshot_read_failed",
                );
            }
            return None;
        }
    };

    Some(LocalOpenAiChatDecisionInput {
        auth_context,
        requested_model,
        auth_snapshot,
    })
}
