use tracing::warn;

use super::decision::{
    materialize_local_openai_cli_candidate_attempts,
    maybe_build_local_openai_cli_decision_payload_for_candidate,
    resolve_local_openai_cli_decision_input, LocalOpenAiCliSpec,
};
use crate::gateway::ai_pipeline::planner::plan_builders::{
    build_openai_cli_stream_plan_from_decision, build_openai_cli_sync_plan_from_decision,
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::{
    OPENAI_CLI_STREAM_PLAN_KIND, OPENAI_CLI_SYNC_PLAN_KIND, OPENAI_COMPACT_STREAM_PLAN_KIND,
    OPENAI_COMPACT_SYNC_PLAN_KIND,
};
use crate::gateway::{AppState, GatewayControlDecision, GatewayError};

pub(super) fn resolve_sync_spec(plan_kind: &str) -> Option<LocalOpenAiCliSpec> {
    match plan_kind {
        OPENAI_CLI_SYNC_PLAN_KIND => Some(LocalOpenAiCliSpec {
            api_format: "openai:cli",
            decision_kind: OPENAI_CLI_SYNC_PLAN_KIND,
            report_kind: "openai_cli_sync_success",
            compact: false,
            require_streaming: false,
        }),
        OPENAI_COMPACT_SYNC_PLAN_KIND => Some(LocalOpenAiCliSpec {
            api_format: "openai:compact",
            decision_kind: OPENAI_COMPACT_SYNC_PLAN_KIND,
            report_kind: "openai_cli_sync_success",
            compact: true,
            require_streaming: false,
        }),
        _ => None,
    }
}

pub(super) fn resolve_stream_spec(plan_kind: &str) -> Option<LocalOpenAiCliSpec> {
    match plan_kind {
        OPENAI_CLI_STREAM_PLAN_KIND => Some(LocalOpenAiCliSpec {
            api_format: "openai:cli",
            decision_kind: OPENAI_CLI_STREAM_PLAN_KIND,
            report_kind: "openai_cli_stream_success",
            compact: false,
            require_streaming: true,
        }),
        OPENAI_COMPACT_STREAM_PLAN_KIND => Some(LocalOpenAiCliSpec {
            api_format: "openai:compact",
            decision_kind: OPENAI_COMPACT_STREAM_PLAN_KIND,
            report_kind: "openai_cli_stream_success",
            compact: true,
            require_streaming: true,
        }),
        _ => None,
    }
}

pub(super) async fn build_local_sync_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    spec: LocalOpenAiCliSpec,
) -> Result<Vec<LocalSyncPlanAndReport>, GatewayError> {
    let Some(input) =
        resolve_local_openai_cli_decision_input(state, trace_id, decision, body_json).await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_openai_cli_candidate_attempts(state, trace_id, &input, spec).await?;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_openai_cli_decision_payload_for_candidate(
            state, parts, trace_id, body_json, &input, attempt, spec,
        )
        .await
        else {
            continue;
        };

        match build_openai_cli_sync_plan_from_decision(parts, body_json, payload, spec.compact) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    api_format = spec.api_format,
                    error = ?err,
                    "gateway local openai cli sync decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}

pub(super) async fn build_local_stream_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    spec: LocalOpenAiCliSpec,
) -> Result<Vec<LocalStreamPlanAndReport>, GatewayError> {
    let Some(input) =
        resolve_local_openai_cli_decision_input(state, trace_id, decision, body_json).await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_openai_cli_candidate_attempts(state, trace_id, &input, spec).await?;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_openai_cli_decision_payload_for_candidate(
            state, parts, trace_id, body_json, &input, attempt, spec,
        )
        .await
        else {
            continue;
        };

        match build_openai_cli_stream_plan_from_decision(parts, body_json, payload, spec.compact) {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    api_format = spec.api_format,
                    error = ?err,
                    "gateway local openai cli stream decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}
