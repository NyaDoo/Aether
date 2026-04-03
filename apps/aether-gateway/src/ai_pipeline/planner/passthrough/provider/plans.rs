use tracing::warn;

use super::{
    materialize_local_same_format_provider_candidate_attempts,
    maybe_build_local_same_format_provider_decision_payload_for_candidate,
    resolve_local_same_format_provider_decision_input, AppState, GatewayControlDecision,
    GatewayError, LocalSameFormatProviderFamily, LocalSameFormatProviderSpec,
    LocalStreamPlanAndReport, LocalSyncPlanAndReport,
};
use crate::gateway::ai_pipeline::planner::plan_builders::{
    build_gemini_stream_plan_from_decision, build_gemini_sync_plan_from_decision,
    build_standard_stream_plan_from_decision, build_standard_sync_plan_from_decision,
};
use crate::gateway::ai_pipeline::planner::{
    CLAUDE_CHAT_STREAM_PLAN_KIND, CLAUDE_CHAT_SYNC_PLAN_KIND, CLAUDE_CLI_STREAM_PLAN_KIND,
    CLAUDE_CLI_SYNC_PLAN_KIND, GEMINI_CHAT_STREAM_PLAN_KIND, GEMINI_CHAT_SYNC_PLAN_KIND,
    GEMINI_CLI_STREAM_PLAN_KIND, GEMINI_CLI_SYNC_PLAN_KIND,
};

pub(crate) fn resolve_sync_spec(plan_kind: &str) -> Option<LocalSameFormatProviderSpec> {
    match plan_kind {
        CLAUDE_CHAT_SYNC_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "claude:chat",
            decision_kind: CLAUDE_CHAT_SYNC_PLAN_KIND,
            report_kind: "claude_chat_sync_success",
            family: LocalSameFormatProviderFamily::Standard,
            require_streaming: false,
        }),
        CLAUDE_CLI_SYNC_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "claude:cli",
            decision_kind: CLAUDE_CLI_SYNC_PLAN_KIND,
            report_kind: "claude_cli_sync_success",
            family: LocalSameFormatProviderFamily::Standard,
            require_streaming: false,
        }),
        GEMINI_CHAT_SYNC_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "gemini:chat",
            decision_kind: GEMINI_CHAT_SYNC_PLAN_KIND,
            report_kind: "gemini_chat_sync_success",
            family: LocalSameFormatProviderFamily::Gemini,
            require_streaming: false,
        }),
        GEMINI_CLI_SYNC_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "gemini:cli",
            decision_kind: GEMINI_CLI_SYNC_PLAN_KIND,
            report_kind: "gemini_cli_sync_success",
            family: LocalSameFormatProviderFamily::Gemini,
            require_streaming: false,
        }),
        _ => None,
    }
}

pub(crate) fn resolve_stream_spec(plan_kind: &str) -> Option<LocalSameFormatProviderSpec> {
    match plan_kind {
        CLAUDE_CHAT_STREAM_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "claude:chat",
            decision_kind: CLAUDE_CHAT_STREAM_PLAN_KIND,
            report_kind: "claude_chat_stream_success",
            family: LocalSameFormatProviderFamily::Standard,
            require_streaming: true,
        }),
        CLAUDE_CLI_STREAM_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "claude:cli",
            decision_kind: CLAUDE_CLI_STREAM_PLAN_KIND,
            report_kind: "claude_cli_stream_success",
            family: LocalSameFormatProviderFamily::Standard,
            require_streaming: true,
        }),
        GEMINI_CHAT_STREAM_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "gemini:chat",
            decision_kind: GEMINI_CHAT_STREAM_PLAN_KIND,
            report_kind: "gemini_chat_stream_success",
            family: LocalSameFormatProviderFamily::Gemini,
            require_streaming: true,
        }),
        GEMINI_CLI_STREAM_PLAN_KIND => Some(LocalSameFormatProviderSpec {
            api_format: "gemini:cli",
            decision_kind: GEMINI_CLI_STREAM_PLAN_KIND,
            report_kind: "gemini_cli_stream_success",
            family: LocalSameFormatProviderFamily::Gemini,
            require_streaming: true,
        }),
        _ => None,
    }
}

pub(crate) async fn build_local_sync_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    spec: LocalSameFormatProviderSpec,
) -> Result<Vec<LocalSyncPlanAndReport>, GatewayError> {
    let Some(input) = resolve_local_same_format_provider_decision_input(
        state, parts, trace_id, decision, body_json, spec,
    )
    .await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_same_format_provider_candidate_attempts(state, trace_id, &input, spec)
            .await?;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_same_format_provider_decision_payload_for_candidate(
            state, parts, trace_id, body_json, &input, attempt, spec,
        )
        .await
        else {
            continue;
        };

        let built = match spec.family {
            LocalSameFormatProviderFamily::Standard => {
                build_standard_sync_plan_from_decision(parts, body_json, payload)
            }
            LocalSameFormatProviderFamily::Gemini => {
                build_gemini_sync_plan_from_decision(parts, body_json, payload)
            }
        };

        match built {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    api_format = spec.api_format,
                    error = ?err,
                    "gateway local same-format sync decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}

pub(crate) async fn build_local_stream_plan_and_reports(
    state: &AppState,
    parts: &http::request::Parts,
    trace_id: &str,
    decision: &GatewayControlDecision,
    body_json: &serde_json::Value,
    spec: LocalSameFormatProviderSpec,
) -> Result<Vec<LocalStreamPlanAndReport>, GatewayError> {
    let Some(input) = resolve_local_same_format_provider_decision_input(
        state, parts, trace_id, decision, body_json, spec,
    )
    .await
    else {
        return Ok(Vec::new());
    };

    let attempts =
        materialize_local_same_format_provider_candidate_attempts(state, trace_id, &input, spec)
            .await?;

    let mut plans = Vec::new();
    for attempt in attempts {
        let Some(payload) = maybe_build_local_same_format_provider_decision_payload_for_candidate(
            state, parts, trace_id, body_json, &input, attempt, spec,
        )
        .await
        else {
            continue;
        };

        let built = match spec.family {
            LocalSameFormatProviderFamily::Standard => {
                build_standard_stream_plan_from_decision(parts, body_json, payload, false)
            }
            LocalSameFormatProviderFamily::Gemini => {
                build_gemini_stream_plan_from_decision(parts, body_json, payload)
            }
        };

        match built {
            Ok(Some(value)) => plans.push(value),
            Ok(None) => {}
            Err(err) => {
                warn!(
                    trace_id = %trace_id,
                    api_format = spec.api_format,
                    error = ?err,
                    "gateway local same-format stream decision plan build failed"
                );
            }
        }
    }

    Ok(plans)
}
