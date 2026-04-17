use aether_contracts::{ExecutionPlan, ExecutionTelemetry};
use aether_usage_runtime::{
    build_stream_terminal_usage_outcome, build_sync_terminal_usage_outcome, TerminalUsageOutcome,
};
use serde_json::Value;
use std::collections::BTreeMap;
use tracing::warn;

use crate::ai_pipeline::extract_pool_sticky_session_token;
use crate::handlers::shared::provider_pool::admin_provider_pool_config_from_config_value;
use crate::handlers::shared::provider_pool::{
    record_admin_provider_pool_error, record_admin_provider_pool_stream_timeout,
    record_admin_provider_pool_success, AdminProviderPoolConfig,
};
use crate::usage::{GatewayStreamReportRequest, GatewaySyncReportRequest};
use crate::AppState;

struct PoolFeedbackContext {
    runner: aether_data::redis::RedisKvRunner,
    pool_config: AdminProviderPoolConfig,
    sticky_session_token: Option<String>,
}

fn pool_feedback_request_body<'a>(
    plan: &'a ExecutionPlan,
    report_context: Option<&'a Value>,
) -> Option<&'a Value> {
    report_context
        .and_then(Value::as_object)
        .and_then(|object| object.get("original_request_body"))
        .filter(|value| !value.is_null())
        .or(plan.body.json_body.as_ref())
}

async fn resolve_pool_feedback_context(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
) -> Option<PoolFeedbackContext> {
    let Some(runner) = state.redis_kv_runner() else {
        return None;
    };

    let transport = match state
        .read_provider_transport_snapshot(&plan.provider_id, &plan.endpoint_id, &plan.key_id)
        .await
    {
        Ok(Some(transport)) => transport,
        Ok(None) => return None,
        Err(err) => {
            warn!(
                "gateway execution runtime pool feedback: failed to read transport snapshot for provider {} endpoint {} key {}: {:?}",
                plan.provider_id, plan.endpoint_id, plan.key_id, err
            );
            return None;
        }
    };

    let Some(pool_config) =
        admin_provider_pool_config_from_config_value(transport.provider.config.as_ref())
    else {
        return None;
    };

    let sticky_session_token = pool_feedback_request_body(plan, report_context)
        .and_then(extract_pool_sticky_session_token);

    Some(PoolFeedbackContext {
        runner,
        pool_config,
        sticky_session_token,
    })
}

fn total_tokens_used(outcome: &TerminalUsageOutcome) -> u64 {
    outcome
        .standardized_usage
        .as_ref()
        .map(|usage| {
            usage
                .input_tokens
                .saturating_add(usage.output_tokens)
                .max(0) as u64
        })
        .unwrap_or(0)
}

fn resolve_ttfb_ms(telemetry: Option<&ExecutionTelemetry>) -> Option<u64> {
    telemetry.and_then(|telemetry| telemetry.ttfb_ms.or(telemetry.elapsed_ms))
}

pub(crate) async fn record_sync_pool_success_feedback(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    payload: &GatewaySyncReportRequest,
) {
    let Some(context) = resolve_pool_feedback_context(state, plan, report_context).await else {
        return;
    };

    let usage_outcome = build_sync_terminal_usage_outcome(plan, report_context, payload);
    record_admin_provider_pool_success(
        &context.runner,
        &plan.provider_id,
        &plan.key_id,
        &context.pool_config,
        context.sticky_session_token.as_deref(),
        total_tokens_used(&usage_outcome),
        resolve_ttfb_ms(payload.telemetry.as_ref()),
    )
    .await;
}

pub(crate) async fn record_stream_pool_success_feedback(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    payload: &GatewayStreamReportRequest,
) {
    let Some(context) = resolve_pool_feedback_context(state, plan, report_context).await else {
        return;
    };

    let usage_outcome = build_stream_terminal_usage_outcome(plan, report_context, payload);
    record_admin_provider_pool_success(
        &context.runner,
        &plan.provider_id,
        &plan.key_id,
        &context.pool_config,
        context.sticky_session_token.as_deref(),
        total_tokens_used(&usage_outcome),
        resolve_ttfb_ms(payload.telemetry.as_ref()),
    )
    .await;
}

pub(crate) async fn record_pool_error_feedback(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_code: u16,
    headers: &BTreeMap<String, String>,
    error_body: Option<&str>,
) {
    if status_code < 400 {
        return;
    }

    let Some(context) = resolve_pool_feedback_context(state, plan, report_context).await else {
        return;
    };

    if status_code == 401 {
        let _ = state
            .invalidate_local_oauth_refresh_entry(&plan.key_id)
            .await;
    }

    record_admin_provider_pool_error(
        &context.runner,
        &plan.provider_id,
        &plan.key_id,
        &context.pool_config,
        status_code,
        error_body,
        Some(headers),
    )
    .await;
}

pub(crate) async fn record_pool_stream_timeout_feedback(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
) {
    let Some(context) = resolve_pool_feedback_context(state, plan, report_context).await else {
        return;
    };

    record_admin_provider_pool_stream_timeout(
        &context.runner,
        &plan.provider_id,
        &plan.key_id,
        &context.pool_config,
    )
    .await;
}
