use std::collections::BTreeMap;
use std::io::Error as IoError;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aether_ai_serving::{AiAttemptExecutionOutcome, AiAttemptRetryScope, UPSTREAM_IS_STREAM_KEY};
use aether_contracts::{
    ExecutionError, ExecutionErrorKind, ExecutionPhase, ExecutionPlan,
    ExecutionResponseObservation, ExecutionResult, ExecutionTelemetry,
};
use aether_data_contracts::repository::candidates::RequestCandidateStatus;
use aether_scheduler_core::{
    execution_error_details, parse_request_candidate_report_context,
    SchedulerRequestCandidateStatusUpdate,
};
use aether_usage_runtime::{
    build_lifecycle_usage_seed, build_sync_terminal_usage_payload_seed,
    build_terminal_usage_context_seed, build_usage_event_data_seed, UsageEvent, UsageEventType,
};
use async_stream::stream;
use axum::body::{to_bytes, Body, Bytes};
use axum::http::header::{CACHE_CONTROL, CONTENT_ENCODING, CONTENT_LENGTH, CONTENT_TYPE};
use axum::http::{HeaderName, HeaderValue, Response, StatusCode};
use futures_util::StreamExt;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::time::MissedTickBehavior;
use tracing::{debug, warn};

use crate::ai_serving::api::{
    build_core_error_body_for_client_format, extract_provider_private_stream_error_body,
    implicit_sync_finalize_report_kind, maybe_build_sync_finalize_outcome, LocalCoreSyncErrorKind,
    LocalCoreSyncFinalizeOutcome,
};
use crate::api::response::{
    attach_control_metadata_headers, build_client_response, build_client_response_from_parts,
    build_client_response_from_parts_with_mutator,
};
use crate::async_task::finalize_video_task_if_terminal;
use crate::clock::current_unix_ms as current_request_candidate_unix_ms;
use crate::control::GatewayControlDecision;
use crate::execution_runtime::chatgpt_web_image::maybe_execute_chatgpt_web_image_sync;
use crate::execution_runtime::grok::maybe_execute_grok_sync;
use crate::execution_runtime::kiro_cache::{
    build_kiro_prompt_cache_profile, compute_kiro_prompt_cache_usage,
    estimate_kiro_prompt_input_tokens, kiro_simulated_cache_enabled_from_provider_config,
    kiro_simulated_cache_enabled_from_report_context, KiroPromptCacheUsage,
    KIRO_SIMULATED_CACHE_ENABLED_CONTEXT_FIELD, TERMINAL_KIRO_CACHE_LOOKUP_TIMEOUT,
};
use crate::execution_runtime::oauth_retry::refresh_oauth_plan_auth_for_retry;
#[cfg(test)]
use crate::execution_runtime::remote_compat::post_sync_plan_to_remote_execution_runtime;
use crate::execution_runtime::submission::{
    resolve_local_sync_error_status_code,
    submit_local_core_error_or_sync_finalize_after_terminal_usage,
};
use crate::execution_runtime::transport::{
    append_upstream_response_body_chunk_with_limit, build_execution_response_body,
    build_request_body, collect_response_headers, decode_response_body_bytes_with_limit,
    execution_plan_response_body_limit_bytes, execution_response_body_mode,
    format_hyper_error_chain, format_upstream_request_error, format_wreq_upstream_request_error,
    response_body_is_json, send_request, DirectHttpResponse, DirectSyncExecutionRuntime,
    ExecutionRuntimeTransportError,
};
use crate::execution_runtime::windsurf::maybe_execute_windsurf_sync;
use crate::execution_runtime::{
    ai_attempt_retry_scope_from_failure_disposition, analyze_local_candidate_failover_sync,
    apply_endpoint_response_header_rules_best_effort,
    attach_provider_response_headers_to_report_context,
    get_or_register_stream_candidate_watchdog_progress, local_failover_response_text,
    resolve_core_sync_error_finalize_report_kind, should_fallback_to_control_sync,
    should_finalize_sync_response, stream_candidate_watchdog_progress_for_request,
    try_claim_stream_candidate_intermediate_for_request, LocalFailoverDecision,
    StreamCandidateWatchdogProgress,
};
use crate::log_ids::short_request_id;
use crate::orchestration::{
    apply_local_execution_effect, apply_local_stream_failure_effects_with_analysis,
    build_local_error_flow_metadata, spawn_local_oauth_success_effect,
    trace_upstream_response_body, with_error_flow_report_context,
    with_upstream_response_report_context, LocalAdaptiveRateLimitEffect,
    LocalAdaptiveSuccessEffect, LocalAttemptFailureEffect, LocalExecutionEffect,
    LocalExecutionEffectContext, LocalHealthFailureEffect, LocalHealthSuccessEffect,
    LocalOAuthInvalidationEffect, LocalOAuthSuccessEffect, LocalPoolErrorEffect,
    LocalStreamFailureEffect,
};
use crate::provider_pool_demand::acquire_provider_pool_in_flight_guard;
use crate::request_candidate_runtime::{
    ensure_execution_request_candidate_slot, record_local_request_candidate_extra_data,
    record_local_request_candidate_status, record_local_request_candidate_status_snapshot,
    snapshot_local_request_candidate_status, spawn_candidate_persistence_retry_after_usage_handoff,
    spawn_terminal_candidate_reconciliation,
};
use crate::request_diagnostics::{
    attach_current_request_diagnostics_and_candidate_start_timing_to_report_context,
    attach_request_diagnostics_to_report_context, calibrate_candidate_first_byte_elapsed_ms,
    current_request_diagnostics, RequestDiagnostics,
};
use crate::usage::{
    spawn_sync_report_after_terminal_usage, submit_sync_report_after_terminal_usage,
};
use crate::video_tasks::VideoTaskSyncReportMode;
use crate::{usage::GatewaySyncReportRequest, AppState, GatewayError};

#[path = "execution/policy.rs"]
mod policy;
#[path = "execution/response.rs"]
mod response;

use policy::decode_execution_result_body;
pub(crate) use response::{
    maybe_build_local_sync_finalize_response, maybe_build_local_video_error_response,
    maybe_build_local_video_success_outcome, resolve_local_sync_error_background_report_kind,
    resolve_local_sync_success_background_report_kind, LocalVideoSyncSuccessBuild,
    LocalVideoSyncSuccessOutcome,
};

const OPENAI_IMAGE_SYNC_PLAN_KIND: &str = "openai_image_sync";
const OPENAI_IMAGE_SYNC_DEFAULT_TOTAL_TIMEOUT_MS: u64 = 900_000;
const SYNC_EXECUTION_IDLE_LOG_INTERVAL: Duration = Duration::from_secs(60);
const OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_BYTES: &[u8] = b"\n";
const OPENAI_IMAGE_SYNC_PROGRESS_WRITE_INTERVAL: Duration = Duration::from_secs(5);
const INVALID_GEMINI_PROVIDER_SUCCESS_MESSAGE: &str = "Provider returned HTTP 200 but the Gemini response did not contain visible model output; refusing to finalize it as a successful response.";

fn elapsed_ms_since(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn calibrated_sync_candidate_first_byte_elapsed_ms(
    candidate_started_at: Instant,
    result: &ExecutionResult,
) -> Option<u64> {
    let telemetry = result.telemetry.as_ref()?;
    calibrate_candidate_first_byte_elapsed_ms(
        elapsed_ms_since(candidate_started_at),
        telemetry.elapsed_ms,
        telemetry.ttfb_ms,
    )
}

#[derive(Debug)]
struct SyncExecutionFailure {
    error_type: &'static str,
    message: String,
    status_code: Option<u16>,
    latency_ms: Option<u64>,
    fallback_kind: Option<SyncExecutionFailureFallbackKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncExecutionFailureFallbackKind {
    UpstreamResponseTooLarge,
    UpstreamResponseDecode,
}

impl SyncExecutionFailureFallbackKind {
    fn error_type(self) -> &'static str {
        match self {
            Self::UpstreamResponseTooLarge => "upstream_response_too_large",
            Self::UpstreamResponseDecode => "upstream_response_decode_failed",
        }
    }

    fn client_message(self) -> &'static str {
        match self {
            Self::UpstreamResponseTooLarge => "Upstream response too large",
            Self::UpstreamResponseDecode => "Failed to decode upstream response",
        }
    }
}

struct SyncAttemptTerminalGuard {
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
    // Sync attempts do not run through the stream watchdog, but they have the
    // same cancellation window between the pending skeleton and terminal
    // usage write.  Reuse the request/candidate-scoped owner protocol so a
    // dropped sync task cannot race a late success/failure handoff with a
    // synthetic 499.
    watchdog_progress: Arc<StreamCandidateWatchdogProgress>,
    owns_watchdog_progress_registration: bool,
    candidate_started_unix_ms: u64,
    candidate_started_at: Instant,
    armed: bool,
}

impl SyncAttemptTerminalGuard {
    fn new(
        state: &AppState,
        plan: &ExecutionPlan,
        report_context: Option<Value>,
        candidate_started_unix_ms: u64,
        candidate_started_at: Instant,
    ) -> Self {
        let (watchdog_progress, owns_watchdog_progress_registration) =
            get_or_register_stream_candidate_watchdog_progress(
                plan.request_id.as_str(),
                plan.candidate_id.as_deref(),
            );
        Self {
            state: state.clone(),
            plan: plan.clone(),
            report_context,
            request_diagnostics: current_request_diagnostics(),
            watchdog_progress,
            owns_watchdog_progress_registration,
            candidate_started_unix_ms,
            candidate_started_at,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }

    /// Disarm the cancellation fallback without releasing the watchdog owner.
    /// Terminal callers must invoke `release_watchdog_progress_registration`
    /// only after their candidate-status handoff has returned; releasing from
    /// `disarm` used to leave a window where another scope installed a fresh
    /// owner while the candidate row was still being written.
    fn disarm_and_release(&mut self) {
        self.armed = false;
        self.release_watchdog_progress_registration();
    }

    fn release_watchdog_progress_registration(&mut self) {
        if !self.owns_watchdog_progress_registration {
            return;
        }
        self.owns_watchdog_progress_registration = false;
        crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
            self.plan.request_id.as_str(),
            self.plan.candidate_id.as_deref(),
            &self.watchdog_progress,
        );
    }

    async fn fail_and_disarm(&mut self, error: &GatewayError) {
        if !self.armed {
            return;
        }
        // If a concurrent terminal/retry owner already won, it owns the
        // request lifecycle and this guard must not emit a second terminal
        // usage event.
        if !self.watchdog_progress.mark_terminal_started() {
            self.disarm();
            return;
        }
        self.armed = false;
        let state = self.state.clone();
        let plan = self.plan.clone();
        let report_context = self.report_context.clone();
        let request_diagnostics = self.request_diagnostics.clone();
        let watchdog_progress = Arc::clone(&self.watchdog_progress);
        let candidate_started_unix_ms = self.candidate_started_unix_ms;
        let candidate_started_at = self.candidate_started_at;
        let error_message =
            format!("Local sync attempt failed before terminal finalization: {error:?}");
        // The request future is already on an error/cancellation path.  Keep
        // the terminal handoff on the process-lifetime usage runtime so a
        // cancellation of this future cannot strand the pending skeleton.
        spawn_sync_attempt_forced_terminal_state(
            state,
            plan,
            report_context,
            request_diagnostics,
            Some(watchdog_progress),
            candidate_started_unix_ms,
            candidate_started_at,
            UsageEventType::Failed,
            RequestCandidateStatus::Failed,
            StatusCode::INTERNAL_SERVER_ERROR.as_u16(),
            "local_sync_attempt_aborted",
            error_message,
        );
    }
}

impl Drop for SyncAttemptTerminalGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let state = self.state.clone();
        let plan = self.plan.clone();
        let report_context = self.report_context.clone();
        let request_diagnostics = self.request_diagnostics.clone();
        let watchdog_progress = Arc::clone(&self.watchdog_progress);
        let candidate_started_unix_ms = self.candidate_started_unix_ms;
        let candidate_started_at = self.candidate_started_at;
        // A stop-on-transport watchdog reserves STOP_REQUESTED immediately
        // before aborting the attempt.  If that outer owner is cancelled in
        // the handoff window, promote its reservation to STOP_FALLBACK so the
        // pending usage row still receives a failed/504 terminal event.
        // Otherwise, a sync failover reserves RETRY_ABORT before it writes the
        // candidate's intermediate Failed status.  If this task is cancelled
        // in that write window, promote the bare retry state to a detached
        // void-499 handoff; an externally-owned retry reservation is left to
        // its candidate-loop owner.
        let owns_stop_fallback = if watchdog_progress.stop_requested() {
            watchdog_progress.try_claim_stop_fallback()
        } else {
            false
        };
        let owns_cancel_fallback = !owns_stop_fallback
            && if watchdog_progress.retry_abort() {
                if watchdog_progress.retry_owner_external()
                    || watchdog_progress.retry_handoff_complete()
                {
                    false
                } else {
                    watchdog_progress.try_claim_retry_cancel_fallback()
                }
            } else {
                watchdog_progress.try_claim_cancel_fallback()
            };
        if !owns_stop_fallback && !owns_cancel_fallback {
            // A terminal handoff or intermediate retry owner already won.
            // Do not enqueue a competing terminal event from Drop.
            return;
        }
        // Do not use the request Tokio runtime here: Drop commonly runs while
        // that runtime is cancelling tasks or shutting down.  A process-
        // lifetime usage worker is the only reliable owner of the 499 write.
        if owns_stop_fallback {
            spawn_sync_attempt_forced_terminal_state(
                state,
                plan,
                report_context,
                request_diagnostics,
                Some(watchdog_progress),
                candidate_started_unix_ms,
                candidate_started_at,
                UsageEventType::Failed,
                RequestCandidateStatus::Failed,
                StatusCode::GATEWAY_TIMEOUT.as_u16(),
                "local_sync_watchdog_timeout",
                "Local sync attempt exceeded its watchdog timeout before terminal finalization.",
            );
        } else {
            spawn_sync_attempt_forced_terminal_state(
                state,
                plan,
                report_context,
                request_diagnostics,
                Some(watchdog_progress),
                candidate_started_unix_ms,
                candidate_started_at,
                UsageEventType::Cancelled,
                RequestCandidateStatus::Cancelled,
                499,
                "local_sync_attempt_cancelled",
                "Local sync attempt was dropped before terminal finalization, usually because the client disconnected or the request task was cancelled.",
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_sync_attempt_forced_terminal_state(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
    watchdog_progress: Option<Arc<StreamCandidateWatchdogProgress>>,
    candidate_started_unix_ms: u64,
    candidate_started_at: Instant,
    usage_event_type: UsageEventType,
    candidate_status: RequestCandidateStatus,
    status_code: u16,
    error_type: &'static str,
    error_message: impl Into<String>,
) {
    let error_message = error_message.into();
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        record_sync_attempt_forced_terminal_state(
            state,
            plan,
            report_context,
            request_diagnostics,
            watchdog_progress,
            candidate_started_unix_ms,
            candidate_started_at,
            usage_event_type,
            candidate_status,
            status_code,
            error_type,
            error_message,
        )
        .await;
    });
}

#[allow(clippy::too_many_arguments)]
async fn record_sync_attempt_forced_terminal_state(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
    watchdog_progress: Option<Arc<StreamCandidateWatchdogProgress>>,
    candidate_started_unix_ms: u64,
    candidate_started_at: Instant,
    usage_event_type: UsageEventType,
    candidate_status: RequestCandidateStatus,
    status_code: u16,
    error_type: &'static str,
    error_message: impl Into<String>,
) {
    if let Some(progress) = watchdog_progress.as_ref() {
        let already_owned =
            progress.terminal_started() || progress.cancel_fallback() || progress.stop_fallback();
        if !already_owned && !progress.mark_terminal_started() {
            // A candidate retry or another terminal owner won the race.
            // Leave the request to that owner; emitting a second usage event
            // here could overwrite the authoritative outcome.
            return;
        }
    }
    let error_message = error_message.into();
    let report_context =
        attach_request_diagnostics_to_report_context(report_context, request_diagnostics.as_ref());
    let terminal_unix_ms = current_request_candidate_unix_ms();
    let latency_ms = elapsed_ms_since(candidate_started_at);
    // Keep the same invariant as normal sync execution: usage terminal first,
    // candidate terminal second. The previous guard path did the reverse and
    // could leave a cancelled/failed candidate with no corresponding usage row
    // when the request task was reclaimed during a database/queue stall.
    let usage_persisted = if !state.usage_runtime.is_enabled() {
        true
    } else {
        let mut usage_data = build_usage_event_data_seed(&plan, report_context.as_ref());
        usage_data.status_code = Some(status_code);
        usage_data.error_message = Some(error_message.clone());
        usage_data.error_category = Some(
            match usage_event_type {
                UsageEventType::Cancelled => "cancelled",
                _ => "server_error",
            }
            .to_string(),
        );
        usage_data.response_time_ms = Some(latency_ms);
        let error_body = json!({
            "error": {
                "type": error_type,
                "message": error_message,
                "code": status_code
            }
        });
        usage_data.response_headers = Some(json!({"content-type": "application/json"}));
        usage_data.response_body = Some(error_body.clone());
        usage_data.client_response_headers = Some(json!({"content-type": "application/json"}));
        usage_data.client_response_body = Some(error_body);

        state
            .usage_runtime
            .record_terminal_event_direct_with_handoff(
                state.usage_lifecycle_data_state().as_ref(),
                UsageEvent::new(usage_event_type, plan.request_id.clone(), usage_data),
            )
            .await
    };

    // A candidate terminal row is only meaningful together with its usage
    // terminal row.  This applies to failures/cancellations as well as
    // successes: otherwise cleanup (or a later report) can observe a terminal
    // candidate while the usage row is still just the pending skeleton.  Leave
    // the candidate in Streaming until the next cleanup/retry pass can see the
    // durable usage event.
    let status = usage_persisted
        .then_some(candidate_status)
        .unwrap_or(RequestCandidateStatus::Streaming);
    let (candidate_error_type, candidate_error_message) = if usage_persisted {
        (Some(error_type.to_string()), Some(error_message.clone()))
    } else {
        (
            Some("usage_terminal_handoff_unconfirmed".to_string()),
            Some(format!(
                "{error_message}; terminal usage persistence was not confirmed"
            )),
        )
    };
    let terminal_reconciliation_update = SchedulerRequestCandidateStatusUpdate {
        status: candidate_status,
        status_code: Some(status_code),
        error_type: Some(error_type.to_string()),
        error_message: Some(error_message.clone()),
        latency_ms: Some(latency_ms),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: Some(terminal_unix_ms),
    };
    let terminal_update = SchedulerRequestCandidateStatusUpdate {
        status,
        status_code: Some(status_code),
        error_type: candidate_error_type,
        error_message: candidate_error_message,
        latency_ms: Some(latency_ms),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: usage_persisted.then_some(terminal_unix_ms),
    };
    if !usage_persisted {
        spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            report_context.clone(),
            terminal_reconciliation_update.clone(),
        );
    }
    let candidate_persisted = record_local_request_candidate_status(
        &state,
        &plan,
        report_context.as_ref(),
        terminal_update,
    )
    .await;
    if usage_persisted && !candidate_persisted {
        spawn_candidate_persistence_retry_after_usage_handoff(
            state.clone(),
            plan.clone(),
            report_context.clone(),
            terminal_reconciliation_update,
            watchdog_progress.clone(),
        );
    }
    // Keep a failed handoff's owner registered.  Its terminal retry/reconciliation
    // may still be in flight, and removing the owner here would let a later
    // candidate install a fresh CAS domain and race the retained event.  The
    // registry TTL is the conservative cleanup for this detached path.
    if usage_persisted && candidate_persisted {
        if let Some(progress) = watchdog_progress {
            // This helper is used only by detached guard fallbacks, whose
            // exact owner is already terminal.  Retain the tombstone when a
            // different outer scope owns the same Arc; pointer equality keeps
            // an unrelated registration untouched.
            crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                plan.request_id.as_str(),
                plan.candidate_id.as_deref(),
                &progress,
            );
        }
    }
}

impl SyncExecutionFailure {
    fn from_transport(err: ExecutionRuntimeTransportError) -> Self {
        let fallback_kind = match &err {
            ExecutionRuntimeTransportError::UpstreamResponseTooLarge { .. } => {
                Some(SyncExecutionFailureFallbackKind::UpstreamResponseTooLarge)
            }
            ExecutionRuntimeTransportError::UpstreamResponseDecode { .. } => {
                Some(SyncExecutionFailureFallbackKind::UpstreamResponseDecode)
            }
            _ => None,
        };
        Self {
            error_type: fallback_kind
                .map(SyncExecutionFailureFallbackKind::error_type)
                .unwrap_or("execution_runtime_unavailable"),
            message: err.to_string(),
            status_code: fallback_kind.map(|_| StatusCode::BAD_GATEWAY.as_u16()),
            latency_ms: None,
            fallback_kind,
        }
    }

    fn image_sync_total_timeout(timeout_ms: u64, elapsed_ms: u64) -> Self {
        Self {
            error_type: "image_sync_total_timeout",
            message: format!(
                "OpenAI image sync execution exceeded total timeout of {timeout_ms}ms"
            ),
            status_code: Some(StatusCode::GATEWAY_TIMEOUT.as_u16()),
            latency_ms: Some(elapsed_ms),
            fallback_kind: None,
        }
    }
}

fn build_sync_execution_failure_fallback_body(
    client_api_format: &str,
    kind: SyncExecutionFailureFallbackKind,
) -> Value {
    let message = kind.client_message();
    let error_type = kind.error_type();
    match crate::ai_serving::normalize_api_format_alias(client_api_format).as_str() {
        "claude:messages" => json!({
            "type": "error",
            "error": {
                "type": "upstream_error",
                "message": message,
            }
        }),
        "gemini:generate_content" => json!({
            "error": {
                "code": StatusCode::BAD_GATEWAY.as_u16(),
                "message": message,
                "status": "BAD_GATEWAY",
            }
        }),
        _ => json!({
            "error": {
                "type": "upstream_error",
                "message": message,
                "code": error_type,
            }
        }),
    }
}

fn build_sync_execution_failure_fallback_response(
    failure: &SyncExecutionFailure,
    plan: &ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(kind) = failure.fallback_kind else {
        return Ok(None);
    };
    let body_json = build_sync_execution_failure_fallback_body(&plan.client_api_format, kind);
    let body_bytes = serde_json::to_vec(&body_json)
        .map_err(|error| GatewayError::Internal(error.to_string()))?;
    let headers = BTreeMap::from([
        ("content-type".to_string(), "application/json".to_string()),
        ("content-length".to_string(), body_bytes.len().to_string()),
    ]);
    let response = build_client_response_from_parts(
        StatusCode::BAD_GATEWAY.as_u16(),
        &headers,
        Body::from(body_bytes),
        trace_id,
        Some(decision),
    )?;
    attach_control_metadata_headers(
        response,
        Some(plan.request_id.as_str()),
        plan.candidate_id.as_deref(),
    )
    .map(Some)
}

fn maybe_store_sync_execution_failure_fallback(
    failure: &SyncExecutionFailure,
    plan: &ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
    retry_scope_out: &mut Option<&mut AiAttemptRetryScope>,
    retry_fallback_out: &mut Option<&mut Option<Response<Body>>>,
) -> Result<(), GatewayError> {
    if failure.fallback_kind.is_none() {
        return Ok(());
    }
    if let Some(retry_scope) = retry_scope_out.as_deref_mut() {
        *retry_scope = AiAttemptRetryScope::Candidate;
    }
    if let Some(retry_fallback) = retry_fallback_out.as_deref_mut() {
        *retry_fallback =
            build_sync_execution_failure_fallback_response(failure, plan, trace_id, decision)?;
    }
    Ok(())
}

async fn maybe_build_sync_transport_error_stop_response(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    trace_id: &str,
    decision: &GatewayControlDecision,
    error_type: &str,
    error_message: &str,
    elapsed_ms: u64,
) -> Result<Option<Response<Body>>, GatewayError> {
    let analysis = crate::orchestration::resolve_local_transport_failover_analysis_for_attempt(
        state,
        plan,
        report_context,
    )
    .await;
    if !matches!(analysis.decision, LocalFailoverDecision::StopLocalFailover) {
        return Ok(None);
    }

    crate::execution_runtime::build_transport_error_stop_response(
        state,
        plan,
        report_context,
        trace_id,
        decision,
        StatusCode::BAD_GATEWAY.as_u16(),
        error_type,
        error_message,
        elapsed_ms,
    )
    .await
    .map(Some)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncWatchdogIntermediateOwnership {
    /// Another retry owner has reserved this candidate's intermediate
    /// transition.  The caller must return the normal candidate-retry signal
    /// and must not publish a second Failed row.
    Retry,
    /// A terminal/cancellation owner has already reserved the request-level
    /// lifecycle.  Returning the retry signal here would let the outer
    /// candidate loop start another provider attempt and race that terminal
    /// handoff, so the caller returns a client-only transport response.
    Terminal,
}

fn sync_watchdog_intermediate_ownership(
    plan: &ExecutionPlan,
) -> Option<SyncWatchdogIntermediateOwnership> {
    let progress = stream_candidate_watchdog_progress_for_request(
        plan.request_id.as_str(),
        plan.candidate_id.as_deref(),
    )?;
    if progress.retry_abort() {
        Some(SyncWatchdogIntermediateOwnership::Retry)
    } else if progress.terminal_owner_active()
        || progress.stop_requested()
        || progress.stop_fallback()
    {
        Some(SyncWatchdogIntermediateOwnership::Terminal)
    } else {
        None
    }
}

fn build_sync_watchdog_owned_client_response(
    plan: &ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
) -> Result<Response<Body>, GatewayError> {
    crate::execution_runtime::build_transport_error_client_response(
        plan,
        trace_id,
        decision,
        StatusCode::BAD_GATEWAY.as_u16(),
    )
}

/// Resolve the transport failover decision before publishing a terminal
/// candidate state.  When local failover stops, the shared stop-response
/// helper persists terminal usage first; when it retries, this function records
/// only the intermediate candidate failure (the request-level usage row stays
/// open for the next candidate).
#[allow(clippy::too_many_arguments)]
async fn stop_or_record_sync_transport_failure(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    trace_id: &str,
    decision: &GatewayControlDecision,
    error_type: &str,
    error_message: &str,
    elapsed_ms: u64,
    candidate_started_unix_ms: u64,
    has_candidate_retry_scope: bool,
) -> Result<Option<Response<Body>>, GatewayError> {
    let response = maybe_build_sync_transport_error_stop_response(
        state,
        plan,
        report_context,
        trace_id,
        decision,
        error_type,
        error_message,
        elapsed_ms,
    )
    .await?;
    if response.is_none() && !has_candidate_retry_scope {
        // The direct/internal sync entry point has no outer candidate loop to
        // own exhaustion.  Never return `None` there: close the usage row and
        // return a client-visible terminal transport response instead.
        return crate::execution_runtime::build_transport_error_stop_response(
            state,
            plan,
            report_context,
            trace_id,
            decision,
            StatusCode::BAD_GATEWAY.as_u16(),
            error_type,
            error_message,
            elapsed_ms,
        )
        .await
        .map(Some);
    }
    if response.is_none() {
        // The watchdog/retry owner is shared with the stream/sync terminal
        // guards.  Claim the intermediate slot before writing the candidate;
        // a timeout/cancellation finalizer that won the race must not be
        // overwritten by a late Failed transition.
        if !try_claim_stream_candidate_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        ) {
            return match sync_watchdog_intermediate_ownership(plan) {
                Some(SyncWatchdogIntermediateOwnership::Terminal) => {
                    build_sync_watchdog_owned_client_response(plan, trace_id, decision).map(Some)
                }
                Some(SyncWatchdogIntermediateOwnership::Retry) | None => Ok(None),
            };
        }
        let terminal_unix_ms = current_request_candidate_unix_ms();
        record_local_request_candidate_status(
            state,
            plan,
            report_context,
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Failed,
                status_code: None,
                error_type: Some(error_type.to_string()),
                error_message: Some(error_message.to_string()),
                latency_ms: Some(elapsed_ms),
                started_at_unix_ms: Some(candidate_started_unix_ms),
                finished_at_unix_ms: Some(terminal_unix_ms),
            },
        )
        .await;
    }
    Ok(response)
}

struct ImplicitSyncFinalizeOutcome {
    payload: GatewaySyncReportRequest,
    outcome: LocalCoreSyncFinalizeOutcome,
}

fn spawn_sync_candidate_status_update(
    state: AppState,
    snapshot: crate::request_candidate_runtime::LocalRequestCandidateStatusSnapshot,
    status_update: SchedulerRequestCandidateStatusUpdate,
) {
    tokio::spawn(async move {
        record_local_request_candidate_status_snapshot(&state, &snapshot, status_update).await;
    });
}

fn record_sync_response_started(
    state: &AppState,
    lifecycle_seed: aether_usage_runtime::LifecycleUsageSeed,
    request_candidate_status_snapshot: Option<
        crate::request_candidate_runtime::LocalRequestCandidateStatusSnapshot,
    >,
    candidate_started_unix_ms: u64,
    status_code: u16,
    ttfb_ms: u64,
) {
    state.usage_runtime.record_stream_started_immediate_async(
        state.usage_lifecycle_data_state().as_ref(),
        lifecycle_seed,
        status_code,
        Some(ExecutionTelemetry {
            ttfb_ms: Some(ttfb_ms),
            elapsed_ms: Some(ttfb_ms),
            upstream_bytes: None,
        }),
    );

    if let Some(snapshot) = request_candidate_status_snapshot {
        spawn_sync_candidate_status_update(
            state.clone(),
            snapshot,
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Streaming,
                status_code: Some(status_code),
                error_type: None,
                error_message: None,
                latency_ms: Some(ttfb_ms),
                started_at_unix_ms: Some(candidate_started_unix_ms),
                finished_at_unix_ms: None,
            },
        );
    }
}

fn record_sync_execution_active(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    candidate_started_unix_ms: u64,
) {
    let lifecycle_seed = build_lifecycle_usage_seed(plan, report_context);
    state.usage_runtime.record_sync_active_immediate_async(
        state.usage_lifecycle_data_state().as_ref(),
        lifecycle_seed,
    );

    if let Some(snapshot) = snapshot_local_request_candidate_status(plan, report_context) {
        spawn_sync_candidate_status_update(
            state.clone(),
            snapshot,
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Streaming,
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(candidate_started_unix_ms),
                finished_at_unix_ms: None,
            },
        );
    }
}

async fn record_sync_terminal_usage(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
    payload: &GatewaySyncReportRequest,
    candidate_started_at: Instant,
    candidate_first_byte_elapsed_ms: Option<u64>,
    watchdog_progress: Arc<StreamCandidateWatchdogProgress>,
) -> (bool, bool) {
    // Terminal usage is a single-owner operation.  A cancellation fallback or
    // an intermediate retry that won first must not be overwritten by a late
    // sync response.  Use the exact Arc held by the attempt guard; looking it
    // up again by request id can miss the owner after a concurrent registry
    // cleanup and silently bypass the CAS.
    if !watchdog_progress.mark_terminal_started() {
        return (false, false);
    }
    let report_context_with_diagnostics =
        attach_current_request_diagnostics_and_candidate_start_timing_to_report_context(
            report_context,
            candidate_started_at,
            candidate_first_byte_elapsed_ms,
        );
    let context_seed = build_terminal_usage_context_seed(
        plan,
        report_context_with_diagnostics.as_ref().or(report_context),
    );
    let payload_seed = build_sync_terminal_usage_payload_seed(payload);
    let persisted = state
        .usage_runtime
        .record_sync_terminal_with_handoff(
            state.usage_lifecycle_data_state().as_ref(),
            context_seed,
            payload_seed,
        )
        .await;
    (persisted, true)
}

fn sync_terminal_candidate_error_fields(
    payload: &GatewaySyncReportRequest,
) -> (Option<String>, Option<String>) {
    let Some(body) = payload.body_json.as_ref().and_then(Value::as_object) else {
        return (None, None);
    };
    let error = body.get("error").and_then(Value::as_object);
    let error_type = error
        .and_then(|value| value.get("type"))
        .and_then(Value::as_str)
        .or_else(|| body.get("type").and_then(Value::as_str))
        .map(ToOwned::to_owned);
    let error_message = error
        .and_then(|value| value.get("message"))
        .and_then(Value::as_str)
        .or_else(|| body.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned);
    (error_type, error_message)
}

fn sync_terminal_candidate_status(status_code: u16) -> RequestCandidateStatus {
    if status_code == 499 {
        RequestCandidateStatus::Cancelled
    } else if status_code >= 400 {
        RequestCandidateStatus::Failed
    } else {
        RequestCandidateStatus::Success
    }
}

async fn record_sync_terminal_usage_and_disarm_guard(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
    payload: &GatewaySyncReportRequest,
    candidate_started_unix_ms: u64,
    candidate_started_at: Instant,
    candidate_first_byte_elapsed_ms: Option<u64>,
    terminal_guard: &mut SyncAttemptTerminalGuard,
) -> bool {
    let (persisted, terminal_owner_claimed) = record_sync_terminal_usage(
        state,
        plan,
        report_context,
        payload,
        candidate_started_at,
        candidate_first_byte_elapsed_ms,
        Arc::clone(&terminal_guard.watchdog_progress),
    )
    .await;

    // Candidate terminal persistence is intentionally downstream of the usage handoff.  A queue
    // append alone is not enough: if the durable usage write could not be confirmed, retain a
    // streaming candidate so cleanup cannot reinterpret a skeletal row as a successful request.
    let desired_status = sync_terminal_candidate_status(payload.status_code);
    // Never publish *any* terminal candidate status before the handoff is
    // confirmed.  The previous ordering only downgraded successful 2xx
    // responses; a failed/499 payload could still race the pending usage row.
    let status = if persisted {
        desired_status
    } else {
        RequestCandidateStatus::Streaming
    };
    let (desired_error_type, desired_error_message) = sync_terminal_candidate_error_fields(payload);
    let reconciliation_update = SchedulerRequestCandidateStatusUpdate {
        status: desired_status,
        status_code: Some(payload.status_code),
        error_type: desired_error_type.clone(),
        error_message: desired_error_message.clone(),
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: Some(current_request_candidate_unix_ms()),
    };
    let (mut error_type, mut error_message) = (desired_error_type, desired_error_message);
    if status == RequestCandidateStatus::Streaming {
        error_type = Some("usage_terminal_handoff_unconfirmed".to_string());
        error_message = Some(
            "terminal usage persistence was not confirmed before candidate finalization"
                .to_string(),
        );
    }
    let terminal_update = SchedulerRequestCandidateStatusUpdate {
        status,
        status_code: Some(payload.status_code),
        error_type,
        error_message,
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: (status != RequestCandidateStatus::Streaming)
            .then_some(current_request_candidate_unix_ms()),
    };
    if !persisted {
        spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            reconciliation_update.clone(),
        );
    }
    // Candidate persistence is the second half of the usage→candidate
    // handoff.  Run it on the process-lifetime usage runtime and only wait a
    // bounded amount in the request task; if the request is cancelled, the
    // JoinHandle is detached and the write still completes.  This closes the
    // cancellation window without scheduling a long reconciliation task for
    // every healthy request.
    let terminal_owner_progress =
        terminal_owner_claimed.then(|| Arc::clone(&terminal_guard.watchdog_progress));
    let release_watchdog_registration =
        terminal_owner_claimed && terminal_guard.owns_watchdog_progress_registration;
    let candidate_write = aether_usage_runtime::spawn_on_usage_background_runtime({
        let state = state.clone();
        let plan = plan.clone();
        let report_context = report_context.cloned();
        let terminal_owner_progress = terminal_owner_progress.clone();
        let reconciliation_update = reconciliation_update.clone();
        async move {
            let candidate_persisted = record_local_request_candidate_status(
                &state,
                &plan,
                report_context.as_ref(),
                terminal_update,
            )
            .await;
            if persisted && !candidate_persisted {
                spawn_candidate_persistence_retry_after_usage_handoff(
                    state.clone(),
                    plan.clone(),
                    report_context.clone(),
                    reconciliation_update,
                    release_watchdog_registration
                        .then_some(terminal_owner_progress.clone())
                        .flatten(),
                );
            }
            if release_watchdog_registration && persisted && candidate_persisted {
                if let Some(progress) = terminal_owner_progress {
                    crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                        plan.request_id.as_str(),
                        plan.candidate_id.as_deref(),
                        &progress,
                    );
                }
            }
        }
    });
    let _ = tokio::time::timeout(Duration::from_secs(5), candidate_write).await;
    // The candidate write runs on the process-lifetime runtime.  Disarm the
    // request cancellation guard after dispatch; the detached writer owns the
    // exact progress Arc and releases the registry only after its write.
    terminal_guard.disarm();
    persisted
}

fn with_sync_error_trace_context(
    report_context: Option<&serde_json::Value>,
    status_code: u16,
    headers: &BTreeMap<String, String>,
    body_json: Option<&serde_json::Value>,
    body_bytes: &[u8],
    response_text: Option<&str>,
    local_failover_analysis: crate::orchestration::LocalFailoverAnalysis,
) -> Option<serde_json::Value> {
    let body = trace_upstream_response_body(body_json, body_bytes);
    let upstream_context = with_upstream_response_report_context(
        report_context,
        status_code,
        Some(headers),
        body.as_ref(),
        None,
        None,
    );
    with_error_flow_report_context(
        upstream_context.as_ref().or(report_context),
        build_local_error_flow_metadata(status_code, response_text, local_failover_analysis),
    )
}

fn build_sync_report_payload(
    trace_id: &str,
    report_kind: String,
    report_context: Option<serde_json::Value>,
    status_code: u16,
    headers: BTreeMap<String, String>,
    body_json: Option<serde_json::Value>,
    body_base64: Option<String>,
    telemetry: Option<ExecutionTelemetry>,
) -> GatewaySyncReportRequest {
    GatewaySyncReportRequest {
        trace_id: trace_id.to_string(),
        report_kind,
        report_context,
        status_code,
        headers,
        body_json,
        client_body_json: None,
        body_base64,
        telemetry,
    }
}

fn seed_kiro_sync_report_context_input_tokens(
    plan: &ExecutionPlan,
    report_context: &mut Option<Value>,
) {
    if !plan
        .provider_name
        .as_deref()
        .is_some_and(|provider_name| provider_name.eq_ignore_ascii_case("Kiro"))
    {
        return;
    }

    let Some(context) = report_context.as_mut().and_then(Value::as_object_mut) else {
        return;
    };
    if context
        .get("input_tokens")
        .and_then(Value::as_u64)
        .is_some_and(|input_tokens| input_tokens > 0)
    {
        return;
    }

    let Some(original_request_body) = context.get("original_request_body").cloned() else {
        return;
    };
    let estimated_input_tokens = estimate_kiro_prompt_input_tokens(&original_request_body);
    context.insert(
        "input_tokens".to_string(),
        Value::from(estimated_input_tokens),
    );
}

async fn seed_kiro_sync_simulated_cache_enabled(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: &mut Option<Value>,
) {
    if !plan
        .provider_name
        .as_deref()
        .is_some_and(|provider_name| provider_name.eq_ignore_ascii_case("Kiro"))
    {
        return;
    }

    let enabled = match tokio::time::timeout(
        TERMINAL_KIRO_CACHE_LOOKUP_TIMEOUT,
        state.read_provider_catalog_providers_by_ids(std::slice::from_ref(&plan.provider_id)),
    )
    .await
    {
        Err(_) => {
            warn!(
                event_name = "kiro_simulated_cache_config_read_timeout",
                log_type = "ops",
                request_id = %plan.request_id,
                provider_id = %plan.provider_id,
                timeout_ms = TERMINAL_KIRO_CACHE_LOOKUP_TIMEOUT.as_millis() as u64,
                "timed out reading Kiro simulated cache provider config; defaulting disabled"
            );
            false
        }
        Ok(Err(err)) => {
            warn!(
                event_name = "kiro_simulated_cache_config_read_failed",
                log_type = "event",
                request_id = %plan.request_id,
                provider_id = %plan.provider_id,
                error = ?err,
                "failed to read Kiro simulated cache provider config; defaulting disabled"
            );
            false
        }
        Ok(Ok(providers)) => providers
            .iter()
            .find(|provider| provider.id == plan.provider_id)
            .filter(|provider| provider.provider_type.eq_ignore_ascii_case("kiro"))
            .is_some_and(|provider| {
                kiro_simulated_cache_enabled_from_provider_config(provider.config.as_ref())
            }),
    };

    let Some(context) = report_context.as_mut().and_then(Value::as_object_mut) else {
        return;
    };
    if enabled {
        context.insert(
            KIRO_SIMULATED_CACHE_ENABLED_CONTEXT_FIELD.to_string(),
            Value::Bool(true),
        );
    } else {
        context.remove(KIRO_SIMULATED_CACHE_ENABLED_CONTEXT_FIELD);
    }
}

async fn seed_kiro_sync_report_context_prompt_cache_usage(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: &mut Option<Value>,
) {
    if !plan
        .provider_name
        .as_deref()
        .is_some_and(|provider_name| provider_name.eq_ignore_ascii_case("Kiro"))
    {
        return;
    }

    let simulated_cache_enabled =
        kiro_simulated_cache_enabled_from_report_context(report_context.as_ref());
    let Some(context) = report_context.as_mut().and_then(Value::as_object_mut) else {
        return;
    };
    if context
        .get("kiro_web_search_mcp")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return;
    }
    if !simulated_cache_enabled {
        return;
    }
    if kiro_cache_usage_from_context_object(context).is_some() {
        return;
    }

    let Some(original_request_body) = context.get("original_request_body").cloned() else {
        return;
    };
    let input_tokens = context
        .get("input_tokens")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            let estimated = estimate_kiro_prompt_input_tokens(&original_request_body);
            context.insert("input_tokens".to_string(), Value::from(estimated));
            estimated
        });
    let Some(profile) = build_kiro_prompt_cache_profile(&original_request_body, input_tokens)
    else {
        return;
    };

    let cache_usage = match tokio::time::timeout(
        TERMINAL_KIRO_CACHE_LOOKUP_TIMEOUT,
        compute_kiro_prompt_cache_usage(
            state.runtime_state(),
            kiro_sync_cache_credential_id(plan),
            &profile,
        ),
    )
    .await
    {
        Ok(cache_usage) => cache_usage,
        Err(_) => {
            warn!(
                event_name = "kiro_prompt_cache_usage_lookup_timeout",
                log_type = "ops",
                request_id = %plan.request_id,
                provider_id = %plan.provider_id,
                timeout_ms = TERMINAL_KIRO_CACHE_LOOKUP_TIMEOUT.as_millis() as u64,
                "gateway skipped Kiro prompt-cache enrichment before terminal usage handoff"
            );
            KiroPromptCacheUsage::default()
        }
    };
    if cache_usage.cache_creation_input_tokens == 0 && cache_usage.cache_read_input_tokens == 0 {
        return;
    }
    context.insert(
        "cache_creation_input_tokens".to_string(),
        Value::from(cache_usage.cache_creation_input_tokens),
    );
    context.insert(
        "cache_read_input_tokens".to_string(),
        Value::from(cache_usage.cache_read_input_tokens),
    );
}

fn kiro_sync_cache_credential_id(plan: &ExecutionPlan) -> String {
    format!("{}:{}:{}", plan.provider_id, plan.endpoint_id, plan.key_id)
}

fn kiro_cache_usage_from_context_object(
    context: &serde_json::Map<String, Value>,
) -> Option<KiroPromptCacheUsage> {
    let cache_creation_input_tokens = context
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read_input_tokens = context
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (cache_creation_input_tokens > 0 || cache_read_input_tokens > 0).then_some(
        KiroPromptCacheUsage {
            cache_creation_input_tokens,
            cache_read_input_tokens,
        },
    )
}

fn invalid_gemini_provider_success_message(
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_code: u16,
    body_json: Option<&Value>,
) -> Option<&'static str> {
    if status_code >= 400 {
        return None;
    }
    if !provider_api_format_is_gemini_generate_content(plan, report_context) {
        return None;
    }
    let body_json = body_json?;
    if body_json
        .as_object()
        .is_some_and(|object| object.get("error").is_some_and(|error| !error.is_null()))
    {
        return None;
    }
    let normalized_body_json = report_context
        .filter(|context| {
            context
                .get("has_envelope")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .and_then(|context| {
            crate::ai_serving::normalize_provider_private_response_value(body_json.clone(), context)
        });
    let body_json = normalized_body_json.as_ref().unwrap_or(body_json);
    if crate::ai_serving::gemini_generate_content_response_has_visible_output(body_json) {
        return None;
    }
    Some(INVALID_GEMINI_PROVIDER_SUCCESS_MESSAGE)
}

fn invalid_gemini_provider_stream_success_message(
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_code: u16,
    body_json: Option<&Value>,
    body_bytes: &[u8],
    has_body_bytes: bool,
) -> Option<&'static str> {
    if status_code >= 400 || body_json.is_some() || !has_body_bytes {
        return None;
    }
    if !provider_api_format_is_gemini_generate_content(plan, report_context) {
        return None;
    }
    let Some(body_json) = crate::ai_serving::aggregate_gemini_stream_sync_response(body_bytes)
    else {
        return Some(INVALID_GEMINI_PROVIDER_SUCCESS_MESSAGE);
    };
    if crate::ai_serving::gemini_generate_content_response_has_visible_output(&body_json) {
        return None;
    }
    Some(INVALID_GEMINI_PROVIDER_SUCCESS_MESSAGE)
}

fn provider_api_format_is_gemini_generate_content(
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
) -> bool {
    let provider_api_format = report_context
        .and_then(|value| value.get("provider_api_format"))
        .and_then(Value::as_str)
        .unwrap_or(plan.provider_api_format.as_str());
    crate::ai_serving::normalize_api_format_alias(provider_api_format) == "gemini:generate_content"
}

fn invalid_gemini_provider_success_execution_error(message: &str) -> ExecutionError {
    ExecutionError {
        kind: ExecutionErrorKind::Upstream5xx,
        phase: ExecutionPhase::Finalize,
        message: message.to_string(),
        upstream_status: Some(StatusCode::OK.as_u16()),
        retryable: true,
        failover_recommended: true,
    }
}

fn build_invalid_provider_success_body(
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    message: &str,
) -> Option<Value> {
    let client_api_format = report_context
        .and_then(|value| value.get("client_api_format"))
        .and_then(Value::as_str)
        .unwrap_or(plan.client_api_format.as_str());
    build_core_error_body_for_client_format(
        client_api_format,
        message,
        Some("invalid_provider_success_response"),
        LocalCoreSyncErrorKind::ServerError,
    )
}

fn provider_private_error_details(body_json: &Value) -> (Option<String>, Option<String>) {
    let body_object = body_json.as_object();
    let error_object = body_object
        .and_then(|object| object.get("error"))
        .and_then(Value::as_object);
    let error_type =
        first_non_empty_error_text(error_object, body_object, &["type", "code", "status"]);
    let error_message = first_non_empty_error_text(
        error_object,
        body_object,
        &["message", "detail", "reason", "status", "type", "code"],
    );
    (error_type, error_message)
}

fn first_non_empty_error_text(
    error_object: Option<&serde_json::Map<String, Value>>,
    body_object: Option<&serde_json::Map<String, Value>>,
    keys: &[&str],
) -> Option<String> {
    for object in [error_object, body_object].into_iter().flatten() {
        for key in keys {
            let Some(value) = object.get(*key) else {
                continue;
            };
            match value {
                Value::String(text) if !text.trim().is_empty() => {
                    return Some(text.trim().to_string());
                }
                Value::Number(number) => return Some(number.to_string()),
                _ => {}
            }
        }
    }
    None
}

#[derive(Debug, Clone)]
struct OpenAiImageSyncProgressSnapshot {
    phase: &'static str,
    upstream_ttfb_ms: Option<u64>,
    upstream_sse_frame_count: u64,
    last_upstream_event: Option<String>,
    last_upstream_frame_at_unix_ms: Option<u64>,
    partial_image_count: u64,
    last_client_visible_event: Option<String>,
    downstream_heartbeat_count: u64,
    last_downstream_heartbeat_at_unix_ms: Option<u64>,
    downstream_heartbeat_interval_ms: Option<u64>,
}

struct OpenAiImageSyncProgressRecorder<'a> {
    state: &'a AppState,
    plan: &'a ExecutionPlan,
    report_context: Option<&'a Value>,
    snapshot: Arc<Mutex<OpenAiImageSyncProgressSnapshot>>,
    buffer: Vec<u8>,
    last_persist_at: Option<Instant>,
}

/// Progress updates are written while the request-level usage terminal event
/// is still pending.  A terminal candidate status here would let cleanup (or
/// a concurrent report) observe `failed` before the durable usage handoff and
/// recreate the candidate-before-billing race.  Keep the detailed phase in
/// `extra_data.image_progress`; the execution finalizer is the sole owner of
/// terminal candidate statuses.
fn non_terminal_image_progress_status(status: RequestCandidateStatus) -> RequestCandidateStatus {
    match status {
        RequestCandidateStatus::Success
        | RequestCandidateStatus::Failed
        | RequestCandidateStatus::Cancelled
        | RequestCandidateStatus::Skipped => RequestCandidateStatus::Streaming,
        non_terminal => non_terminal,
    }
}

#[derive(Clone)]
struct OpenAiImageSyncJsonHeartbeatContext {
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    snapshot: Arc<Mutex<OpenAiImageSyncProgressSnapshot>>,
    started_at: Instant,
    trace_id: String,
    request_id_for_log: String,
    candidate_id: Option<String>,
}

#[derive(Debug)]
struct OpenAiImageSyncSseFrame {
    event_name: String,
    is_partial_image: bool,
    is_completed: bool,
    is_failed: bool,
    client_visible_event: Option<&'static str>,
}

impl OpenAiImageSyncProgressSnapshot {
    fn new() -> Self {
        Self {
            phase: "upstream_connecting",
            upstream_ttfb_ms: None,
            upstream_sse_frame_count: 0,
            last_upstream_event: None,
            last_upstream_frame_at_unix_ms: None,
            partial_image_count: 0,
            last_client_visible_event: None,
            downstream_heartbeat_count: 0,
            last_downstream_heartbeat_at_unix_ms: None,
            downstream_heartbeat_interval_ms: None,
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "phase": self.phase,
            "upstream_ttfb_ms": self.upstream_ttfb_ms,
            "upstream_sse_frame_count": self.upstream_sse_frame_count,
            "last_upstream_event": self.last_upstream_event,
            "last_upstream_frame_at_unix_ms": self.last_upstream_frame_at_unix_ms,
            "partial_image_count": self.partial_image_count,
            "last_client_visible_event": self.last_client_visible_event,
            "downstream_heartbeat_count": self.downstream_heartbeat_count,
            "last_downstream_heartbeat_at_unix_ms": self.last_downstream_heartbeat_at_unix_ms,
            "downstream_heartbeat_interval_ms": self.downstream_heartbeat_interval_ms,
        })
    }
}

impl<'a> OpenAiImageSyncProgressRecorder<'a> {
    fn new(
        state: &'a AppState,
        plan: &'a ExecutionPlan,
        report_context: Option<&'a Value>,
        snapshot: Option<Arc<Mutex<OpenAiImageSyncProgressSnapshot>>>,
    ) -> Self {
        Self {
            state,
            plan,
            report_context,
            snapshot: snapshot
                .unwrap_or_else(|| Arc::new(Mutex::new(OpenAiImageSyncProgressSnapshot::new()))),
            buffer: Vec::new(),
            last_persist_at: None,
        }
    }

    async fn persist(
        &mut self,
        status: RequestCandidateStatus,
        status_code: Option<u16>,
        latency_ms: Option<u64>,
        force: bool,
    ) {
        let now = Instant::now();
        if !force
            && self.last_persist_at.is_some_and(|last| {
                now.duration_since(last) < OPENAI_IMAGE_SYNC_PROGRESS_WRITE_INTERVAL
            })
        {
            return;
        }
        let snapshot = self.snapshot.lock().await.clone();
        let extra_data = json!({
            "image_progress": snapshot.to_json(),
        });
        record_local_request_candidate_extra_data(
            self.state,
            self.plan,
            self.report_context,
            non_terminal_image_progress_status(status),
            status_code,
            latency_ms,
            extra_data,
        )
        .await;
        self.last_persist_at = Some(now);
    }

    async fn record_connecting(&mut self) {
        self.snapshot.lock().await.phase = "upstream_connecting";
        self.persist(RequestCandidateStatus::Pending, None, None, true)
            .await;
    }

    async fn record_response_started(&mut self, status_code: u16, ttfb_ms: u64) {
        {
            let mut snapshot = self.snapshot.lock().await;
            snapshot.phase = if status_code >= 400 {
                "failed"
            } else {
                "upstream_streaming"
            };
            snapshot.upstream_ttfb_ms = Some(ttfb_ms);
        }
        self.persist(
            if status_code >= 400 {
                RequestCandidateStatus::Failed
            } else {
                RequestCandidateStatus::Streaming
            },
            Some(status_code),
            Some(ttfb_ms),
            true,
        )
        .await;
    }

    async fn observe_chunk(&mut self, chunk: &[u8], status_code: u16, elapsed_ms: u64) {
        if chunk.is_empty() {
            return;
        }
        self.buffer.extend_from_slice(chunk);
        let mut force_persist = false;
        while let Some(block_end) = find_sse_block_end(&self.buffer) {
            let block = self.buffer.drain(..block_end).collect::<Vec<_>>();
            let Some(frame) = parse_openai_image_sync_sse_frame(&block) else {
                continue;
            };
            {
                let mut snapshot = self.snapshot.lock().await;
                snapshot.upstream_sse_frame_count =
                    snapshot.upstream_sse_frame_count.saturating_add(1);
                snapshot.last_upstream_event = Some(frame.event_name);
                snapshot.last_upstream_frame_at_unix_ms = Some(current_request_candidate_unix_ms());
                if frame.is_partial_image {
                    snapshot.partial_image_count = snapshot.partial_image_count.saturating_add(1);
                }
                if let Some(client_visible_event) = frame.client_visible_event {
                    snapshot.last_client_visible_event = Some(client_visible_event.to_string());
                    force_persist = true;
                }
                if frame.is_failed || status_code >= 400 {
                    snapshot.phase = "failed";
                    force_persist = true;
                } else if frame.is_completed {
                    snapshot.phase = "upstream_completed";
                    force_persist = true;
                } else {
                    snapshot.phase = "upstream_streaming";
                }
            }
        }
        let phase = self.snapshot.lock().await.phase;
        self.persist(
            if phase == "failed" {
                RequestCandidateStatus::Failed
            } else {
                RequestCandidateStatus::Streaming
            },
            Some(status_code),
            Some(elapsed_ms),
            force_persist,
        )
        .await;
    }

    async fn finish(&mut self, status_code: u16, elapsed_ms: u64) {
        {
            let mut snapshot = self.snapshot.lock().await;
            if status_code >= 400 || snapshot.phase == "failed" {
                snapshot.phase = "failed";
            } else {
                snapshot.phase = "upstream_completed";
            }
        }
        self.persist(
            if status_code >= 400 {
                RequestCandidateStatus::Failed
            } else {
                RequestCandidateStatus::Streaming
            },
            Some(status_code),
            Some(elapsed_ms),
            true,
        )
        .await;
    }

    async fn fail(&mut self, status_code: Option<u16>, elapsed_ms: u64) {
        self.snapshot.lock().await.phase = "failed";
        self.persist(
            RequestCandidateStatus::Failed,
            status_code,
            Some(elapsed_ms),
            true,
        )
        .await;
    }
}

impl OpenAiImageSyncJsonHeartbeatContext {
    async fn record_heartbeat(&self, heartbeat_kind: &'static str, heartbeat_interval: Duration) {
        let now_unix_ms = current_request_candidate_unix_ms();
        let elapsed_ms = self.started_at.elapsed().as_millis() as u64;
        let interval_ms = heartbeat_interval
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX);
        let (count, phase, progress_json) = {
            let mut snapshot = self.snapshot.lock().await;
            snapshot.downstream_heartbeat_count =
                snapshot.downstream_heartbeat_count.saturating_add(1);
            snapshot.last_downstream_heartbeat_at_unix_ms = Some(now_unix_ms);
            snapshot.downstream_heartbeat_interval_ms = Some(interval_ms);
            (
                snapshot.downstream_heartbeat_count,
                snapshot.phase,
                snapshot.to_json(),
            )
        };
        let status = match phase {
            // The phase remains visible in `image_progress`; do not publish a
            // terminal candidate while terminal usage is still in flight.
            "failed" => RequestCandidateStatus::Streaming,
            "upstream_connecting" => RequestCandidateStatus::Pending,
            _ => RequestCandidateStatus::Streaming,
        };
        record_local_request_candidate_extra_data(
            &self.state,
            &self.plan,
            self.report_context.as_ref(),
            status,
            None,
            Some(elapsed_ms),
            json!({ "image_progress": progress_json }),
        )
        .await;
        debug!(
            event_name = "openai_image_sync_json_heartbeat_sent",
            log_type = "event",
            trace_id = %self.trace_id,
            request_id = %self.request_id_for_log,
            candidate_id = self.candidate_id.as_deref().unwrap_or("-"),
            heartbeat_kind,
            heartbeat_count = count,
            heartbeat_interval_ms = interval_ms,
            elapsed_ms,
            phase,
            "gateway emitted OpenAI image sync JSON whitespace heartbeat"
        );
    }
}

fn find_sse_block_end(buffer: &[u8]) -> Option<usize> {
    buffer
        .windows(2)
        .position(|window| window == b"\n\n")
        .map(|index| index + 2)
        .or_else(|| {
            buffer
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
                .map(|index| index + 4)
        })
}

fn parse_openai_image_sync_sse_frame(block: &[u8]) -> Option<OpenAiImageSyncSseFrame> {
    let text = std::str::from_utf8(block).ok()?.trim();
    if text.is_empty() {
        return None;
    }

    let mut event_name = None;
    let mut data_lines = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r').trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim().to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.trim().to_string());
        }
    }

    let data_text = data_lines.join("\n");
    if data_text.trim().eq("[DONE]") {
        let event_name = event_name.unwrap_or_else(|| "done".to_string());
        return Some(OpenAiImageSyncSseFrame {
            event_name,
            is_partial_image: false,
            is_completed: true,
            is_failed: false,
            client_visible_event: None,
        });
    }

    let data_event_name = serde_json::from_str::<Value>(&data_text)
        .ok()
        .and_then(|value| {
            value
                .get("type")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    value
                        .get("error")
                        .and_then(Value::as_object)
                        .map(|_| "error".to_string())
                })
        });
    let event_name = event_name.or(data_event_name)?;
    let is_partial_image = event_name == "response.image_generation_call.partial_image";
    let is_completed = event_name == "response.completed";
    let is_failed = event_name == "response.failed"
        || event_name == "response.error"
        || event_name == "error"
        || event_name.ends_with(".failed");
    let client_visible_event = if is_partial_image {
        Some("image_generation.partial_image")
    } else if is_completed {
        Some("image_generation.completed")
    } else if is_failed {
        Some("image_generation.failed")
    } else {
        None
    };

    Some(OpenAiImageSyncSseFrame {
        event_name,
        is_partial_image,
        is_completed,
        is_failed,
        client_visible_event,
    })
}

#[allow(clippy::too_many_arguments)]
async fn execute_direct_sync_runtime_candidate(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    trace_id: &str,
    plan_kind: &str,
    candidate_started_unix_ms: u64,
    request_id_for_log: &str,
    candidate_id: Option<&str>,
    provider_name: &str,
    endpoint_id: &str,
    key_id: &str,
    model_name: &str,
    candidate_index: &str,
    progress_snapshot: Option<Arc<Mutex<OpenAiImageSyncProgressSnapshot>>>,
) -> Result<ExecutionResult, SyncExecutionFailure> {
    if let Some(result) = maybe_execute_windsurf_sync(state, plan, report_context)
        .await
        .map_err(SyncExecutionFailure::from_transport)?
    {
        return Ok(result);
    }
    if !should_track_openai_image_sync_upstream_sse(plan_kind, plan, report_context) {
        let state_for_response_started = state.clone();
        let response_started_lifecycle_seed = build_lifecycle_usage_seed(plan, report_context);
        let response_started_candidate_snapshot =
            snapshot_local_request_candidate_status(plan, report_context);
        return DirectSyncExecutionRuntime::new()
            .execute_sync_with_response_started(plan, move |event| {
                record_sync_response_started(
                    &state_for_response_started,
                    response_started_lifecycle_seed,
                    response_started_candidate_snapshot,
                    candidate_started_unix_ms,
                    event.status_code,
                    event.ttfb_ms,
                );
                spawn_local_oauth_success_effect(
                    state_for_response_started.clone(),
                    plan,
                    report_context,
                    LocalOAuthSuccessEffect {
                        status_code: event.status_code,
                        request_started_at_unix_ms: Some(
                            event.response_observation.request_started_at_unix_ms,
                        ),
                        request_order_id: Some(&event.response_observation.request_order_id),
                    },
                );
            })
            .await
            .map_err(SyncExecutionFailure::from_transport);
    }

    let started_at = Instant::now();
    let timeout_ms = resolve_openai_image_sync_total_timeout_ms(plan);
    let mut execution = Box::pin(execute_openai_image_sync_upstream_sse_candidate(
        state,
        plan,
        report_context,
        progress_snapshot.clone(),
    ));
    let mut idle_interval = tokio::time::interval(SYNC_EXECUTION_IDLE_LOG_INTERVAL);
    idle_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    idle_interval.tick().await;
    let mut total_timeout = Box::pin(tokio::time::sleep(Duration::from_millis(timeout_ms)));

    loop {
        tokio::select! {
            result = execution.as_mut() => {
                match result {
                    Ok(result) => return Ok(result),
                    Err(err) => {
                        let elapsed_ms = started_at.elapsed().as_millis() as u64;
                        let status_code = err.status_code;
                        record_openai_image_sync_failed_progress(
                            state,
                            plan,
                            report_context,
                            status_code,
                            elapsed_ms,
                            progress_snapshot.clone(),
                        )
                        .await;
                        return Err(err);
                    }
                }
            }
            _ = idle_interval.tick() => {
                warn!(
                    event_name = "openai_image_sync_execution_idle",
                    log_type = "ops",
                    trace_id = %trace_id,
                    request_id = %request_id_for_log,
                    candidate_id = candidate_id.unwrap_or("-"),
                    provider_name,
                    endpoint_id,
                    key_id,
                    model_name,
                    candidate_index,
                    elapsed_ms = started_at.elapsed().as_millis() as u64,
                    timeout_ms,
                    "gateway OpenAI image sync execution still waiting for upstream response"
                );
            }
            _ = total_timeout.as_mut() => {
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                record_openai_image_sync_failed_progress(
                    state,
                    plan,
                    report_context,
                    Some(StatusCode::GATEWAY_TIMEOUT.as_u16()),
                    elapsed_ms,
                    progress_snapshot.clone(),
                )
                .await;
                warn!(
                    event_name = "openai_image_sync_total_timeout",
                    log_type = "ops",
                    trace_id = %trace_id,
                    request_id = %request_id_for_log,
                    candidate_id = candidate_id.unwrap_or("-"),
                    provider_name,
                    endpoint_id,
                    key_id,
                    model_name,
                    candidate_index,
                    elapsed_ms,
                    timeout_ms,
                    "gateway OpenAI image sync execution exceeded total timeout"
                );
                return Err(SyncExecutionFailure::image_sync_total_timeout(
                    timeout_ms,
                    elapsed_ms,
                ));
            }
        }
    }
}

async fn execute_openai_image_sync_upstream_sse_candidate(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    progress_snapshot: Option<Arc<Mutex<OpenAiImageSyncProgressSnapshot>>>,
) -> Result<ExecutionResult, SyncExecutionFailure> {
    let request_body = build_request_body(plan).map_err(SyncExecutionFailure::from_transport)?;
    let response_body_limit_bytes = execution_plan_response_body_limit_bytes(plan);
    let started_at = Instant::now();
    let mut progress =
        OpenAiImageSyncProgressRecorder::new(state, plan, report_context, progress_snapshot);
    progress.record_connecting().await;

    let request_started_at_unix_ms = current_request_candidate_unix_ms();
    let request_order_id = uuid::Uuid::now_v7().to_string();
    let response = send_request(plan, request_body)
        .await
        .map_err(SyncExecutionFailure::from_transport)?;
    let ttfb_ms = started_at.elapsed().as_millis() as u64;
    let response_headers_observed_at_unix_ms = current_request_candidate_unix_ms();
    let status_code = response.status_code();
    let headers = response.headers();
    spawn_local_oauth_success_effect(
        state.clone(),
        plan,
        report_context,
        LocalOAuthSuccessEffect {
            status_code,
            request_started_at_unix_ms: Some(request_started_at_unix_ms),
            request_order_id: Some(&request_order_id),
        },
    );
    progress.record_response_started(status_code, ttfb_ms).await;

    let mut body_bytes = Vec::new();
    match response {
        DirectHttpResponse::Reqwest(response) => {
            let mut upstream_stream = response.bytes_stream();
            while let Some(chunk) = upstream_stream.next().await {
                let chunk = chunk.map_err(|err| {
                    SyncExecutionFailure::from_transport(
                        ExecutionRuntimeTransportError::UpstreamRequest(
                            format_upstream_request_error(&err),
                        ),
                    )
                })?;
                append_upstream_response_body_chunk_with_limit(
                    &mut body_bytes,
                    &chunk,
                    response_body_limit_bytes,
                )
                .map_err(SyncExecutionFailure::from_transport)?;
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                progress
                    .observe_chunk(&chunk, status_code, elapsed_ms)
                    .await;
            }
        }
        DirectHttpResponse::HyperH2c(response) => {
            let mut upstream_stream = response.into_body().into_data_stream();
            while let Some(chunk) = upstream_stream.next().await {
                let chunk = chunk.map_err(|err| {
                    SyncExecutionFailure::from_transport(
                        ExecutionRuntimeTransportError::UpstreamRequest(format_hyper_error_chain(
                            &err,
                        )),
                    )
                })?;
                append_upstream_response_body_chunk_with_limit(
                    &mut body_bytes,
                    &chunk,
                    response_body_limit_bytes,
                )
                .map_err(SyncExecutionFailure::from_transport)?;
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                progress
                    .observe_chunk(&chunk, status_code, elapsed_ms)
                    .await;
            }
        }
        DirectHttpResponse::BrowserWreq(response) => {
            let mut upstream_stream = response.bytes_stream();
            while let Some(chunk) = upstream_stream.next().await {
                let chunk = chunk.map_err(|err| {
                    SyncExecutionFailure::from_transport(
                        ExecutionRuntimeTransportError::UpstreamRequest(
                            format_wreq_upstream_request_error(&err),
                        ),
                    )
                })?;
                append_upstream_response_body_chunk_with_limit(
                    &mut body_bytes,
                    &chunk,
                    response_body_limit_bytes,
                )
                .map_err(SyncExecutionFailure::from_transport)?;
                let elapsed_ms = started_at.elapsed().as_millis() as u64;
                progress
                    .observe_chunk(&chunk, status_code, elapsed_ms)
                    .await;
            }
        }
    }

    let decoded_body_bytes =
        decode_response_body_bytes_with_limit(&headers, &body_bytes, response_body_limit_bytes)
            .map_err(SyncExecutionFailure::from_transport)?;
    let elapsed_ms = started_at.elapsed().as_millis() as u64;
    let upstream_bytes = body_bytes.len() as u64;
    progress.finish(status_code, elapsed_ms).await;

    let body = build_execution_response_body(
        &headers,
        &body_bytes,
        decoded_body_bytes.as_ref(),
        plan.stream,
        execution_response_body_mode(plan),
    )
    .map_err(SyncExecutionFailure::from_transport)?;

    Ok(ExecutionResult {
        request_id: plan.request_id.clone(),
        candidate_id: plan.candidate_id.clone(),
        status_code,
        headers,
        response_observation: Some(ExecutionResponseObservation {
            request_started_at_unix_ms,
            response_headers_observed_at_unix_ms,
            request_order_id,
        }),
        body,
        telemetry: Some(ExecutionTelemetry {
            ttfb_ms: Some(ttfb_ms),
            elapsed_ms: Some(elapsed_ms),
            upstream_bytes: Some(upstream_bytes),
        }),
        error: None,
    })
}

async fn record_openai_image_sync_failed_progress(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_code: Option<u16>,
    elapsed_ms: u64,
    progress_snapshot: Option<Arc<Mutex<OpenAiImageSyncProgressSnapshot>>>,
) {
    let mut progress =
        OpenAiImageSyncProgressRecorder::new(state, plan, report_context, progress_snapshot);
    progress.fail(status_code, elapsed_ms).await;
}

fn resolve_openai_image_sync_total_timeout_ms(plan: &ExecutionPlan) -> u64 {
    plan.timeouts
        .as_ref()
        .and_then(|timeouts| timeouts.total_ms)
        .unwrap_or(OPENAI_IMAGE_SYNC_DEFAULT_TOTAL_TIMEOUT_MS)
        .max(1)
}

fn report_context_upstream_is_stream(report_context: Option<&Value>) -> bool {
    report_context
        .and_then(|value| value.get(UPSTREAM_IS_STREAM_KEY))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn should_track_openai_image_sync_upstream_sse(
    plan_kind: &str,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
) -> bool {
    plan_kind == OPENAI_IMAGE_SYNC_PLAN_KIND
        && (plan.stream || report_context_upstream_is_stream(report_context))
}

fn should_enable_openai_image_sync_json_heartbeat(
    _plan_kind: &str,
    _plan: &ExecutionPlan,
    _report_context: Option<&Value>,
) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
fn build_openai_image_sync_json_heartbeat_response(
    state: AppState,
    request_path: String,
    plan: ExecutionPlan,
    trace_id: String,
    decision: GatewayControlDecision,
    plan_kind: String,
    report_kind: Option<String>,
    report_context: Option<Value>,
) -> Result<Response<Body>, GatewayError> {
    let request_id = plan.request_id.clone();
    let candidate_id = plan.candidate_id.clone();
    let trace_id_for_response = trace_id.clone();
    let decision_for_response = decision.clone();
    let progress_snapshot = Arc::new(Mutex::new(OpenAiImageSyncProgressSnapshot::new()));
    let heartbeat_context = OpenAiImageSyncJsonHeartbeatContext {
        state: state.clone(),
        plan: plan.clone(),
        report_context: report_context.clone(),
        snapshot: progress_snapshot.clone(),
        started_at: Instant::now(),
        trace_id: trace_id.clone(),
        request_id_for_log: short_request_id(request_id.as_str()),
        candidate_id: candidate_id.clone(),
    };
    let (tx, rx) = mpsc::channel::<Result<Bytes, IoError>>(1);

    tokio::spawn(async move {
        let bytes = openai_image_sync_json_heartbeat_final_bytes(
            execute_execution_runtime_sync_impl(
                &state,
                request_path.as_str(),
                plan,
                trace_id.as_str(),
                &decision,
                plan_kind.as_str(),
                report_kind,
                report_context,
                false,
                Some(progress_snapshot),
                None,
                None,
            )
            .await,
        )
        .await;
        let _ = tx.send(Ok(Bytes::from(bytes))).await;
    });

    let headers = BTreeMap::from([(
        CONTENT_TYPE.as_str().to_string(),
        "application/json".to_string(),
    )]);
    let response = build_client_response_from_parts_with_mutator(
        StatusCode::OK.as_u16(),
        &headers,
        Body::from_stream(build_json_whitespace_heartbeat_stream(
            rx,
            OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_INTERVAL,
            Some(heartbeat_context),
        )),
        trace_id_for_response.as_str(),
        Some(&decision_for_response),
        |headers| {
            headers.remove(CONTENT_LENGTH);
            headers.remove(CONTENT_ENCODING);
            headers.insert(
                CACHE_CONTROL,
                HeaderValue::from_static("no-cache, no-transform"),
            );
            headers.insert(
                HeaderName::from_static("x-accel-buffering"),
                HeaderValue::from_static("no"),
            );
            Ok(())
        },
    )?;
    attach_control_metadata_headers(response, Some(request_id.as_str()), candidate_id.as_deref())
}

fn build_json_whitespace_heartbeat_stream(
    mut rx: mpsc::Receiver<Result<Bytes, IoError>>,
    heartbeat_interval: Duration,
    heartbeat_context: Option<OpenAiImageSyncJsonHeartbeatContext>,
) -> impl futures_util::Stream<Item = Result<Bytes, IoError>> + Send + 'static {
    stream! {
        if let Some(context) = heartbeat_context.as_ref() {
            context.record_heartbeat("initial", heartbeat_interval).await;
        }
        yield Ok(Bytes::from_static(OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_BYTES));

        let mut heartbeat = tokio::time::interval(heartbeat_interval);
        heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
        heartbeat.tick().await;
        loop {
            tokio::select! {
                biased;
                item = rx.recv() => {
                    let Some(item) = item else {
                        break;
                    };
                    yield item;
                    break;
                }
                _ = heartbeat.tick() => {
                    if let Some(context) = heartbeat_context.as_ref() {
                        context.record_heartbeat("interval", heartbeat_interval).await;
                    }
                    yield Ok(Bytes::from_static(OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_BYTES));
                }
            }
        }
    }
}

pub(crate) fn build_sync_json_whitespace_heartbeat_stream(
    rx: mpsc::Receiver<Result<Bytes, IoError>>,
) -> impl futures_util::Stream<Item = Result<Bytes, IoError>> + Send + 'static {
    build_json_whitespace_heartbeat_stream(rx, OPENAI_IMAGE_SYNC_JSON_HEARTBEAT_INTERVAL, None)
}

pub(crate) fn build_openai_image_sync_json_whitespace_heartbeat_stream(
    rx: mpsc::Receiver<Result<Bytes, IoError>>,
) -> impl futures_util::Stream<Item = Result<Bytes, IoError>> + Send + 'static {
    build_sync_json_whitespace_heartbeat_stream(rx)
}

async fn openai_image_sync_json_heartbeat_final_bytes(
    result: Result<Option<Response<Body>>, GatewayError>,
) -> Vec<u8> {
    match result {
        Ok(Some(response)) => match to_bytes(
            response.into_body(),
            crate::headers::max_internal_buffered_body_bytes(),
        )
        .await
        {
            Ok(bytes) if !bytes.is_empty() => bytes.to_vec(),
            Ok(_) => openai_image_sync_json_heartbeat_error_body("empty sync image response"),
            Err(err) => openai_image_sync_json_heartbeat_error_body(&err.to_string()),
        },
        Ok(None) => openai_image_sync_json_heartbeat_error_body(
            "sync image execution ended without a local response",
        ),
        Err(err) => openai_image_sync_json_heartbeat_error_body(&format!("{err:?}")),
    }
}

fn openai_image_sync_json_heartbeat_error_body(message: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "error": {
            "type": "aether_gateway_error",
            "message": message,
        }
    }))
    .unwrap_or_else(|_| b"{\"error\":{\"type\":\"aether_gateway_error\"}}".to_vec())
}

async fn apply_sync_success_effects(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&serde_json::Value>,
    payload: &GatewaySyncReportRequest,
) {
    if let Some(report_context) = report_context {
        crate::ai_serving::persist_converted_response_history(
            state.runtime_state(),
            report_context,
            payload
                .client_body_json
                .as_ref()
                .or(payload.body_json.as_ref()),
        )
        .await;
    }
    apply_local_execution_effect(
        state,
        LocalExecutionEffectContext {
            plan,
            report_context,
        },
        LocalExecutionEffect::HealthSuccess(LocalHealthSuccessEffect),
    )
    .await;
    apply_local_execution_effect(
        state,
        LocalExecutionEffectContext {
            plan,
            report_context,
        },
        LocalExecutionEffect::AdaptiveSuccess(LocalAdaptiveSuccessEffect),
    )
    .await;
    apply_local_execution_effect(
        state,
        LocalExecutionEffectContext {
            plan,
            report_context,
        },
        LocalExecutionEffect::PoolSuccessSync { payload },
    )
    .await;
}

async fn apply_sync_failure_effects(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_code: u16,
    headers: &BTreeMap<String, String>,
    response_text: Option<&str>,
    analysis: crate::orchestration::LocalFailoverAnalysis,
) {
    apply_local_stream_failure_effects_with_analysis(
        state,
        LocalExecutionEffectContext {
            plan,
            report_context,
        },
        LocalStreamFailureEffect::new(status_code, headers, response_text),
        analysis,
    )
    .await;
}

#[cfg(test)]
enum RemoteSyncFallbackOutcome {
    Executed(ExecutionResult),
    ClientResponse(Response<Body>),
    Unavailable,
}

#[allow(clippy::too_many_arguments)] // internal function, grouping would add unnecessary indirection
pub(crate) async fn execute_execution_runtime_sync(
    state: &AppState,
    request_path: &str,
    mut plan: ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    report_kind: Option<String>,
    mut report_context: Option<serde_json::Value>,
) -> Result<Option<Response<Body>>, GatewayError> {
    execute_execution_runtime_sync_impl(
        state,
        request_path,
        plan,
        trace_id,
        decision,
        plan_kind,
        report_kind,
        report_context,
        true,
        None,
        None,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_execution_runtime_sync_with_retry_scope(
    state: &AppState,
    request_path: &str,
    plan: ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    report_kind: Option<String>,
    report_context: Option<serde_json::Value>,
) -> Result<AiAttemptExecutionOutcome<Response<Body>>, GatewayError> {
    let mut retry_scope = AiAttemptRetryScope::Candidate;
    let mut fallback_response = None;
    let response = execute_execution_runtime_sync_impl(
        state,
        request_path,
        plan,
        trace_id,
        decision,
        plan_kind,
        report_kind,
        report_context,
        true,
        None,
        Some(&mut retry_scope),
        Some(&mut fallback_response),
    )
    .await?;
    Ok(match response {
        Some(response) => AiAttemptExecutionOutcome::Responded(
            crate::executor::mark_sync_response_usage_terminal_owner(response),
        ),
        None => AiAttemptExecutionOutcome::Retry {
            scope: retry_scope,
            fallback_response,
        },
    })
}

#[allow(clippy::too_many_arguments)] // internal function, grouping would add unnecessary indirection
async fn execute_execution_runtime_sync_impl(
    state: &AppState,
    request_path: &str,
    mut plan: ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    report_kind: Option<String>,
    mut report_context: Option<serde_json::Value>,
    allow_json_heartbeat: bool,
    progress_snapshot: Option<Arc<Mutex<OpenAiImageSyncProgressSnapshot>>>,
    mut retry_scope_out: Option<&mut AiAttemptRetryScope>,
    mut retry_fallback_out: Option<&mut Option<Response<Body>>>,
) -> Result<Option<Response<Body>>, GatewayError> {
    if allow_json_heartbeat
        && should_enable_openai_image_sync_json_heartbeat(plan_kind, &plan, report_context.as_ref())
    {
        return build_openai_image_sync_json_heartbeat_response(
            state.clone(),
            request_path.to_string(),
            plan,
            trace_id.to_string(),
            decision.clone(),
            plan_kind.to_string(),
            report_kind,
            report_context,
        )
        .map(Some);
    }

    ensure_execution_request_candidate_slot(state, &mut plan, &mut report_context).await;
    let plan_request_id = plan.request_id.clone();
    let plan_request_id_for_log = short_request_id(plan_request_id.as_str());
    let plan_candidate_id = plan.candidate_id.clone();
    let provider_name = plan
        .provider_name
        .clone()
        .unwrap_or_else(|| "-".to_string());
    let endpoint_id = plan.endpoint_id.clone();
    let key_id = plan.key_id.clone();
    let model_name = plan.model_name.clone().unwrap_or_else(|| "-".to_string());
    let candidate_index = parse_request_candidate_report_context(report_context.as_ref())
        .and_then(|context| context.candidate_index)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let candidate_started_at = Instant::now();
    let candidate_started_unix_secs = current_request_candidate_unix_ms();
    let lifecycle_seed = build_lifecycle_usage_seed(&plan, report_context.as_ref());
    // Arm before either pending usage or candidate-state persistence.  Both
    // operations can await a queue/database; cancellation in either write
    // window must still have a process-lifetime terminal fallback.
    let mut terminal_guard = SyncAttemptTerminalGuard::new(
        state,
        &plan,
        report_context.clone(),
        candidate_started_unix_secs,
        candidate_started_at,
    );
    // The exact per-attempt Drop guard now owns cancellation. Pause the
    // request-gap owner before the first lifecycle await so it cannot race
    // with an older candidate snapshot.
    crate::executor::pause_current_request_terminal_gap_for_attempt();
    let usage_data = state.usage_lifecycle_data_state().as_ref().clone();
    state
        .usage_runtime
        .record_pending_direct(&usage_data, lifecycle_seed)
        .await;
    record_local_request_candidate_status(
        state,
        &plan,
        report_context.as_ref(),
        SchedulerRequestCandidateStatusUpdate {
            status: RequestCandidateStatus::Pending,
            status_code: None,
            error_type: None,
            error_message: None,
            latency_ms: None,
            started_at_unix_ms: Some(candidate_started_unix_secs),
            finished_at_unix_ms: None,
        },
    )
    .await;
    let result = (async {
    let _provider_pool_in_flight_guard = acquire_provider_pool_in_flight_guard(
        state.runtime_state.clone(),
        &plan.provider_id,
        plan_request_id.as_str(),
        plan_candidate_id.as_deref(),
        key_id.as_str(),
    )
    .await;
    record_sync_execution_active(
        state,
        &plan,
        report_context.as_ref(),
        candidate_started_unix_secs,
    );
    #[cfg(not(test))]
    let mut result = {
        match maybe_execute_grok_sync(&plan, report_context.as_ref()).await {
            Ok(Some(result)) => result,
            Ok(None) => {
                match maybe_execute_chatgpt_web_image_sync(state, &plan, report_context.as_ref())
                    .await
                {
                    Ok(Some(result)) => result,
                    Ok(None) => match execute_direct_sync_runtime_candidate(
                        state,
                        &plan,
                        report_context.as_ref(),
                        trace_id,
                        plan_kind,
                        candidate_started_unix_secs,
                        plan_request_id_for_log.as_str(),
                        plan_candidate_id.as_deref(),
                        provider_name.as_str(),
                        endpoint_id.as_str(),
                        key_id.as_str(),
                        model_name.as_str(),
                        candidate_index.as_str(),
                        progress_snapshot.clone(),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            let failure_error_type = err.error_type;
                            let failure_message = err.message.clone();
                            let failure_latency_ms = err
                                .latency_ms
                                .unwrap_or_else(|| elapsed_ms_since(candidate_started_at));
                            maybe_store_sync_execution_failure_fallback(
                                &err,
                                &plan,
                                trace_id,
                                decision,
                                &mut retry_scope_out,
                                &mut retry_fallback_out,
                            )?;
                            warn!(
                                event_name = "sync_execution_runtime_unavailable",
                                log_type = "ops",
                                trace_id = %trace_id,
                                request_id = %plan_request_id_for_log,
                                candidate_id = ?plan_candidate_id,
                                provider_name,
                                endpoint_id,
                                key_id,
                                model_name,
                                candidate_index = candidate_index.as_str(),
                                error_type = err.error_type,
                                error = %err.message,
                                "gateway in-process sync execution unavailable"
                            );
                            if let Some(response) = stop_or_record_sync_transport_failure(
                                state,
                                &plan,
                                report_context.as_ref(),
                                trace_id,
                                decision,
                                failure_error_type,
                                failure_message.as_str(),
                                failure_latency_ms,
                                candidate_started_unix_secs,
                                retry_scope_out.is_some(),
                            )
                            .await?
                            {
                                return Ok(Some(response));
                            }
                            return Ok(None);
                        }
                    },
                    Err(err) => {
                        let transport_error_message = err.to_string();
                        warn!(
                            event_name = "chatgpt_web_image_execution_unavailable",
                            log_type = "ops",
                            trace_id = %trace_id,
                            request_id = %plan_request_id_for_log,
                            candidate_id = ?plan_candidate_id,
                            provider_name,
                            endpoint_id,
                            key_id,
                            model_name,
                            candidate_index = candidate_index.as_str(),
                            error = %err,
                            "gateway ChatGPT-Web image execution unavailable"
                        );
                        if let Some(response) = stop_or_record_sync_transport_failure(
                            state,
                            &plan,
                            report_context.as_ref(),
                            trace_id,
                            decision,
                            "chatgpt_web_image_execution_unavailable",
                            transport_error_message.as_str(),
                            elapsed_ms_since(candidate_started_at),
                            candidate_started_unix_secs,
                            retry_scope_out.is_some(),
                        )
                        .await?
                        {
                            return Ok(Some(response));
                        }
                        return Ok(None);
                    }
                }
            }
            Err(err) => {
                let transport_error_message = err.to_string();
                warn!(
                    event_name = "grok_execution_unavailable",
                    log_type = "ops",
                    trace_id = %trace_id,
                    request_id = %plan_request_id_for_log,
                    candidate_id = ?plan_candidate_id,
                    provider_name,
                    endpoint_id,
                    key_id,
                    model_name,
                    candidate_index = candidate_index.as_str(),
                    error = %err,
                    "gateway Grok execution unavailable"
                );
                if let Some(response) = stop_or_record_sync_transport_failure(
                    state,
                    &plan,
                    report_context.as_ref(),
                    trace_id,
                    decision,
                    "grok_execution_unavailable",
                    transport_error_message.as_str(),
                    elapsed_ms_since(candidate_started_at),
                    candidate_started_unix_secs,
                    retry_scope_out.is_some(),
                )
                .await?
                {
                    return Ok(Some(response));
                }
                return Ok(None);
            }
        }
    };
    #[cfg(test)]
    let mut result = {
        if let Some(override_fn) = state.execution_runtime_sync_override.as_ref() {
            match (override_fn.0)(&plan) {
                Ok(result) => result,
                Err(err) => {
                    let transport_error_message = format!("{err:?}");
                    warn!(
                        event_name = "sync_execution_runtime_test_override_failed",
                        log_type = "ops",
                        trace_id = %trace_id,
                        request_id = %plan_request_id_for_log,
                        candidate_id = ?plan_candidate_id,
                        provider_name,
                        endpoint_id,
                        key_id,
                        model_name,
                        candidate_index = candidate_index.as_str(),
                        error = ?err,
                        "gateway test sync execution override failed"
                    );
                    if let Some(response) = stop_or_record_sync_transport_failure(
                        state,
                        &plan,
                        report_context.as_ref(),
                        trace_id,
                        decision,
                        "execution_runtime_unavailable",
                        transport_error_message.as_str(),
                        elapsed_ms_since(candidate_started_at),
                        candidate_started_unix_secs,
                        retry_scope_out.is_some(),
                    )
                    .await?
                    {
                        return Ok(Some(response));
                    }
                    return Ok(None);
                }
            }
        } else if state
            .execution_runtime_override_base_url()
            .unwrap_or_default()
            .trim()
            .is_empty()
        {
            match maybe_execute_grok_sync(&plan, report_context.as_ref()).await {
                Ok(Some(result)) => result,
                Ok(None) => match maybe_execute_chatgpt_web_image_sync(
                    state,
                    &plan,
                    report_context.as_ref(),
                )
                .await
                {
                    Ok(Some(result)) => result,
                    Ok(None) => match execute_direct_sync_runtime_candidate(
                        state,
                        &plan,
                        report_context.as_ref(),
                        trace_id,
                        plan_kind,
                        candidate_started_unix_secs,
                        plan_request_id_for_log.as_str(),
                        plan_candidate_id.as_deref(),
                        provider_name.as_str(),
                        endpoint_id.as_str(),
                        key_id.as_str(),
                        model_name.as_str(),
                        candidate_index.as_str(),
                        progress_snapshot.clone(),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(err) => {
                            let failure_error_type = err.error_type;
                            let failure_message = err.message.clone();
                            let failure_latency_ms = err
                                .latency_ms
                                .unwrap_or_else(|| elapsed_ms_since(candidate_started_at));
                            maybe_store_sync_execution_failure_fallback(
                                &err,
                                &plan,
                                trace_id,
                                decision,
                                &mut retry_scope_out,
                                &mut retry_fallback_out,
                            )?;
                            warn!(
                                event_name = "sync_execution_runtime_unavailable",
                                log_type = "ops",
                                trace_id = %trace_id,
                                request_id = %plan_request_id_for_log,
                                candidate_id = ?plan_candidate_id,
                                provider_name,
                                endpoint_id,
                                key_id,
                                model_name,
                                candidate_index = candidate_index.as_str(),
                                error_type = err.error_type,
                                error = %err.message,
                                "gateway in-process sync execution unavailable"
                            );
                            if let Some(response) = stop_or_record_sync_transport_failure(
                                state,
                                &plan,
                                report_context.as_ref(),
                                trace_id,
                                decision,
                                failure_error_type,
                                failure_message.as_str(),
                                failure_latency_ms,
                                candidate_started_unix_secs,
                                retry_scope_out.is_some(),
                            )
                            .await?
                            {
                                return Ok(Some(response));
                            }
                            return Ok(None);
                        }
                    },
                    Err(err) => {
                        let transport_error_message = err.to_string();
                        warn!(
                            event_name = "chatgpt_web_image_execution_unavailable",
                            log_type = "ops",
                            trace_id = %trace_id,
                            request_id = %plan_request_id_for_log,
                            candidate_id = ?plan_candidate_id,
                            provider_name,
                            endpoint_id,
                            key_id,
                            model_name,
                            candidate_index = candidate_index.as_str(),
                            error = %err,
                            "gateway ChatGPT-Web image execution unavailable"
                        );
                        if let Some(response) = stop_or_record_sync_transport_failure(
                            state,
                            &plan,
                            report_context.as_ref(),
                            trace_id,
                            decision,
                            "chatgpt_web_image_execution_unavailable",
                            transport_error_message.as_str(),
                            elapsed_ms_since(candidate_started_at),
                            candidate_started_unix_secs,
                            retry_scope_out.is_some(),
                        )
                        .await?
                        {
                            return Ok(Some(response));
                        }
                        return Ok(None);
                    }
                },
                Err(err) => {
                    let transport_error_message = err.to_string();
                    warn!(
                        event_name = "grok_execution_unavailable",
                        log_type = "ops",
                        trace_id = %trace_id,
                        request_id = %plan_request_id_for_log,
                        candidate_id = ?plan_candidate_id,
                        provider_name,
                        endpoint_id,
                        key_id,
                        model_name,
                        candidate_index = candidate_index.as_str(),
                        error = %err,
                        "gateway Grok execution unavailable"
                    );
                    if let Some(response) = stop_or_record_sync_transport_failure(
                        state,
                        &plan,
                        report_context.as_ref(),
                        trace_id,
                        decision,
                        "grok_execution_unavailable",
                        transport_error_message.as_str(),
                        elapsed_ms_since(candidate_started_at),
                        candidate_started_unix_secs,
                        retry_scope_out.is_some(),
                    )
                    .await?
                    {
                        return Ok(Some(response));
                    }
                    return Ok(None);
                }
            }
        } else {
            let remote_execution_runtime_base_url = state
                .execution_runtime_override_base_url()
                .unwrap_or_default();
            let remote_outcome = execute_sync_via_remote_execution_runtime(
                state,
                remote_execution_runtime_base_url,
                trace_id,
                decision,
                &plan,
                plan_request_id.as_str(),
                plan_candidate_id.as_deref(),
                report_context.as_ref(),
                candidate_started_unix_secs,
                candidate_started_at,
                retry_scope_out.is_some(),
                Some(Arc::clone(&terminal_guard.watchdog_progress)),
            )
            .await?;
            match remote_outcome {
                RemoteSyncFallbackOutcome::Executed(result) => result,
                RemoteSyncFallbackOutcome::ClientResponse(response) => return Ok(Some(response)),
                RemoteSyncFallbackOutcome::Unavailable => {
                    if retry_scope_out.is_none() {
                        return crate::execution_runtime::build_transport_error_stop_response(
                            state,
                            &plan,
                            report_context.as_ref(),
                            trace_id,
                            decision,
                            StatusCode::BAD_GATEWAY.as_u16(),
                            "execution_runtime_unavailable",
                            "execution runtime remained unavailable after the terminal fallback",
                            elapsed_ms_since(candidate_started_at),
                        )
                        .await
                        .map(Some);
                    }
                    return Ok(None);
                }
            }
        }
    };
    let mut candidate_first_byte_elapsed_ms =
        calibrated_sync_candidate_first_byte_elapsed_ms(candidate_started_at, &result);
    let initial_response_observed_at_unix_ms = current_request_candidate_unix_ms();
    let mut provider_response_observation =
        result
            .response_observation
            .clone()
            .unwrap_or(ExecutionResponseObservation {
                request_started_at_unix_ms: candidate_started_unix_secs,
                response_headers_observed_at_unix_ms: initial_response_observed_at_unix_ms,
                request_order_id: uuid::Uuid::now_v7().to_string(),
            });
    let mut oauth_retry_attempted = false;
    let (
        result_error_type,
        result_error_message,
        result_latency_ms,
        headers,
        body_bytes,
        body_json,
        body_base64,
        local_failover_response_text,
        local_failover_analysis,
    ) = loop {
        spawn_local_oauth_success_effect(
            state.clone(),
            &plan,
            report_context.as_ref(),
            LocalOAuthSuccessEffect {
                status_code: result.status_code,
                request_started_at_unix_ms: Some(
                    provider_response_observation.request_started_at_unix_ms,
                ),
                request_order_id: Some(&provider_response_observation.request_order_id),
            },
        );
        let result_latency_ms = result
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms);
        let mut headers = std::mem::take(&mut result.headers);
        let (body_bytes, mut body_json, body_base64) =
            decode_execution_result_body(result.body.take(), &mut headers)?;
        if let Some(message) = invalid_gemini_provider_success_message(
            &plan,
            report_context.as_ref(),
            result.status_code,
            body_json.as_ref(),
        )
        .or_else(|| {
            invalid_gemini_provider_stream_success_message(
                &plan,
                report_context.as_ref(),
                result.status_code,
                body_json.as_ref(),
                &body_bytes,
                body_base64.is_some(),
            )
        }) {
            result.status_code = StatusCode::BAD_GATEWAY.as_u16();
            result.error = Some(invalid_gemini_provider_success_execution_error(message));
            if let Some(error_body) =
                build_invalid_provider_success_body(&plan, report_context.as_ref(), message)
            {
                body_json = Some(error_body);
                headers.insert("content-type".to_string(), "application/json".to_string());
            }
        }
        let (mut result_error_type, mut result_error_message) =
            execution_error_details(result.error.as_ref(), body_json.as_ref());
        if result.status_code < 400 && body_json.is_none() {
            if let Some(error_body_json) =
                extract_provider_private_stream_error_body(report_context.as_ref(), &body_bytes)
            {
                result.status_code =
                    resolve_local_sync_error_status_code(result.status_code, &error_body_json);
                let (private_error_type, private_error_message) =
                    provider_private_error_details(&error_body_json);
                result_error_type = private_error_type.or(result_error_type);
                result_error_message = private_error_message.or(result_error_message);
                body_json = Some(error_body_json);
            }
        }
        let local_failover_response_text = local_failover_response_text(
            body_json.as_ref(),
            &body_bytes,
            result.error.as_ref().map(|error| error.message.as_str()),
        );

        if result.status_code >= 400
            && !oauth_retry_attempted
            && refresh_oauth_plan_auth_for_retry(
                state,
                &mut plan,
                result.status_code,
                local_failover_response_text.as_deref(),
                trace_id,
                report_context.as_ref(),
                Some(provider_response_observation.request_started_at_unix_ms),
                Some(&provider_response_observation.request_order_id),
            )
            .await
        {
            oauth_retry_attempted = true;
            let retry_started_at_unix_ms = current_request_candidate_unix_ms();
            let retry_request_order_id = uuid::Uuid::now_v7().to_string();
            match crate::execution_runtime::execute_execution_runtime_sync_plan(
                state,
                Some(trace_id),
                &plan,
            )
            .await
            {
                Ok(retry_result) => {
                    let retry_response_observed_at_unix_ms = current_request_candidate_unix_ms();
                    provider_response_observation = retry_result
                        .response_observation
                        .clone()
                        .unwrap_or(ExecutionResponseObservation {
                            request_started_at_unix_ms: retry_started_at_unix_ms,
                            response_headers_observed_at_unix_ms:
                                retry_response_observed_at_unix_ms,
                            request_order_id: retry_request_order_id,
                        });
                    candidate_first_byte_elapsed_ms =
                        calibrated_sync_candidate_first_byte_elapsed_ms(
                            candidate_started_at,
                            &retry_result,
                        );
                    result = retry_result;
                    continue;
                }
                Err(err) => {
                    warn!(
                        event_name = "local_sync_oauth_retry_execution_failed",
                        log_type = "ops",
                        trace_id = %trace_id,
                        request_id = %plan_request_id_for_log,
                        candidate_id = ?plan_candidate_id,
                        provider_name,
                        endpoint_id,
                        key_id,
                        model_name,
                        candidate_index = candidate_index.as_str(),
                        error = ?err,
                        "gateway oauth retry sync execution failed"
                    );
                }
            }
        }

        let local_failover_analysis = analyze_local_candidate_failover_sync(
            state,
            &plan,
            plan_kind,
            report_context.as_ref(),
            &result,
            local_failover_response_text.as_deref(),
        )
        .await;
        break (
            result_error_type,
            result_error_message,
            result_latency_ms,
            headers,
            body_bytes,
            body_json,
            body_base64,
            local_failover_response_text,
            local_failover_analysis,
        );
    };
    let mut report_context = attach_provider_response_headers_to_report_context(
        report_context,
        &headers,
        provider_response_observation.request_started_at_unix_ms,
        provider_response_observation.response_headers_observed_at_unix_ms,
        &provider_response_observation.request_order_id,
    );
    if retry_scope_out.is_some()
        && matches!(
            local_failover_analysis.decision,
            LocalFailoverDecision::RetryNextCandidate
        )
    {
        if result.status_code >= 400 {
            apply_sync_failure_effects(
                state,
                &plan,
                report_context.as_ref(),
                result.status_code,
                &headers,
                local_failover_response_text.as_deref(),
                local_failover_analysis,
            )
            .await;
        }
        let failure_disposition = crate::orchestration::classify_failure_disposition(
            &plan.provider_api_format,
            local_failover_analysis.classification,
            result.status_code,
        );
        if let Some(retry_scope) = retry_scope_out.as_deref_mut() {
            *retry_scope =
                ai_attempt_retry_scope_from_failure_disposition(failure_disposition);
        }
        if failure_disposition.preserve_upstream_error {
            if let Some(retry_fallback) = retry_fallback_out.as_deref_mut() {
                let mut fallback_headers = headers.clone();
                apply_endpoint_response_header_rules_best_effort(
                    state,
                    &plan,
                    &mut fallback_headers,
                    body_json.as_ref(),
                )
                .await;
                *retry_fallback = Some(attach_control_metadata_headers(
                    build_client_response_from_parts(
                        result.status_code,
                        &fallback_headers,
                        Body::from(body_bytes.clone()),
                        trace_id,
                        Some(decision),
                    )?,
                    Some(plan.request_id.as_str()),
                    plan.candidate_id.as_deref(),
                )?);
            }
        }
        let terminal_unix_secs = current_request_candidate_unix_ms();
        let error_trace_report_context = with_sync_error_trace_context(
            report_context.as_ref(),
            result.status_code,
            &headers,
            body_json.as_ref(),
            &body_bytes,
            local_failover_response_text.as_deref(),
            local_failover_analysis,
        );
        let intermediate_claimed = try_claim_stream_candidate_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
        if !intermediate_claimed {
            if matches!(
                sync_watchdog_intermediate_ownership(&plan),
                Some(SyncWatchdogIntermediateOwnership::Terminal)
            ) {
                return Ok(Some(build_sync_watchdog_owned_client_response(
                    &plan, trace_id, decision,
                )?));
            }
        } else {
            record_local_request_candidate_status(
                state,
                &plan,
                error_trace_report_context
                    .as_ref()
                    .or(report_context.as_ref()),
                SchedulerRequestCandidateStatusUpdate {
                    status: RequestCandidateStatus::Failed,
                    status_code: Some(result.status_code),
                    error_type: result_error_type.clone(),
                    error_message: result_error_message.clone(),
                    latency_ms: result_latency_ms,
                    started_at_unix_ms: Some(candidate_started_unix_secs),
                    finished_at_unix_ms: Some(terminal_unix_secs),
                },
            )
            .await;
        }
        warn!(
            event_name = "local_sync_candidate_retry_scheduled",
            log_type = "event",
            trace_id = %trace_id,
            request_id = %plan_request_id_for_log,
            status_code = result.status_code,
            provider_name,
            endpoint_id,
            key_id,
            model_name,
            candidate_index = candidate_index.as_str(),
            "gateway local sync decision retrying next candidate after retryable execution runtime result"
        );
        return Ok(None);
    }
    let status_code = result.status_code;
    let has_body_bytes = body_base64.is_some();
    if (200..300).contains(&status_code) {
        seed_kiro_sync_simulated_cache_enabled(state, &plan, &mut report_context).await;
        if kiro_simulated_cache_enabled_from_report_context(report_context.as_ref()) {
            seed_kiro_sync_report_context_input_tokens(&plan, &mut report_context);
        }
        seed_kiro_sync_report_context_prompt_cache_usage(state, &plan, &mut report_context).await;
    }
    let mut client_headers = headers.clone();
    apply_endpoint_response_header_rules_best_effort(
        state,
        &plan,
        &mut client_headers,
        body_json.as_ref(),
    )
    .await;
    let explicit_finalize = should_finalize_sync_response(report_kind.as_deref());
    let mapped_error_finalize_kind =
        resolve_core_sync_error_finalize_report_kind(plan_kind, &result, body_json.as_ref());
    let implicit_finalize = if !explicit_finalize && mapped_error_finalize_kind.is_none() {
        maybe_build_implicit_sync_finalize_outcome(
            trace_id,
            decision,
            plan_kind,
            &report_context,
            status_code,
            &client_headers,
            &body_json,
            &body_base64,
            &result.telemetry,
        )?
    } else {
        None
    };
    if retry_scope_out.is_some()
        && !matches!(
            local_failover_analysis.decision,
            LocalFailoverDecision::StopLocalFailover
        )
        && should_fallback_to_control_sync(
        plan_kind,
        &result,
        body_json.as_ref(),
        has_body_bytes,
        explicit_finalize || implicit_finalize.is_some(),
        mapped_error_finalize_kind.is_some(),
    ) {
        if result.status_code >= 400 {
            apply_sync_failure_effects(
                state,
                &plan,
                report_context.as_ref(),
                result.status_code,
                &headers,
                local_failover_response_text.as_deref(),
                local_failover_analysis,
            )
            .await;
        }
        let terminal_unix_secs = current_request_candidate_unix_ms();
        let error_trace_report_context = with_sync_error_trace_context(
            report_context.as_ref(),
            result.status_code,
            &headers,
            body_json.as_ref(),
            &body_bytes,
            local_failover_response_text.as_deref(),
            local_failover_analysis,
        );
        let intermediate_claimed = try_claim_stream_candidate_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
        if !intermediate_claimed {
            if matches!(
                sync_watchdog_intermediate_ownership(&plan),
                Some(SyncWatchdogIntermediateOwnership::Terminal)
            ) {
                return Ok(Some(build_sync_watchdog_owned_client_response(
                    &plan, trace_id, decision,
                )?));
            }
        } else {
            record_local_request_candidate_status(
                state,
                &plan,
                error_trace_report_context
                    .as_ref()
                    .or(report_context.as_ref()),
                SchedulerRequestCandidateStatusUpdate {
                    status: RequestCandidateStatus::Failed,
                    status_code: Some(result.status_code),
                    error_type: result_error_type.clone(),
                    error_message: result_error_message.clone(),
                    latency_ms: result_latency_ms,
                    started_at_unix_ms: Some(candidate_started_unix_secs),
                    finished_at_unix_ms: Some(terminal_unix_secs),
                },
            )
            .await;
        }
        return Ok(None);
    }

    let terminal_unix_secs = current_request_candidate_unix_ms();
    let error_flow_report_context = (result.status_code >= 400)
        .then(|| {
            with_sync_error_trace_context(
                report_context.as_ref(),
                result.status_code,
                &headers,
                body_json.as_ref(),
                &body_bytes,
                local_failover_response_text.as_deref(),
                local_failover_analysis,
            )
        })
        .flatten();
    // Keep the candidate non-terminal until the usage terminal handoff below has confirmed a
    // durable usage row.  Writing `success` here used to let cleanup observe a skeletal usage
    // record while the terminal billing event was still in flight.
    record_local_request_candidate_status(
        state,
        &plan,
        error_flow_report_context
            .as_ref()
            .or(report_context.as_ref()),
        SchedulerRequestCandidateStatusUpdate {
            status: RequestCandidateStatus::Streaming,
            status_code: Some(result.status_code),
            error_type: None,
            error_message: None,
            latency_ms: result_latency_ms,
            started_at_unix_ms: Some(candidate_started_unix_secs),
            finished_at_unix_ms: None,
        },
    )
    .await;

    let request_id_owned = result.request_id;
    let candidate_id_owned = result.candidate_id;
    let request_id = (!request_id_owned.trim().is_empty())
        .then_some(request_id_owned.as_str())
        .or(Some(plan_request_id.as_str()));
    let request_id_for_log = short_request_id(request_id.unwrap_or("-"));
    let candidate_id = candidate_id_owned
        .as_deref()
        .or(plan_candidate_id.as_deref());
    let report_context = report_context;
    let body_json = body_json;
    let telemetry = result.telemetry;

    if let Some(implicit_finalize) = implicit_finalize {
        let usage_payload = implicit_finalize
            .outcome
            .background_report
            .as_ref()
            .unwrap_or(&implicit_finalize.payload);
        let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
            state,
            &plan,
            implicit_finalize.payload.report_context.as_ref(),
            usage_payload,
            candidate_started_unix_secs,
            candidate_started_at,
            candidate_first_byte_elapsed_ms,
            &mut terminal_guard,
        )
        .await;
        apply_sync_success_effects(
            state,
            &plan,
            implicit_finalize.payload.report_context.as_ref(),
            usage_payload,
        )
        .await;
        if let Some(report_payload) = implicit_finalize.outcome.background_report {
            spawn_sync_report_after_terminal_usage(
                state.clone(),
                report_payload,
                usage_handoff_persisted,
            );
        } else {
            warn!(
                event_name = "local_core_finalize_missing_success_report_mapping",
                log_type = "event",
                trace_id = %trace_id,
                report_kind = %implicit_finalize.payload.report_kind,
                "gateway implicit local core finalize produced response without background success report mapping"
            );
        }
        return Ok(Some(attach_control_metadata_headers(
            implicit_finalize.outcome.response,
            request_id,
            candidate_id,
        )?));
    }

    let finalize_report_kind = if explicit_finalize {
        report_kind.clone()
    } else {
        mapped_error_finalize_kind
    };

    if let Some(finalize_report_kind) = finalize_report_kind {
        let mut payload = build_sync_report_payload(
            trace_id,
            finalize_report_kind,
            report_context,
            status_code,
            client_headers,
            body_json,
            body_base64,
            telemetry,
        );
        if let Some(outcome) = maybe_build_sync_finalize_outcome(trace_id, decision, &payload)? {
            let usage_payload = outcome.background_report.as_ref().unwrap_or(&payload);
            let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
                state,
                &plan,
                payload.report_context.as_ref(),
                usage_payload,
                candidate_started_unix_secs,
                candidate_started_at,
                candidate_first_byte_elapsed_ms,
                &mut terminal_guard,
            )
            .await;
            if status_code < 400 {
                apply_sync_success_effects(
                    state,
                    &plan,
                    payload.report_context.as_ref(),
                    usage_payload,
                )
                .await;
            } else {
                apply_sync_failure_effects(
                    state,
                    &plan,
                    payload.report_context.as_ref(),
                    status_code,
                    &payload.headers,
                    local_failover_response_text.as_deref(),
                    local_failover_analysis,
                )
                .await;
            }
            if let Some(report_payload) = outcome.background_report {
                spawn_sync_report_after_terminal_usage(
                    state.clone(),
                    report_payload,
                    usage_handoff_persisted,
                );
            } else {
                warn!(
                    event_name = "local_core_finalize_missing_success_report_mapping",
                    log_type = "event",
                    trace_id = %trace_id,
                    report_kind = %payload.report_kind,
                    "gateway local core finalize produced response without background success report mapping"
                );
            }
            return Ok(Some(attach_control_metadata_headers(
                outcome.response,
                request_id,
                candidate_id,
            )?));
        }
        let mut payload = match maybe_build_local_video_success_outcome(
            trace_id,
            decision,
            payload,
            &state.video_tasks,
            &plan,
        )? {
            LocalVideoSyncSuccessBuild::Handled(outcome) => {
                let LocalVideoSyncSuccessOutcome {
                    response,
                    report_payload,
                    original_report_context,
                    report_mode,
                    local_task_snapshot,
                } = outcome;
                let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
                    state,
                    &plan,
                    original_report_context.as_ref(),
                    &report_payload,
                    candidate_started_unix_secs,
                    candidate_started_at,
                    candidate_first_byte_elapsed_ms,
                    &mut terminal_guard,
                )
                .await;
                apply_sync_success_effects(
                    state,
                    &plan,
                    original_report_context.as_ref(),
                    &report_payload,
                )
                .await;
                if let Some(snapshot) = local_task_snapshot {
                    let stored = state.upsert_video_task_snapshot(&snapshot).await?;
                    if let Some(stored) = stored.as_ref() {
                        // Async create normally remains pending, but a provider is
                        // allowed to return an immediate terminal state. Settle the
                        // original create request in that case as well.
                        finalize_video_task_if_terminal(state, stored).await;
                    }
                }
                match report_mode {
                    VideoTaskSyncReportMode::InlineSync => {
                        submit_sync_report_after_terminal_usage(
                            state,
                            report_payload,
                            usage_handoff_persisted,
                        )
                        .await?;
                    }
                    VideoTaskSyncReportMode::Background => {
                        spawn_sync_report_after_terminal_usage(
                            state.clone(),
                            report_payload,
                            usage_handoff_persisted,
                        );
                    }
                }
                return Ok(Some(attach_control_metadata_headers(
                    response,
                    request_id,
                    candidate_id,
                )?));
            }
            LocalVideoSyncSuccessBuild::NotHandled(payload) => {
                // A video create that reports success but produces no task would
                // silently never reach the database, leaving the task invisible
                // in the admin UI with no trace of why.
                if payload.report_kind.starts_with("doubao_video_create")
                    || payload.report_kind.starts_with("openai_video_create")
                    || payload.report_kind.starts_with("openai_video_remix")
                    || payload.report_kind.starts_with("gemini_video_create")
                {
                    warn!(
                        event_name = "local_video_task_not_persisted",
                        log_type = "event",
                        trace_id = %trace_id,
                        report_kind = %payload.report_kind,
                        status_code = payload.status_code,
                        provider_api_format = %plan.provider_api_format,
                        upstream_url = %plan.url,
                        truth_source_rust_authoritative = state.video_tasks.is_rust_authoritative(),
                        has_body_json = payload.body_json.is_some(),
                        "gateway video create succeeded upstream but produced no local task"
                    );
                }
                payload
            }
        };
        if let Some(response) =
            maybe_build_local_sync_finalize_response(trace_id, decision, &mut payload)?
        {
            let background_success_report_kind =
                resolve_local_sync_success_background_report_kind(payload.report_kind.as_str());
            let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
                state,
                &plan,
                payload.report_context.as_ref(),
                &payload,
                candidate_started_unix_secs,
                candidate_started_at,
                candidate_first_byte_elapsed_ms,
                &mut terminal_guard,
            )
            .await;
            apply_sync_success_effects(state, &plan, payload.report_context.as_ref(), &payload)
                .await;
            let snapshot = payload
                .report_context
                .as_ref()
                .and_then(|context| context.get("task_id"))
                .and_then(Value::as_str)
                .and_then(|internal_task_id| {
                    state
                        .video_tasks
                        .apply_finalize_mutation_for_internal_task_id(
                            internal_task_id,
                            payload.report_kind.as_str(),
                        )
                })
                .or_else(|| {
                    state
                        .video_tasks
                        .apply_finalize_mutation(request_path, payload.report_kind.as_str());
                    state
                        .video_tasks
                        .snapshot_for_route(decision.route_family.as_deref(), request_path)
                });
            if let Some(snapshot) = snapshot {
                if let Some(stored) = state.upsert_video_task_snapshot(&snapshot).await? {
                    if payload.report_kind.ends_with("_cancel_sync_finalize") {
                        // The follow-up execution uses its own trace request id.
                        // The task row retains the original create request id, so
                        // cancellation settlement must be driven from the task.
                        // A delete of an already completed/failed task must not
                        // rewrite that task's settled generation usage as cancelled.
                        finalize_video_task_if_terminal(state, &stored).await;
                    }
                }
            }
            if let Some(success_report_kind) = background_success_report_kind {
                payload.report_kind = success_report_kind.to_string();
            }
            if background_success_report_kind.is_some() {
                spawn_sync_report_after_terminal_usage(
                    state.clone(),
                    payload,
                    usage_handoff_persisted,
                );
            } else {
                warn!(
                    event_name = "local_video_finalize_missing_success_report_mapping",
                    log_type = "ops",
                    trace_id = %trace_id,
                    request_id = %request_id_for_log,
                    candidate_id = ?candidate_id,
                    report_kind = %payload.report_kind,
                    "gateway local video finalize produced response without background success report mapping"
                );
            }
            return Ok(Some(attach_control_metadata_headers(
                response,
                request_id,
                candidate_id,
            )?));
        }
        if let Some(response) =
            maybe_build_local_video_error_response(trace_id, decision, &payload)?
        {
            let background_error_report_kind =
                resolve_local_sync_error_background_report_kind(payload.report_kind.as_str());
            if let Some(error_report_kind) = background_error_report_kind {
                payload.report_kind = error_report_kind.to_string();
            }
            let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
                state,
                &plan,
                payload.report_context.as_ref(),
                &payload,
                candidate_started_unix_secs,
                candidate_started_at,
                candidate_first_byte_elapsed_ms,
                &mut terminal_guard,
            )
            .await;
            apply_sync_failure_effects(
                state,
                &plan,
                payload.report_context.as_ref(),
                payload.status_code,
                &payload.headers,
                local_failover_response_text.as_deref(),
                local_failover_analysis,
            )
            .await;
            if background_error_report_kind.is_some() {
                spawn_sync_report_after_terminal_usage(
                    state.clone(),
                    payload,
                    usage_handoff_persisted,
                );
            } else {
                warn!(
                    event_name = "local_video_finalize_missing_error_report_mapping",
                    log_type = "ops",
                    trace_id = %trace_id,
                    request_id = %request_id_for_log,
                    candidate_id = ?candidate_id,
                    report_kind = %payload.report_kind,
                    "gateway local video finalize produced response without background error report mapping"
                );
            }
            return Ok(Some(attach_control_metadata_headers(
                response,
                request_id,
                candidate_id,
            )?));
        }
        let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
            state,
            &plan,
            payload.report_context.as_ref(),
            &payload,
            candidate_started_unix_secs,
            candidate_started_at,
            candidate_first_byte_elapsed_ms,
        &mut terminal_guard,
    )
    .await;
        if payload.status_code >= 400 {
            apply_sync_failure_effects(
                state,
                &plan,
                payload.report_context.as_ref(),
                payload.status_code,
                &payload.headers,
                local_failover_response_text.as_deref(),
                local_failover_analysis,
            )
            .await;
        }
        let response = submit_local_core_error_or_sync_finalize_after_terminal_usage(
            state,
            trace_id,
            decision,
            payload,
            usage_handoff_persisted,
        )
        .await?;
        return Ok(Some(attach_control_metadata_headers(
            response,
            request_id,
            candidate_id,
        )?));
    }

    let usage_payload = build_sync_report_payload(
        trace_id,
        report_kind.unwrap_or_default(),
        report_context,
        status_code,
        client_headers,
        body_json,
        body_base64,
        telemetry,
    );
    let usage_handoff_persisted = record_sync_terminal_usage_and_disarm_guard(
        state,
        &plan,
        usage_payload.report_context.as_ref(),
        &usage_payload,
        candidate_started_unix_secs,
        candidate_started_at,
        candidate_first_byte_elapsed_ms,
        &mut terminal_guard,
    )
    .await;
    if status_code < 400 {
        apply_sync_success_effects(
            state,
            &plan,
            usage_payload.report_context.as_ref(),
            &usage_payload,
        )
        .await;
    } else {
        apply_sync_failure_effects(
            state,
            &plan,
            usage_payload.report_context.as_ref(),
            status_code,
            &usage_payload.headers,
            local_failover_response_text.as_deref(),
            local_failover_analysis,
        )
        .await;
    }
    let response = attach_control_metadata_headers(
        build_client_response_from_parts(
            status_code,
            &usage_payload.headers,
            Body::from(body_bytes),
            trace_id,
            Some(decision),
        )?,
        request_id,
        candidate_id,
    )?;
    if !usage_payload.report_kind.trim().is_empty() {
        if status_code >= 400 {
            let report_kind = usage_payload.report_kind.clone();
            if let Err(err) = submit_sync_report_after_terminal_usage(
                state,
                usage_payload,
                usage_handoff_persisted,
            )
            .await
            {
                warn!(
                    event_name = "local_sync_error_report_submit_failed",
                    log_type = "ops",
                    trace_id = %trace_id,
                    report_kind = %report_kind,
                    "gateway failed to submit local sync error report before returning response: {err:?}"
                );
            }
        } else {
            spawn_sync_report_after_terminal_usage(
                state.clone(),
                usage_payload,
                usage_handoff_persisted,
            );
        }
    }

    Ok(Some(response))
    })
    .await;
    match result.as_ref() {
        Err(error) => terminal_guard.fail_and_disarm(error).await,
        Ok(Some(_)) => terminal_guard.disarm(),
        Ok(None) => {
            // `None` is the candidate-loop retry signal.  All candidates for
            // one request share the same usage row, so finalizing this attempt
            // as Failed would freeze the request-level row and prevent a later
            // candidate from recording its tokens.  The final exhaustion path
            // owns the request-level terminal event; keep this attempt's
            // pending row open until then.
            //
            // If the intermediate Failed candidate handoff itself was not
            // accepted by the writer, however, the RETRY_ABORT reservation is
            // still live.  Keep this guard armed so its Drop path can promote
            // that reservation to a detached void-499 handoff if the outer
            // candidate loop is cancelled before another owner takes over.
            let retry_handoff_incomplete = terminal_guard.watchdog_progress.retry_abort()
                && !terminal_guard.watchdog_progress.retry_owner_external()
                && !terminal_guard.watchdog_progress.retry_handoff_complete();
            if !retry_handoff_incomplete {
                terminal_guard.disarm();
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)] // mirrors sync execution context
fn maybe_build_implicit_sync_finalize_outcome(
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan_kind: &str,
    report_context: &Option<serde_json::Value>,
    status_code: u16,
    headers: &BTreeMap<String, String>,
    body_json: &Option<serde_json::Value>,
    body_base64: &Option<String>,
    telemetry: &Option<ExecutionTelemetry>,
) -> Result<Option<ImplicitSyncFinalizeOutcome>, GatewayError> {
    if status_code >= 400 || body_json.is_some() || body_base64.is_none() {
        return Ok(None);
    }

    let Some(report_kind) = implicit_sync_finalize_report_kind(plan_kind) else {
        return Ok(None);
    };

    let payload = GatewaySyncReportRequest {
        trace_id: trace_id.to_string(),
        report_kind: report_kind.to_string(),
        report_context: report_context.clone(),
        status_code,
        headers: headers.clone(),
        body_json: body_json.clone(),
        client_body_json: None,
        body_base64: body_base64.clone(),
        telemetry: telemetry.clone(),
    };
    let Some(outcome) = maybe_build_sync_finalize_outcome(trace_id, decision, &payload)? else {
        return Ok(None);
    };

    Ok(Some(ImplicitSyncFinalizeOutcome { payload, outcome }))
}

#[allow(clippy::too_many_arguments)] // internal helper mirroring execute path context
#[cfg(test)]
async fn execute_sync_via_remote_execution_runtime(
    state: &AppState,
    remote_execution_runtime_base_url: &str,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan: &ExecutionPlan,
    plan_request_id: &str,
    plan_candidate_id: Option<&str>,
    report_context: Option<&serde_json::Value>,
    candidate_started_unix_secs: u64,
    candidate_started_at: Instant,
    has_candidate_retry_scope: bool,
    watchdog_progress: Option<Arc<StreamCandidateWatchdogProgress>>,
) -> Result<RemoteSyncFallbackOutcome, GatewayError> {
    let remote_request_started_at_unix_ms = current_request_candidate_unix_ms();
    let remote_request_order_id = uuid::Uuid::now_v7().to_string();
    let response = match post_sync_plan_to_remote_execution_runtime(
        state,
        remote_execution_runtime_base_url,
        Some(trace_id),
        plan,
    )
    .await
    {
        Ok(response) => response,
        Err(err) => {
            warn!(
                event_name = "sync_execution_runtime_remote_unavailable",
                log_type = "ops",
                trace_id = %trace_id,
                request_id = %short_request_id(plan_request_id),
                candidate_id = ?plan_candidate_id,
                error = ?err,
                "gateway remote execution runtime sync unavailable"
            );
            let transport_error_message = format!("{err:?}");
            if let Some(response) = maybe_build_sync_transport_error_stop_response(
                state,
                plan,
                report_context,
                trace_id,
                decision,
                "execution_runtime_unavailable",
                transport_error_message.as_str(),
                elapsed_ms_since(candidate_started_at),
            )
            .await?
            {
                return Ok(RemoteSyncFallbackOutcome::ClientResponse(response));
            }
            if !has_candidate_retry_scope {
                let response = crate::execution_runtime::build_transport_error_stop_response(
                    state,
                    plan,
                    report_context,
                    trace_id,
                    decision,
                    StatusCode::BAD_GATEWAY.as_u16(),
                    "execution_runtime_unavailable",
                    transport_error_message.as_str(),
                    elapsed_ms_since(candidate_started_at),
                )
                .await?;
                return Ok(RemoteSyncFallbackOutcome::ClientResponse(response));
            }
            if !watchdog_progress
                .as_ref()
                .is_none_or(|progress| progress.try_mark_retry_abort())
            {
                // A concurrent terminal owner already won.  Do not publish a
                // late Failed candidate from this compatibility-runtime
                // error; let that owner finish the request lifecycle.
                return Ok(RemoteSyncFallbackOutcome::Unavailable);
            }
            let terminal_unix_secs = current_request_candidate_unix_ms();
            record_local_request_candidate_status(
                state,
                plan,
                report_context,
                SchedulerRequestCandidateStatusUpdate {
                    status: RequestCandidateStatus::Failed,
                    status_code: None,
                    error_type: Some("execution_runtime_unavailable".to_string()),
                    error_message: Some(transport_error_message),
                    latency_ms: Some(elapsed_ms_since(candidate_started_at)),
                    started_at_unix_ms: Some(candidate_started_unix_secs),
                    finished_at_unix_ms: Some(terminal_unix_secs),
                },
            )
            .await;
            return Ok(RemoteSyncFallbackOutcome::Unavailable);
        }
    };

    if response.status() != http::StatusCode::OK {
        let status_code = response.status().as_u16();
        let error_message = format!("execution runtime returned HTTP {status_code}");
        // A non-OK response from the compatibility runtime is already the
        // terminal response for this attempt (the body is forwarded below),
        // so close the request-level usage lifecycle before exposing a Failed
        // candidate.  The previous path only wrote the candidate and left the
        // pending usage skeleton without a terminal billing event.
        record_sync_attempt_forced_terminal_state(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            current_request_diagnostics(),
            watchdog_progress,
            candidate_started_unix_secs,
            candidate_started_at,
            UsageEventType::Failed,
            RequestCandidateStatus::Failed,
            status_code,
            "execution_runtime_http_error",
            error_message.clone(),
        )
        .await;
        return Ok(RemoteSyncFallbackOutcome::ClientResponse(
            attach_control_metadata_headers(
                build_client_response(response, trace_id, Some(decision))?,
                Some(plan_request_id),
                plan_candidate_id,
            )?,
        ));
    }

    let remote_response_observed_at_unix_ms = current_request_candidate_unix_ms();
    let mut result = response
        .json::<ExecutionResult>()
        .await
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    result
        .response_observation
        .get_or_insert(ExecutionResponseObservation {
            request_started_at_unix_ms: remote_request_started_at_unix_ms,
            response_headers_observed_at_unix_ms: remote_response_observed_at_unix_ms,
            request_order_id: remote_request_order_id,
        });
    Ok(RemoteSyncFallbackOutcome::Executed(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aether_data::repository::candidates::InMemoryRequestCandidateRepository;
    use aether_data::repository::usage::InMemoryUsageReadRepository;
    use aether_data_contracts::repository::candidates::RequestCandidateReadRepository;
    use aether_data_contracts::repository::usage::UsageReadRepository;
    use aether_usage_runtime::UsageRuntimeConfig;
    use futures_util::{pin_mut, StreamExt as _};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn test_openai_image_plan(stream: bool) -> ExecutionPlan {
        ExecutionPlan {
            request_id: "req-1".to_string(),
            candidate_id: Some("candidate-1".to_string()),
            provider_name: Some("OpenAI".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            method: "POST".to_string(),
            url: "https://chatgpt.com/backend-api/codex/responses".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: aether_contracts::RequestBody::from_json(json!({"stream": true})),
            stream,
            client_api_format: "openai:image".to_string(),
            provider_api_format: "openai:image".to_string(),
            model_name: Some("gpt-image-2".to_string()),
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    fn test_gemini_chat_plan() -> ExecutionPlan {
        let mut plan = test_openai_image_plan(false);
        plan.client_api_format = "openai:chat".to_string();
        plan.provider_api_format = "gemini:generate_content".to_string();
        plan.model_name = Some("gemini-3-flash-preview".to_string());
        plan
    }

    fn test_decision() -> GatewayControlDecision {
        GatewayControlDecision::synthetic(
            "/v1/chat/completions",
            Some("ai_public".to_string()),
            Some("openai".to_string()),
            Some("chat".to_string()),
            Some("openai:chat".to_string()),
        )
        .with_execution_runtime_candidate(true)
    }

    #[tokio::test]
    async fn oversized_upstream_response_builds_claude_502_retry_fallback() {
        let mut plan = test_openai_image_plan(false);
        plan.client_api_format = "claude:messages".to_string();
        plan.provider_api_format = "claude:messages".to_string();
        let decision = GatewayControlDecision::synthetic(
            "/v1/messages",
            Some("ai_public".to_string()),
            Some("claude".to_string()),
            Some("messages".to_string()),
            Some("claude:messages".to_string()),
        )
        .with_execution_runtime_candidate(true);
        let failure = SyncExecutionFailure::from_transport(
            ExecutionRuntimeTransportError::UpstreamResponseTooLarge {
                phase: crate::execution_runtime::transport::UpstreamResponseBodyPhase::Wire,
                limit_bytes: 8,
            },
        );

        assert_eq!(failure.status_code, Some(StatusCode::BAD_GATEWAY.as_u16()));
        assert_eq!(
            failure.fallback_kind,
            Some(SyncExecutionFailureFallbackKind::UpstreamResponseTooLarge)
        );
        let mut retry_scope = AiAttemptRetryScope::Provider;
        let mut retry_fallback = None;
        {
            let mut retry_scope_out = Some(&mut retry_scope);
            let mut retry_fallback_out = Some(&mut retry_fallback);
            maybe_store_sync_execution_failure_fallback(
                &failure,
                &plan,
                "trace-too-large",
                &decision,
                &mut retry_scope_out,
                &mut retry_fallback_out,
            )
            .expect("fallback response should build");
        }

        assert_eq!(retry_scope, AiAttemptRetryScope::Candidate);
        let response = retry_fallback.expect("oversized response should provide a fallback");

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), 1024)
            .await
            .expect("fallback body should read");
        let body: Value = serde_json::from_slice(&body).expect("fallback body should be json");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "upstream_error");
        assert_eq!(body["error"]["message"], "Upstream response too large");
    }

    fn test_kiro_sync_plan() -> ExecutionPlan {
        ExecutionPlan {
            request_id: "req-kiro-sync-cache-1".to_string(),
            candidate_id: Some("candidate-kiro-sync-cache-1".to_string()),
            provider_name: Some("Kiro".to_string()),
            provider_id: "provider-kiro-sync-1".to_string(),
            endpoint_id: "endpoint-kiro-sync-1".to_string(),
            key_id: "key-kiro-sync-1".to_string(),
            method: "POST".to_string(),
            url: "https://kiro.example/generateAssistantResponse".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: aether_contracts::RequestBody::from_json(json!({
                "model": "claude-sonnet-4",
                "messages": [{"role": "user", "content": "hello kiro"}],
            })),
            stream: false,
            client_api_format: "claude:messages".to_string(),
            provider_api_format: "claude:messages".to_string(),
            model_name: Some("claude-sonnet-4".to_string()),
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    fn test_kiro_sync_cacheable_request_body() -> serde_json::Value {
        json!({
            "model": "claude-sonnet-4",
            "system": [{
                "type": "text",
                "text": format!("sync cacheable prompt {}", "cacheable prompt chunk ".repeat(300)),
                "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": "reuse this Kiro prompt"}]
        })
    }

    #[test]
    fn invalid_gemini_provider_success_uses_plan_format_when_context_is_missing() {
        let plan = test_gemini_chat_plan();
        let body = json!({
            "candidates": [{
                "content": {"role": "model"},
                "finishReason": "MAX_TOKENS"
            }],
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 25,
                "totalTokenCount": 34
            }
        });

        let message = invalid_gemini_provider_success_message(
            &plan,
            None,
            StatusCode::OK.as_u16(),
            Some(&body),
        )
        .expect("empty Gemini 200 response should be rejected from plan format");

        assert!(message.contains("visible model output"));
    }

    #[test]
    fn invalid_gemini_provider_success_error_is_retryable_candidate_failure() {
        let error = invalid_gemini_provider_success_execution_error(
            INVALID_GEMINI_PROVIDER_SUCCESS_MESSAGE,
        );

        assert_eq!(error.kind, ExecutionErrorKind::Upstream5xx);
        assert_eq!(error.phase, ExecutionPhase::Finalize);
        assert_eq!(error.upstream_status, Some(StatusCode::OK.as_u16()));
        assert!(error.retryable);
        assert!(error.failover_recommended);
    }

    #[test]
    fn invalid_gemini_provider_success_accepts_antigravity_chunks_with_visible_output() {
        let plan = test_gemini_chat_plan();
        let report_context = json!({
            "has_envelope": true,
            "envelope_name": "antigravity:v1internal",
            "provider_api_format": "gemini:generate_content",
        });
        let body = json!({
            "chunks": [{
                "response": {
                    "responseId": "resp_antigravity_chunks_123",
                    "candidates": [{
                        "content": {
                            "parts": [{"text": "Hello Gemini"}],
                            "role": "model"
                        },
                        "finishReason": "STOP",
                        "index": 0
                    }],
                    "modelVersion": "gemini-3-flash-agent",
                    "usageMetadata": {
                        "promptTokenCount": 2,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 4
                    }
                },
                "traceId": "trace-antigravity-chunks"
            }],
            "metadata": {
                "stream": true,
                "stored_chunks": 1,
                "total_chunks": 1
            }
        });

        let message = invalid_gemini_provider_success_message(
            &plan,
            Some(&report_context),
            StatusCode::OK.as_u16(),
            Some(&body),
        );

        assert!(message.is_none());
    }

    #[test]
    fn invalid_gemini_provider_success_unwraps_gemini_cli_v1internal_envelope() {
        let plan = test_gemini_chat_plan();
        let report_context = json!({
            "has_envelope": true,
            "envelope_name": "gemini_cli:v1internal",
            "provider_api_format": "gemini:generate_content",
        });
        let body = json!({
            "response": {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello from Gemini CLI"}]
                    },
                    "finishReason": "STOP"
                }]
            },
            "remainingCredits": 41,
            "consumedCredits": 1,
            "traceId": "trace-upstream-sync-1"
        });

        let message = invalid_gemini_provider_success_message(
            &plan,
            Some(&report_context),
            StatusCode::OK.as_u16(),
            Some(&body),
        );

        assert!(message.is_none());
    }

    #[test]
    fn sync_retry_claim_reports_terminal_owner_without_retrying() {
        let mut plan = test_openai_image_plan(false);
        plan.request_id = "sync-watchdog-ownership-test".to_string();
        plan.candidate_id = Some("sync-watchdog-ownership-candidate".to_string());

        let retry_progress = StreamCandidateWatchdogProgress::shared();
        crate::execution_runtime::register_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            retry_progress.clone(),
        );
        assert_eq!(
            sync_watchdog_intermediate_ownership(&plan),
            None,
            "an unclaimed watchdog slot should not imply ownership"
        );
        assert!(retry_progress.try_mark_retry_abort());
        assert_eq!(
            sync_watchdog_intermediate_ownership(&plan),
            Some(SyncWatchdogIntermediateOwnership::Retry)
        );
        crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            &retry_progress,
        );

        let terminal_progress = StreamCandidateWatchdogProgress::shared();
        crate::execution_runtime::register_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            terminal_progress.clone(),
        );
        assert!(terminal_progress.mark_terminal_started());
        assert_eq!(
            sync_watchdog_intermediate_ownership(&plan),
            Some(SyncWatchdogIntermediateOwnership::Terminal)
        );
        crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            &terminal_progress,
        );
    }

    #[test]
    fn sync_terminal_guard_reuses_outer_watchdog_registration() {
        let state = AppState::new().expect("gateway state should build");
        let mut plan = test_openai_image_plan(false);
        plan.request_id = "sync-watchdog-reuse-test".to_string();
        plan.candidate_id = Some("sync-watchdog-reuse-candidate".to_string());
        let existing = StreamCandidateWatchdogProgress::shared();
        crate::execution_runtime::register_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            Arc::clone(&existing),
        );

        let mut guard = SyncAttemptTerminalGuard::new(
            &state,
            &plan,
            None,
            current_request_candidate_unix_ms(),
            Instant::now(),
        );
        assert!(Arc::ptr_eq(&guard.watchdog_progress, &existing));
        assert!(!guard.owns_watchdog_progress_registration);
        guard.disarm();
        let still_registered = stream_candidate_watchdog_progress_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        )
        .expect("outer watchdog registration must remain owned by caller");
        assert!(Arc::ptr_eq(&still_registered, &existing));
        crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
            &existing,
        );
    }

    #[test]
    fn sync_watchdog_stop_owner_promotes_to_failed_terminal_fallback() {
        let progress = StreamCandidateWatchdogProgress::shared();
        assert!(progress.try_mark_stop_requested());
        assert!(progress.try_claim_stop_fallback());
        assert!(progress.stop_fallback());
        assert!(!progress.try_claim_cancel_fallback());
        assert!(!progress.mark_terminal_started());
    }

    #[tokio::test]
    async fn sync_attempt_terminal_guard_marks_dropped_pending_attempt_cancelled() {
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let request_candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&request_candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(UsageRuntimeConfig {
                enabled: true,
                ..UsageRuntimeConfig::default()
            });
        let mut plan = test_openai_image_plan(false);
        plan.request_id = "sync-cancel-guard-request".to_string();
        plan.candidate_id = None;
        let mut report_context = Some(json!({
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-cancel",
            "api_key_id": "api-key-cancel",
            "client_api_format": "openai:image",
            "provider_api_format": "openai:image",
            "request_path": "/v1/images/generations",
            "request_path_and_query": "/v1/images/generations",
            "upstream_url": "https://example.test/v1/images/generations",
            "mapped_model": "gpt-image-2",
        }));

        ensure_execution_request_candidate_slot(&state, &mut plan, &mut report_context).await;
        let candidate_started_at = Instant::now();
        let started_at = current_request_candidate_unix_ms();
        state.usage_runtime.record_pending(
            state.usage_lifecycle_data_state().as_ref(),
            build_lifecycle_usage_seed(&plan, report_context.as_ref()),
        );
        record_local_request_candidate_status(
            &state,
            &plan,
            report_context.as_ref(),
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Pending,
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(started_at),
                finished_at_unix_ms: None,
            },
        )
        .await;

        crate::request_diagnostics::scope_request_diagnostics(async {
            crate::request_diagnostics::record_request_accepted_at(
                Instant::now() - Duration::from_millis(25),
            );
            let _guard = SyncAttemptTerminalGuard::new(
                &state,
                &plan,
                report_context.clone(),
                started_at,
                candidate_started_at,
            );
        })
        .await;

        let mut stored_usage = None;
        for _ in 0..50 {
            if let Some(usage) = usage_repository
                .find_by_request_id("sync-cancel-guard-request")
                .await
                .expect("usage should read")
            {
                if usage.status == "cancelled" {
                    stored_usage = Some(usage);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let stored_usage = stored_usage.expect("cancelled usage should be recorded");
        assert_eq!(stored_usage.status, "cancelled");
        assert_eq!(stored_usage.billing_status, "void");
        assert_eq!(stored_usage.status_code, Some(499));
        assert_eq!(stored_usage.error_category.as_deref(), Some("cancelled"));
        let request_metadata = stored_usage
            .request_metadata
            .as_ref()
            .expect("cancelled usage should retain request diagnostics");
        assert!(request_metadata
            .get("end_to_end_time_ms")
            .and_then(Value::as_u64)
            .is_some());
        assert!(request_metadata
            .get("end_to_end_first_byte_time_ms")
            .is_none());

        let stored_candidates = request_candidate_repository
            .list_by_request_id("sync-cancel-guard-request")
            .await
            .expect("request candidates should read");
        assert_eq!(stored_candidates.len(), 1);
        assert_eq!(
            stored_candidates[0].status,
            RequestCandidateStatus::Cancelled
        );
        assert_eq!(stored_candidates[0].status_code, Some(499));
        assert_eq!(
            stored_candidates[0].error_type.as_deref(),
            Some("local_sync_attempt_cancelled")
        );
    }

    #[tokio::test]
    async fn sync_attempt_terminal_guard_closes_retry_abort_window_as_cancelled() {
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let request_candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&request_candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(UsageRuntimeConfig {
                enabled: true,
                ..UsageRuntimeConfig::default()
            });
        let mut plan = test_openai_image_plan(false);
        plan.request_id = "sync-retry-abort-cancel-guard-request".to_string();
        plan.candidate_id = Some("sync-retry-abort-cancel-guard-candidate".to_string());
        let report_context = Some(json!({
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-retry-abort",
            "api_key_id": "api-key-retry-abort",
            "candidate_id": "sync-retry-abort-cancel-guard-candidate",
            "client_api_format": "openai:image",
            "provider_api_format": "openai:image",
            "request_path": "/v1/images/generations",
            "request_path_and_query": "/v1/images/generations",
            "upstream_url": "https://example.test/v1/images/generations",
            "mapped_model": "gpt-image-2",
        }));
        let candidate_started_at = Instant::now();
        let started_at = current_request_candidate_unix_ms();
        state
            .usage_runtime
            .record_pending_direct(
                state.usage_lifecycle_data_state().as_ref(),
                build_lifecycle_usage_seed(&plan, report_context.as_ref()),
            )
            .await;
        record_local_request_candidate_status(
            &state,
            &plan,
            report_context.as_ref(),
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Pending,
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(started_at),
                finished_at_unix_ms: None,
            },
        )
        .await;

        {
            let guard = SyncAttemptTerminalGuard::new(
                &state,
                &plan,
                report_context.clone(),
                started_at,
                candidate_started_at,
            );
            // Model the failover owner having claimed RETRY_ABORT immediately
            // before its intermediate Failed candidate write.  Dropping the
            // owner in this window must still settle the request as a void 499.
            assert!(guard.watchdog_progress.try_mark_retry_abort());
        }

        let mut stored_usage = None;
        for _ in 0..100 {
            if let Some(usage) = usage_repository
                .find_by_request_id("sync-retry-abort-cancel-guard-request")
                .await
                .expect("usage should read")
            {
                if usage.status == "cancelled" {
                    stored_usage = Some(usage);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let stored_usage = stored_usage.expect("retry-abort cancellation should settle usage");
        assert_eq!(stored_usage.status, "cancelled");
        assert_eq!(stored_usage.billing_status, "void");
        assert_eq!(stored_usage.status_code, Some(499));

        let stored_candidates = request_candidate_repository
            .list_by_request_id("sync-retry-abort-cancel-guard-request")
            .await
            .expect("request candidates should read");
        assert_eq!(stored_candidates.len(), 1);
        assert_eq!(
            stored_candidates[0].status,
            RequestCandidateStatus::Cancelled
        );
        assert_eq!(stored_candidates[0].status_code, Some(499));
    }

    #[tokio::test]
    async fn sync_direct_response_start_marks_usage_and_candidate_active_before_body_finishes() {
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let request_candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&request_candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(UsageRuntimeConfig {
                enabled: true,
                ..UsageRuntimeConfig::default()
            });

        let listener = crate::test_support::bind_loopback_listener()
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("local addr should resolve");
        let (headers_tx, headers_rx) = tokio::sync::oneshot::channel();
        let (body_tx, body_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("client should connect");
            let mut request = [0_u8; 4096];
            let _ = socket
                .read(&mut request)
                .await
                .expect("request should read");
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 11\r\n\r\n",
                )
                .await
                .expect("headers should write");
            socket.flush().await.expect("headers should flush");
            let _ = headers_tx.send(());
            let _ = body_rx.await;
            socket
                .write_all(br#"{"ok":true}"#)
                .await
                .expect("body should write");
        });

        let mut plan = test_gemini_chat_plan();
        plan.request_id = "sync-response-start-active-request".to_string();
        plan.candidate_id = Some("sync-response-start-active-candidate".to_string());
        plan.url = format!("http://{addr}/chat");
        plan.provider_api_format = "openai:chat".to_string();
        plan.model_name = Some("gpt-5".to_string());
        plan.body = aether_contracts::RequestBody::from_json(json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "slow body"}],
        }));
        let report_context = Some(json!({
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-active",
            "api_key_id": "api-key-active",
            "client_api_format": "openai:chat",
            "provider_api_format": "openai:chat",
            "request_path": "/v1/chat/completions",
            "request_path_and_query": "/v1/chat/completions",
            "upstream_url": plan.url.clone(),
            "mapped_model": "gpt-5",
        }));
        let started_at = current_request_candidate_unix_ms();
        state
            .usage_runtime
            .record_pending_direct(
                state.usage_lifecycle_data_state().as_ref(),
                build_lifecycle_usage_seed(&plan, report_context.as_ref()),
            )
            .await;
        record_local_request_candidate_status(
            &state,
            &plan,
            report_context.as_ref(),
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Pending,
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(started_at),
                finished_at_unix_ms: None,
            },
        )
        .await;

        let state_for_exec = state.clone();
        let plan_for_exec = plan.clone();
        let report_context_for_exec = report_context.clone();
        let exec = tokio::spawn(async move {
            execute_direct_sync_runtime_candidate(
                &state_for_exec,
                &plan_for_exec,
                report_context_for_exec.as_ref(),
                "trace-response-start-active",
                "openai_chat_sync",
                started_at,
                "sync-response-start-active-request",
                plan_for_exec.candidate_id.as_deref(),
                "openai",
                "endpoint-1",
                "key-1",
                "gpt-5",
                "0",
                None,
            )
            .await
        });

        headers_rx.await.expect("headers should be written");
        let mut active_usage = None;
        for _ in 0..50 {
            if let Some(usage) = usage_repository
                .find_by_request_id("sync-response-start-active-request")
                .await
                .expect("usage should read")
            {
                if usage.status == "streaming" {
                    active_usage = Some(usage);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let active_usage = active_usage.expect("usage should become active before body finishes");
        assert_eq!(active_usage.status_code, Some(200));
        assert!(active_usage.first_byte_time_ms.is_some());
        assert!(active_usage.response_time_ms.is_some());

        let stored_candidates = request_candidate_repository
            .list_by_request_id("sync-response-start-active-request")
            .await
            .expect("candidate should read");
        let active_candidate = stored_candidates
            .iter()
            .find(|candidate| candidate.id == "sync-response-start-active-candidate")
            .expect("candidate should exist");
        assert_eq!(active_candidate.status, RequestCandidateStatus::Streaming);
        assert_eq!(active_candidate.status_code, Some(200));

        let _ = body_tx.send(());
        let result = tokio::time::timeout(Duration::from_secs(2), exec)
            .await
            .expect("sync execution should finish")
            .expect("sync execution task should not panic")
            .expect("sync execution should succeed");
        assert_eq!(result.status_code, 200);
        server.abort();
    }

    #[tokio::test]
    async fn sync_execution_active_marks_usage_before_response_headers() {
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let request_candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                crate::data::GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&request_candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(UsageRuntimeConfig {
                enabled: true,
                ..UsageRuntimeConfig::default()
            });

        let listener = crate::test_support::bind_loopback_listener()
            .await
            .expect("listener should bind");
        let addr = listener.local_addr().expect("local addr should resolve");
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("client should connect");
            let mut request = [0_u8; 4096];
            let _ = socket
                .read(&mut request)
                .await
                .expect("request should read");
            let _ = request_seen_tx.send(());
            let _ = finish_rx.await;
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 11\r\n\r\n{\"ok\":true}",
                )
                .await
                .expect("response should write");
        });

        let mut plan = test_gemini_chat_plan();
        plan.request_id = "sync-active-before-headers-request".to_string();
        plan.candidate_id = Some("sync-active-before-headers-candidate".to_string());
        plan.url = format!("http://{addr}/chat");
        plan.provider_name = Some("OpenAI".to_string());
        plan.provider_api_format = "openai:chat".to_string();
        plan.model_name = Some("gpt-5".to_string());
        plan.body = aether_contracts::RequestBody::from_json(json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "slow headers"}],
        }));
        let report_context = Some(json!({
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-active-before-headers",
            "api_key_id": "api-key-active-before-headers",
            "candidate_id": "sync-active-before-headers-candidate",
            "provider_id": "provider-1",
            "endpoint_id": "endpoint-1",
            "key_id": "key-1",
            "provider_name": "OpenAI",
            "client_api_format": "openai:chat",
            "provider_api_format": "openai:chat",
            "request_path": "/v1/chat/completions",
            "request_path_and_query": "/v1/chat/completions",
            "upstream_url": plan.url.clone(),
            "mapped_model": "gpt-5",
        }));
        let state_for_exec = state.clone();
        let plan_for_exec = plan.clone();
        let report_context_for_exec = report_context.clone();
        let exec = tokio::spawn(async move {
            execute_execution_runtime_sync(
                &state_for_exec,
                "/v1/chat/completions",
                plan_for_exec,
                "trace-active-before-headers",
                &test_decision(),
                "openai_chat_sync",
                Some("openai_chat_sync".to_string()),
                report_context_for_exec,
            )
            .await
        });

        request_seen_rx
            .await
            .expect("upstream request should be observed");
        let mut active_usage = None;
        for _ in 0..50 {
            if let Some(usage) = usage_repository
                .find_by_request_id("sync-active-before-headers-request")
                .await
                .expect("usage should read")
            {
                if usage.status == "streaming" {
                    active_usage = Some(usage);
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let active_usage =
            active_usage.expect("usage should become active before upstream headers");
        assert_eq!(active_usage.status_code, None);
        assert_eq!(active_usage.first_byte_time_ms, None);
        assert_eq!(active_usage.response_time_ms, None);

        let stored_candidates = request_candidate_repository
            .list_by_request_id("sync-active-before-headers-request")
            .await
            .expect("candidate should read");
        let active_candidate = stored_candidates
            .iter()
            .find(|candidate| candidate.id == "sync-active-before-headers-candidate")
            .expect("candidate should exist");
        assert_eq!(active_candidate.status, RequestCandidateStatus::Streaming);
        assert_eq!(active_candidate.status_code, None);
        assert!(active_candidate.started_at_unix_ms.is_some());
        assert!(active_candidate.finished_at_unix_ms.is_none());

        let _ = finish_tx.send(());
        let response = tokio::time::timeout(Duration::from_secs(2), exec)
            .await
            .expect("sync execution should finish")
            .expect("sync execution task should not panic")
            .expect("sync execution should succeed")
            .expect("sync execution should produce a response");
        assert_eq!(response.status(), StatusCode::OK);
        server.abort();
    }

    #[test]
    fn kiro_sync_report_context_seeds_input_tokens_from_original_request_body() {
        let plan = test_kiro_sync_plan();
        let mut report_context = Some(json!({
            "original_request_body": test_kiro_sync_cacheable_request_body(),
        }));

        seed_kiro_sync_report_context_input_tokens(&plan, &mut report_context);

        assert!(report_context
            .as_ref()
            .and_then(|value| value.get("input_tokens"))
            .and_then(Value::as_u64)
            .is_some_and(|tokens| tokens > 0));
    }

    #[tokio::test]
    async fn kiro_sync_report_context_applies_prompt_cache_usage_from_tracker() {
        let state = AppState::new().expect("gateway state should build");
        let plan = test_kiro_sync_plan();

        let mut first_report_context = Some(json!({
            "original_request_body": test_kiro_sync_cacheable_request_body(),
            "kiro_simulated_cache_enabled": true,
        }));
        seed_kiro_sync_report_context_input_tokens(&plan, &mut first_report_context);
        seed_kiro_sync_report_context_prompt_cache_usage(&state, &plan, &mut first_report_context)
            .await;
        let first_creation = first_report_context
            .as_ref()
            .and_then(|value| value.get("cache_creation_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let first_read = first_report_context
            .as_ref()
            .and_then(|value| value.get("cache_read_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        assert!(first_creation > 0);
        assert_eq!(first_read, 0);

        let mut second_report_context = Some(json!({
            "original_request_body": test_kiro_sync_cacheable_request_body(),
            "kiro_simulated_cache_enabled": true,
        }));
        seed_kiro_sync_report_context_input_tokens(&plan, &mut second_report_context);
        seed_kiro_sync_report_context_prompt_cache_usage(&state, &plan, &mut second_report_context)
            .await;
        let second_creation = second_report_context
            .as_ref()
            .and_then(|value| value.get("cache_creation_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        let second_read = second_report_context
            .as_ref()
            .and_then(|value| value.get("cache_read_input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or_default();
        assert_eq!(second_creation, 0);
        assert!(second_read > 0);
    }

    #[tokio::test]
    async fn json_whitespace_heartbeat_stream_prefixes_final_json() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, IoError>>(1);
        tx.send(Ok(Bytes::from_static(br#"{"data":[]}"#)))
            .await
            .expect("final body should send");
        drop(tx);

        let body = to_bytes(
            Body::from_stream(build_json_whitespace_heartbeat_stream(
                rx,
                Duration::from_secs(60),
                None,
            )),
            usize::MAX,
        )
        .await
        .expect("body should collect");

        assert!(body.starts_with(b"\n"));
        let parsed: Value =
            serde_json::from_slice(&body).expect("leading whitespace is valid JSON");
        assert_eq!(parsed, json!({"data": []}));
    }

    #[tokio::test]
    async fn json_whitespace_heartbeat_stream_emits_interval_whitespace() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, IoError>>(1);
        let stream = build_json_whitespace_heartbeat_stream(rx, Duration::from_millis(5), None);
        pin_mut!(stream);

        let first = stream
            .next()
            .await
            .expect("initial whitespace")
            .expect("initial whitespace ok");
        assert_eq!(first, Bytes::from_static(b"\n"));

        let second = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .expect("interval heartbeat")
            .expect("interval heartbeat item")
            .expect("interval heartbeat ok");
        assert_eq!(second, Bytes::from_static(b"\n"));

        tx.send(Ok(Bytes::from_static(br#"{"data":[{"b64_json":"x"}]}"#)))
            .await
            .expect("final body should send");
        let final_body = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .expect("final body")
            .expect("final body item")
            .expect("final body ok");
        assert_eq!(
            serde_json::from_slice::<Value>(&final_body).expect("final body json"),
            json!({"data": [{"b64_json": "x"}]})
        );
    }

    #[test]
    fn openai_image_sync_sse_parser_tracks_partial_and_completed_frames() {
        let partial = parse_openai_image_sync_sse_frame(
            concat!(
                "event: response.image_generation_call.partial_image\n",
                "data: {\"type\":\"response.image_generation_call.partial_image\",\"partial_image_index\":0}\n\n"
            )
            .as_bytes(),
        )
        .expect("partial frame");
        assert_eq!(
            partial.event_name,
            "response.image_generation_call.partial_image"
        );
        assert!(partial.is_partial_image);
        assert_eq!(
            partial.client_visible_event,
            Some("image_generation.partial_image")
        );

        let completed = parse_openai_image_sync_sse_frame(
            b"data: {\"type\":\"response.completed\",\"response\":{}}\n\n",
        )
        .expect("completed frame");
        assert_eq!(completed.event_name, "response.completed");
        assert!(completed.is_completed);
        assert_eq!(
            completed.client_visible_event,
            Some("image_generation.completed")
        );
    }

    #[test]
    fn openai_image_sync_progress_tracks_upstream_stream_without_json_heartbeat_wrapper() {
        let plan = test_openai_image_plan(false);
        let report_context = json!({"upstream_is_stream": true});

        assert!(should_track_openai_image_sync_upstream_sse(
            OPENAI_IMAGE_SYNC_PLAN_KIND,
            &plan,
            Some(&report_context),
        ));
        assert!(!should_enable_openai_image_sync_json_heartbeat(
            OPENAI_IMAGE_SYNC_PLAN_KIND,
            &plan,
            Some(&report_context),
        ));
    }

    #[test]
    fn openai_image_sync_progress_ignores_non_stream_upstream() {
        let plan = test_openai_image_plan(false);
        let report_context = json!({"upstream_is_stream": false});

        assert!(!should_track_openai_image_sync_upstream_sse(
            OPENAI_IMAGE_SYNC_PLAN_KIND,
            &plan,
            Some(&report_context),
        ));
        assert!(!should_enable_openai_image_sync_json_heartbeat(
            OPENAI_IMAGE_SYNC_PLAN_KIND,
            &plan,
            Some(&report_context),
        ));
    }

    #[test]
    fn openai_image_sync_progress_never_publishes_terminal_candidate_status() {
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Success),
            RequestCandidateStatus::Streaming
        );
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Failed),
            RequestCandidateStatus::Streaming
        );
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Cancelled),
            RequestCandidateStatus::Streaming
        );
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Skipped),
            RequestCandidateStatus::Streaming
        );
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Pending),
            RequestCandidateStatus::Pending
        );
        assert_eq!(
            non_terminal_image_progress_status(RequestCandidateStatus::Streaming),
            RequestCandidateStatus::Streaming
        );
    }

    #[test]
    fn sync_candidate_status_distinguishes_cancel_command_from_cancelled_request() {
        assert_eq!(
            sync_terminal_candidate_status(StatusCode::OK.as_u16()),
            RequestCandidateStatus::Success
        );
        assert_eq!(
            sync_terminal_candidate_status(499),
            RequestCandidateStatus::Cancelled
        );
        assert_eq!(
            sync_terminal_candidate_status(StatusCode::BAD_GATEWAY.as_u16()),
            RequestCandidateStatus::Failed
        );
    }
}
