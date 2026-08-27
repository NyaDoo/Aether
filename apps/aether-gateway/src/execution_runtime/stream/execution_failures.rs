use aether_ai_serving::AiAttemptRetryScope;
use aether_contracts::{
    ExecutionError, ExecutionErrorKind, ExecutionPhase, ExecutionPlan, ExecutionTelemetry,
};
use aether_data_contracts::repository::candidates::RequestCandidateStatus;
use aether_scheduler_core::SchedulerRequestCandidateStatusUpdate;
use aether_usage_runtime::{
    build_sync_terminal_usage_payload_seed, build_terminal_usage_context_seed,
};
use axum::body::Body;
use axum::http::Response;
use base64::Engine as _;
use serde::Serialize;
use serde_json::{Map, Value};
use tracing::warn;

use crate::api::response::attach_control_metadata_headers;
use crate::clock::current_unix_ms as current_request_candidate_unix_ms;
use crate::control::GatewayControlDecision;
use crate::execution_runtime::ai_attempt_retry_scope_from_failure_disposition;
use crate::execution_runtime::submission::{
    resolve_core_error_background_report_kind,
    submit_local_core_error_or_sync_finalize_after_terminal_usage,
};
use crate::log_ids::short_request_id;
use crate::orchestration::{
    apply_local_execution_effect, apply_local_stream_failure_effects_with_analysis,
    classify_failure_disposition, resolve_local_failover_analysis_for_attempt,
    resolve_local_transport_failover_analysis_for_attempt, with_upstream_response_report_context,
    LocalAdaptiveRateLimitEffect, LocalAttemptFailureEffect, LocalExecutionEffect,
    LocalExecutionEffectContext, LocalFailoverAnalysis, LocalFailoverDecision,
    LocalHealthFailureEffect, LocalOAuthInvalidationEffect, LocalPoolErrorEffect,
    LocalStreamFailureEffect,
};
use crate::request_candidate_runtime::{
    record_local_request_candidate_status, record_report_request_candidate_status,
    spawn_report_candidate_persistence_retry_after_usage_handoff,
    spawn_terminal_candidate_reconciliation,
};
use crate::request_diagnostics::attach_current_request_diagnostics_and_candidate_timing_to_report_context;
use crate::usage::submit_sync_report_after_terminal_usage;
use crate::{usage::GatewaySyncReportRequest, AppState, GatewayError};

#[derive(Debug, Clone)]
pub(super) struct StreamFailureReport {
    pub(super) status_code: u16,
    pub(super) error_type: String,
    pub(super) error_message: String,
    upstream_status_code: Option<u16>,
    transport_error: bool,
    honor_http_failover: bool,
    extra_error_fields: Map<String, Value>,
    provider_body_json: Option<Value>,
}

#[derive(Serialize)]
struct StreamFailureBody<'a> {
    error: StreamFailureBodyFields<'a>,
}

#[derive(Serialize)]
struct StreamFailureBodyFields<'a> {
    #[serde(rename = "type")]
    error_type: &'a str,
    message: &'a str,
    code: u16,
    #[serde(flatten)]
    extra_error_fields: &'a Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamFailureHandling {
    Terminal,
    HonorLocalFailover,
}

impl StreamFailureReport {
    fn into_body_jsons(self) -> (Value, Option<Value>) {
        let Self {
            status_code,
            error_type,
            error_message,
            upstream_status_code: _,
            transport_error: _,
            honor_http_failover: _,
            mut extra_error_fields,
            provider_body_json,
        } = self;
        extra_error_fields.insert("type".to_string(), Value::String(error_type));
        extra_error_fields.insert("message".to_string(), Value::String(error_message));
        extra_error_fields.insert("code".to_string(), Value::from(status_code));
        let normalized_body = Value::Object(Map::from_iter([(
            "error".to_string(),
            Value::Object(extra_error_fields),
        )]));
        match provider_body_json {
            Some(provider_body) if provider_body != normalized_body => {
                (provider_body, Some(normalized_body))
            }
            Some(provider_body) => (provider_body, None),
            None => (normalized_body, None),
        }
    }

    pub(super) fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string(&StreamFailureBody {
            error: StreamFailureBodyFields {
                error_type: self.error_type.as_str(),
                message: self.error_message.as_str(),
                code: self.status_code,
                extra_error_fields: &self.extra_error_fields,
            },
        })
    }
}

pub(super) fn build_stream_failure_report(
    error_type: impl Into<String>,
    error_message: impl Into<String>,
    status_code: u16,
) -> StreamFailureReport {
    let error_type = error_type.into();
    let error_message = error_message.into();
    StreamFailureReport {
        status_code,
        error_type,
        error_message,
        upstream_status_code: Some(status_code),
        transport_error: false,
        honor_http_failover: false,
        extra_error_fields: Map::new(),
        provider_body_json: None,
    }
}

pub(super) fn build_stream_transport_failure_report(
    error_type: impl Into<String>,
    error_message: impl Into<String>,
    status_code: u16,
) -> StreamFailureReport {
    StreamFailureReport {
        status_code,
        error_type: error_type.into(),
        error_message: error_message.into(),
        upstream_status_code: None,
        transport_error: true,
        honor_http_failover: false,
        extra_error_fields: Map::new(),
        provider_body_json: None,
    }
}

pub(super) fn build_stream_failure_from_execution_error(
    error: &ExecutionError,
) -> StreamFailureReport {
    let transport_error = execution_error_is_transport(error);
    let fallback_status_code = if matches!(
        error.kind,
        ExecutionErrorKind::ConnectTimeout
            | ExecutionErrorKind::FirstByteTimeout
            | ExecutionErrorKind::ReadTimeout
    ) {
        504
    } else {
        502
    };
    let status_code = error.upstream_status.unwrap_or(fallback_status_code);
    let error_type = serde_json::to_value(&error.kind)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "internal".to_string());
    let error_message = error.message.trim().to_string();
    let phase = serde_json::to_value(&error.phase).unwrap_or(Value::Null);
    let mut error_object = Map::from_iter([
        ("phase".to_string(), phase),
        ("retryable".to_string(), Value::Bool(error.retryable)),
        (
            "failover_recommended".to_string(),
            Value::Bool(error.failover_recommended),
        ),
    ]);
    if let Some(upstream_status) = error.upstream_status {
        error_object.insert("upstream_status".to_string(), Value::from(upstream_status));
    }

    StreamFailureReport {
        status_code,
        error_type,
        error_message,
        upstream_status_code: error.upstream_status,
        transport_error,
        honor_http_failover: error.upstream_status.is_some(),
        extra_error_fields: error_object,
        provider_body_json: None,
    }
}

pub(super) fn build_stream_failure_from_provider_error_body(
    status_code: u16,
    body_json: &Value,
) -> StreamFailureReport {
    let body_object = body_json.as_object();
    let error_object = body_object
        .and_then(|object| object.get("error"))
        .and_then(Value::as_object);
    let error_type =
        first_non_empty_error_text(error_object, body_object, &["type", "code", "status"])
            .unwrap_or_else(|| "upstream_error".to_string());
    let error_message = first_non_empty_error_text(
        error_object,
        body_object,
        &["message", "detail", "reason", "status", "type", "code"],
    )
    .unwrap_or_else(|| format!("upstream stream returned error status {status_code}"));

    StreamFailureReport {
        status_code,
        error_type,
        error_message,
        upstream_status_code: Some(status_code),
        transport_error: false,
        honor_http_failover: true,
        extra_error_fields: Map::new(),
        provider_body_json: Some(body_json.clone()),
    }
}

fn execution_error_is_transport(error: &ExecutionError) -> bool {
    if error.upstream_status.is_some() {
        return false;
    }
    let explicit_transport_kind = matches!(
        error.kind,
        ExecutionErrorKind::ConnectTimeout
            | ExecutionErrorKind::FirstByteTimeout
            | ExecutionErrorKind::ReadTimeout
            | ExecutionErrorKind::TlsError
            | ExecutionErrorKind::ProxyError
            | ExecutionErrorKind::ProtocolError
    );
    let retryable_internal_transport_phase = matches!(error.kind, ExecutionErrorKind::Internal)
        && (error.retryable || error.failover_recommended)
        && matches!(
            error.phase,
            ExecutionPhase::Connect
                | ExecutionPhase::Handshake
                | ExecutionPhase::Write
                | ExecutionPhase::FirstByte
                | ExecutionPhase::StreamRead
        );
    explicit_transport_kind || retryable_internal_transport_phase
}

fn first_non_empty_error_text(
    error_object: Option<&Map<String, Value>>,
    body_object: Option<&Map<String, Value>>,
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

fn build_stream_failure_sync_payload(
    trace_id: &str,
    report_kind: String,
    report_context: Option<Value>,
    mut headers: std::collections::BTreeMap<String, String>,
    telemetry: Option<ExecutionTelemetry>,
    provider_buffered_body: &[u8],
    failure: StreamFailureReport,
) -> GatewaySyncReportRequest {
    let status_code = failure.status_code;
    let upstream_status_code = failure.upstream_status_code;
    let transport_error = failure.transport_error;
    let (body, client_body) = failure.into_body_jsons();
    headers.retain(|name, _| {
        !name.eq_ignore_ascii_case("content-encoding")
            && !name.eq_ignore_ascii_case("content-length")
            && !name.eq_ignore_ascii_case("content-type")
    });
    headers.insert("content-type".to_string(), "application/json".to_string());
    let report_context = upstream_status_code
        .and_then(|upstream_status_code| {
            with_upstream_response_report_context(
                report_context.as_ref(),
                upstream_status_code,
                Some(&headers),
                Some(&body),
                None,
                None,
            )
        })
        .or(report_context);
    let report_context = report_context.map(|mut context| {
        if let Some(object) = context.as_object_mut() {
            let response_headers = serde_json::to_value(&headers).unwrap_or(Value::Null);
            if upstream_status_code.is_some() {
                object.insert(
                    "provider_response_headers".to_string(),
                    response_headers.clone(),
                );
            }
            object.insert("client_response_headers".to_string(), response_headers);
            if transport_error {
                object.insert("transport_error".to_string(), Value::Bool(true));
            }
        }
        context
    });

    GatewaySyncReportRequest {
        trace_id: trace_id.to_string(),
        report_kind,
        report_context,
        status_code,
        headers,
        body_json: Some(body),
        client_body_json: client_body,
        body_base64: (!provider_buffered_body.is_empty())
            .then(|| base64::engine::general_purpose::STANDARD.encode(provider_buffered_body)),
        telemetry,
    }
}

fn stream_failure_body_field<'a>(
    payload: &'a GatewaySyncReportRequest,
    field: &str,
) -> Option<&'a str> {
    payload
        .client_body_json
        .as_ref()
        .or(payload.body_json.as_ref())
        .and_then(|body_json| body_json.get("error"))
        .and_then(|value| value.get(field))
        .and_then(Value::as_str)
}

async fn record_stream_sync_failure(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    payload: &GatewaySyncReportRequest,
    candidate_status_code: Option<u16>,
    started_at_unix_ms: Option<u64>,
    handling: StreamFailureHandling,
) -> (LocalFailoverAnalysis, bool, bool) {
    let error_type = stream_failure_body_field(payload, "type").unwrap_or("internal");
    let error_message = stream_failure_body_field(payload, "message").unwrap_or_default();
    let error_body = payload
        .body_json
        .as_ref()
        .and_then(|body_json| serde_json::to_string(body_json).ok());
    let failure_analysis = resolve_local_failover_analysis_for_attempt(
        state,
        plan,
        report_context,
        payload.status_code,
        error_body.as_deref(),
    )
    .await;
    let retrying_next_candidate = matches!(
        failure_analysis.decision,
        LocalFailoverDecision::RetryNextCandidate
    );
    // Resolve the owner once and carry the exact Arc through the terminal
    // usage write. A second request-id lookup can miss a concurrent registry
    // entry transition and let a terminal helper bypass the ownership CAS.
    let watchdog_progress =
        crate::execution_runtime::stream_candidate_watchdog_progress_for_current_or_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
    // Only the prefetch path that explicitly honors local failover has an
    // outer candidate owner.  Mid-stream terminal failures still settle the
    // request even when the policy classification happens to say
    // `RetryNextCandidate`; treating those as intermediate here would put
    // proxy/provider effects ahead of the terminal usage handoff.
    let intermediate_retry = matches!(handling, StreamFailureHandling::HonorLocalFailover)
        && retrying_next_candidate
        && crate::execution_runtime::stream_candidate_watchdog_allows_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        )
        && crate::execution_runtime::try_claim_stream_candidate_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
    let mut failure_effect =
        LocalStreamFailureEffect::new(payload.status_code, &payload.headers, error_body.as_deref());
    if matches!(error_type, "first_byte_timeout" | "read_timeout") {
        failure_effect = failure_effect.with_stream_timeout();
    }
    // A retrying candidate has no request-level terminal usage yet, so retain
    // the historical intermediate-effect ordering.  True terminal paths do
    // this only after the durable usage handoff below.
    if intermediate_retry {
        apply_local_stream_failure_effects_with_analysis(
            state,
            LocalExecutionEffectContext {
                plan,
                report_context,
            },
            failure_effect,
            failure_analysis,
        )
        .await;
    }
    let terminal_owner_claimed = !intermediate_retry
        && crate::execution_runtime::mark_stream_candidate_watchdog_terminal_started_with_progress(
            watchdog_progress.as_ref(),
        );
    let usage_handoff_persisted = if intermediate_retry {
        // A retrying candidate intentionally has no terminal usage event yet;
        // its candidate failure is part of failover bookkeeping, not the final
        // attempt settlement.
        true
    } else if terminal_owner_claimed {
        let report_context_with_diagnostics =
            attach_current_request_diagnostics_and_candidate_timing_to_report_context(
                report_context,
                payload
                    .telemetry
                    .as_ref()
                    .and_then(|telemetry| telemetry.elapsed_ms),
                payload
                    .telemetry
                    .as_ref()
                    .and_then(|telemetry| telemetry.ttfb_ms),
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
        persisted
    } else {
        // Another watchdog owner (for example a 499 cancellation fallback)
        // won the terminal race.  This path must not project a failed
        // candidate or effects as if its usage handoff had succeeded.
        false
    };
    if !intermediate_retry && usage_handoff_persisted {
        // Usage is the billing source of truth.  Project provider/key effects
        // only after its terminal handoff has at least been attempted, so a
        // slow effect cannot recreate the old "candidate finished, usage only
        // has first-byte timing" race.
        apply_local_stream_failure_effects_with_analysis(
            state,
            LocalExecutionEffectContext {
                plan,
                report_context,
            },
            failure_effect,
            failure_analysis,
        )
        .await;
    }
    let terminal_unix_secs = current_request_candidate_unix_ms();
    // Keep the intended terminal transition separate from the in-band
    // candidate update.  When the usage runtime cannot confirm its terminal
    // handoff, the candidate must remain streaming in this critical path, but
    // a detached reconciliation task still needs the original Failed update
    // so it can promote the candidate once the usage row becomes durable.
    let reconciliation_update = SchedulerRequestCandidateStatusUpdate {
        status: RequestCandidateStatus::Failed,
        status_code: candidate_status_code,
        error_type: Some(error_type.to_string()),
        error_message: Some(error_message.to_string()),
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms),
        started_at_unix_ms: started_at_unix_ms.or(Some(terminal_unix_secs)),
        finished_at_unix_ms: Some(terminal_unix_secs),
    };
    if !usage_handoff_persisted {
        spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            reconciliation_update.clone(),
        );
    }
    let candidate_status = if usage_handoff_persisted {
        RequestCandidateStatus::Failed
    } else {
        RequestCandidateStatus::Streaming
    };
    let candidate_update = SchedulerRequestCandidateStatusUpdate {
        status: candidate_status,
        status_code: candidate_status_code,
        error_type: if usage_handoff_persisted {
            Some(error_type.to_string())
        } else {
            Some("usage_terminal_handoff_unconfirmed".to_string())
        },
        error_message: if usage_handoff_persisted {
            Some(error_message.to_string())
        } else {
            Some(
                "terminal usage persistence was not confirmed before candidate finalization"
                    .to_string(),
            )
        },
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms),
        started_at_unix_ms: started_at_unix_ms.or(Some(terminal_unix_secs)),
        finished_at_unix_ms: usage_handoff_persisted.then_some(terminal_unix_secs),
    };
    let candidate_persisted = if intermediate_retry {
        // This is deliberately an intermediate candidate failure.  The request-level
        // usage row must remain pending so the next candidate can own the terminal
        // settlement; the report-driven durability gate is only for true request
        // terminal outcomes and would otherwise defer this diagnostic transition until
        // a later request terminal exists.
        record_local_request_candidate_status(state, plan, report_context, candidate_update.clone())
            .await
    } else {
        record_report_request_candidate_status(state, report_context, candidate_update.clone())
            .await
    };
    if terminal_owner_claimed && usage_handoff_persisted && candidate_persisted {
        if let Some(progress) = watchdog_progress {
            // Keep the owner registered through the candidate handoff.  A
            // terminal usage write can finish before this status update; only
            // now is it safe to let a later scope create a new owner.
            crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                plan.request_id.as_str(),
                plan.candidate_id.as_deref(),
                &progress,
            );
        }
    } else if terminal_owner_claimed && usage_handoff_persisted && !candidate_persisted {
        // Usage is terminal but the candidate side channel was not accepted.
        // Keep the same owner reservation while reconciliation retries; a late
        // disconnect must not install a second terminal owner in this gap.
        spawn_report_candidate_persistence_retry_after_usage_handoff(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            reconciliation_update,
            watchdog_progress,
        );
    }
    (
        failure_analysis,
        usage_handoff_persisted,
        intermediate_retry,
    )
}

#[allow(clippy::too_many_arguments)] // internal helper for prefetch error handling
pub(super) async fn handle_prefetch_provider_private_stream_error(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan: &ExecutionPlan,
    report_context: Option<Value>,
    request_id: &str,
    candidate_id: Option<&str>,
    report_kind: &str,
    mut headers: std::collections::BTreeMap<String, String>,
    telemetry: Option<ExecutionTelemetry>,
    buffered_body: &[u8],
    upstream_status_code: u16,
    status_code: u16,
    body_json: Value,
    retry_scope_out: Option<&mut AiAttemptRetryScope>,
    retry_fallback_out: Option<&mut Option<Response<Body>>>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let upstream_headers = headers.clone();
    headers.remove("content-encoding");
    headers.remove("content-length");
    headers.insert("content-type".to_string(), "application/json".to_string());

    let payload = GatewaySyncReportRequest {
        trace_id: trace_id.to_string(),
        report_kind: report_kind.to_string(),
        report_context,
        status_code,
        headers,
        body_json: Some(body_json),
        client_body_json: None,
        body_base64: (!buffered_body.is_empty())
            .then(|| base64::engine::general_purpose::STANDARD.encode(buffered_body)),
        telemetry,
    };
    // A direct/internal prefetch invocation has no outer candidate owner.  It
    // must settle request-level usage even when policy classification says
    // RetryNextCandidate; only the candidate-loop variant may leave usage
    // pending and return `None` for failover.
    let honor_local_failover = retry_scope_out.is_some();
    let (failure_analysis, usage_handoff_persisted, intermediate_retry) =
        record_stream_sync_failure(
            state,
            plan,
            payload.report_context.as_ref(),
            &payload,
            Some(status_code),
            None,
            if honor_local_failover {
                StreamFailureHandling::HonorLocalFailover
            } else {
                StreamFailureHandling::Terminal
            },
        )
        .await;
    if intermediate_retry {
        let failure_disposition = classify_failure_disposition(
            &plan.provider_api_format,
            failure_analysis.classification,
            status_code,
        );
        if let Some(retry_scope) = retry_scope_out {
            *retry_scope = ai_attempt_retry_scope_from_failure_disposition(failure_disposition);
        }
        if failure_disposition.preserve_upstream_error {
            if let Some(retry_fallback) = retry_fallback_out {
                *retry_fallback = Some(attach_control_metadata_headers(
                    crate::api::response::build_client_response_from_parts(
                        upstream_status_code,
                        &upstream_headers,
                        Body::from(buffered_body.to_vec()),
                        trace_id,
                        Some(decision),
                    )?,
                    Some(request_id),
                    candidate_id,
                )?);
            }
        }
        warn!(
            event_name = "local_stream_candidate_retry_scheduled",
            log_type = "event",
            trace_id = %trace_id,
            request_id = %request_id,
            candidate_id = ?candidate_id,
            status_code,
            failover_classification = failure_analysis.classification.as_str(),
            "gateway local stream decision retrying next candidate after prefetched provider error"
        );
        return Ok(None);
    }

    let response = submit_local_core_error_or_sync_finalize_after_terminal_usage(
        state,
        trace_id,
        decision,
        payload,
        usage_handoff_persisted,
    )
    .await?;
    Ok(Some(attach_control_metadata_headers(
        response,
        Some(request_id),
        candidate_id,
    )?))
}

#[allow(clippy::too_many_arguments)] // internal helper for prefetch error handling
pub(super) async fn handle_prefetch_stream_failure(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan: &ExecutionPlan,
    report_context: Option<Value>,
    request_id: &str,
    candidate_id: Option<&str>,
    report_kind: &str,
    headers: std::collections::BTreeMap<String, String>,
    telemetry: Option<ExecutionTelemetry>,
    buffered_body: &[u8],
    candidate_started_unix_ms: u64,
    candidate_elapsed_ms: u64,
    failure: StreamFailureReport,
    retry_scope_out: Option<&mut AiAttemptRetryScope>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let transport_error = failure.transport_error;
    let candidate_status_code = failure.upstream_status_code;
    let honor_http_failover = failure.honor_http_failover;
    let mut payload = build_stream_failure_sync_payload(
        trace_id,
        report_kind.to_string(),
        report_context,
        headers,
        telemetry,
        buffered_body,
        failure,
    );
    if transport_error {
        let telemetry = payload.telemetry.get_or_insert(ExecutionTelemetry {
            ttfb_ms: None,
            elapsed_ms: None,
            upstream_bytes: None,
        });
        telemetry.elapsed_ms.get_or_insert(candidate_elapsed_ms);
        return handle_prefetch_transport_stream_failure(
            state,
            trace_id,
            decision,
            plan,
            request_id,
            candidate_id,
            payload,
            candidate_started_unix_ms,
            candidate_elapsed_ms,
            retry_scope_out,
        )
        .await;
    }
    let honor_local_failover = honor_http_failover && retry_scope_out.is_some();
    let (failure_analysis, usage_handoff_persisted, intermediate_retry) =
        record_stream_sync_failure(
            state,
            plan,
            payload.report_context.as_ref(),
            &payload,
            candidate_status_code,
            None,
            if honor_local_failover {
                StreamFailureHandling::HonorLocalFailover
            } else {
                StreamFailureHandling::Terminal
            },
        )
        .await;
    if intermediate_retry {
        let failure_disposition = classify_failure_disposition(
            &plan.provider_api_format,
            failure_analysis.classification,
            payload.status_code,
        );
        if let Some(retry_scope) = retry_scope_out {
            *retry_scope = ai_attempt_retry_scope_from_failure_disposition(failure_disposition);
        }
        warn!(
            event_name = "local_stream_candidate_retry_scheduled",
            log_type = "event",
            trace_id = %trace_id,
            request_id = %request_id,
            candidate_id = ?candidate_id,
            status_code = payload.status_code,
            failover_classification = failure_analysis.classification.as_str(),
            "gateway local stream decision retrying next candidate after prefetched execution error"
        );
        return Ok(None);
    }

    let response = submit_local_core_error_or_sync_finalize_after_terminal_usage(
        state,
        trace_id,
        decision,
        payload,
        usage_handoff_persisted,
    )
    .await?;
    Ok(Some(attach_control_metadata_headers(
        response,
        Some(request_id),
        candidate_id,
    )?))
}

#[allow(clippy::too_many_arguments)]
async fn handle_prefetch_transport_stream_failure(
    state: &AppState,
    trace_id: &str,
    decision: &GatewayControlDecision,
    plan: &ExecutionPlan,
    request_id: &str,
    candidate_id: Option<&str>,
    payload: GatewaySyncReportRequest,
    candidate_started_unix_ms: u64,
    candidate_elapsed_ms: u64,
    retry_scope_out: Option<&mut AiAttemptRetryScope>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let error_type = stream_failure_body_field(&payload, "type").unwrap_or("internal");
    let error_message = stream_failure_body_field(&payload, "message").unwrap_or_default();
    let analysis = resolve_local_transport_failover_analysis_for_attempt(
        state,
        plan,
        payload.report_context.as_ref(),
    )
    .await;
    let watchdog_progress =
        crate::execution_runtime::stream_candidate_watchdog_progress_for_current_or_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
    let retrying_next_candidate = retry_scope_out.is_some()
        && matches!(analysis.decision, LocalFailoverDecision::RetryNextCandidate)
        && crate::execution_runtime::stream_candidate_watchdog_allows_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        )
        && crate::execution_runtime::try_claim_stream_candidate_intermediate_for_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        );
    let timeout_failure = matches!(error_type, "first_byte_timeout" | "read_timeout");
    if retrying_next_candidate && timeout_failure {
        apply_local_execution_effect(
            state,
            LocalExecutionEffectContext {
                plan,
                report_context: payload.report_context.as_ref(),
            },
            LocalExecutionEffect::PoolStreamTimeout,
        )
        .await;
    }
    let terminal_owner_claimed = !retrying_next_candidate
        && crate::execution_runtime::mark_stream_candidate_watchdog_terminal_started_with_progress(
            watchdog_progress.as_ref(),
        );
    let usage_handoff_persisted = if retrying_next_candidate {
        true
    } else if terminal_owner_claimed {
        let report_context_with_diagnostics =
            attach_current_request_diagnostics_and_candidate_timing_to_report_context(
                payload.report_context.as_ref(),
                payload
                    .telemetry
                    .as_ref()
                    .and_then(|telemetry| telemetry.elapsed_ms)
                    .or(Some(candidate_elapsed_ms)),
                payload
                    .telemetry
                    .as_ref()
                    .and_then(|telemetry| telemetry.ttfb_ms),
            );
        let context_seed = build_terminal_usage_context_seed(
            plan,
            report_context_with_diagnostics
                .as_ref()
                .or(payload.report_context.as_ref()),
        );
        let payload_seed = build_sync_terminal_usage_payload_seed(&payload);
        let persisted = state
            .usage_runtime
            .record_sync_terminal_with_handoff(
                state.usage_lifecycle_data_state().as_ref(),
                context_seed,
                payload_seed,
            )
            .await;
        persisted
    } else {
        // The terminal owner was already claimed by another path; leave this
        // candidate non-terminal until that durable outcome is observed.
        false
    };
    if !retrying_next_candidate && timeout_failure && usage_handoff_persisted {
        // Keep the terminal usage handoff ahead of timeout/lease side effects;
        // otherwise a stalled pool write can strand the request at streaming.
        apply_local_execution_effect(
            state,
            LocalExecutionEffectContext {
                plan,
                report_context: payload.report_context.as_ref(),
            },
            LocalExecutionEffect::PoolStreamTimeout,
        )
        .await;
    }

    let terminal_unix_ms = current_request_candidate_unix_ms();
    let reconciliation_update = SchedulerRequestCandidateStatusUpdate {
        status: RequestCandidateStatus::Failed,
        status_code: None,
        error_type: Some(error_type.to_string()),
        error_message: Some(error_message.to_string()),
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms)
            .or(Some(candidate_elapsed_ms)),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: Some(terminal_unix_ms),
    };
    if !usage_handoff_persisted {
        spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            payload.report_context.clone(),
            reconciliation_update.clone(),
        );
    }
    let candidate_status = if usage_handoff_persisted {
        RequestCandidateStatus::Failed
    } else {
        RequestCandidateStatus::Streaming
    };
    let candidate_update = SchedulerRequestCandidateStatusUpdate {
        status: candidate_status,
        status_code: None,
        error_type: if usage_handoff_persisted {
            Some(error_type.to_string())
        } else {
            Some("usage_terminal_handoff_unconfirmed".to_string())
        },
        error_message: if usage_handoff_persisted {
            Some(error_message.to_string())
        } else {
            Some(
                "terminal usage persistence was not confirmed before candidate finalization"
                    .to_string(),
            )
        },
        latency_ms: payload
            .telemetry
            .as_ref()
            .and_then(|telemetry| telemetry.elapsed_ms)
            .or(Some(candidate_elapsed_ms)),
        started_at_unix_ms: Some(candidate_started_unix_ms),
        finished_at_unix_ms: usage_handoff_persisted.then_some(terminal_unix_ms),
    };
    let candidate_persisted = if retrying_next_candidate {
        record_local_request_candidate_status(
            state,
            plan,
            payload.report_context.as_ref(),
            candidate_update.clone(),
        )
        .await
    } else {
        record_report_request_candidate_status(
            state,
            payload.report_context.as_ref(),
            candidate_update.clone(),
        )
        .await
    };
    if terminal_owner_claimed && usage_handoff_persisted && candidate_persisted {
        if let Some(progress) = watchdog_progress {
            crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                plan.request_id.as_str(),
                plan.candidate_id.as_deref(),
                &progress,
            );
        }
    } else if terminal_owner_claimed && usage_handoff_persisted && !candidate_persisted {
        spawn_report_candidate_persistence_retry_after_usage_handoff(
            state.clone(),
            plan.clone(),
            payload.report_context.clone(),
            reconciliation_update,
            watchdog_progress,
        );
    }

    if retrying_next_candidate {
        if let Some(retry_scope) = retry_scope_out {
            *retry_scope = AiAttemptRetryScope::Candidate;
        }
        warn!(
            event_name = "local_stream_transport_retry_scheduled",
            log_type = "event",
            trace_id = %trace_id,
            request_id = %request_id,
            candidate_id = ?candidate_id,
            transport_classification = analysis.classification.as_str(),
            "gateway retrying next candidate after precommit transport failure"
        );
        return Ok(None);
    }

    let response = submit_local_core_error_or_sync_finalize_after_terminal_usage(
        state,
        trace_id,
        decision,
        payload,
        usage_handoff_persisted,
    )
    .await?;
    Ok(Some(attach_control_metadata_headers(
        response,
        Some(request_id),
        candidate_id,
    )?))
}

const FALLBACK_STREAM_TERMINAL_ERROR_REPORT_KIND: &str = "stream_terminal_error";

/// Resolve the report kind used by the mid-stream terminal path without
/// making usage persistence conditional on a planner/report mapping.  The
/// direct-finalize kind is preferred because it carries the exact client
/// contract.  Older/custom attempt sources may omit it (and sometimes only
/// provide a success stream kind); those paths still need a terminal usage
/// event, so map the known success aliases to their sync error counterparts
/// and retain a neutral kind for completely unknown formats.
fn resolve_midstream_stream_error_report_kind(
    direct_stream_finalize_kind: Option<&str>,
    fallback_report_kind: Option<&str>,
) -> String {
    if let Some(report_kind) =
        direct_stream_finalize_kind.and_then(resolve_core_error_background_report_kind)
    {
        return report_kind;
    }

    let Some(report_kind) = fallback_report_kind
        .map(str::trim)
        .filter(|report_kind| !report_kind.is_empty())
    else {
        return FALLBACK_STREAM_TERMINAL_ERROR_REPORT_KIND.to_string();
    };

    let mapped = match report_kind {
        "openai_chat_stream_success" | "openai_chat_sync_success" => "openai_chat_sync_error",
        "claude_chat_stream_success" | "claude_chat_sync_success" => "claude_chat_sync_error",
        "gemini_chat_stream_success" | "gemini_chat_sync_success" => "gemini_chat_sync_error",
        "gemini_interactions_stream_success" | "gemini_interactions_sync_success" => {
            "gemini_interactions_sync_error"
        }
        "openai_responses_stream_success" | "openai_responses_sync_success" => {
            "openai_responses_sync_error"
        }
        "openai_responses_compact_stream_success" | "openai_responses_compact_sync_success" => {
            "openai_responses_compact_sync_error"
        }
        "openai_image_stream_success" | "openai_image_sync_success" => "openai_image_sync_error",
        "openai_cli_stream_success" => "openai_cli_sync_error",
        "claude_cli_stream_success" => "claude_cli_sync_error",
        "gemini_cli_stream_success" => "gemini_cli_sync_error",
        _ => report_kind,
    };
    mapped.to_string()
}

pub(super) async fn submit_midstream_stream_failure(
    state: &AppState,
    trace_id: &str,
    plan: &ExecutionPlan,
    direct_stream_finalize_kind: Option<&str>,
    fallback_report_kind: Option<&str>,
    report_context: Option<Value>,
    headers: std::collections::BTreeMap<String, String>,
    telemetry: Option<ExecutionTelemetry>,
    buffered_body: &[u8],
    started_at_unix_ms: u64,
    failure: StreamFailureReport,
) {
    let report_kind = resolve_midstream_stream_error_report_kind(
        direct_stream_finalize_kind,
        fallback_report_kind,
    );

    let candidate_status_code = failure.upstream_status_code;
    let payload = build_stream_failure_sync_payload(
        trace_id,
        report_kind,
        report_context,
        headers,
        telemetry,
        buffered_body,
        failure,
    );
    let (_failure_analysis, usage_handoff_persisted, _intermediate_retry) =
        record_stream_sync_failure(
            state,
            plan,
            payload.report_context.as_ref(),
            &payload,
            candidate_status_code,
            Some(started_at_unix_ms),
            StreamFailureHandling::Terminal,
        )
        .await;
    if let Err(err) =
        submit_sync_report_after_terminal_usage(state, payload, usage_handoff_persisted).await
    {
        let request_id = short_request_id(plan.request_id.as_str());
        warn!(
            event_name = "execution_report_submit_failed",
            log_type = "ops",
            trace_id = %trace_id,
            request_id = %request_id,
            candidate_id = ?plan.candidate_id,
            report_scope = "stream_failure",
            error = ?err,
            "gateway failed to submit sync execution report for terminal stream failure"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aether_contracts::{ExecutionError, ExecutionErrorKind, ExecutionPhase};
    use base64::Engine as _;
    use serde_json::json;

    use super::{
        build_stream_failure_from_execution_error, build_stream_failure_from_provider_error_body,
        build_stream_failure_sync_payload, build_stream_transport_failure_report,
        resolve_midstream_stream_error_report_kind,
    };

    #[test]
    fn midstream_failure_report_kind_has_a_terminal_fallback_without_mapping() {
        assert_eq!(
            resolve_midstream_stream_error_report_kind(None, None),
            "stream_terminal_error"
        );
        assert_eq!(
            resolve_midstream_stream_error_report_kind(
                None,
                Some("openai_responses_stream_success")
            ),
            "openai_responses_sync_error"
        );
        assert_eq!(
            resolve_midstream_stream_error_report_kind(
                Some("openai_chat_sync_finalize"),
                Some("gemini_chat_stream_success")
            ),
            "openai_chat_sync_error"
        );
    }

    #[test]
    fn committed_transport_failure_has_no_upstream_status() {
        for status_code in [502, 504] {
            let failure = build_stream_transport_failure_report(
                "execution_runtime_stream_read_error",
                "upstream disconnected",
                status_code,
            );

            assert_eq!(failure.status_code, status_code);
            assert_eq!(failure.upstream_status_code, None);
            assert!(failure.transport_error);
            assert!(!failure.honor_http_failover);
        }
    }

    #[test]
    fn precommit_protocol_error_is_transport_without_upstream_status() {
        let failure = build_stream_failure_from_execution_error(&ExecutionError {
            kind: ExecutionErrorKind::ProtocolError,
            phase: ExecutionPhase::StreamRead,
            message: "connection reset".to_string(),
            upstream_status: None,
            retryable: true,
            failover_recommended: true,
        });

        assert!(failure.transport_error);
        assert_eq!(failure.upstream_status_code, None);
        assert_eq!(failure.status_code, 502);
    }

    #[test]
    fn cancelled_stream_is_not_reclassified_as_transport_retry() {
        let failure = build_stream_failure_from_execution_error(&ExecutionError {
            kind: ExecutionErrorKind::Cancelled,
            phase: ExecutionPhase::StreamRead,
            message: "downstream cancelled".to_string(),
            upstream_status: None,
            retryable: true,
            failover_recommended: true,
        });

        assert!(!failure.transport_error);
    }

    #[test]
    fn midstream_failure_trace_uses_terminal_error_instead_of_buffered_sse() {
        let provider_buffered_body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"instructions\":\"AGENTS.md secret prompt\",\"tools\":[{\"name\":\"update_plan\"}]}}\n\n",
            "event: response.failed\n",
            "data: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\",\"error\":{\"type\":\"invalid_request\",\"message\":\"This content was flagged for possible cybersecurity risk.\",\"code\":\"cyber_policy_violation\",\"param\":\"input\",\"details\":{\"policy_category\":\"cybersecurity\",\"appeal_allowed\":true}}}}\n\n",
        )
        .as_bytes();
        let terminal_error = crate::ai_serving::api::extract_provider_private_stream_error_body(
            None,
            provider_buffered_body,
        )
        .expect("raw upstream SSE should expose its terminal provider error JSON");
        let failure = build_stream_failure_from_provider_error_body(400, &terminal_error);

        let payload = build_stream_failure_sync_payload(
            "trace-cyber-policy",
            "openai_responses_sync_error".to_string(),
            Some(json!({"request_id": "request-cyber-policy"})),
            BTreeMap::from([
                ("Content-Encoding".to_string(), "gzip".to_string()),
                ("Content-Length".to_string(), "4096".to_string()),
                ("Content-Type".to_string(), "text/event-stream".to_string()),
                (
                    "x-request-id".to_string(),
                    "req_usage-cyber-risk-demo".to_string(),
                ),
            ]),
            None,
            provider_buffered_body,
            failure,
        );

        let trace_body = payload
            .report_context
            .as_ref()
            .and_then(|context| context.pointer("/upstream_response/body"))
            .expect("candidate trace should include the terminal error body");
        assert_eq!(
            trace_body,
            payload.body_json.as_ref().expect("usage error body")
        );
        assert_eq!(trace_body, &terminal_error);
        assert_eq!(trace_body["error"]["type"], json!("invalid_request"));
        assert_eq!(
            trace_body["error"]["message"],
            json!("This content was flagged for possible cybersecurity risk.")
        );
        assert_eq!(trace_body["error"]["code"], json!("cyber_policy_violation"));
        assert_eq!(trace_body["error"]["param"], json!("input"));
        assert_eq!(
            trace_body["error"]["details"],
            json!({
                "policy_category": "cybersecurity",
                "appeal_allowed": true
            })
        );
        assert_eq!(
            payload
                .report_context
                .as_ref()
                .and_then(|context| context.pointer("/upstream_response/headers/content-type")),
            Some(&json!("application/json"))
        );
        assert_eq!(
            payload
                .report_context
                .as_ref()
                .and_then(|context| context.pointer("/upstream_response/headers/x-request-id")),
            Some(&json!("req_usage-cyber-risk-demo"))
        );
        assert_eq!(
            payload
                .report_context
                .as_ref()
                .and_then(|context| context.pointer("/provider_response_headers/content-type")),
            Some(&json!("application/json"))
        );
        assert_eq!(
            payload
                .report_context
                .as_ref()
                .and_then(|context| context.pointer("/client_response_headers/content-type")),
            Some(&json!("application/json"))
        );
        let trace_headers = payload
            .report_context
            .as_ref()
            .and_then(|context| context.pointer("/upstream_response/headers"))
            .and_then(serde_json::Value::as_object)
            .expect("candidate trace should include terminal JSON headers");
        assert!(!trace_headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("content-encoding")));
        assert!(!trace_headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("content-length")));
        assert_eq!(
            payload.headers.get("content-type").map(String::as_str),
            Some("application/json")
        );
        assert!(!payload
            .headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("content-encoding")));
        assert!(!payload
            .headers
            .keys()
            .any(|name| name.eq_ignore_ascii_case("content-length")));
        assert_eq!(
            payload
                .client_body_json
                .as_ref()
                .and_then(|body| body.pointer("/error/message")),
            Some(&json!(
                "This content was flagged for possible cybersecurity risk."
            ))
        );
        assert_eq!(
            payload
                .client_body_json
                .as_ref()
                .and_then(|body| body.pointer("/error/code")),
            Some(&json!(400))
        );
        assert_ne!(payload.client_body_json.as_ref(), Some(&terminal_error));
        assert!(!trace_body.to_string().contains("AGENTS.md secret prompt"));

        let raw_capture = payload
            .body_base64
            .as_deref()
            .and_then(|body| base64::engine::general_purpose::STANDARD.decode(body).ok())
            .expect("raw provider stream should remain available for usage auditing");
        assert_eq!(raw_capture, provider_buffered_body);
        assert!(String::from_utf8_lossy(&raw_capture).contains("AGENTS.md secret prompt"));
    }
}
