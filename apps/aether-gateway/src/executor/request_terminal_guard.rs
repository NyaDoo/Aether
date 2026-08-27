use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aether_contracts::{ExecutionPlan, ExecutionTelemetry};
use aether_data_contracts::repository::usage::UsageBodyCaptureState;
use aether_usage_runtime::{
    build_sync_terminal_usage_outcome, build_terminal_usage_event_from_outcome,
    GatewaySyncReportRequest, UsageEventType,
};
use axum::body::{to_bytes, Body, Bytes};
use axum::http::{HeaderMap, Response};
use base64::Engine as _;
use hyper::body::Body as _;
use serde_json::{json, Value};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::request_diagnostics::{
    attach_request_diagnostics_to_report_context, current_request_diagnostics, RequestDiagnostics,
};
use crate::{AppState, GatewayError};

const OWNER_ACTIVE: u8 = 0;
const OWNER_DISPATCHED: u8 = 1;
const OWNER_TRANSFERRED: u8 = 2;
const TERMINAL_RETRY_DELAYS_MS: &[u64] = &[0, 50, 200, 500, 1_000, 2_000];
const INLINE_TERMINAL_HANDOFF_WAIT: Duration = Duration::from_secs(2);

tokio::task_local! {
    static REQUEST_TERMINAL_OWNER: Arc<RequestTerminalOwnershipState>;
}

#[derive(Clone)]
pub(crate) struct CapturedRequestTerminalOwner(Option<Arc<RequestTerminalOwnershipState>>);

impl CapturedRequestTerminalOwner {
    pub(crate) async fn scope<F>(self, future: F) -> F::Output
    where
        F: Future,
    {
        match self.0 {
            Some(owner) => REQUEST_TERMINAL_OWNER.scope(owner, future).await,
            None => future.await,
        }
    }
}

pub(crate) fn capture_current_request_terminal_owner() -> CapturedRequestTerminalOwner {
    CapturedRequestTerminalOwner(REQUEST_TERMINAL_OWNER.try_with(Arc::clone).ok())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseUsageTerminalOwner {
    SyncTerminal,
    StreamBody,
}

#[derive(Clone)]
struct RequestTerminalSnapshot {
    plan: ExecutionPlan,
    report_context: Option<Value>,
    observed_at: Instant,
}

struct RequestTerminalOwnershipState {
    phase: AtomicU8,
    gap_armed: AtomicBool,
    snapshot: Mutex<Option<RequestTerminalSnapshot>>,
}

impl RequestTerminalOwnershipState {
    fn new() -> Self {
        Self {
            phase: AtomicU8::new(OWNER_ACTIVE),
            gap_armed: AtomicBool::new(false),
            snapshot: Mutex::new(None),
        }
    }

    fn note_retry(&self, plan: &ExecutionPlan, report_context: Option<Value>) {
        if self.phase.load(Ordering::Acquire) != OWNER_ACTIVE {
            return;
        }
        let Ok(mut snapshot) = self.snapshot.lock() else {
            return;
        };
        *snapshot = Some(RequestTerminalSnapshot {
            plan: plan.clone(),
            report_context,
            observed_at: Instant::now(),
        });
        self.gap_armed.store(true, Ordering::Release);
    }

    fn pause_for_attempt(&self) {
        self.gap_armed.store(false, Ordering::Release);
    }

    fn snapshot(&self) -> Option<RequestTerminalSnapshot> {
        self.snapshot.lock().ok().and_then(|value| value.clone())
    }

    fn transfer(&self) {
        let _ = self.phase.compare_exchange(
            OWNER_ACTIVE,
            OWNER_TRANSFERRED,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn claim_dispatch(&self) -> Option<RequestTerminalSnapshot> {
        if !self.gap_armed.load(Ordering::Acquire) {
            return None;
        }
        let snapshot = self.snapshot()?;
        self.phase
            .compare_exchange(
                OWNER_ACTIVE,
                OWNER_DISPATCHED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok()
            .map(|_| snapshot)
    }

    fn is_active(&self) -> bool {
        self.phase.load(Ordering::Acquire) == OWNER_ACTIVE
    }
}

#[derive(Clone)]
struct TerminalClientAudit {
    status_code: u16,
    headers: BTreeMap<String, String>,
    body: Option<Value>,
    body_state: UsageBodyCaptureState,
    event_type: UsageEventType,
    error_type: &'static str,
    error_message: String,
}

impl TerminalClientAudit {
    fn cancelled() -> Self {
        let status_code = 499;
        let error_type = "request_cancelled_between_execution_candidates";
        let error_message =
            "request was cancelled while no execution attempt owned terminal usage".to_string();
        Self {
            status_code,
            headers: BTreeMap::from([("content-type".to_string(), "application/json".to_string())]),
            body: Some(json!({
                "error": {
                    "type": error_type,
                    "message": error_message.clone(),
                    "code": status_code,
                }
            })),
            body_state: UsageBodyCaptureState::Inline,
            event_type: UsageEventType::Cancelled,
            error_type,
            error_message,
        }
    }

    fn failed_from_error(error: &GatewayError) -> Self {
        let status_code = error.status_code().as_u16();
        let error_type = "request_failed_between_execution_candidates";
        let (mut headers, body) = gateway_error_audit_response(error);
        headers
            .entry("content-type".to_string())
            .or_insert_with(|| "application/json".to_string());
        Self {
            status_code,
            headers,
            body: Some(body),
            body_state: UsageBodyCaptureState::Inline,
            event_type: UsageEventType::Failed,
            error_type,
            error_message: format!("local execution failed before terminal response: {error:?}"),
        }
    }

    fn failed_from_response(
        status_code: u16,
        headers: BTreeMap<String, String>,
        body: Option<Value>,
        body_state: UsageBodyCaptureState,
    ) -> Self {
        let error_type = "request_failed_after_candidate_retry";
        let error_message =
            format!("request returned HTTP {status_code} after all retryable execution candidates");
        Self {
            status_code,
            headers,
            body,
            body_state,
            event_type: UsageEventType::Failed,
            error_type,
            error_message,
        }
    }
}

/// Owns the request-level gap between two execution attempts and the gap
/// between an exhausted attempt loop and the proxy's final usage write.
///
/// Per-attempt guards deliberately yield after an intermediate candidate has
/// durably become `Failed`, because the next candidate must be allowed to
/// record the one request-level terminal event.  This guard is the owner while
/// no attempt is active.  Its Drop path never writes a candidate row; the
/// latest candidate remains the truthful per-attempt `Failed` outcome.
pub(crate) struct RequestTerminalOwnershipGuard {
    state: AppState,
    request_id: String,
    owner: Arc<RequestTerminalOwnershipState>,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
}

impl RequestTerminalOwnershipGuard {
    pub(crate) fn new(state: AppState, request_id: impl Into<String>) -> Self {
        Self {
            state,
            request_id: request_id.into(),
            owner: Arc::new(RequestTerminalOwnershipState::new()),
            request_diagnostics: current_request_diagnostics(),
        }
    }

    pub(crate) async fn scope<F>(&self, future: F) -> F::Output
    where
        F: Future,
    {
        REQUEST_TERMINAL_OWNER
            .scope(Arc::clone(&self.owner), future)
            .await
    }

    pub(crate) async fn settle_response(
        &mut self,
        response: Response<Body>,
    ) -> Result<Response<Body>, GatewayError> {
        if !self.owner.is_active() {
            return Ok(response);
        }
        if response_has_usage_terminal_owner(&response) {
            self.owner.transfer();
            return Ok(response);
        }
        if self.owner.snapshot().is_none() {
            self.owner.transfer();
            return Ok(response);
        }

        // An active gap owner means this is a non-stream fallback/planning
        // response: all real sync/stream terminal owners attach an explicit
        // response extension above.  Buffer only when Body advertises a
        // bounded size within the existing error-capture limit. Unknown or
        // oversized bodies pass through untouched and are audited with an
        // explicit unavailable/truncated state.
        let advertised_upper = response.body().size_hint().upper();
        if advertised_upper.is_none_or(|upper| upper > crate::MAX_ERROR_BODY_BYTES as u64) {
            let body_state = if advertised_upper.is_some() {
                UsageBodyCaptureState::Truncated
            } else {
                UsageBodyCaptureState::Unavailable
            };
            let body = advertised_upper.map(|size| {
                json!({
                    "truncated": true,
                    "reason": "client_response_body_exceeds_audit_limit",
                    "observed_size_bytes": size,
                    "limit_bytes": crate::MAX_ERROR_BODY_BYTES,
                })
            });
            let audit = TerminalClientAudit::failed_from_response(
                response.status().as_u16(),
                header_map_to_btree(response.headers()),
                body,
                body_state,
            );
            await_terminal_handoff_bounded(self.dispatch(audit)).await;
            return Ok(response);
        }

        let (parts, body) = response.into_parts();
        let body_bytes = match to_bytes(body, crate::MAX_ERROR_BODY_BYTES).await {
            Ok(body_bytes) => body_bytes,
            Err(error) => {
                let error = GatewayError::Internal(format!(
                    "failed to read bounded terminal fallback response: {error}"
                ));
                await_terminal_handoff_bounded(
                    self.dispatch(TerminalClientAudit::failed_from_error(&error)),
                )
                .await;
                return Err(error);
            }
        };
        let audit = TerminalClientAudit::failed_from_response(
            parts.status.as_u16(),
            header_map_to_btree(&parts.headers),
            capture_body(&body_bytes),
            if body_bytes.is_empty() {
                UsageBodyCaptureState::None
            } else {
                UsageBodyCaptureState::Inline
            },
        );
        await_terminal_handoff_bounded(self.dispatch(audit)).await;
        Ok(Response::from_parts(parts, Body::from(body_bytes)))
    }

    pub(crate) async fn settle_error(&mut self, error: &GatewayError) {
        await_terminal_handoff_bounded(
            self.dispatch(TerminalClientAudit::failed_from_error(error)),
        )
        .await;
    }

    fn dispatch(&mut self, audit: TerminalClientAudit) -> Option<JoinHandle<bool>> {
        let snapshot = self.owner.claim_dispatch()?;
        let usage_request_id = snapshot.plan.request_id.clone();
        Some(spawn_terminal_if_still_pending(
            self.state.clone(),
            usage_request_id,
            self.request_id.clone(),
            snapshot,
            self.request_diagnostics.clone(),
            audit,
        ))
    }
}

impl Drop for RequestTerminalOwnershipGuard {
    fn drop(&mut self) {
        let Some(snapshot) = self.owner.claim_dispatch() else {
            return;
        };
        let usage_request_id = snapshot.plan.request_id.clone();
        let _ = spawn_terminal_if_still_pending(
            self.state.clone(),
            usage_request_id,
            self.request_id.clone(),
            snapshot,
            self.request_diagnostics.clone(),
            TerminalClientAudit::cancelled(),
        );
    }
}

pub(crate) fn mark_sync_response_usage_terminal_owner(
    mut response: Response<Body>,
) -> Response<Body> {
    response
        .extensions_mut()
        .insert(ResponseUsageTerminalOwner::SyncTerminal);
    response
}

pub(crate) fn mark_stream_response_usage_terminal_owner(
    mut response: Response<Body>,
) -> Response<Body> {
    response
        .extensions_mut()
        .insert(ResponseUsageTerminalOwner::StreamBody);
    response
}

pub(crate) fn response_has_usage_terminal_owner(response: &Response<Body>) -> bool {
    response
        .extensions()
        .get::<ResponseUsageTerminalOwner>()
        .is_some()
}

pub(crate) fn note_current_request_retry_terminal_gap(
    plan: &ExecutionPlan,
    report_context: Option<Value>,
) {
    let _ = REQUEST_TERMINAL_OWNER.try_with(|owner| owner.note_retry(plan, report_context));
}

/// Refresh the request-gap snapshot to the candidate that is about to start.
/// This synchronous handoff happens before `record_attempt_started` awaits, so
/// a cancellation can never attribute the next attempt to the previous
/// candidate. The per-attempt guard pauses this owner as soon as it is armed.
pub(crate) fn note_current_request_attempt_terminal_gap(
    plan: &ExecutionPlan,
    report_context: Option<Value>,
) {
    let _ = REQUEST_TERMINAL_OWNER.try_with(|owner| owner.note_retry(plan, report_context));
}

pub(crate) fn pause_current_request_terminal_gap_for_attempt() {
    let _ = REQUEST_TERMINAL_OWNER.try_with(|owner| owner.pause_for_attempt());
}

pub(crate) fn transfer_current_request_terminal_owner(response: &Response<Body>) {
    if !response_has_usage_terminal_owner(response) {
        return;
    }
    let _ = REQUEST_TERMINAL_OWNER.try_with(|owner| owner.transfer());
}

async fn await_terminal_handoff_bounded(join: Option<JoinHandle<bool>>) {
    let Some(join) = join else {
        return;
    };
    // Dropping a JoinHandle detaches the process-lifetime usage task. The
    // client wait is bounded while reconciliation keeps ownership of the
    // complete event until persistence/readback finishes.
    let _ = tokio::time::timeout(INLINE_TERMINAL_HANDOFF_WAIT, join).await;
}

fn spawn_terminal_if_still_pending(
    state: AppState,
    request_id: String,
    ingress_trace_id: String,
    snapshot: RequestTerminalSnapshot,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
    audit: TerminalClientAudit,
) -> JoinHandle<bool> {
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        terminal_if_still_pending(
            &state,
            request_id.as_str(),
            ingress_trace_id.as_str(),
            snapshot,
            request_diagnostics,
            audit,
        )
        .await
    })
}

async fn terminal_if_still_pending(
    state: &AppState,
    request_id: &str,
    ingress_trace_id: &str,
    snapshot: RequestTerminalSnapshot,
    request_diagnostics: Option<Arc<RequestDiagnostics>>,
    audit: TerminalClientAudit,
) -> bool {
    if !state.usage_runtime.is_enabled() {
        return true;
    }

    let report_context = enrich_report_context_from_exact_candidate(
        state,
        request_id,
        snapshot.plan.candidate_id.as_deref(),
        snapshot.report_context.clone(),
    )
    .await;
    let report_context =
        attach_request_diagnostics_to_report_context(report_context, request_diagnostics.as_ref());

    let terminal_event = match build_terminal_gap_event(
        request_id,
        ingress_trace_id,
        &snapshot,
        report_context,
        request_diagnostics.as_ref(),
        &audit,
    ) {
        Ok(event) => event,
        Err(error) => {
            warn!(
                event_name = "request_terminal_gap_event_build_failed",
                log_type = "ops",
                request_id,
                candidate_id = ?snapshot.plan.candidate_id,
                error = ?error,
                "gateway could not build a complete request terminal gap event"
            );
            return false;
        }
    };

    for (index, delay_ms) in TERMINAL_RETRY_DELAYS_MS.iter().enumerate() {
        if *delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(*delay_ms)).await;
        }
        let final_attempt = index + 1 == TERMINAL_RETRY_DELAYS_MS.len();
        let usage_read = state
            .data
            .find_request_usage_by_request_id_shallow(request_id)
            .await;
        let should_write = match usage_read {
            Ok(Some(usage))
                if usage_is_unfinalized(
                    &usage.status,
                    &usage.billing_status,
                    usage.finalized_at_unix_secs,
                ) =>
            {
                true
            }
            Ok(Some(usage)) => {
                debug!(
                    event_name = "request_terminal_gap_already_finalized",
                    log_type = "event",
                    request_id,
                    status = usage.status.as_str(),
                    billing_status = usage.billing_status.as_str(),
                    "gateway skipped a competing request terminal gap event"
                );
                return true;
            }
            // A pending insert can lag this process-lifetime task. Wait for
            // it on early probes, but the final probe must submit the full
            // terminal event so a missing row can be created instead of
            // silently reproducing the production hole.
            Ok(None) => final_attempt,
            Err(error) => {
                warn!(
                    event_name = "request_terminal_gap_usage_read_failed",
                    log_type = "ops",
                    request_id,
                    error = ?error,
                    "gateway could not read durable usage while closing a request terminal gap"
                );
                final_attempt
            }
        };
        if !should_write {
            continue;
        }

        let _ = state
            .usage_runtime
            .record_terminal_event_direct_with_handoff(
                state.usage_lifecycle_data_state().as_ref(),
                terminal_event.clone(),
            )
            .await;

        match state
            .data
            .find_request_usage_by_request_id_shallow(request_id)
            .await
        {
            Ok(Some(stored))
                if !usage_is_unfinalized(
                    &stored.status,
                    &stored.billing_status,
                    stored.finalized_at_unix_secs,
                ) =>
            {
                return true;
            }
            Ok(_) => {}
            Err(error) => warn!(
                event_name = "request_terminal_gap_usage_readback_failed",
                log_type = "ops",
                request_id,
                error = ?error,
                "gateway could not confirm the request terminal gap usage write"
            ),
        }
    }

    warn!(
        event_name = "request_terminal_gap_reconciliation_exhausted",
        log_type = "ops",
        request_id,
        candidate_id = ?snapshot.plan.candidate_id,
        "gateway exhausted retries while closing a request terminal gap"
    );
    false
}

fn build_terminal_gap_event(
    request_id: &str,
    ingress_trace_id: &str,
    snapshot: &RequestTerminalSnapshot,
    report_context: Option<Value>,
    request_diagnostics: Option<&Arc<RequestDiagnostics>>,
    audit: &TerminalClientAudit,
) -> Result<aether_usage_runtime::UsageEvent, aether_data_contracts::DataLayerError> {
    let mut context = report_context
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let upstream = context
        .get("upstream_response")
        .and_then(Value::as_object)
        .cloned();
    let provider_headers = context
        .get("provider_response_headers")
        .cloned()
        .or_else(|| {
            upstream
                .as_ref()
                .and_then(|value| value.get("headers").cloned())
        });
    let provider_body = context.get("provider_response").cloned().or_else(|| {
        upstream
            .as_ref()
            .and_then(|value| value.get("body").cloned())
    });
    let provider_body_ref = context
        .get("response_body_ref")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            upstream
                .as_ref()
                .and_then(|value| value.get("body_ref"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        });
    let provider_body_state = context
        .get("provider_response_body_state")
        .cloned()
        .or_else(|| {
            upstream
                .as_ref()
                .and_then(|value| value.get("body_state").cloned())
        })
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_else(|| {
            body_capture_state(provider_body.as_ref(), provider_body_ref.as_deref())
        });
    if let Some(headers) = provider_headers {
        context.insert("provider_response_headers".to_string(), headers);
    }
    if let Some(body_ref) = provider_body_ref.as_ref() {
        context
            .entry("response_body_ref".to_string())
            .or_insert_with(|| Value::String(body_ref.clone()));
    }

    let explicit_client_body = context
        .get("client_response")
        .cloned()
        .or_else(|| context.get("client_response_body").cloned());
    let client_body_ref = context
        .get("client_response_body_ref")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    // Failed exhaustion fallbacks preserve the most recent provider error
    // when no converted client capture exists. Cancellation never borrows a
    // provider body: its synthetic 499 belongs only to client-response fields.
    let client_body = audit.body.clone().or(explicit_client_body).or_else(|| {
        (audit.event_type == UsageEventType::Failed)
            .then(|| provider_body.clone())
            .flatten()
    });
    let client_body_state = if audit.body.is_some()
        || matches!(
            audit.body_state,
            UsageBodyCaptureState::Truncated | UsageBodyCaptureState::Unavailable
        ) {
        audit.body_state
    } else {
        context
            .get("client_response_body_state")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
            .unwrap_or_else(|| body_capture_state(client_body.as_ref(), client_body_ref.as_deref()))
    };
    context.insert(
        "client_response_headers".to_string(),
        serde_json::to_value(&audit.headers).unwrap_or(Value::Null),
    );
    context.insert(
        "terminal_gap_error_type".to_string(),
        Value::String(audit.error_type.to_string()),
    );
    if ingress_trace_id != request_id {
        context.insert(
            "terminal_gap_ingress_trace_id".to_string(),
            Value::String(ingress_trace_id.to_string()),
        );
    }

    let elapsed_ms = request_diagnostics
        .and_then(|diagnostics| diagnostics.request_accepted_elapsed_ms())
        .unwrap_or_else(|| snapshot.observed_at.elapsed().as_millis() as u64);
    let payload = GatewaySyncReportRequest {
        trace_id: ingress_trace_id.to_string(),
        report_kind: match audit.event_type {
            UsageEventType::Cancelled => "request_terminal_gap_cancelled",
            _ => "request_terminal_gap_failed",
        }
        .to_string(),
        report_context: Some(Value::Object(context)),
        status_code: audit.status_code,
        // `headers` is the legacy provider-header fallback. Client headers
        // are explicit in report_context above; leaving this empty prevents a
        // gateway 499/5xx response head from being copied into provider audit.
        headers: BTreeMap::new(),
        body_json: provider_body,
        client_body_json: client_body,
        body_base64: None,
        telemetry: Some(ExecutionTelemetry {
            ttfb_ms: None,
            elapsed_ms: Some(elapsed_ms),
            upstream_bytes: None,
        }),
    };
    let mut event = build_terminal_usage_event_from_outcome(build_sync_terminal_usage_outcome(
        &snapshot.plan,
        payload.report_context.as_ref(),
        &payload,
    ))?;
    // Never let a 2xx header alone manufacture success. This owner is entered
    // only for cancellation or an unowned failure response, and that semantic
    // terminal state wins over the builder's status-only inference.
    event.event_type = audit.event_type;
    event.data.status_code = Some(audit.status_code);
    event.data.error_message = Some(audit.error_message.clone());
    event.data.error_category = Some(
        if audit.event_type == UsageEventType::Cancelled {
            "cancelled"
        } else {
            "server_error"
        }
        .to_string(),
    );
    event.data.response_time_ms = Some(elapsed_ms);
    event.data.response_body_state = Some(provider_body_state);
    event.data.client_response_body_state = Some(client_body_state);
    Ok(event)
}

fn usage_is_unfinalized(status: &str, billing_status: &str, finalized_at: Option<u64>) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "pending" | "streaming"
    ) && billing_status.trim().eq_ignore_ascii_case("pending")
        && finalized_at.is_none()
}

async fn enrich_report_context_from_exact_candidate(
    state: &AppState,
    request_id: &str,
    candidate_id: Option<&str>,
    report_context: Option<Value>,
) -> Option<Value> {
    let Some(candidate_id) = candidate_id
        .map(str::trim)
        .filter(|candidate_id| !candidate_id.is_empty())
    else {
        return report_context;
    };
    let candidates = match state
        .data
        .list_request_candidates_by_request_id(request_id)
        .await
    {
        Ok(candidates) => candidates,
        Err(error) => {
            warn!(
                event_name = "request_terminal_gap_candidate_read_failed",
                log_type = "ops",
                request_id,
                candidate_id,
                error = ?error,
                "gateway could not reload exact candidate audit for request terminal gap"
            );
            return report_context;
        }
    };
    let Some(extra_data) = candidates
        .into_iter()
        .find(|candidate| candidate.id == candidate_id)
        .and_then(|candidate| candidate.extra_data)
        .and_then(|value| value.as_object().cloned())
    else {
        return report_context;
    };
    let mut context = report_context
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    for (key, value) in extra_data {
        context.insert(key, value);
    }
    Some(Value::Object(context))
}

fn body_capture_state(body: Option<&Value>, body_ref: Option<&str>) -> UsageBodyCaptureState {
    if body.is_some() {
        UsageBodyCaptureState::Inline
    } else if body_ref.is_some() {
        UsageBodyCaptureState::Reference
    } else {
        UsageBodyCaptureState::None
    }
}

fn header_map_to_btree(headers: &HeaderMap) -> BTreeMap<String, String> {
    let mut object: BTreeMap<String, String> = BTreeMap::new();
    for (name, value) in headers {
        let value = value.to_str().unwrap_or_default().to_string();
        object
            .entry(name.as_str().to_string())
            .and_modify(|existing| {
                existing.push_str(", ");
                existing.push_str(value.as_str());
            })
            .or_insert(value);
    }
    object
}

fn gateway_error_audit_response(error: &GatewayError) -> (BTreeMap<String, String>, Value) {
    let mut headers = BTreeMap::new();
    match error {
        GatewayError::UpstreamUnavailable { trace_id, .. } => {
            headers.insert(
                crate::constants::TRACE_ID_HEADER.to_string(),
                trace_id.clone(),
            );
            headers.insert(
                crate::constants::GATEWAY_HEADER.to_string(),
                "rust-phase3b".into(),
            );
            (
                headers,
                json!({"error": {"message": "gateway proxy unavailable", "trace_id": trace_id}}),
            )
        }
        GatewayError::ControlUnavailable { trace_id, .. } => {
            headers.insert(
                crate::constants::TRACE_ID_HEADER.to_string(),
                trace_id.clone(),
            );
            headers.insert(
                crate::constants::GATEWAY_HEADER.to_string(),
                "rust-phase3b".into(),
            );
            (
                headers,
                json!({"error": {"message": "gateway control unavailable", "trace_id": trace_id}}),
            )
        }
        GatewayError::LocalExecutionPlanningTimeout { trace_id, .. } => {
            headers.insert(
                crate::constants::TRACE_ID_HEADER.to_string(),
                trace_id.clone(),
            );
            headers.insert(
                crate::constants::GATEWAY_HEADER.to_string(),
                "rust-phase3b".into(),
            );
            (
                headers,
                json!({"error": {"message": "gateway local execution planning timed out", "trace_id": trace_id}}),
            )
        }
        GatewayError::AdmissionTimeout { trace_id, .. } => {
            headers.insert(
                crate::constants::TRACE_ID_HEADER.to_string(),
                trace_id.clone(),
            );
            headers.insert(
                crate::constants::GATEWAY_HEADER.to_string(),
                "rust-phase3b".into(),
            );
            headers.insert("retry-after".to_string(), "1".to_string());
            (
                headers,
                json!({"error": {"message": "gateway admission queue timed out", "trace_id": trace_id}}),
            )
        }
        GatewayError::Client { message, .. } | GatewayError::Internal(message) => {
            (headers, json!({"error": {"message": message}}))
        }
    }
}

fn capture_body(body: &Bytes) -> Option<Value> {
    if body.is_empty() {
        return None;
    }
    serde_json::from_slice(body).ok().or_else(|| {
        std::str::from_utf8(body)
            .ok()
            .map(|text| Value::String(text.to_string()))
            .or_else(|| {
                Some(json!({
                    "encoding": "base64",
                    "data": base64::engine::general_purpose::STANDARD.encode(body),
                }))
            })
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Instant;

    use aether_contracts::{ExecutionPlan, RequestBody};
    use aether_data_contracts::repository::usage::UsageBodyCaptureState;
    use aether_usage_runtime::UsageEventType;
    use serde_json::json;

    use super::{
        build_terminal_gap_event, capture_current_request_terminal_owner,
        pause_current_request_terminal_gap_for_attempt, usage_is_unfinalized,
        RequestTerminalOwnershipState, RequestTerminalSnapshot, TerminalClientAudit,
        REQUEST_TERMINAL_OWNER,
    };

    fn test_plan(candidate_id: &str) -> ExecutionPlan {
        ExecutionPlan {
            request_id: "req-terminal-gap".to_string(),
            candidate_id: Some(candidate_id.to_string()),
            provider_name: Some("provider".to_string()),
            provider_id: "provider-id".to_string(),
            endpoint_id: "endpoint-id".to_string(),
            key_id: "key-id".to_string(),
            method: "POST".to_string(),
            url: "https://provider.invalid/v1/chat/completions".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: RequestBody::from_json(json!({"model": "test"})),
            stream: false,
            client_api_format: "openai:chat".to_string(),
            provider_api_format: "openai:chat".to_string(),
            model_name: Some("test".to_string()),
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    #[test]
    fn durable_usage_gate_accepts_only_unfinalized_pending_or_streaming() {
        assert!(usage_is_unfinalized("pending", "pending", None));
        assert!(usage_is_unfinalized("streaming", "pending", None));
        assert!(!usage_is_unfinalized("completed", "pending", None));
        assert!(!usage_is_unfinalized("pending", "void", None));
        assert!(!usage_is_unfinalized("streaming", "pending", Some(1)));
    }

    #[test]
    fn next_attempt_pause_never_dispatches_the_previous_candidate_snapshot() {
        let owner = RequestTerminalOwnershipState::new();
        owner.note_retry(&test_plan("candidate-previous"), None);

        // The synchronous starting-attempt hook refreshes identity before any
        // candidate-start await; the per-attempt guard then pauses this owner.
        owner.note_retry(&test_plan("candidate-current"), None);
        assert_eq!(
            owner
                .snapshot()
                .and_then(|snapshot| snapshot.plan.candidate_id),
            Some("candidate-current".to_string())
        );
        owner.pause_for_attempt();
        assert!(owner.claim_dispatch().is_none());

        // A Retry from the current attempt re-arms the same exact identity
        // before the loop performs planning/cleanup awaits.
        owner.note_retry(&test_plan("candidate-current"), None);
        assert_eq!(
            owner
                .claim_dispatch()
                .and_then(|snapshot| snapshot.plan.candidate_id),
            Some("candidate-current".to_string())
        );
    }

    #[test]
    fn terminal_event_separates_sanitized_provider_and_client_headers_and_refs() {
        let snapshot = RequestTerminalSnapshot {
            plan: test_plan("candidate-audit"),
            report_context: None,
            observed_at: Instant::now(),
        };
        let mut audit = TerminalClientAudit::cancelled();
        audit.headers.insert(
            "set-cookie".to_string(),
            "client-session-secret".to_string(),
        );
        audit
            .headers
            .insert("x-client-only".to_string(), "client".to_string());
        let event = build_terminal_gap_event(
            "req-terminal-gap",
            "ingress-trace-that-differs",
            &snapshot,
            Some(json!({
                "upstream_response": {
                    "headers": {
                        "authorization": "Bearer provider-secret",
                        "x-provider-only": "provider"
                    },
                    "body_ref": "usage://provider/body",
                    "body_state": "reference"
                }
            })),
            None,
            &audit,
        )
        .expect("terminal event should build");

        let provider_headers = event
            .data
            .response_headers
            .as_ref()
            .and_then(serde_json::Value::as_object)
            .expect("provider headers should be captured");
        let client_headers = event
            .data
            .client_response_headers
            .as_ref()
            .and_then(serde_json::Value::as_object)
            .expect("client headers should be captured");
        assert_eq!(provider_headers["x-provider-only"], "provider");
        assert!(!provider_headers.contains_key("x-client-only"));
        assert_eq!(client_headers["x-client-only"], "client");
        assert!(!client_headers.contains_key("x-provider-only"));
        assert_ne!(provider_headers["authorization"], "Bearer provider-secret");
        assert_ne!(client_headers["set-cookie"], "client-session-secret");
        assert_eq!(
            event.data.response_body_state,
            Some(UsageBodyCaptureState::Reference)
        );
        assert_eq!(event.event_type, UsageEventType::Cancelled);
        assert_eq!(event.request_id, "req-terminal-gap");
        assert_eq!(event.data.status_code, Some(499));
    }

    #[tokio::test]
    async fn watchdog_spawn_context_propagates_terminal_owner_and_request_diagnostics() {
        let owner = Arc::new(RequestTerminalOwnershipState::new());
        owner.note_retry(&test_plan("candidate-spawned"), None);
        let diagnostics = Arc::new(crate::request_diagnostics::RequestDiagnostics::default());
        let diagnostics_for_child = Arc::clone(&diagnostics);

        crate::request_diagnostics::scope_request_diagnostics_with(
            Some(Arc::clone(&diagnostics)),
            REQUEST_TERMINAL_OWNER.scope(Arc::clone(&owner), async move {
                let captured_owner = capture_current_request_terminal_owner();
                let captured_diagnostics =
                    crate::request_diagnostics::current_request_diagnostics();
                tokio::spawn(captured_owner.scope(
                    crate::request_diagnostics::scope_request_diagnostics_with(
                        captured_diagnostics,
                        async move {
                            let child_diagnostics =
                                crate::request_diagnostics::current_request_diagnostics()
                                    .expect("request diagnostics must cross watchdog spawn");
                            assert!(Arc::ptr_eq(&child_diagnostics, &diagnostics_for_child));
                            pause_current_request_terminal_gap_for_attempt();
                        },
                    ),
                ))
                .await
                .expect("watchdog child task should complete");
            }),
        )
        .await;

        assert!(
            owner.claim_dispatch().is_none(),
            "spawned per-attempt owner must pause the outer request-gap owner"
        );
    }

    #[test]
    fn unowned_two_hundred_response_is_failed_not_header_inferred_success() {
        let snapshot = RequestTerminalSnapshot {
            plan: test_plan("candidate-semantic-failure"),
            report_context: None,
            observed_at: Instant::now(),
        };
        let audit = TerminalClientAudit::failed_from_response(
            200,
            BTreeMap::new(),
            Some(json!({"error": {"message": "missing terminal owner"}})),
            UsageBodyCaptureState::Inline,
        );
        let event = build_terminal_gap_event(
            "req-terminal-gap",
            "req-terminal-gap",
            &snapshot,
            None,
            None,
            &audit,
        )
        .expect("terminal event should build");

        assert_eq!(event.event_type, UsageEventType::Failed);
        assert_eq!(event.data.error_category.as_deref(), Some("server_error"));
    }
}
