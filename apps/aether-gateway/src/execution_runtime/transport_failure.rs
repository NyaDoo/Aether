use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex as StdMutex, OnceLock,
};
use std::time::{Duration, Instant};

use aether_usage_runtime::{build_usage_event_data_seed, UsageEvent, UsageEventType};
use axum::body::Body;
use axum::http::Response;
use serde_json::{json, Value};

use crate::ai_serving::{build_core_error_body_for_client_format, LocalCoreSyncErrorKind};
use crate::api::response::{attach_control_metadata_headers, build_client_response_from_parts};
use crate::clock::current_unix_ms;
use crate::control::GatewayControlDecision;
use crate::request_candidate_runtime::{
    record_local_request_candidate_status, spawn_candidate_persistence_retry_after_usage_handoff,
    spawn_terminal_candidate_reconciliation,
};
use crate::request_diagnostics::attach_current_request_diagnostics_and_candidate_timing_to_report_context;
use crate::{AppState, GatewayError};
use aether_data_contracts::repository::candidates::RequestCandidateStatus;
use aether_scheduler_core::SchedulerRequestCandidateStatusUpdate;

const TRANSPORT_ERROR_CLIENT_MESSAGE: &str =
    "Upstream transport failed before an HTTP response was received";
const WATCHDOG_PROGRESS_REGISTRY_TTL: Duration = Duration::from_secs(24 * 60 * 60);

struct RegisteredWatchdogProgress {
    progress: Arc<StreamCandidateWatchdogProgress>,
    registered_at: Instant,
}

fn watchdog_progress_registry() -> &'static StdMutex<HashMap<String, RegisteredWatchdogProgress>> {
    static REGISTRY: OnceLock<StdMutex<HashMap<String, RegisteredWatchdogProgress>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| StdMutex::new(HashMap::new()))
}

fn watchdog_progress_registry_key(request_id: &str, candidate_id: Option<&str>) -> String {
    format!(
        "{}\u{1f}{}",
        request_id.trim(),
        candidate_id.unwrap_or_default().trim()
    )
}

/// Task-local values do not cross the `tokio::spawn` used by stream body
/// pumps.  Keep a request-scoped strong reference as a second transport for
/// terminal ownership; terminal helpers resolve it by request id when their
/// child task has no task-local context.
pub(crate) fn register_stream_candidate_watchdog_progress(
    request_id: &str,
    candidate_id: Option<&str>,
    progress: Arc<StreamCandidateWatchdogProgress>,
) {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return;
    }
    let Ok(mut registry) = watchdog_progress_registry().lock() else {
        return;
    };
    // Every entry is request-scoped.  A still-NONE entry can otherwise stay
    // in the process forever when a caller is cancelled before the watchdog
    // state changes, slowly turning this safety registry into a leak.
    registry.retain(|_, entry| entry.registered_at.elapsed() <= WATCHDOG_PROGRESS_REGISTRY_TTL);
    registry.insert(
        watchdog_progress_registry_key(request_id, candidate_id),
        RegisteredWatchdogProgress {
            progress,
            registered_at: Instant::now(),
        },
    );
}

/// Return the request/candidate watchdog owner already registered by an
/// outer candidate loop, or atomically install a new owner when this is the
/// first execution scope for the pair.  Sync execution uses this instead of
/// blindly calling `register_*`: replacing an outer owner's `Arc` would split
/// the terminal CAS state and let the two scopes both finalize usage.
pub(crate) fn get_or_register_stream_candidate_watchdog_progress(
    request_id: &str,
    candidate_id: Option<&str>,
) -> (Arc<StreamCandidateWatchdogProgress>, bool) {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return (StreamCandidateWatchdogProgress::shared(), false);
    }
    let key = watchdog_progress_registry_key(request_id, candidate_id);
    if let Ok(mut registry) = watchdog_progress_registry().lock() {
        registry.retain(|_, entry| entry.registered_at.elapsed() <= WATCHDOG_PROGRESS_REGISTRY_TTL);
        if let Some(entry) = registry.get(&key) {
            return (Arc::clone(&entry.progress), false);
        }
        let progress = StreamCandidateWatchdogProgress::shared();
        registry.insert(
            key,
            RegisteredWatchdogProgress {
                progress: Arc::clone(&progress),
                registered_at: Instant::now(),
            },
        );
        return (progress, true);
    }

    // A poisoned registry must not prevent the request from acquiring a
    // local owner.  There is no entry to unregister in this fallback path.
    (StreamCandidateWatchdogProgress::shared(), false)
}

pub(crate) fn stream_candidate_watchdog_progress_for_request(
    request_id: &str,
    candidate_id: Option<&str>,
) -> Option<Arc<StreamCandidateWatchdogProgress>> {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return None;
    }
    watchdog_progress_registry()
        .lock()
        .ok()
        .and_then(|registry| {
            registry
                .get(&watchdog_progress_registry_key(request_id, candidate_id))
                .map(|entry| Arc::clone(&entry.progress))
        })
}

/// Claim terminal ownership on an already-resolved watchdog progress object.
///
/// Terminal writers must use the exact `Arc` that was resolved by their
/// caller. Looking the owner up a second time by request id leaves a race in
/// which a concurrent cleanup/re-registration can make the helper silently
/// bypass the compare/exchange and emit a competing terminal event.
pub(crate) fn mark_stream_candidate_watchdog_terminal_started_with_progress(
    progress: Option<&Arc<StreamCandidateWatchdogProgress>>,
) -> bool {
    progress
        .map(|progress| progress.mark_terminal_started())
        .unwrap_or(true)
}

pub(crate) fn unregister_stream_candidate_watchdog_progress(
    request_id: &str,
    candidate_id: Option<&str>,
    progress: &Arc<StreamCandidateWatchdogProgress>,
) {
    let Ok(mut registry) = watchdog_progress_registry().lock() else {
        return;
    };
    let key = watchdog_progress_registry_key(request_id, candidate_id);
    if registry
        .get(&key)
        .is_some_and(|entry| Arc::ptr_eq(&entry.progress, progress))
    {
        registry.remove(&key);
    }
}

/// Atomically move an owner from a request-level registry key to its
/// candidate-specific key.  `StreamAttemptTerminalGuard::arm_for` performs
/// this transition after candidate-slot creation; removing the old key and
/// inserting the new one in separate lock sections briefly made terminal
/// helpers observe no owner and bypass their CAS.
pub(crate) fn rekey_stream_candidate_watchdog_progress(
    request_id: &str,
    from_candidate_id: Option<&str>,
    to_candidate_id: Option<&str>,
    progress: Arc<StreamCandidateWatchdogProgress>,
) -> Arc<StreamCandidateWatchdogProgress> {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return progress;
    }
    let from_key = watchdog_progress_registry_key(request_id, from_candidate_id);
    let to_key = watchdog_progress_registry_key(request_id, to_candidate_id);
    if from_key == to_key {
        return progress;
    }
    let Ok(mut registry) = watchdog_progress_registry().lock() else {
        return progress;
    };
    registry.retain(|_, entry| entry.registered_at.elapsed() <= WATCHDOG_PROGRESS_REGISTRY_TTL);

    // If a candidate-specific owner already exists, it is canonical.  Never
    // overwrite it with a second Arc; callers must continue using that shared
    // state so terminal ownership remains a single CAS domain.
    if let Some(entry) = registry.get(&to_key) {
        let canonical = Arc::clone(&entry.progress);
        if registry
            .get(&from_key)
            .is_some_and(|source| Arc::ptr_eq(&source.progress, &canonical))
        {
            registry.remove(&from_key);
        }
        return canonical;
    }

    let canonical = registry
        .get(&from_key)
        .filter(|entry| Arc::ptr_eq(&entry.progress, &progress))
        .map(|entry| Arc::clone(&entry.progress))
        .unwrap_or(progress);
    registry.insert(
        to_key,
        RegisteredWatchdogProgress {
            progress: Arc::clone(&canonical),
            registered_at: Instant::now(),
        },
    );
    if registry
        .get(&from_key)
        .is_some_and(|entry| Arc::ptr_eq(&entry.progress, &canonical))
    {
        registry.remove(&from_key);
    }
    canonical
}

/// Build the client-visible transport response without taking ownership of
/// the request's terminal usage event.  The stream watchdog uses this only
/// after an already-started terminal finalizer has been detached: emitting a
/// second terminal usage event there could overwrite the real provider
/// outcome (or double-settle billing) merely because persistence was slow.
pub(crate) fn build_transport_error_client_response(
    plan: &aether_contracts::ExecutionPlan,
    trace_id: &str,
    decision: &GatewayControlDecision,
    client_status_code: u16,
) -> Result<Response<Body>, GatewayError> {
    let client_body = build_core_error_body_for_client_format(
        &plan.client_api_format,
        TRANSPORT_ERROR_CLIENT_MESSAGE,
        Some("upstream_transport_error"),
        LocalCoreSyncErrorKind::ServerError,
    )
    .unwrap_or_else(|| {
        json!({
            "error": {
                "type": "server_error",
                "message": TRANSPORT_ERROR_CLIENT_MESSAGE,
                "code": "upstream_transport_error",
            }
        })
    });
    let body_bytes =
        serde_json::to_vec(&client_body).map_err(|err| GatewayError::Internal(err.to_string()))?;
    let headers = BTreeMap::from([
        ("content-type".to_string(), "application/json".to_string()),
        ("content-length".to_string(), body_bytes.len().to_string()),
    ]);
    attach_control_metadata_headers(
        build_client_response_from_parts(
            client_status_code,
            &headers,
            Body::from(body_bytes),
            trace_id,
            Some(decision),
        )?,
        Some(plan.request_id.as_str()),
        plan.candidate_id.as_deref(),
    )
}

const WATCHDOG_STATE_NONE: u8 = 0;
const WATCHDOG_STATE_TERMINAL_STARTED: u8 = 1;
const WATCHDOG_STATE_RETRY_ABORT: u8 = 2;
const WATCHDOG_STATE_CANCEL_FALLBACK: u8 = 3;
const WATCHDOG_STATE_STOP_REQUESTED: u8 = 4;
const WATCHDOG_STATE_STOP_FALLBACK: u8 = 5;
const WATCHDOG_STATE_RETRY_ABORT_EXTERNAL: u8 = 6;

/// Shared ownership state for the small race window around a stream watchdog
/// timeout.  A boolean was not sufficient: the watchdog can abort an
/// intermediate candidate while the stream task is concurrently starting a
/// terminal handoff, and a cancellation guard may also be dropping at the
/// same time.  Exactly one of those owners must win the compare/exchange.
#[derive(Debug)]
pub(crate) struct StreamCandidateWatchdogProgress {
    state: AtomicU8,
    /// Set after the intermediate candidate transition has returned from its
    /// persistence call.  A stream task that is dropped while that call is in
    /// flight can then safely promote RETRY_ABORT to the 499 fallback.
    retry_handoff_complete: std::sync::atomic::AtomicBool,
}

impl Default for StreamCandidateWatchdogProgress {
    fn default() -> Self {
        Self {
            state: AtomicU8::new(WATCHDOG_STATE_NONE),
            retry_handoff_complete: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

tokio::task_local! {
    static STREAM_CANDIDATE_WATCHDOG_PROGRESS: Arc<StreamCandidateWatchdogProgress>;
}

impl StreamCandidateWatchdogProgress {
    pub(crate) fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Claim terminal ownership for an execution finalizer.  A retry abort or
    /// cancellation fallback that won first is never overwritten.
    pub(crate) fn mark_terminal_started(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_NONE,
                WATCHDOG_STATE_TERMINAL_STARTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn terminal_started(&self) -> bool {
        self.state.load(Ordering::Acquire) == WATCHDOG_STATE_TERMINAL_STARTED
    }

    pub(crate) fn terminal_owner_active(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire),
            WATCHDOG_STATE_TERMINAL_STARTED
                | WATCHDOG_STATE_CANCEL_FALLBACK
                | WATCHDOG_STATE_STOP_FALLBACK
        )
    }

    /// Claim an intermediate retry abort.  If a terminal finalizer has
    /// already started, the caller must not abort it as a retry.
    pub(crate) fn try_mark_retry_abort(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_NONE,
                WATCHDOG_STATE_RETRY_ABORT,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn retry_abort(&self) -> bool {
        matches!(
            self.state.load(Ordering::Acquire),
            WATCHDOG_STATE_RETRY_ABORT | WATCHDOG_STATE_RETRY_ABORT_EXTERNAL
        )
    }

    /// Claim the request-level cancellation fallback after a watchdog had
    /// reserved an intermediate retry but the retry owner itself was dropped
    /// before it could publish the candidate transition.  The retry state is
    /// intentionally not treated as terminal by the stream execution guard:
    /// normal watchdog timeouts still need to return a candidate retry.  A
    /// separate owner (the candidate-loop handoff guard) uses this CAS only
    /// when that retry handoff is cancelled, closing the small RETRY_ABORT
    /// window without manufacturing a competing 499 during normal failover.
    pub(crate) fn try_claim_retry_cancel_fallback(&self) -> bool {
        if self.retry_handoff_complete.load(Ordering::Acquire) {
            return false;
        }
        loop {
            let current = self.state.load(Ordering::Acquire);
            if !matches!(
                current,
                WATCHDOG_STATE_RETRY_ABORT | WATCHDOG_STATE_RETRY_ABORT_EXTERNAL
            ) {
                return false;
            }
            match self.state.compare_exchange(
                current,
                WATCHDOG_STATE_CANCEL_FALLBACK,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(_) => {
                    if self.retry_handoff_complete.load(Ordering::Acquire) {
                        return false;
                    }
                }
            }
        }
    }

    pub(crate) fn retry_owner_external(&self) -> bool {
        self.state.load(Ordering::Acquire) == WATCHDOG_STATE_RETRY_ABORT_EXTERNAL
    }

    /// Atomically reserve an intermediate retry whose candidate write will be
    /// performed by the outer watchdog task after the stream task is aborted.
    /// Encoding this owner kind in the shared state prevents the stream
    /// cancellation guard from observing a bare RETRY_ABORT before the outer
    /// owner has marked itself external.
    pub(crate) fn try_mark_retry_abort_external(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_NONE,
                WATCHDOG_STATE_RETRY_ABORT_EXTERNAL,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn mark_retry_handoff_complete(&self) -> bool {
        if !self.retry_abort() {
            return false;
        }
        self.retry_handoff_complete.store(true, Ordering::Release);
        true
    }

    pub(crate) fn retry_handoff_complete(&self) -> bool {
        self.retry_handoff_complete.load(Ordering::Acquire)
    }

    pub(crate) fn cancel_fallback(&self) -> bool {
        self.state.load(Ordering::Acquire) == WATCHDOG_STATE_CANCEL_FALLBACK
    }

    pub(crate) fn stop_requested(&self) -> bool {
        self.state.load(Ordering::Acquire) == WATCHDOG_STATE_STOP_REQUESTED
    }

    pub(crate) fn stop_fallback(&self) -> bool {
        self.state.load(Ordering::Acquire) == WATCHDOG_STATE_STOP_FALLBACK
    }

    /// Reserve a stop-on-transport terminal handoff before aborting the
    /// execution task.  The guard can then convert the reservation into a
    /// detached failed/504 handoff if the outer request is cancelled.
    pub(crate) fn try_mark_stop_requested(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_NONE,
                WATCHDOG_STATE_STOP_REQUESTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn try_claim_stop_fallback(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_STOP_REQUESTED,
                WATCHDOG_STATE_STOP_FALLBACK,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) fn try_claim_stop_terminal(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_STOP_REQUESTED,
                WATCHDOG_STATE_TERMINAL_STARTED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Claim the cancellation/499 fallback.  The transport timeout helper
    /// observes this claim and returns only a client response, preventing a
    /// second terminal usage event.
    pub(crate) fn try_claim_cancel_fallback(&self) -> bool {
        self.state
            .compare_exchange(
                WATCHDOG_STATE_NONE,
                WATCHDOG_STATE_CANCEL_FALLBACK,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    pub(crate) async fn scope<F>(self: Arc<Self>, future: F) -> F::Output
    where
        F: Future,
    {
        STREAM_CANDIDATE_WATCHDOG_PROGRESS.scope(self, future).await
    }
}

pub(crate) fn current_stream_candidate_watchdog_progress(
) -> Option<Arc<StreamCandidateWatchdogProgress>> {
    STREAM_CANDIDATE_WATCHDOG_PROGRESS.try_with(Arc::clone).ok()
}

pub(crate) fn stream_candidate_watchdog_progress_for_current_or_request(
    request_id: &str,
    candidate_id: Option<&str>,
) -> Option<Arc<StreamCandidateWatchdogProgress>> {
    // A non-empty request id is an explicit identity boundary.  Never fall
    // back to an inherited task-local owner when its exact registry entry is
    // absent: body pumps and detached finalizers can outlive the candidate
    // task, and a stale task-local Arc could otherwise claim a different
    // request/candidate's terminal state.  Callers that intentionally operate
    // inside the current candidate scope can use
    // `current_stream_candidate_watchdog_progress()` directly.
    if !request_id.trim().is_empty() {
        return stream_candidate_watchdog_progress_for_request(request_id, candidate_id);
    }
    current_stream_candidate_watchdog_progress()
}

pub(crate) fn mark_stream_candidate_watchdog_terminal_started() -> bool {
    current_stream_candidate_watchdog_progress()
        .map(|progress| progress.mark_terminal_started())
        .unwrap_or(true)
}

pub(crate) fn mark_stream_candidate_watchdog_terminal_started_for_request(
    request_id: &str,
    candidate_id: Option<&str>,
) -> bool {
    stream_candidate_watchdog_progress_for_current_or_request(request_id, candidate_id)
        .map(|progress| progress.mark_terminal_started())
        .unwrap_or(true)
}

/// Returns whether the current stream-candidate watchdog has already taken
/// ownership of terminal handling.  Cancellation guards use this to avoid
/// manufacturing a competing 499 while the watchdog is about to publish its
/// own timeout/transport outcome.
pub(crate) fn stream_candidate_watchdog_terminal_started() -> bool {
    STREAM_CANDIDATE_WATCHDOG_PROGRESS
        .try_with(|progress| progress.terminal_started())
        .unwrap_or(false)
}

pub(crate) fn stream_candidate_watchdog_retry_aborted() -> bool {
    STREAM_CANDIDATE_WATCHDOG_PROGRESS
        .try_with(|progress| progress.retry_abort())
        .unwrap_or(false)
}

/// Mark an intermediate candidate transition complete for the watchdog owner
/// registered under this request/candidate pair.  Candidate persistence calls
/// use this after their await returns so a stream task dropped during the
/// write can be distinguished from a normally completed retry handoff.
pub(crate) fn mark_stream_candidate_watchdog_retry_handoff_complete_for_request(
    request_id: &str,
    candidate_id: Option<&str>,
) {
    if let Some(progress) =
        stream_candidate_watchdog_progress_for_current_or_request(request_id, candidate_id)
    {
        // Keep the completed retry tombstone registered for the remainder of
        // its TTL. Detached body/error tasks can still report after the
        // candidate write returns; removing the key would let one of them
        // create a fresh owner and emit a competing request terminal event.
        // Candidate ids are attempt-scoped, so retaining this small entry does
        // not block a later candidate from taking over the request.
        let _ = progress.mark_retry_handoff_complete();
    }
}

pub(crate) fn stream_candidate_watchdog_stop_requested() -> bool {
    STREAM_CANDIDATE_WATCHDOG_PROGRESS
        .try_with(|progress| progress.stop_requested())
        .unwrap_or(false)
}

pub(crate) fn stream_candidate_watchdog_allows_intermediate_for_request(
    request_id: &str,
    candidate_id: Option<&str>,
) -> bool {
    stream_candidate_watchdog_progress_for_current_or_request(request_id, candidate_id).is_none_or(
        |progress| {
            !progress.terminal_owner_active()
                && !progress.cancel_fallback()
                && !progress.stop_requested()
                && !progress.stop_fallback()
        },
    )
}

/// Reserve the shared watchdog slot for an intermediate candidate retry.  The
/// reservation makes the retry decision mutually exclusive with a concurrent
/// 499/504 terminal fallback; callers must not write a Failed candidate when
/// this returns false.
pub(crate) fn try_claim_stream_candidate_intermediate_for_request(
    request_id: &str,
    candidate_id: Option<&str>,
) -> bool {
    let Some(progress) =
        stream_candidate_watchdog_progress_for_current_or_request(request_id, candidate_id)
    else {
        // Direct/internal paths without a watchdog still retain their normal
        // failover behavior.
        return true;
    };
    // Only the caller that wins the CAS may publish the intermediate
    // candidate failure.  Returning `true` for an already-owned retry made
    // concurrent prefetch/error handlers both write a Failed row and could
    // race a request-level terminal fallback.
    progress.try_mark_retry_abort()
}

/// Try to claim the request-level transport terminal owner.  `false` means a
/// stream finalizer, retry abort, or cancellation fallback already owns the
/// lifecycle and the caller must only build a client response.
pub(crate) fn try_claim_stream_candidate_terminal_owner() -> bool {
    try_claim_stream_candidate_terminal_owner_with_progress(
        current_stream_candidate_watchdog_progress(),
    )
}

fn try_claim_stream_candidate_terminal_owner_with_progress(
    progress: Option<Arc<StreamCandidateWatchdogProgress>>,
) -> bool {
    progress
        .map(|progress| {
            if progress.stop_requested() {
                // The watchdog reserved this owner immediately before
                // aborting.  Promote that reservation to the actual terminal
                // owner so a dropping stream guard cannot start a duplicate.
                progress.try_claim_stop_terminal()
            } else if progress.terminal_started()
                || progress.cancel_fallback()
                || progress.stop_fallback()
                || progress.retry_abort()
            {
                false
            } else {
                progress.mark_terminal_started()
            }
        })
        .unwrap_or(true)
}

#[allow(clippy::too_many_arguments)]
async fn record_transport_error_terminal(
    state: &AppState,
    plan: &aether_contracts::ExecutionPlan,
    report_context: Option<&Value>,
    client_status_code: u16,
    error_type: &str,
    error_message: &str,
    elapsed_ms: u64,
    watchdog_progress: Option<Arc<StreamCandidateWatchdogProgress>>,
) -> bool {
    let client_body = build_core_error_body_for_client_format(
        &plan.client_api_format,
        TRANSPORT_ERROR_CLIENT_MESSAGE,
        Some("upstream_transport_error"),
        LocalCoreSyncErrorKind::ServerError,
    )
    .unwrap_or_else(|| {
        json!({
            "error": {
                "type": "server_error",
                "message": TRANSPORT_ERROR_CLIENT_MESSAGE,
                "code": "upstream_transport_error",
            }
        })
    });
    let usage_handoff_persisted = if state.usage_runtime.is_enabled() {
        let report_context_with_diagnostics =
            attach_current_request_diagnostics_and_candidate_timing_to_report_context(
                report_context,
                Some(elapsed_ms),
                None,
            );
        let mut usage_data = build_usage_event_data_seed(
            plan,
            report_context_with_diagnostics.as_ref().or(report_context),
        );
        usage_data.status_code = Some(client_status_code);
        usage_data.error_message = Some(error_message.to_string());
        usage_data.error_category = Some("server_error".to_string());
        usage_data.response_time_ms = Some(elapsed_ms);
        usage_data.response_headers = Some(json!({"content-type": "application/json"}));
        usage_data.response_body = Some(client_body.clone());
        usage_data.client_response_headers = Some(json!({"content-type": "application/json"}));
        usage_data.client_response_body = Some(client_body);
        let mut request_metadata = match usage_data.request_metadata.take() {
            Some(Value::Object(object)) => object,
            Some(other) => serde_json::Map::from_iter([("seed".to_string(), other)]),
            None => serde_json::Map::new(),
        };
        request_metadata.insert("transport_error".to_string(), Value::Bool(true));
        request_metadata.insert(
            "transport_error_type".to_string(),
            Value::String(error_type.to_string()),
        );
        usage_data.request_metadata = Some(Value::Object(request_metadata));
        state
            .usage_runtime
            .record_terminal_event_direct_with_handoff(
                state.usage_lifecycle_data_state().as_ref(),
                UsageEvent::new(UsageEventType::Failed, plan.request_id.clone(), usage_data),
            )
            .await
    } else {
        true
    };

    // A transport watchdog can be the only terminal path for an attempt. Keep
    // its candidate transition behind the same usage handoff as normal sync
    // and stream finalizers; otherwise a 502/504 candidate can become terminal
    // while the usage row is still the pending skeleton.
    let candidate_status = usage_handoff_persisted
        .then_some(RequestCandidateStatus::Failed)
        .unwrap_or(RequestCandidateStatus::Streaming);
    let terminal_at_unix_ms = current_unix_ms();
    let terminal_reconciliation_update = SchedulerRequestCandidateStatusUpdate {
        status: RequestCandidateStatus::Failed,
        status_code: Some(client_status_code),
        error_type: Some(error_type.to_string()),
        error_message: Some(error_message.to_string()),
        latency_ms: Some(elapsed_ms),
        started_at_unix_ms: Some(terminal_at_unix_ms.saturating_sub(elapsed_ms)),
        finished_at_unix_ms: Some(terminal_at_unix_ms),
    };
    let terminal_update = SchedulerRequestCandidateStatusUpdate {
        status: candidate_status,
        status_code: Some(client_status_code),
        error_type: Some(error_type.to_string()),
        error_message: Some(if usage_handoff_persisted {
            error_message.to_string()
        } else {
            format!("{error_message}; terminal usage persistence was not confirmed")
        }),
        latency_ms: Some(elapsed_ms),
        started_at_unix_ms: Some(terminal_at_unix_ms.saturating_sub(elapsed_ms)),
        finished_at_unix_ms: usage_handoff_persisted.then_some(terminal_at_unix_ms),
    };
    if !usage_handoff_persisted {
        spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            terminal_reconciliation_update.clone(),
        );
    }
    let candidate_persisted =
        record_local_request_candidate_status(state, plan, report_context, terminal_update).await;
    if usage_handoff_persisted && !candidate_persisted {
        spawn_candidate_persistence_retry_after_usage_handoff(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            terminal_reconciliation_update,
            watchdog_progress.clone(),
        );
    }

    // Keep the exact owner registered until the candidate write above has
    // returned.  Removing it immediately after the usage write lets a
    // concurrent finalizer create a fresh progress object and bypass the
    // terminal CAS while the candidate row is still being persisted.
    if usage_handoff_persisted && candidate_persisted {
        if let Some(progress) = watchdog_progress {
            unregister_stream_candidate_watchdog_progress(
                plan.request_id.as_str(),
                plan.candidate_id.as_deref(),
                &progress,
            );
        }
    }

    usage_handoff_persisted
}

/// Spawn a process-lifetime transport terminal handoff.  This is used by the
/// stream cancellation guard after a watchdog has reserved a stop timeout but
/// the request task itself is being dropped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_transport_error_terminal_handoff(
    state: AppState,
    plan: aether_contracts::ExecutionPlan,
    report_context: Option<Value>,
    client_status_code: u16,
    error_type: String,
    error_message: String,
    elapsed_ms: u64,
    watchdog_progress: Option<Arc<StreamCandidateWatchdogProgress>>,
) {
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _ = record_transport_error_terminal(
            &state,
            &plan,
            report_context.as_ref(),
            client_status_code,
            error_type.as_str(),
            error_message.as_str(),
            elapsed_ms,
            watchdog_progress,
        )
        .await;
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_transport_error_stop_response(
    state: &AppState,
    plan: &aether_contracts::ExecutionPlan,
    report_context: Option<&Value>,
    trace_id: &str,
    decision: &GatewayControlDecision,
    client_status_code: u16,
    error_type: &str,
    error_message: &str,
    elapsed_ms: u64,
) -> Result<Response<Body>, GatewayError> {
    let watchdog_progress = stream_candidate_watchdog_progress_for_current_or_request(
        plan.request_id.as_str(),
        plan.candidate_id.as_deref(),
    );
    build_transport_error_stop_response_with_progress(
        state,
        plan,
        report_context,
        trace_id,
        decision,
        client_status_code,
        error_type,
        error_message,
        elapsed_ms,
        watchdog_progress,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_transport_error_stop_response_with_progress(
    state: &AppState,
    plan: &aether_contracts::ExecutionPlan,
    report_context: Option<&Value>,
    trace_id: &str,
    decision: &GatewayControlDecision,
    client_status_code: u16,
    error_type: &str,
    error_message: &str,
    elapsed_ms: u64,
    progress: Option<Arc<StreamCandidateWatchdogProgress>>,
) -> Result<Response<Body>, GatewayError> {
    let progress = progress.or_else(|| {
        stream_candidate_watchdog_progress_for_current_or_request(
            plan.request_id.as_str(),
            plan.candidate_id.as_deref(),
        )
    });
    if !try_claim_stream_candidate_terminal_owner_with_progress(progress.clone()) {
        return build_transport_error_client_response(plan, trace_id, decision, client_status_code);
    }
    let _ = record_transport_error_terminal(
        state,
        plan,
        report_context,
        client_status_code,
        error_type,
        error_message,
        elapsed_ms,
        progress,
    )
    .await;

    // `build_transport_error_client_response` deterministically builds the
    // same client payload and owns the shared response-header construction.
    build_transport_error_client_response(plan, trace_id, decision, client_status_code)
}

#[cfg(test)]
mod tests {
    use super::StreamCandidateWatchdogProgress;

    #[test]
    fn intermediate_retry_claim_has_one_owner_and_handoff_closes_fallback() {
        let progress = StreamCandidateWatchdogProgress::default();

        // A second failure observer must not publish another Failed candidate
        // update for the same attempt.  The first CAS owns the retry lane.
        assert!(progress.try_mark_retry_abort());
        assert!(!progress.try_mark_retry_abort());

        // Once the candidate write returns, cancellation must not manufacture
        // a competing 499 fallback.
        assert!(progress.mark_retry_handoff_complete());
        assert!(!progress.try_claim_retry_cancel_fallback());
    }
}
