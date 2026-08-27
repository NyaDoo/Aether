//! Shared, short-window dashboard request/token metrics.
//!
//! The historical dashboard card queried persisted usage rows.  That made the
//! value laggy, double-counted failed attempts, and was inherently process
//! local when the gateway ran without a shared counter store.  This module
//! keeps a real rolling window in `RuntimeState`: Redis uses atomic Lua event
//! updates, while the memory backend is an explicit single-process fallback.

use std::sync::Arc;
use std::time::Duration;

use aether_runtime_state::{DataLayerError, RealtimeBucket, RuntimeState};
use aether_usage_runtime::{
    spawn_on_usage_background_runtime, UsageEvent, UsageEventData, UsageEventType,
    UsageRealtimeMetricsSink, UsageRealtimeTokenDelta,
};
use chrono::{TimeZone, Utc};
use serde::Serialize;
use uuid::Uuid;

pub(crate) const REALTIME_WINDOW_SECONDS: u64 = 60;
pub(crate) const REALTIME_WINDOW_MILLIS: u64 = REALTIME_WINDOW_SECONDS.saturating_mul(1_000);
/// Internal lifecycle marker used by a provider-attempt retry.  A transparent
/// quota retry is still one client request for RPM purposes; the first
/// attempt's terminal failure must therefore not remove the logical admission
/// before the replacement attempt has had a chance to finish.
pub(crate) const REALTIME_ADMISSION_DEFER_FAILURE_METADATA_KEY: &str =
    "realtime_admission_defer_failure";
const REALTIME_ATTEMPT_ID_METADATA_KEY: &str = "provider_request_order_id";
const REALTIME_BUCKET_TTL_SECONDS: u64 = REALTIME_WINDOW_SECONDS.saturating_mul(2);
// Admission markers must survive the longest configured execution so a late
// terminal event can still settle the original request identity. The exact
// event itself is retained only for the rolling window; keeping the marker
// longer does not affect the exported RPM.
const REALTIME_MARKER_TTL_SECONDS: u64 =
    aether_contracts::MAX_EXECUTION_REQUEST_TIMEOUT_SECS + REALTIME_WINDOW_SECONDS;
const REALTIME_KEY_PREFIX: &str = "dashboard:realtime:v1";
const REALTIME_EVENT_TTL_SECONDS: u64 = REALTIME_BUCKET_TTL_SECONDS;
// The reconciliation state outlives an individual stream callback so a
// delayed terminal lifecycle event can still claim only its unseen remainder.
const REALTIME_TOKEN_LEDGER_TTL_SECONDS: u64 =
    aether_contracts::MAX_EXECUTION_REQUEST_TIMEOUT_SECS + REALTIME_WINDOW_SECONDS;
const REALTIME_TOKEN_WRITE_RETRIES: usize = 3;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct RealtimeSemantics {
    pub rpm: &'static str,
    pub tpm: &'static str,
    pub window: &'static str,
    pub failed_requests: &'static str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct RealtimeMetrics {
    pub rpm: u64,
    pub tpm: u64,
    pub window_seconds: u64,
    pub as_of: String,
    pub semantics: RealtimeSemantics,
    pub storage_scope: &'static str,
}

/// Runtime-owned observer installed on `UsageRuntime`.  It contains only the
/// shared counter backend; the HTTP response DTO above stays independent from
/// the write path.
#[derive(Debug, Clone)]
pub(crate) struct RealtimeMetricsRecorder {
    runtime: Arc<RuntimeState>,
}

impl RealtimeMetricsRecorder {
    pub(crate) fn new(runtime: Arc<RuntimeState>) -> Self {
        Self { runtime }
    }

    pub(crate) fn runtime(&self) -> &RuntimeState {
        self.runtime.as_ref()
    }
}

impl UsageRealtimeMetricsSink for RealtimeMetricsRecorder {
    fn observe_usage_event(&self, event: &UsageEvent) {
        // Copy only the fields needed by the counter task.  Usage events may
        // contain captured request/response bodies and must not be cloned in
        // their entirety for a dashboard metric.
        let observation = UsageRealtimeObservation {
            request_id: event.request_id.clone(),
            admission_id: realtime_admission_id_from_metadata(event.data.request_metadata.as_ref()),
            attempt_id: realtime_attempt_id_from_metadata(event.data.request_metadata.as_ref()),
            defer_admission_failure: realtime_admission_failure_deferred(
                event.data.request_metadata.as_ref(),
            ),
            event_type: event.event_type,
            timestamp_ms: event.timestamp_ms,
            token_delta: token_delta_from_usage_data(&event.data),
            // Candidate IDs are stable across lifecycle retries and distinguish
            // separate provider attempts that happen to share the client trace
            // ID.  Keep this small identity fragment instead of cloning the
            // full usage payload onto the background task.
            candidate_id: event
                .data
                .candidate_id
                .clone()
                .or_else(|| candidate_id_from_metadata(event.data.request_metadata.as_ref())),
        };
        let runtime = Arc::clone(&self.runtime);
        spawn_on_usage_background_runtime(async move {
            if let Err(err) = record_usage_observation(runtime.as_ref(), &observation).await {
                tracing::warn!(
                    event_name = "dashboard_realtime_usage_observation_failed",
                    log_type = "ops",
                    request_id = %observation.request_id,
                    error = %err,
                    "failed to update dashboard realtime usage counters"
                );
            }
        });
    }

    fn observe_token_delta(&self, observation: UsageRealtimeTokenDelta) {
        // Streaming observers emit already de-duplicated increments before a
        // terminal UsageEvent exists.  Keep these writes on the same shared
        // event collection as lifecycle tokens, but use a deterministic
        // sequence identity whenever the transport supplied one so retries of
        // a frame cannot inflate TPM.
        let event_id = stream_token_event_id(&observation);
        let ledger_identity = token_ledger_identity_from_parts(
            &observation.request_id,
            observation.admission_id.as_deref(),
            observation.candidate_id.as_deref(),
            observation.attempt_id.as_deref(),
        );
        let runtime = Arc::clone(&self.runtime);
        spawn_on_usage_background_runtime(async move {
            if let Err(err) = record_stream_token_observation(
                runtime.as_ref(),
                &ledger_identity,
                &event_id,
                observation.timestamp_ms,
                observation.token_delta,
            )
            .await
            {
                tracing::warn!(
                    event_name = "dashboard_realtime_stream_token_observation_failed",
                    log_type = "ops",
                    request_id = %observation.request_id,
                    error = %err,
                    "failed to update dashboard realtime streaming token counters"
                );
            }
        });
    }
}

#[derive(Debug, Clone)]
struct UsageRealtimeObservation {
    request_id: String,
    admission_id: Option<String>,
    attempt_id: Option<String>,
    defer_admission_failure: bool,
    event_type: UsageEventType,
    timestamp_ms: u64,
    token_delta: u64,
    candidate_id: Option<String>,
}

impl RealtimeMetrics {
    fn from_bucket(
        bucket: RealtimeBucket,
        as_of_unix_secs: u64,
        storage_scope: &'static str,
    ) -> Self {
        Self::from_bucket_at_millis(bucket, as_of_unix_secs.saturating_mul(1_000), storage_scope)
    }

    fn from_bucket_at_millis(
        bucket: RealtimeBucket,
        as_of_unix_ms: u64,
        storage_scope: &'static str,
    ) -> Self {
        let as_of = Utc
            .timestamp_millis_opt(i64::try_from(as_of_unix_ms).unwrap_or(i64::MAX))
            .single()
            .unwrap_or_else(Utc::now)
            .to_rfc3339();
        Self {
            rpm: bucket.requests.max(0) as u64,
            tpm: bucket.tokens.max(0) as u64,
            window_seconds: REALTIME_WINDOW_SECONDS,
            as_of,
            semantics: RealtimeSemantics {
                rpm: "accepted_non_failed_requests",
                // TPM is an observation stream: tokens already emitted by a
                // provider remain visible even if the eventual request
                // settles as failed/cancelled.  The RPM admission is the
                // metric that is compensated on failure.
                tpm: "observed_token_deltas_including_failed",
                window: "trailing_60_seconds",
                failed_requests: "excluded_from_rpm_only",
            },
            storage_scope,
        }
    }
}

/// A request reservation made after authentication, owner forwarding, and
/// frontdoor rate limiting have succeeded.  Dropping an armed guard schedules
/// a compensating decrement; terminal lifecycle code can instead call
/// `handoff` and settle the marker explicitly.
#[derive(Debug)]
pub(crate) struct RealtimeAdmissionGuard {
    runtime: RuntimeState,
    request_id: String,
    armed: bool,
}

impl RealtimeAdmissionGuard {
    /// Leave the marker for the terminal lifecycle sink. The guard no longer
    /// compensates on drop, because the sink owns the eventual outcome.
    pub(crate) fn handoff(mut self) {
        self.armed = false;
    }

    /// Settle this reservation immediately. `failed=true` removes the RPM
    /// admission; successful requests add the supplied token delta.
    pub(crate) async fn settle(
        mut self,
        failed: bool,
        token_delta: u64,
    ) -> Result<bool, DataLayerError> {
        self.armed = false;
        settle_admission(&self.runtime, &self.request_id, failed, token_delta).await
    }

    pub(crate) fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl Drop for RealtimeAdmissionGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let runtime = self.runtime.clone();
        let request_id = self.request_id.clone();
        // Use the process-lifetime usage runtime so a dropped request future
        // during shutdown still has a chance to compensate the shared bucket.
        aether_usage_runtime::spawn_on_usage_background_runtime(async move {
            if let Err(err) = settle_admission(&runtime, &request_id, true, 0).await {
                tracing::warn!(
                    event_name = "dashboard_realtime_admission_rollback_failed",
                    log_type = "ops",
                    request_id = %request_id,
                    error = %err,
                    "failed to compensate dropped realtime admission"
                );
            }
        });
    }
}

/// Reserve one accepted AI request at the current millisecond. A duplicate
/// request ID is ignored, making retries/duplicate terminal reports safe.
pub(crate) async fn reserve_admission(
    runtime: &RuntimeState,
    request_id: &str,
) -> Result<Option<RealtimeAdmissionGuard>, DataLayerError> {
    reserve_admission_at(runtime, request_id, current_unix_millis()).await
}

/// Reserve one accepted AI request at an explicit millisecond timestamp.
///
/// The marker keeps both the event identity and its timestamp (`event_id|
/// timestamp_ms`) so a later terminal lifecycle event can remove exactly the
/// admission event, even when the request crosses a second boundary.  The
/// event store performs the idempotent insert atomically across gateway
/// instances.
pub(crate) async fn reserve_admission_at(
    runtime: &RuntimeState,
    request_id: &str,
    timestamp_ms: u64,
) -> Result<Option<RealtimeAdmissionGuard>, DataLayerError> {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return Ok(None);
    }
    let marker_key = admission_marker_key(request_id);
    let event_id = admission_event_id(request_id);
    let marker_ttl = Duration::from_secs(REALTIME_MARKER_TTL_SECONDS);
    if !runtime
        .kv_set_if_absent(
            &marker_key,
            admission_marker_value(&event_id, timestamp_ms),
            marker_ttl,
        )
        .await?
    {
        return Ok(None);
    }

    match runtime
        .realtime_event_add(
            REALTIME_KEY_PREFIX,
            &event_id,
            timestamp_ms,
            1,
            0,
            Duration::from_secs(REALTIME_EVENT_TTL_SECONDS),
        )
        .await
    {
        Ok(true) => Ok(Some(RealtimeAdmissionGuard {
            runtime: runtime.clone(),
            request_id: request_id.to_string(),
            armed: true,
        })),
        Ok(false) => {
            // The marker may have expired while the event remained live (or
            // an operator may have restored only one side of the pair).  Do
            // not leave a fresh marker for a reservation we did not win.
            let _ = runtime.kv_delete(&marker_key).await;
            Ok(None)
        }
        Err(err) => {
            // Do not leave an admission marker behind when the event write was
            // unavailable; a later retry should be able to reserve normally.
            let _ = runtime.kv_delete(&marker_key).await;
            Err(err)
        }
    }
}

/// Apply a terminal result to a prior admission marker. Returns `true` only
/// when this call consumed the marker (i.e. it won the idempotency race).
pub(crate) async fn settle_admission(
    runtime: &RuntimeState,
    request_id: &str,
    failed: bool,
    token_delta: u64,
) -> Result<bool, DataLayerError> {
    let request_id = request_id.trim();
    if request_id.is_empty() {
        return Ok(false);
    }
    let marker_key = admission_marker_key(request_id);
    let Some(marker_value) = runtime.kv_take(&marker_key).await? else {
        return Ok(false);
    };
    let marker = parse_admission_marker(&marker_value);
    let settlement_timestamp_ms = match &marker {
        AdmissionMarker::Event { timestamp_ms, .. } => *timestamp_ms,
        AdmissionMarker::LegacyBucket(bucket) => bucket.saturating_mul(1_000),
    };

    // A direct guard settlement can carry a terminal token total. Reconcile it
    // against any live stream increments before removing/consuming the
    // admission marker so emitted tokens remain visible even when the request
    // ultimately fails.
    let token_result = if token_delta > 0 {
        let ledger_identity =
            token_ledger_identity_from_parts(request_id, Some(request_id), None, None);
        let token_event_id = settlement_token_event_id(request_id);
        record_terminal_token_remainder(
            runtime,
            &ledger_identity,
            &token_event_id,
            settlement_timestamp_ms,
            token_delta,
        )
        .await
    } else {
        Ok(())
    };
    if let Err(err) = token_result {
        let _ = runtime
            .kv_set_if_absent(
                &marker_key,
                marker_value,
                Duration::from_secs(REALTIME_MARKER_TTL_SECONDS),
            )
            .await;
        return Err(err);
    }
    let update_result = match (marker, failed) {
        (AdmissionMarker::Event { event_id, .. }, true) => runtime
            .realtime_event_remove(REALTIME_KEY_PREFIX, &event_id)
            .await
            .map(|_| ()),
        (AdmissionMarker::Event { .. }, false) => Ok(()),
        // Markers written by the previous bucket implementation may survive a
        // rolling deploy.  Keep their compensation path for compatibility;
        // all new reservations use the exact event marker above.
        (AdmissionMarker::LegacyBucket(bucket), true) => runtime
            .realtime_bucket_add(
                &bucket_key(bucket),
                -1,
                0,
                Duration::from_secs(REALTIME_BUCKET_TTL_SECONDS),
            )
            .await
            .map(|_| ()),
        (AdmissionMarker::LegacyBucket(_), false) => Ok(()),
    };
    if let Err(err) = update_result {
        // `kv_take` and the bucket mutation are separate primitives for the
        // memory and Redis adapters.  Restore the marker when the second step
        // fails so a lifecycle retry can make the same compensation.  A
        // concurrent retry may already have recreated/consumed the marker;
        // SETNX keeps this recovery idempotent in that case.
        let _ = runtime
            .kv_set_if_absent(
                &marker_key,
                marker_value,
                Duration::from_secs(REALTIME_MARKER_TTL_SECONDS),
            )
            .await;
        return Err(err);
    }
    Ok(true)
}

pub(crate) async fn rollback_admission(
    runtime: &RuntimeState,
    request_id: &str,
) -> Result<bool, DataLayerError> {
    settle_admission(runtime, request_id, true, 0).await
}

/// Record an observed token increment at a compatibility second timestamp.
/// New lifecycle paths use `record_token_delta_event` directly; this wrapper
/// keeps the old test/helper surface while still writing an exact millisecond
/// event.
pub(crate) async fn record_token_delta_at(
    runtime: &RuntimeState,
    timestamp_unix_secs: u64,
    token_delta: u64,
) -> Result<(), DataLayerError> {
    let event_id = format!("{REALTIME_KEY_PREFIX}:compat-token:{}", Uuid::new_v4());
    record_token_delta_event(
        runtime,
        &event_id,
        timestamp_unix_secs.saturating_mul(1_000),
        token_delta,
    )
    .await
    .map(|_| ())
}

async fn record_token_delta_event(
    runtime: &RuntimeState,
    event_id: &str,
    timestamp_ms: u64,
    token_delta: u64,
) -> Result<bool, DataLayerError> {
    if token_delta == 0 {
        return Ok(false);
    }
    runtime
        .realtime_event_add(
            REALTIME_KEY_PREFIX,
            event_id,
            timestamp_ms,
            0,
            i64::try_from(token_delta).unwrap_or(i64::MAX),
            Duration::from_secs(REALTIME_EVENT_TTL_SECONDS),
        )
        .await
}

/// Persist one live stream increment with a bounded retry around the two
/// shared stores (reconciliation ledger first, exact event second).  Keep the
/// accepted delta in the local task while retrying: the ledger's idempotency
/// marker intentionally returns zero on a replay, so re-running the ledger
/// call would otherwise lose the delta after an ambiguous event-store reply.
async fn record_stream_token_observation(
    runtime: &RuntimeState,
    ledger_identity: &str,
    event_id: &str,
    timestamp_ms: u64,
    token_delta: u64,
) -> Result<(), DataLayerError> {
    let mut accepted_delta = None;
    let mut last_error = None;
    for attempt in 0..REALTIME_TOKEN_WRITE_RETRIES {
        if accepted_delta.is_none() {
            match runtime
                .realtime_token_stream_add_once(
                    ledger_identity,
                    event_id,
                    token_delta,
                    Duration::from_secs(REALTIME_TOKEN_LEDGER_TTL_SECONDS),
                )
                .await
            {
                Ok(accepted) if accepted > 0 => accepted_delta = Some(accepted),
                Ok(_) => return Ok(()),
                Err(error) => {
                    last_error = Some(error);
                    if attempt + 1 < REALTIME_TOKEN_WRITE_RETRIES {
                        retry_realtime_token_write(attempt).await;
                        continue;
                    }
                    break;
                }
            }
        }

        let Some(accepted) = accepted_delta else {
            continue;
        };
        match record_token_delta_event(runtime, event_id, timestamp_ms, accepted).await {
            Ok(_) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < REALTIME_TOKEN_WRITE_RETRIES {
                    retry_realtime_token_write(attempt).await;
                    continue;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        DataLayerError::UnexpectedValue(
            "realtime stream token write exhausted its retry budget".to_string(),
        )
    }))
}

/// Reconcile a terminal cumulative total through a retryable prepare/commit
/// protocol.  The ledger and exact event collection use different Redis hash
/// tags, so they cannot be one cross-slot Lua transaction; keeping the pending
/// remainder in the ledger means an event-write or commit failure is safely
/// replayable instead of permanently losing TPM.
async fn record_terminal_token_remainder(
    runtime: &RuntimeState,
    ledger_identity: &str,
    token_event_id: &str,
    timestamp_ms: u64,
    terminal_total: u64,
) -> Result<(), DataLayerError> {
    let mut last_error = None;
    for attempt in 0..REALTIME_TOKEN_WRITE_RETRIES {
        let remainder = match runtime
            .realtime_token_terminal_prepare(
                ledger_identity,
                token_event_id,
                terminal_total,
                Duration::from_secs(REALTIME_TOKEN_LEDGER_TTL_SECONDS),
            )
            .await
        {
            Ok(remainder) => remainder,
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < REALTIME_TOKEN_WRITE_RETRIES {
                    retry_realtime_token_write(attempt).await;
                    continue;
                }
                break;
            }
        };
        if remainder == 0 {
            return Ok(());
        }
        // `realtime_event_add` is idempotent.  A false result means a prior
        // retry already made the event durable, so it is safe to commit the
        // pending ledger in either case.
        if let Err(error) =
            record_token_delta_event(runtime, token_event_id, timestamp_ms, remainder).await
        {
            last_error = Some(error);
            if attempt + 1 < REALTIME_TOKEN_WRITE_RETRIES {
                retry_realtime_token_write(attempt).await;
                continue;
            }
            break;
        }
        match runtime
            .realtime_token_terminal_commit(ledger_identity, token_event_id)
            .await
        {
            Ok(_) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                if attempt + 1 < REALTIME_TOKEN_WRITE_RETRIES {
                    retry_realtime_token_write(attempt).await;
                    continue;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        DataLayerError::UnexpectedValue(
            "realtime terminal token write exhausted its retry budget".to_string(),
        )
    }))
}

async fn retry_realtime_token_write(attempt: usize) {
    // Keep transient Redis/network failures off the request path while giving
    // the cross-slot prepare → event → commit sequence a bounded opportunity
    // to complete before the lifecycle is forgotten.  A pending prepare is
    // still replayable by a later terminal lifecycle event.
    let delay_ms = 10_u64.saturating_mul((attempt as u64).saturating_add(1));
    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
}

async fn record_usage_observation(
    runtime: &RuntimeState,
    observation: &UsageRealtimeObservation,
) -> Result<(), DataLayerError> {
    // A terminal event can be retried by the usage writer without being a new
    // token observation.  The exact event store uses the stable lifecycle key
    // as its idempotency identity; streaming notifications are deltas and get
    // a fresh identity for each observed chunk (two equal chunks in one
    // millisecond are still two legitimate increments).
    if observation.token_delta > 0 {
        let ledger_identity = token_ledger_identity(observation);
        if matches!(observation.event_type, UsageEventType::Streaming) {
            let token_event_id = format!("{REALTIME_KEY_PREFIX}:stream-token:{}", Uuid::new_v4());
            record_stream_token_observation(
                runtime,
                &ledger_identity,
                &token_event_id,
                observation.timestamp_ms,
                observation.token_delta,
            )
            .await?;
        } else if matches!(
            observation.event_type,
            UsageEventType::Completed | UsageEventType::Failed | UsageEventType::Cancelled
        ) {
            // Terminal usage is cumulative. Claim only the unseen remainder
            // after any live stream increments for this request/candidate.
            let token_event_id = token_observation_key(observation);
            record_terminal_token_remainder(
                runtime,
                &ledger_identity,
                &token_event_id,
                observation.timestamp_ms,
                observation.token_delta,
            )
            .await?;
        }
    }

    match observation.event_type {
        UsageEventType::Failed | UsageEventType::Cancelled => {
            // Failure compensation is idempotent through the admission
            // marker's GETDEL operation.  It deliberately happens after token
            // recording so tokens already emitted by an upstream provider are
            // retained in TPM even when RPM is rolled back.
            if observation.defer_admission_failure {
                // A transparent provider quota retry is still processing the
                // same client request. Its intermediate failed attempt must
                // not remove the logical RPM admission; the retry owner will
                // either settle it on completion or explicitly roll it back
                // when no replacement attempt can be started.
                return Ok(());
            }
            let admission_id = observation
                .admission_id
                .as_deref()
                .unwrap_or(&observation.request_id);
            if let Err(err) = rollback_admission(runtime, admission_id).await {
                // Keep a successfully-written token marker.  The admission
                // marker is restored by `settle_admission` when its bucket
                // update failed, so a replay can retry only the missing RPM
                // compensation without double-counting TPM.
                return Err(err);
            }
        }
        UsageEventType::Completed => {
            // Completed is the only terminal-success event that owns the
            // admission marker.  Token deltas were recorded above, so pass
            // zero here to avoid counting them a second time.  GETDEL makes
            // duplicate terminal notifications harmless while ensuring a
            // successful request does not suppress a later request that
            // reuses the same trace ID.
            let admission_id = observation
                .admission_id
                .as_deref()
                .unwrap_or(&observation.request_id);
            settle_admission(runtime, admission_id, false, 0).await?;
        }
        // Pending is a deferred terminal skeleton and Streaming is an
        // incremental event.  Neither is allowed to consume the admission
        // marker: a later Completed/Failed event must still settle it.
        UsageEventType::Pending | UsageEventType::Streaming => {}
    }
    Ok(())
}

/// Synchronous-test-friendly entry point for a materialized usage event.  The
/// production sink uses the same function on the usage background runtime.
#[cfg(test)]
pub(crate) async fn observe_usage_event_for_test(
    runtime: &RuntimeState,
    event: &UsageEvent,
) -> Result<(), DataLayerError> {
    let observation = UsageRealtimeObservation {
        request_id: event.request_id.clone(),
        admission_id: realtime_admission_id_from_metadata(event.data.request_metadata.as_ref()),
        attempt_id: realtime_attempt_id_from_metadata(event.data.request_metadata.as_ref()),
        defer_admission_failure: realtime_admission_failure_deferred(
            event.data.request_metadata.as_ref(),
        ),
        event_type: event.event_type,
        timestamp_ms: event.timestamp_ms,
        token_delta: token_delta_from_usage_data(&event.data),
        candidate_id: event
            .data
            .candidate_id
            .clone()
            .or_else(|| candidate_id_from_metadata(event.data.request_metadata.as_ref())),
    };
    record_usage_observation(runtime, &observation).await
}

fn token_observation_key(observation: &UsageRealtimeObservation) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    // Owner forwarding can change the public lifecycle request/trace ID while
    // retaining the private admission identity.  Conversely, clients may
    // reuse a trace ID for multiple accepted requests.  Prefer the admission
    // identity whenever it is present so terminal usage from two accepted
    // requests cannot collapse into one idempotency key; fall back to the
    // lifecycle request ID for older events that predate the marker metadata.
    let identity = observation
        .admission_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(observation.request_id.as_str());
    hasher.update(identity.as_bytes());
    hasher.update([0]);
    // Do not include token_delta or timestamp: a terminal event can be
    // replayed after billing enrichment has corrected its total (and a retry
    // may be materialized at a new wall-clock instant).  The stable lifecycle
    // identity (admission/request + candidate attempt) must map to one
    // observation across terminal phase corrections (for example a failed
    // report followed by a completed replay), regardless of those mutable
    // fields. Streaming deltas use a separate additive key path above and are
    // intentionally not routed through this function.
    if let Some(candidate_id) = observation
        .candidate_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        hasher.update([1]);
        hasher.update(candidate_id.as_bytes());
    } else {
        hasher.update([0]);
    }
    hasher.update([0]);
    if let Some(attempt_id) = observation
        .attempt_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        hasher.update([1]);
        hasher.update(attempt_id.as_bytes());
    } else {
        hasher.update([0]);
    }
    let digest = hasher.finalize();
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:token-observation:{suffix}")
}

fn token_ledger_identity(observation: &UsageRealtimeObservation) -> String {
    token_ledger_identity_from_parts(
        &observation.request_id,
        observation.admission_id.as_deref(),
        observation.candidate_id.as_deref(),
        observation.attempt_id.as_deref(),
    )
}

/// Build the shared identity used by live stream increments and terminal
/// cumulative claims. Attempt IDs remain in the stream event idempotency key,
/// while this identity deliberately follows the lifecycle request/candidate
/// whose terminal usage total is authoritative.
fn token_ledger_identity_from_parts(
    request_id: &str,
    admission_id: Option<&str>,
    candidate_id: Option<&str>,
    attempt_id: Option<&str>,
) -> String {
    use sha2::{Digest, Sha256};
    let identity = admission_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(request_id);
    let mut hasher = Sha256::new();
    hasher.update(identity.as_bytes());
    hasher.update([0]);
    if let Some(candidate_id) = candidate_id.filter(|value| !value.trim().is_empty()) {
        hasher.update(candidate_id.as_bytes());
    }
    hasher.update([0]);
    if let Some(attempt_id) = attempt_id.filter(|value| !value.trim().is_empty()) {
        hasher.update(attempt_id.as_bytes());
    }
    let digest = hasher.finalize();
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:token-ledger:{suffix}")
}

fn stream_token_event_id(observation: &UsageRealtimeTokenDelta) -> String {
    use sha2::{Digest, Sha256};
    let Some(sequence) = observation.sequence else {
        // A transport that cannot expose a sequence has already established
        // that each callback is a distinct increment.  A UUID preserves that
        // additive behavior while the sequence-aware path remains replay-safe.
        return format!("{REALTIME_KEY_PREFIX}:stream-token:{}", Uuid::new_v4());
    };
    let mut hasher = Sha256::new();
    // The private admission identity is the cross-gateway request identity.
    // A forwarded owner may assign a different public/lifecycle request ID;
    // including that ID here would make a replay of the same stream frame
    // look like a new token event.  Fall back to the lifecycle ID only for
    // older transports that do not carry the private identity.
    if let Some(admission_id) = observation
        .admission_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        hasher.update(admission_id.as_bytes());
    } else {
        hasher.update(observation.request_id.as_bytes());
    }
    hasher.update([0]);
    if let Some(candidate_id) = observation
        .candidate_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        hasher.update(candidate_id.as_bytes());
    }
    hasher.update([0]);
    if let Some(attempt_id) = observation
        .attempt_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
    {
        hasher.update(attempt_id.as_bytes());
    }
    hasher.update([0]);
    hasher.update(sequence.to_be_bytes());
    let digest = hasher.finalize();
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:stream-token:{suffix}")
}

fn candidate_id_from_metadata(metadata: Option<&serde_json::Value>) -> Option<String> {
    metadata
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get("candidate_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn realtime_admission_id_from_metadata(metadata: Option<&serde_json::Value>) -> Option<String> {
    metadata
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get("realtime_admission_id"))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn realtime_attempt_id_from_metadata(metadata: Option<&serde_json::Value>) -> Option<String> {
    metadata
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get(REALTIME_ATTEMPT_ID_METADATA_KEY))
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn realtime_admission_failure_deferred(metadata: Option<&serde_json::Value>) -> bool {
    metadata
        .and_then(serde_json::Value::as_object)
        .and_then(|object| {
            object
                .get(REALTIME_ADMISSION_DEFER_FAILURE_METADATA_KEY)
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(false)
}

/// Snapshot the exact trailing 60-second event window at the current UTC
/// millisecond.  This is the endpoint's lightweight read path.
pub(crate) async fn snapshot(runtime: &RuntimeState) -> Result<RealtimeMetrics, DataLayerError> {
    snapshot_at_millis(runtime, current_unix_millis()).await
}

/// Backwards-compatible second-resolution wrapper.  It is still backed by
/// the exact millisecond event window; no legacy second-bucket fallback is
/// allowed because that could mix pre-rollout values into a precise snapshot.
pub(crate) async fn snapshot_at(
    runtime: &RuntimeState,
    as_of_unix_secs: u64,
) -> Result<RealtimeMetrics, DataLayerError> {
    // A second-resolution caller means the whole named second, so use its
    // inclusive millisecond endpoint.  This keeps the wrapper compatible
    // with reservations made anywhere inside that second; callers needing an
    // exact instant should use `snapshot_at_millis`.
    let as_of_unix_ms = as_of_unix_secs.saturating_mul(1_000).saturating_add(999);
    snapshot_at_millis(runtime, as_of_unix_ms).await
}

/// Fixed-clock-friendly exact snapshot helper.  The interval is
/// `(as_of_ms - 60_000, as_of_ms]`, with the lower endpoint excluded and the
/// upper endpoint included.
pub(crate) async fn snapshot_at_millis(
    runtime: &RuntimeState,
    as_of_unix_ms: u64,
) -> Result<RealtimeMetrics, DataLayerError> {
    let start_ms = as_of_unix_ms.saturating_sub(REALTIME_WINDOW_MILLIS);
    let totals = runtime
        .realtime_events_sum(REALTIME_KEY_PREFIX, start_ms, as_of_unix_ms)
        .await?;
    Ok(RealtimeMetrics::from_bucket_at_millis(
        totals,
        as_of_unix_ms,
        if runtime.is_redis() {
            "shared"
        } else {
            "process"
        },
    ))
}

/// Canonical token total from a persisted usage lifecycle event. The usage
/// runtime has already normalized provider-specific fields into
/// `total_tokens`; the component fallback mirrors its canonical
/// `input + output + reasoning` calculation. Cache fields are descriptive
/// dimensions of the input count, not additional tokens, and must not be
/// added again here.
pub(crate) fn token_delta_from_usage_data(data: &UsageEventData) -> u64 {
    data.total_tokens.unwrap_or_else(|| {
        data.input_tokens
            .unwrap_or_default()
            .saturating_add(data.output_tokens.unwrap_or_default())
            .saturating_add(reasoning_tokens_from_metadata(
                data.request_metadata.as_ref(),
            ))
    })
}

fn reasoning_tokens_from_metadata(metadata: Option<&serde_json::Value>) -> u64 {
    let Some(object) = metadata.and_then(serde_json::Value::as_object) else {
        return 0;
    };
    let direct = object
        .get("reasoning_tokens")
        .and_then(value_as_non_negative_u64);
    let dimensions = object
        .get("dimensions")
        .and_then(serde_json::Value::as_object)
        .and_then(|dimensions| {
            dimensions
                .get("reasoning_tokens")
                .and_then(value_as_non_negative_u64)
                .or_else(|| {
                    dimensions
                        .get("output_tokens_details")
                        .or_else(|| dimensions.get("completion_tokens_details"))
                        .and_then(serde_json::Value::as_object)
                        .and_then(|details| {
                            details
                                .get("reasoning_tokens")
                                .and_then(value_as_non_negative_u64)
                        })
                })
        });
    direct.or(dimensions).unwrap_or_default()
}

fn value_as_non_negative_u64(value: &serde_json::Value) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
}

pub(crate) fn bucket_key(unix_secs: u64) -> String {
    format!("{REALTIME_KEY_PREFIX}:bucket:{unix_secs}")
}

pub(crate) fn admission_marker_key(request_id: &str) -> String {
    // Request IDs are normally UUIDs, but hashing keeps arbitrary trace IDs
    // from creating unbounded key lengths or delimiter collisions.
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(request_id.as_bytes());
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:admission:{suffix}")
}

fn admission_event_id(request_id: &str) -> String {
    // Keep event IDs bounded and free of the marker delimiter.  The same
    // request ID maps to one admission event across all gateway instances.
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(request_id.as_bytes());
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:admission-event:{suffix}")
}

fn settlement_token_event_id(request_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(request_id.as_bytes());
    let suffix = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{REALTIME_KEY_PREFIX}:settlement-token:{suffix}")
}

fn admission_marker_value(event_id: &str, timestamp_ms: u64) -> String {
    format!("{event_id}|{timestamp_ms}")
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AdmissionMarker {
    Event { event_id: String, timestamp_ms: u64 },
    LegacyBucket(u64),
}

fn parse_admission_marker(value: &str) -> AdmissionMarker {
    let Some((event_id, timestamp)) = value.split_once('|') else {
        return AdmissionMarker::LegacyBucket(
            value.parse::<u64>().unwrap_or_else(|_| current_unix_secs()),
        );
    };
    let event_id = event_id.trim();
    let timestamp_ms = timestamp
        .parse::<u64>()
        .unwrap_or_else(|_| current_unix_millis());
    if event_id.is_empty() {
        AdmissionMarker::LegacyBucket(timestamp_ms / 1_000)
    } else {
        AdmissionMarker::Event {
            event_id: event_id.to_string(),
            timestamp_ms,
        }
    }
}

fn current_unix_secs() -> u64 {
    u64::try_from(Utc::now().timestamp()).unwrap_or_default()
}

fn current_unix_millis() -> u64 {
    u64::try_from(Utc::now().timestamp_millis()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{admission_marker_key, snapshot_at, token_delta_from_usage_data};
    use aether_runtime_state::{MemoryRuntimeStateConfig, RuntimeState};
    use aether_usage_runtime::UsageEventData;

    #[tokio::test]
    async fn fixed_clock_snapshot_includes_exactly_sixty_buckets() {
        let runtime = RuntimeState::memory(MemoryRuntimeStateConfig::default());
        runtime
            .realtime_event_add(
                "dashboard:realtime:v1",
                "outside",
                940_000,
                1,
                10,
                std::time::Duration::from_secs(120),
            )
            .await
            .expect("event write should succeed");
        runtime
            .realtime_event_add(
                "dashboard:realtime:v1",
                "inside",
                1_000_000,
                2,
                20,
                std::time::Duration::from_secs(120),
            )
            .await
            .expect("event write should succeed");
        runtime
            .realtime_event_add(
                "dashboard:realtime:v1",
                "older",
                939_000,
                100,
                100,
                std::time::Duration::from_secs(120),
            )
            .await
            .expect("event write should succeed");

        let snapshot = snapshot_at(&runtime, 1000)
            .await
            .expect("snapshot should succeed");
        assert_eq!(snapshot.rpm, 2);
        assert_eq!(snapshot.tpm, 20);
        assert_eq!(snapshot.window_seconds, 60);
        assert_eq!(snapshot.as_of, "1970-01-01T00:16:40.999+00:00");
    }

    #[test]
    fn token_fallback_uses_normalized_components_once() {
        let mut data = UsageEventData {
            provider_name: "provider".to_string(),
            model: "model".to_string(),
            input_tokens: Some(4),
            output_tokens: Some(6),
            ..UsageEventData::default()
        };
        assert_eq!(token_delta_from_usage_data(&data), 10);
        data.total_tokens = Some(30);
        assert_eq!(token_delta_from_usage_data(&data), 30);
    }

    #[test]
    fn marker_key_is_stable_and_does_not_expose_trace() {
        let first = admission_marker_key("trace-one");
        assert_eq!(first, admission_marker_key("trace-one"));
        assert!(!first.contains("trace-one"));
    }
}
