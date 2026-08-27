use aether_contracts::ExecutionPlan;
use aether_data_contracts::repository::candidates::{
    RequestCandidateStatus, StoredRequestCandidate, UpsertRequestCandidateRecord,
};
use aether_data_contracts::repository::usage::StoredRequestUsageAudit;
use aether_scheduler_core::{
    build_execution_request_candidate_seed, build_local_request_candidate_status_record,
    build_report_request_candidate_status_record,
    finalize_execution_request_candidate_report_context, parse_request_candidate_report_context,
    resolve_report_request_candidate_slot as resolve_report_request_candidate_slot_from_candidates,
    LocalRequestCandidateStatusRecordInput, ReportRequestCandidateStatusRecordInput,
    SchedulerMinimalCandidateSelectionCandidate, SchedulerRequestCandidateStatusUpdate,
    SchedulerResolvedReportRequestCandidateSlot,
};
use aether_usage_runtime::build_locally_actionable_report_context_from_request_candidate;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::clock::current_unix_ms;
use crate::log_ids::short_request_id;
use crate::{AppState, GatewayError};

const REQUEST_CANDIDATE_PERSISTENCE_ENV: &str = "AETHER_GATEWAY_REQUEST_CANDIDATE_PERSISTENCE";
const REQUEST_CANDIDATE_SEED_WRITE_TIMEOUT_ENV: &str =
    "AETHER_GATEWAY_REQUEST_CANDIDATE_SEED_WRITE_TIMEOUT_MS";
const DEFAULT_REQUEST_CANDIDATE_SEED_WRITE_TIMEOUT_MS: u64 = 10;
const TERMINAL_CANDIDATE_RECONCILIATION_MAX_IN_FLIGHT: usize = 256;
const TERMINAL_CANDIDATE_RECONCILIATION_DELAYS: &[Duration] = &[
    Duration::from_millis(100),
    Duration::from_millis(500),
    Duration::from_secs(2),
    Duration::from_secs(10),
    Duration::from_secs(30),
    Duration::from_secs(60),
    Duration::from_secs(300),
    Duration::from_secs(900),
    Duration::from_secs(1800),
    Duration::from_secs(3600),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestCandidatePersistenceMode {
    Full,
    Terminal,
    None,
}

fn request_candidate_persistence_mode() -> RequestCandidatePersistenceMode {
    static MODE: OnceLock<RequestCandidatePersistenceMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var(REQUEST_CANDIDATE_PERSISTENCE_ENV)
            .ok()
            .map(|value| value.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("terminal") | Some("final") | Some("final_only") | Some("final-only") => {
                RequestCandidatePersistenceMode::Terminal
            }
            Some("none") | Some("off") | Some("disabled") | Some("false") | Some("0") => {
                RequestCandidatePersistenceMode::None
            }
            _ => RequestCandidatePersistenceMode::Full,
        }
    })
}

fn request_candidate_status_is_terminal(status: RequestCandidateStatus) -> bool {
    matches!(
        status,
        RequestCandidateStatus::Success
            | RequestCandidateStatus::Failed
            | RequestCandidateStatus::Cancelled
    )
}

/// A usage row is safe to drive a terminal request-candidate transition only
/// after the accounting write has recorded its durable terminal timestamp.
/// Status alone is insufficient: cleanup/reconciliation can briefly observe a
/// skeleton row with a terminal-looking status while usage/finalization is
/// still in flight.
fn request_usage_is_durable_terminal(status: &str, finalized_at_unix_secs: Option<u64>) -> bool {
    finalized_at_unix_secs.is_some()
        && matches!(
            status.trim().to_ascii_lowercase().as_str(),
            "completed" | "failed" | "cancelled"
        )
}

/// Repository reads populate the typed candidate field with routing-snapshot
/// presence semantics. A present snapshot whose candidate id is NULL is
/// authoritative absence; report reconciliation must not fall back to stale
/// request metadata in that case.
fn authoritative_usage_candidate_id(usage: &StoredRequestUsageAudit) -> Option<&str> {
    usage.candidate_id.as_deref()
}

/// Reconciliation must not apply a stale desired candidate status merely
/// because some terminal usage row appeared for the request.  A later attempt
/// (or a client cancellation) can legitimately win the usage race with a
/// different terminal outcome; promoting the old candidate in that case would
/// make the candidate ledger contradict the durable usage ledger.
fn terminal_candidate_matches_usage(
    candidate_status: RequestCandidateStatus,
    usage_status: &str,
) -> bool {
    let usage_status = usage_status.trim();
    match candidate_status {
        RequestCandidateStatus::Success => usage_status.eq_ignore_ascii_case("completed"),
        RequestCandidateStatus::Failed => usage_status.eq_ignore_ascii_case("failed"),
        RequestCandidateStatus::Cancelled => usage_status.eq_ignore_ascii_case("cancelled"),
        RequestCandidateStatus::Available
        | RequestCandidateStatus::Unused
        | RequestCandidateStatus::Pending
        | RequestCandidateStatus::Streaming
        | RequestCandidateStatus::Skipped => false,
    }
}

/// A request can have several provider candidates under one request id.  A
/// durable terminal usage row is therefore only allowed to settle the
/// candidate that produced it.  Treat a missing id on either side as a
/// mismatch when the other side has one; otherwise a late report from a
/// previous candidate could silently promote the current candidate.
fn terminal_usage_matches_candidate(
    candidate_id: Option<&str>,
    usage_candidate_id: Option<&str>,
) -> bool {
    let candidate_id = candidate_id
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let usage_candidate_id = usage_candidate_id
        .map(str::trim)
        .filter(|value| !value.is_empty());
    match (candidate_id, usage_candidate_id) {
        (Some(candidate_id), Some(usage_candidate_id)) => candidate_id == usage_candidate_id,
        (None, None) => true,
        _ => false,
    }
}

/// Check whether an internal/report-driven terminal candidate write is backed
/// by a durable usage terminal row.  Internal gateway routes may receive a
/// report before the remote executioner's usage write becomes visible; a
/// status-only check would recreate the old candidate-before-billing race.
pub(crate) async fn report_candidate_terminal_usage_is_durable(
    state: &AppState,
    report_context: Option<&Value>,
    candidate_status: RequestCandidateStatus,
) -> bool {
    let Some(request_id) = report_context
        .and_then(|context| context.get("request_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return false;
    };
    let usage = match state
        .usage_lifecycle_data_state()
        .find_request_usage_by_request_id_shallow(request_id)
        .await
    {
        Ok(usage) => usage,
        Err(error) => {
            warn!(
                event_name = "request_candidate_report_usage_lookup_failed",
                log_type = "ops",
                request_id = %short_request_id(request_id),
                error = ?error,
                "gateway will suppress report-driven candidate terminal status"
            );
            return false;
        }
    };
    usage.is_some_and(|usage| {
        request_usage_is_durable_terminal(usage.status.as_str(), usage.finalized_at_unix_secs)
            && terminal_candidate_matches_usage(candidate_status, usage.status.as_str())
            && terminal_usage_matches_candidate(
                report_context
                    .and_then(|context| context.get("candidate_id"))
                    .and_then(Value::as_str),
                authoritative_usage_candidate_id(&usage),
            )
    })
}

fn should_persist_request_candidate_status(status: RequestCandidateStatus) -> bool {
    match request_candidate_persistence_mode() {
        RequestCandidatePersistenceMode::Full => true,
        RequestCandidatePersistenceMode::Terminal => request_candidate_status_is_terminal(status),
        RequestCandidatePersistenceMode::None => false,
    }
}

fn request_candidate_seed_write_timeout() -> Duration {
    static TIMEOUT: OnceLock<Duration> = OnceLock::new();
    *TIMEOUT.get_or_init(|| {
        let millis = std::env::var(REQUEST_CANDIDATE_SEED_WRITE_TIMEOUT_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_REQUEST_CANDIDATE_SEED_WRITE_TIMEOUT_MS);
        Duration::from_millis(millis)
    })
}

#[derive(Debug, Clone)]
pub(crate) struct LocalRequestCandidateStatusSnapshot {
    candidate_id: String,
    request_id: String,
    user_id: Option<String>,
    api_key_id: Option<String>,
    candidate_index: u32,
    retry_index: u32,
    provider_id: String,
    endpoint_id: String,
    key_id: String,
}

#[async_trait]
pub(crate) trait RequestCandidateRuntimeReader {
    async fn read_request_candidates_by_request_id(
        &self,
        request_id: &str,
    ) -> Result<Vec<StoredRequestCandidate>, GatewayError>;
}

#[async_trait]
pub(crate) trait RequestCandidateRuntimeWriter: Sync {
    fn has_request_candidate_data_writer(&self) -> bool;

    async fn upsert_request_candidate(
        &self,
        candidate: UpsertRequestCandidateRecord,
    ) -> Result<Option<StoredRequestCandidate>, GatewayError>;

    async fn enqueue_request_candidate_status(
        &self,
        candidate: UpsertRequestCandidateRecord,
    ) -> Result<Option<()>, GatewayError> {
        self.upsert_request_candidate(candidate)
            .await
            .map(|stored| stored.map(|_| ()))
    }

    fn try_enqueue_request_candidate_status(
        &self,
        candidate: UpsertRequestCandidateRecord,
    ) -> Result<(), UpsertRequestCandidateRecord> {
        Err(candidate)
    }
}

#[async_trait]
pub(crate) trait RequestCandidateRuntimeCapabilityReader {
    async fn read_request_candidate_user_model_capability_settings(
        &self,
        user_id: &str,
    ) -> Result<Option<Value>, GatewayError>;

    async fn read_request_candidate_api_key_force_capabilities(
        &self,
        user_id: &str,
        api_key_id: &str,
    ) -> Result<Option<Value>, GatewayError>;
}

fn terminal_candidate_reconciliation_semaphore() -> &'static Arc<tokio::sync::Semaphore> {
    static SEMAPHORE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();
    SEMAPHORE.get_or_init(|| {
        Arc::new(tokio::sync::Semaphore::new(
            TERMINAL_CANDIDATE_RECONCILIATION_MAX_IN_FLIGHT,
        ))
    })
}

/// Retry a candidate terminal transition after a usage handoff could not be
/// confirmed in the request's critical path. The usage runtime retains the
/// terminal event in its own retry/queue machinery; this companion task only
/// waits for the durable row to become visible and then publishes the matching
/// candidate state. It is bounded and globally gated so a database outage does
/// not create an unbounded task leak.
pub(crate) fn spawn_terminal_candidate_reconciliation(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    mut terminal_update: SchedulerRequestCandidateStatusUpdate,
) {
    let Some(permit) = terminal_candidate_reconciliation_semaphore()
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        warn!(
            event_name = "request_candidate_terminal_reconciliation_saturated",
            log_type = "event",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            "gateway could not schedule terminal candidate reconciliation because the global bound is full"
        );
        return;
    };

    // Reconciliation is part of the terminal usage→candidate handoff.  Keep
    // it on the process-lifetime usage runtime: the request/relay runtime can
    // be torn down after a 499 while the durable usage writer is still
    // retrying, and cancelling this waiter would leave the candidate stuck in
    // `streaming` forever.
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _permit = permit;
        for delay in TERMINAL_CANDIDATE_RECONCILIATION_DELAYS {
            tokio::time::sleep(*delay).await;
            let usage = match state
                .usage_lifecycle_data_state()
                .find_request_usage_by_request_id_shallow(plan.request_id.as_str())
                .await
            {
                Ok(usage) => usage,
                Err(error) => {
                    warn!(
                        event_name = "request_candidate_terminal_reconciliation_usage_lookup_failed",
                        log_type = "event",
                        request_id = %short_request_id(plan.request_id.as_str()),
                        error = ?error,
                        "gateway will retry terminal candidate reconciliation"
                    );
                    continue;
                }
            };
            let Some(usage) = usage else {
                continue;
            };
            if !request_usage_is_durable_terminal(
                usage.status.as_str(),
                usage.finalized_at_unix_secs,
            ) || !terminal_candidate_matches_usage(terminal_update.status, usage.status.as_str())
                || !terminal_usage_matches_candidate(
                    plan.candidate_id.as_deref(),
                    authoritative_usage_candidate_id(&usage),
                )
            {
                continue;
            }
            terminal_update.finished_at_unix_ms = Some(
                terminal_update
                    .finished_at_unix_ms
                    .unwrap_or_else(current_unix_ms),
            );
            let candidate_persisted = record_local_request_candidate_status(
                &state,
                &plan,
                report_context.as_ref(),
                terminal_update.clone(),
            )
            .await;
            if candidate_persisted {
                return;
            }
            debug!(
                event_name = "request_candidate_terminal_reconciliation_write_pending",
                log_type = "event",
                request_id = %short_request_id(plan.request_id.as_str()),
                candidate_id = ?plan.candidate_id,
                "gateway will retry terminal candidate reconciliation because the candidate write was not accepted"
            );
        }
        warn!(
            event_name = "request_candidate_terminal_reconciliation_exhausted",
            log_type = "event",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            fallback = "usage_cleanup",
            "gateway could not observe a durable terminal usage row before reconciliation expired"
        );
    });
}

/// Retry the candidate half of a terminal handoff after the caller has already
/// confirmed that the matching usage event was accepted durably.  Unlike
/// [`spawn_terminal_candidate_reconciliation`], this path must not wait for a
/// terminal usage row: successful async submissions (notably video create)
/// intentionally keep usage pending until the remote task finishes.
pub(crate) fn spawn_candidate_persistence_retry_after_usage_handoff(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    terminal_update: SchedulerRequestCandidateStatusUpdate,
    watchdog_progress: Option<Arc<crate::execution_runtime::StreamCandidateWatchdogProgress>>,
) {
    let Some(permit) = terminal_candidate_reconciliation_semaphore()
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        warn!(
            event_name = "request_candidate_terminal_persistence_retry_saturated",
            log_type = "event",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            "gateway could not schedule terminal candidate persistence retry because the global bound is full"
        );
        return;
    };

    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _permit = permit;
        for delay in TERMINAL_CANDIDATE_RECONCILIATION_DELAYS {
            tokio::time::sleep(*delay).await;
            if record_local_request_candidate_status(
                &state,
                &plan,
                report_context.as_ref(),
                terminal_update.clone(),
            )
            .await
            {
                if let Some(progress) = watchdog_progress {
                    crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                        plan.request_id.as_str(),
                        plan.candidate_id.as_deref(),
                        &progress,
                    );
                }
                return;
            }
        }
        warn!(
            event_name = "request_candidate_terminal_persistence_retry_exhausted",
            log_type = "ops",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            "gateway could not persist the terminal candidate after a durable usage handoff"
        );
    });
}

/// Report-context counterpart of
/// [`spawn_candidate_persistence_retry_after_usage_handoff`].
pub(crate) fn spawn_report_candidate_persistence_retry_after_usage_handoff(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    terminal_update: SchedulerRequestCandidateStatusUpdate,
    watchdog_progress: Option<Arc<crate::execution_runtime::StreamCandidateWatchdogProgress>>,
) {
    let Some(permit) = terminal_candidate_reconciliation_semaphore()
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        warn!(
            event_name = "request_candidate_report_terminal_persistence_retry_saturated",
            log_type = "event",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            "gateway could not schedule report terminal candidate persistence retry because the global bound is full"
        );
        return;
    };

    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _permit = permit;
        for delay in TERMINAL_CANDIDATE_RECONCILIATION_DELAYS {
            tokio::time::sleep(*delay).await;
            if record_report_request_candidate_status(
                &state,
                report_context.as_ref(),
                terminal_update.clone(),
            )
            .await
            {
                if let Some(progress) = watchdog_progress {
                    crate::execution_runtime::unregister_stream_candidate_watchdog_progress(
                        plan.request_id.as_str(),
                        plan.candidate_id.as_deref(),
                        &progress,
                    );
                }
                return;
            }
        }
        warn!(
            event_name = "request_candidate_report_terminal_persistence_retry_exhausted",
            log_type = "ops",
            request_id = %short_request_id(plan.request_id.as_str()),
            candidate_id = ?plan.candidate_id,
            "gateway could not persist the report terminal candidate after a durable usage handoff"
        );
    });
}

/// Detached counterpart for internal/report handlers that do not own an
/// [`ExecutionPlan`].  It waits for the matching durable usage terminal row and
/// then replays the report candidate update.  Until that point the caller's
/// report path must leave the candidate non-terminal.
pub(crate) fn spawn_report_candidate_reconciliation(
    state: AppState,
    report_context: Value,
    terminal_update: SchedulerRequestCandidateStatusUpdate,
) {
    let Some(permit) = terminal_candidate_reconciliation_semaphore()
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        warn!(
            event_name = "request_candidate_report_reconciliation_saturated",
            log_type = "event",
            request_id = %short_request_id(report_request_id_from_context(Some(&report_context))),
            "gateway could not schedule report candidate reconciliation because the global bound is full"
        );
        return;
    };

    // Internal/report reconciliation has the same lifetime requirement as the
    // execution-plan variant above.  A report often arrives while the owning
    // request task is already being cancelled, so do not bind its retries to
    // that task's Tokio runtime.
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _permit = permit;
        let request_id = report_request_id_from_context(Some(&report_context));
        if request_id == "-" {
            return;
        }
        for delay in TERMINAL_CANDIDATE_RECONCILIATION_DELAYS {
            tokio::time::sleep(*delay).await;
            let usage = match state
                .usage_lifecycle_data_state()
                .find_request_usage_by_request_id_shallow(request_id)
                .await
            {
                Ok(usage) => usage,
                Err(error) => {
                    warn!(
                        event_name = "request_candidate_report_reconciliation_usage_lookup_failed",
                        log_type = "ops",
                        request_id = %short_request_id(request_id),
                        error = ?error,
                        "gateway will retry report candidate reconciliation"
                    );
                    continue;
                }
            };
            let Some(usage) = usage else {
                continue;
            };
            let report_candidate_id = parse_request_candidate_report_context(Some(&report_context))
                .and_then(|metadata| metadata.candidate_id);
            if !request_usage_is_durable_terminal(
                usage.status.as_str(),
                usage.finalized_at_unix_secs,
            ) || !terminal_candidate_matches_usage(terminal_update.status, usage.status.as_str())
                || !terminal_usage_matches_candidate(
                    report_candidate_id.as_deref(),
                    authoritative_usage_candidate_id(&usage),
                )
            {
                continue;
            }

            let resolved_context = resolve_locally_actionable_request_candidate_report_context(
                &state,
                &report_context,
            )
            .await;
            let context = resolved_context.as_ref().unwrap_or(&report_context);
            let mut update = terminal_update.clone();
            update.finished_at_unix_ms =
                Some(update.finished_at_unix_ms.unwrap_or_else(current_unix_ms));
            let candidate_persisted =
                record_report_request_candidate_status(&state, Some(context), update).await;
            if candidate_persisted {
                return;
            }
            debug!(
                event_name = "request_candidate_report_reconciliation_write_pending",
                log_type = "event",
                request_id = %short_request_id(request_id),
                candidate_id = ?report_candidate_id,
                "gateway will retry report candidate reconciliation because the candidate write was not accepted"
            );
        }
        warn!(
            event_name = "request_candidate_report_reconciliation_exhausted",
            log_type = "event",
            request_id = %short_request_id(request_id),
            "gateway could not observe a matching durable usage terminal for the internal report"
        );
    });
}

fn report_request_id_from_context(report_context: Option<&Value>) -> &str {
    report_context
        .and_then(|context| context.get("request_id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("-")
}

pub(crate) async fn resolve_request_candidate_required_capabilities(
    state: &(impl RequestCandidateRuntimeCapabilityReader + ?Sized),
    user_id: &str,
    api_key_id: &str,
    requested_model: Option<&str>,
    explicit_required_capabilities: Option<&Value>,
    model_directive_base_model: Option<&str>,
) -> Option<Value> {
    let mut merged = serde_json::Map::new();

    match state
        .read_request_candidate_user_model_capability_settings(user_id)
        .await
    {
        Ok(settings) => merge_capability_object(
            &mut merged,
            select_requested_model_capabilities(
                settings.as_ref(),
                requested_model,
                model_directive_base_model,
            ),
        ),
        Err(error) => {
            warn!(
                user_id = %user_id,
                api_key_id = %api_key_id,
                requested_model = requested_model.unwrap_or_default(),
                error = ?error,
                "gateway request candidate user model capabilities lookup failed"
            );
        }
    }

    match state
        .read_request_candidate_api_key_force_capabilities(user_id, api_key_id)
        .await
    {
        Ok(force_capabilities) => {
            merge_capability_object(&mut merged, force_capabilities.as_ref());
        }
        Err(error) => {
            warn!(
                user_id = %user_id,
                api_key_id = %api_key_id,
                requested_model = requested_model.unwrap_or_default(),
                error = ?error,
                "gateway request candidate api key capabilities lookup failed"
            );
        }
    }

    merge_capability_object(&mut merged, explicit_required_capabilities);

    (!merged.is_empty()).then_some(Value::Object(merged))
}

fn merge_capability_object(target: &mut serde_json::Map<String, Value>, source: Option<&Value>) {
    let Some(source) = source.and_then(Value::as_object) else {
        return;
    };

    for (capability, value) in source {
        if capability.trim().is_empty() {
            continue;
        }
        target.insert(capability.clone(), value.clone());
    }
}

fn select_requested_model_capabilities<'a>(
    settings: Option<&'a Value>,
    requested_model: Option<&str>,
    model_directive_base_model: Option<&str>,
) -> Option<&'a Value> {
    let requested_model = requested_model
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let settings = settings?.as_object()?;

    find_model_capabilities(settings, requested_model).or_else(|| {
        model_directive_base_model
            .map(str::trim)
            .filter(|base_model| !base_model.is_empty() && *base_model != requested_model)
            .and_then(|base_model| find_model_capabilities(settings, base_model))
    })
}

fn find_model_capabilities<'a>(
    settings: &'a serde_json::Map<String, Value>,
    requested_model: &str,
) -> Option<&'a Value> {
    settings.get(requested_model).or_else(|| {
        settings.iter().find_map(|(model_name, capabilities)| {
            model_name
                .trim()
                .eq_ignore_ascii_case(requested_model)
                .then_some(capabilities)
        })
    })
}

fn request_candidate_status_label(status: RequestCandidateStatus) -> &'static str {
    match status {
        RequestCandidateStatus::Available => "available",
        RequestCandidateStatus::Unused => "unused",
        RequestCandidateStatus::Pending => "pending",
        RequestCandidateStatus::Streaming => "streaming",
        RequestCandidateStatus::Success => "success",
        RequestCandidateStatus::Failed => "failed",
        RequestCandidateStatus::Cancelled => "cancelled",
        RequestCandidateStatus::Skipped => "skipped",
    }
}

pub(crate) fn snapshot_local_request_candidate_status(
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
) -> Option<LocalRequestCandidateStatusSnapshot> {
    let candidate_id = plan
        .candidate_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let metadata = parse_request_candidate_report_context(report_context);
    let candidate_index = metadata
        .as_ref()
        .and_then(|metadata| metadata.candidate_index)
        .unwrap_or(0);

    Some(LocalRequestCandidateStatusSnapshot {
        candidate_id: candidate_id.to_string(),
        request_id: plan.request_id.clone(),
        user_id: metadata
            .as_ref()
            .and_then(|metadata| metadata.user_id.clone()),
        api_key_id: metadata
            .as_ref()
            .and_then(|metadata| metadata.api_key_id.clone()),
        candidate_index,
        retry_index: metadata
            .as_ref()
            .map(|metadata| metadata.retry_index)
            .unwrap_or(0),
        provider_id: plan.provider_id.clone(),
        endpoint_id: plan.endpoint_id.clone(),
        key_id: plan.key_id.clone(),
    })
}

pub(crate) async fn persist_local_request_candidate_status_record(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    record: UpsertRequestCandidateRecord,
) -> bool {
    let candidate_id = record.id.clone();
    let request_id = short_request_id(record.request_id.as_str());
    let candidate_index = record.candidate_index;
    let retry_index = record.retry_index;
    let status = record.status;
    // RETRY_ABORT is reserved by the intermediate Failed transition.  Do not
    // let an unrelated terminal Success/Cancelled update mark that handoff
    // complete: those updates can race a dropped retry owner while its usage
    // row is still pending.
    let mark_retry_handoff_complete = matches!(status, RequestCandidateStatus::Failed);
    let watchdog_request_id = record.request_id.clone();

    let should_persist = should_persist_request_candidate_status(status);
    if !should_persist {
        debug!(
            event_name = "request_candidate_status_persistence_skipped",
            log_type = "event",
            request_id = %request_id,
            candidate_id = %candidate_id,
            candidate_index,
            retry_index,
            status = request_candidate_status_label(status),
            source = "local_status",
            "gateway skipped request candidate status update due to persistence mode"
        );
        // A disabled writer is an intentional policy choice, not proof that a
        // durable candidate row exists. Keep the return value false *and* keep
        // any retry owner armed: a cancellation can still arrive before the
        // request-level usage handoff and must be allowed to claim its 499
        // fallback. The owner is released only after an accepted status write.
        return false;
    }

    let mut status_persisted = false;
    match state.enqueue_request_candidate_status(record).await {
        Ok(Some(())) => {
            status_persisted = true;
            debug!(
                event_name = "request_candidate_status_persisted",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = request_candidate_status_label(status),
                source = "local_status",
                "gateway persisted request candidate status update"
            );
        }
        Ok(None) => {
            warn!(
                event_name = "request_candidate_writer_unavailable",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = request_candidate_status_label(status),
                source = "local_status",
                "gateway skipped request candidate persistence because writer is unavailable"
            );
        }
        Err(err) => {
            warn!(
                event_name = "request_candidate_status_persist_failed",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %candidate_id,
                error = ?err,
                "gateway failed to persist request candidate status update"
            );
        }
    }
    if mark_retry_handoff_complete && status_persisted {
        crate::execution_runtime::mark_stream_candidate_watchdog_retry_handoff_complete_for_request(
            watchdog_request_id.as_str(),
            Some(candidate_id.as_str()),
        );
    }
    status_persisted
}

pub(crate) async fn record_local_request_candidate_status(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status_update: SchedulerRequestCandidateStatusUpdate,
) -> bool {
    let Some(record) =
        build_local_request_candidate_status_record(LocalRequestCandidateStatusRecordInput {
            plan,
            report_context,
            status_update,
        })
    else {
        return false;
    };
    persist_local_request_candidate_status_record(state, record).await
}

pub(crate) async fn record_local_request_candidate_extra_data(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    status: RequestCandidateStatus,
    status_code: Option<u16>,
    latency_ms: Option<u64>,
    extra_data: Value,
) -> bool {
    let Some(snapshot) = snapshot_local_request_candidate_status(plan, report_context) else {
        return false;
    };
    let record = UpsertRequestCandidateRecord {
        id: snapshot.candidate_id.clone(),
        request_id: snapshot.request_id.clone(),
        user_id: snapshot.user_id.clone(),
        api_key_id: snapshot.api_key_id.clone(),
        username: None,
        api_key_name: None,
        candidate_index: snapshot.candidate_index,
        retry_index: snapshot.retry_index,
        provider_id: Some(snapshot.provider_id.clone()),
        endpoint_id: Some(snapshot.endpoint_id.clone()),
        key_id: Some(snapshot.key_id.clone()),
        status,
        skip_reason: None,
        is_cached: None,
        status_code,
        error_type: None,
        error_message: None,
        latency_ms,
        concurrent_requests: None,
        extra_data: Some(extra_data),
        required_capabilities: None,
        created_at_unix_ms: None,
        started_at_unix_ms: None,
        finished_at_unix_ms: None,
    };
    persist_local_request_candidate_status_record(state, record).await
}

fn build_local_request_candidate_status_snapshot_record(
    snapshot: &LocalRequestCandidateStatusSnapshot,
    status_update: SchedulerRequestCandidateStatusUpdate,
) -> UpsertRequestCandidateRecord {
    let SchedulerRequestCandidateStatusUpdate {
        status,
        status_code,
        error_type,
        error_message,
        latency_ms,
        started_at_unix_ms,
        finished_at_unix_ms,
    } = status_update;
    UpsertRequestCandidateRecord {
        id: snapshot.candidate_id.clone(),
        request_id: snapshot.request_id.clone(),
        user_id: snapshot.user_id.clone(),
        api_key_id: snapshot.api_key_id.clone(),
        username: None,
        api_key_name: None,
        candidate_index: snapshot.candidate_index,
        retry_index: snapshot.retry_index,
        provider_id: Some(snapshot.provider_id.clone()),
        endpoint_id: Some(snapshot.endpoint_id.clone()),
        key_id: Some(snapshot.key_id.clone()),
        status,
        skip_reason: None,
        is_cached: None,
        status_code,
        error_type,
        error_message,
        latency_ms,
        concurrent_requests: None,
        extra_data: None,
        required_capabilities: None,
        created_at_unix_ms: None,
        started_at_unix_ms,
        finished_at_unix_ms,
    }
}

pub(crate) fn try_enqueue_local_request_candidate_status_snapshot(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    snapshot: &LocalRequestCandidateStatusSnapshot,
    status_update: SchedulerRequestCandidateStatusUpdate,
) -> Result<(), UpsertRequestCandidateRecord> {
    let record = build_local_request_candidate_status_snapshot_record(snapshot, status_update);
    if !should_persist_request_candidate_status(record.status) {
        return Ok(());
    }
    state.try_enqueue_request_candidate_status(record)
}

pub(crate) async fn record_local_request_candidate_status_snapshot(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    snapshot: &LocalRequestCandidateStatusSnapshot,
    status_update: SchedulerRequestCandidateStatusUpdate,
) -> bool {
    let record = build_local_request_candidate_status_snapshot_record(snapshot, status_update);
    persist_local_request_candidate_status_record(state, record).await
}

/// Completes a report-driven terminal candidate transition after the usage
/// runtime's terminal event becomes visible and finalized.  This is separate
/// from the execution-plan reconciliation above because internal report and
/// video-finalize routes may not carry a full `ExecutionPlan` anymore.
fn spawn_report_terminal_candidate_reconciliation(
    state: AppState,
    record: UpsertRequestCandidateRecord,
) {
    let Some(permit) = terminal_candidate_reconciliation_semaphore()
        .clone()
        .try_acquire_owned()
        .ok()
    else {
        warn!(
            event_name = "request_candidate_report_terminal_reconciliation_saturated",
            log_type = "event",
            request_id = %short_request_id(record.request_id.as_str()),
            candidate_id = %record.id,
            "gateway could not schedule report terminal candidate reconciliation because the global bound is full"
        );
        return;
    };

    // This terminal reconciliation may be scheduled from a detached report
    // path.  Use the process-lifetime executor so runtime shutdown cannot
    // discard the final candidate update after usage has become durable.
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        let _permit = permit;
        for delay in TERMINAL_CANDIDATE_RECONCILIATION_DELAYS {
            tokio::time::sleep(*delay).await;
            let usage = match state
                .usage_lifecycle_data_state()
                .find_request_usage_by_request_id_shallow(record.request_id.as_str())
                .await
            {
                Ok(usage) => usage,
                Err(error) => {
                    warn!(
                        event_name = "request_candidate_report_terminal_reconciliation_usage_lookup_failed",
                        log_type = "event",
                        request_id = %short_request_id(record.request_id.as_str()),
                        candidate_id = %record.id,
                        error = ?error,
                        "gateway will retry report terminal candidate reconciliation"
                    );
                    continue;
                }
            };
            let Some(usage) = usage else {
                continue;
            };
            if !request_usage_is_durable_terminal(
                usage.status.as_str(),
                usage.finalized_at_unix_secs,
            ) {
                continue;
            }
            if !terminal_candidate_matches_usage(record.status, usage.status.as_str()) {
                warn!(
                    event_name = "request_candidate_report_terminal_reconciliation_outcome_mismatch",
                    log_type = "ops",
                    request_id = %short_request_id(record.request_id.as_str()),
                    candidate_id = %record.id,
                    candidate_status = request_candidate_status_label(record.status),
                    usage_status = %usage.status,
                    "gateway abandoned report terminal reconciliation because usage finalized with another outcome"
                );
                return;
            }
            if !terminal_usage_matches_candidate(
                Some(record.id.as_str()),
                authoritative_usage_candidate_id(&usage),
            ) {
                warn!(
                    event_name = "request_candidate_report_terminal_reconciliation_candidate_mismatch",
                    log_type = "ops",
                    request_id = %short_request_id(record.request_id.as_str()),
                    candidate_id = %record.id,
                    usage_candidate_id = ?authoritative_usage_candidate_id(&usage),
                    "gateway abandoned report terminal reconciliation because usage finalized for another candidate"
                );
                return;
            }
            let candidate_persisted =
                persist_local_request_candidate_status_record(&state, record.clone()).await;
            if candidate_persisted {
                return;
            }
            debug!(
                event_name = "request_candidate_report_terminal_reconciliation_write_pending",
                log_type = "event",
                request_id = %short_request_id(record.request_id.as_str()),
                candidate_id = %record.id,
                "gateway will retry report terminal candidate reconciliation because the candidate write was not accepted"
            );
        }
        warn!(
            event_name = "request_candidate_report_terminal_reconciliation_exhausted",
            log_type = "ops",
            request_id = %short_request_id(record.request_id.as_str()),
            candidate_id = %record.id,
            fallback = "usage_cleanup",
            "gateway could not observe a durable terminal usage row before report reconciliation expired"
        );
    });
}

pub(crate) async fn record_report_request_candidate_status(
    state: &AppState,
    report_context: Option<&Value>,
    status_update: SchedulerRequestCandidateStatusUpdate,
) -> bool {
    if matches!(
        request_candidate_persistence_mode(),
        RequestCandidatePersistenceMode::None
    ) {
        return false;
    }
    let Some(slot) = resolve_report_request_candidate_slot(state, report_context).await else {
        return false;
    };
    let request_id = slot.request_id.clone();
    let request_id_for_log = short_request_id(request_id.as_str());
    let candidate_index = slot.candidate_index;
    let retry_index = slot.retry_index;
    let record =
        build_report_request_candidate_status_record(ReportRequestCandidateStatusRecordInput {
            slot,
            status_update,
            now_unix_ms: current_unix_ms(),
        });
    let candidate_id = record.id.clone();
    let status = record.status;

    if !should_persist_request_candidate_status(status) {
        debug!(
            event_name = "request_candidate_report_status_persistence_skipped",
            log_type = "event",
            request_id = %request_id_for_log,
            candidate_id = %candidate_id,
            candidate_index,
            retry_index,
            status = request_candidate_status_label(status),
            source = "report_status",
            "gateway skipped report-driven request candidate status update due to persistence mode"
        );
        return false;
    }

    // A report is an asynchronous side channel.  It may arrive after a
    // transport disconnect (or after a retry has already won) while the
    // accounting terminal event is still queued.  Never let such a report
    // publish a terminal candidate state ahead of a durable, matching usage
    // row.  A missing usage row is not proof that no usage event exists: the
    // writer may be delayed, the read may be on a lagging replica, or the
    // usage skeleton may have been dropped before its terminal handoff.  Keep
    // the candidate non-terminal and let reconciliation/cleanup decide; this
    // fail-closed rule applies to report-only/internal paths as well.
    if request_candidate_status_is_terminal(status) {
        match state
            .usage_lifecycle_data_state()
            .find_request_usage_by_request_id_shallow(record.request_id.as_str())
            .await
        {
            Ok(Some(usage))
                if request_usage_is_durable_terminal(
                    usage.status.as_str(),
                    usage.finalized_at_unix_secs,
                ) && terminal_candidate_matches_usage(status, usage.status.as_str())
                    && terminal_usage_matches_candidate(
                        Some(record.id.as_str()),
                        authoritative_usage_candidate_id(&usage),
                    ) => {}
            Ok(Some(usage))
                if request_usage_is_durable_terminal(
                    usage.status.as_str(),
                    usage.finalized_at_unix_secs,
                ) =>
            {
                warn!(
                    event_name = "request_candidate_report_terminal_ignored_usage_mismatch",
                    log_type = "ops",
                    request_id = %request_id_for_log,
                    candidate_id = %candidate_id,
                    candidate_status = request_candidate_status_label(status),
                    usage_status = %usage.status,
                    usage_candidate_id = ?authoritative_usage_candidate_id(&usage),
                    "gateway ignored a report terminal outcome because durable usage belongs to another candidate or finalized differently"
                );
                return false;
            }
            Ok(Some(usage)) => {
                warn!(
                    event_name = "request_candidate_report_terminal_deferred_usage_pending",
                    log_type = "ops",
                    request_id = %request_id_for_log,
                    candidate_id = %candidate_id,
                    candidate_status = request_candidate_status_label(status),
                    usage_status = %usage.status,
                    usage_finalized = usage.finalized_at_unix_secs.is_some(),
                    "gateway deferred report terminal candidate status until usage terminal handoff is durable"
                );
                spawn_report_terminal_candidate_reconciliation(state.clone(), record.clone());
                return false;
            }
            Ok(None) => {
                warn!(
                    event_name = "request_candidate_report_terminal_deferred_usage_missing",
                    log_type = "ops",
                    request_id = %request_id_for_log,
                    candidate_id = %candidate_id,
                    candidate_status = request_candidate_status_label(status),
                    "gateway deferred report terminal candidate status because no usage row was visible"
                );
                spawn_report_terminal_candidate_reconciliation(state.clone(), record.clone());
                return false;
            }
            Err(error) => {
                warn!(
                    event_name = "request_candidate_report_terminal_usage_lookup_failed",
                    log_type = "ops",
                    request_id = %request_id_for_log,
                    candidate_id = %candidate_id,
                    error = ?error,
                    "gateway deferred report terminal candidate status because usage durability could not be checked"
                );
                spawn_report_terminal_candidate_reconciliation(state.clone(), record.clone());
                return false;
            }
        }
    }

    match state.enqueue_request_candidate_status(record).await {
        Ok(Some(())) => {
            debug!(
                event_name = "request_candidate_report_status_persisted",
                log_type = "event",
                request_id = %request_id_for_log,
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = request_candidate_status_label(status),
                source = "report_status",
                "gateway persisted report-driven request candidate status update"
            );
            true
        }
        Ok(None) => {
            warn!(
                event_name = "request_candidate_writer_unavailable",
                log_type = "event",
                request_id = %request_id_for_log,
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = request_candidate_status_label(status),
                source = "report_status",
                "gateway skipped request candidate persistence because writer is unavailable"
            );
            false
        }
        Err(err) => {
            warn!(
                event_name = "request_candidate_report_status_persist_failed",
                log_type = "event",
                request_id = %request_id_for_log,
                candidate_index,
                retry_index,
                error = ?err,
                "gateway failed to persist report-driven request candidate status update"
            );
            false
        }
    }
}

pub(crate) async fn ensure_execution_request_candidate_slot(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    plan: &mut ExecutionPlan,
    report_context: &mut Option<Value>,
) {
    if !state.has_request_candidate_data_writer() {
        warn!(
            event_name = "request_candidate_writer_unavailable",
            log_type = "event",
            request_id = %short_request_id(plan.request_id.as_str()),
            provider_id = %plan.provider_id,
            endpoint_id = %plan.endpoint_id,
            key_id = %plan.key_id,
            source = "seed",
            "gateway skipped request candidate seed because writer is unavailable"
        );
        return;
    }
    let existing_candidate_id = plan
        .candidate_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let report_candidate_id = parse_request_candidate_report_context(report_context.as_ref())
        .and_then(|metadata| metadata.candidate_id);
    if existing_candidate_id.as_deref().is_some()
        && report_candidate_id.as_deref() == existing_candidate_id.as_deref()
    {
        return;
    }

    let seed = build_execution_request_candidate_seed(
        plan,
        report_context.as_ref(),
        current_unix_ms(),
        existing_candidate_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
    );
    let generated_candidate_id = seed.upsert_record.id.clone();
    let request_id = short_request_id(plan.request_id.as_str());

    if !should_persist_request_candidate_status(seed.upsert_record.status) {
        plan.candidate_id = Some(generated_candidate_id.clone());
        *report_context = Some(finalize_execution_request_candidate_report_context(
            seed.report_context,
            &generated_candidate_id,
        ));
        debug!(
            event_name = "request_candidate_slot_seed_persistence_skipped",
            log_type = "event",
            request_id = %request_id,
            candidate_id = %generated_candidate_id,
            provider_id = %plan.provider_id,
            endpoint_id = %plan.endpoint_id,
            key_id = %plan.key_id,
            source = "seed",
            "gateway skipped request candidate seed due to persistence mode"
        );
        return;
    }

    let seed_upsert_record = seed.upsert_record;
    let generated_candidate_id = generated_candidate_id.clone();
    let candidate_id = match tokio::time::timeout(
        request_candidate_seed_write_timeout(),
        state.upsert_request_candidate(seed_upsert_record),
    )
    .await
    {
        Ok(Ok(Some(stored))) => {
            info!(
                event_name = "request_candidate_slot_seeded",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %stored.id,
                provider_id = %plan.provider_id,
                endpoint_id = %plan.endpoint_id,
                key_id = %plan.key_id,
                source = "seed",
                "gateway seeded execution request candidate slot"
            );
            stored.id
        }
        Ok(Ok(None)) => {
            warn!(
                event_name = "request_candidate_writer_unavailable",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %generated_candidate_id,
                provider_id = %plan.provider_id,
                endpoint_id = %plan.endpoint_id,
                key_id = %plan.key_id,
                source = "seed",
                "gateway skipped request candidate seed because writer is unavailable"
            );
            generated_candidate_id
        }
        Ok(Err(err)) => {
            warn!(
                event_name = "request_candidate_slot_seed_failed",
                log_type = "event",
                request_id = %request_id,
                error = ?err,
                "gateway failed to seed execution request candidate slot"
            );
            generated_candidate_id
        }
        Err(_) => {
            let timeout_ms = request_candidate_seed_write_timeout().as_millis() as u64;
            warn!(
                event_name = "request_candidate_slot_seed_timed_out",
                log_type = "event",
                request_id = %request_id,
                candidate_id = %generated_candidate_id,
                provider_id = %plan.provider_id,
                endpoint_id = %plan.endpoint_id,
                key_id = %plan.key_id,
                source = "seed",
                timeout_ms,
                "gateway skipped blocking request candidate seed after timeout"
            );
            generated_candidate_id
        }
    };

    plan.candidate_id = Some(candidate_id.clone());
    *report_context = Some(finalize_execution_request_candidate_report_context(
        seed.report_context,
        &candidate_id,
    ));
}

pub(crate) fn assign_execution_request_candidate_slot(
    plan: &mut ExecutionPlan,
    report_context: &mut Option<Value>,
) {
    let existing_candidate_id = plan
        .candidate_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let report_candidate_id = parse_request_candidate_report_context(report_context.as_ref())
        .and_then(|metadata| metadata.candidate_id);
    if existing_candidate_id.as_deref().is_some()
        && report_candidate_id.as_deref() == existing_candidate_id.as_deref()
    {
        return;
    }

    let seed = build_execution_request_candidate_seed(
        plan,
        report_context.as_ref(),
        current_unix_ms(),
        existing_candidate_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
    );
    let candidate_id = seed.upsert_record.id.clone();
    plan.candidate_id = Some(candidate_id.clone());
    *report_context = Some(finalize_execution_request_candidate_report_context(
        seed.report_context,
        &candidate_id,
    ));
}

pub(crate) async fn persist_available_local_candidate(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    trace_id: &str,
    user_id: &str,
    api_key_id: &str,
    candidate: &SchedulerMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    retry_index: u32,
    candidate_id: &str,
    required_capabilities: Option<&Value>,
    extra_data: Option<serde_json::Value>,
    created_at_unix_ms: u64,
    error_context: &'static str,
) -> String {
    if !should_persist_request_candidate_status(RequestCandidateStatus::Available) {
        return candidate_id.to_string();
    }
    match state
        .upsert_request_candidate(UpsertRequestCandidateRecord {
            id: candidate_id.to_string(),
            request_id: trace_id.to_string(),
            user_id: Some(user_id.to_string()),
            api_key_id: Some(api_key_id.to_string()),
            username: None,
            api_key_name: None,
            candidate_index,
            retry_index,
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
            extra_data,
            required_capabilities: required_capabilities.cloned(),
            created_at_unix_ms: Some(created_at_unix_ms),
            started_at_unix_ms: None,
            finished_at_unix_ms: None,
        })
        .await
    {
        Ok(Some(stored)) => {
            debug!(
                event_name = "request_candidate_status_persisted",
                log_type = "event",
                request_id = %short_request_id(trace_id),
                candidate_id = %stored.id,
                candidate_index,
                retry_index,
                status = "available",
                source = "planner_available",
                provider_id = %candidate.provider_id,
                endpoint_id = %candidate.endpoint_id,
                key_id = %candidate.key_id,
                has_required_capabilities = required_capabilities.is_some(),
                "gateway persisted available local request candidate"
            );
            stored.id
        }
        Ok(None) => {
            warn!(
                event_name = "request_candidate_writer_unavailable",
                log_type = "event",
                request_id = %short_request_id(trace_id),
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = "available",
                source = "planner_available",
                provider_id = %candidate.provider_id,
                endpoint_id = %candidate.endpoint_id,
                key_id = %candidate.key_id,
                "gateway skipped request candidate persistence because writer is unavailable"
            );
            candidate_id.to_string()
        }
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                candidate_id = %candidate_id,
                error = ?err,
                "{error_context}"
            );
            candidate_id.to_string()
        }
    }
}

pub(crate) async fn persist_skipped_local_candidate(
    state: &(impl RequestCandidateRuntimeWriter + ?Sized),
    trace_id: &str,
    user_id: &str,
    api_key_id: &str,
    candidate: &SchedulerMinimalCandidateSelectionCandidate,
    candidate_index: u32,
    retry_index: u32,
    candidate_id: &str,
    required_capabilities: Option<&Value>,
    skip_reason: &str,
    extra_data: Option<serde_json::Value>,
    finished_at_unix_ms: u64,
    error_context: &'static str,
) {
    if !should_persist_request_candidate_status(RequestCandidateStatus::Skipped) {
        return;
    }
    match state
        .upsert_request_candidate(UpsertRequestCandidateRecord {
            id: candidate_id.to_string(),
            request_id: trace_id.to_string(),
            user_id: Some(user_id.to_string()),
            api_key_id: Some(api_key_id.to_string()),
            username: None,
            api_key_name: None,
            candidate_index,
            retry_index,
            provider_id: Some(candidate.provider_id.clone()),
            endpoint_id: Some(candidate.endpoint_id.clone()),
            key_id: Some(candidate.key_id.clone()),
            status: RequestCandidateStatus::Skipped,
            skip_reason: Some(skip_reason.to_string()),
            is_cached: Some(false),
            status_code: None,
            error_type: None,
            error_message: None,
            latency_ms: None,
            concurrent_requests: None,
            extra_data,
            required_capabilities: required_capabilities.cloned(),
            created_at_unix_ms: None,
            started_at_unix_ms: None,
            finished_at_unix_ms: Some(finished_at_unix_ms),
        })
        .await
    {
        Ok(Some(stored)) => {
            debug!(
                event_name = "request_candidate_status_persisted",
                log_type = "event",
                request_id = %short_request_id(trace_id),
                candidate_id = %stored.id,
                candidate_index,
                retry_index,
                status = "skipped",
                skip_reason,
                source = "planner_skipped",
                provider_id = %candidate.provider_id,
                endpoint_id = %candidate.endpoint_id,
                key_id = %candidate.key_id,
                has_required_capabilities = required_capabilities.is_some(),
                "gateway persisted skipped local request candidate"
            );
        }
        Ok(None) => {
            warn!(
                event_name = "request_candidate_writer_unavailable",
                log_type = "event",
                request_id = %short_request_id(trace_id),
                candidate_id = %candidate_id,
                candidate_index,
                retry_index,
                status = "skipped",
                skip_reason,
                source = "planner_skipped",
                provider_id = %candidate.provider_id,
                endpoint_id = %candidate.endpoint_id,
                key_id = %candidate.key_id,
                "gateway skipped request candidate persistence because writer is unavailable"
            );
        }
        Err(err) => {
            warn!(
                trace_id = %trace_id,
                candidate_id = %candidate_id,
                skip_reason,
                error = ?err,
                "{error_context}"
            );
        }
    }
}

pub(crate) async fn resolve_locally_actionable_request_candidate_report_context(
    state: &(impl RequestCandidateRuntimeReader + ?Sized),
    context: &Value,
) -> Option<Value> {
    let request_id = context
        .get("request_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let existing_candidates = state
        .read_request_candidates_by_request_id(request_id)
        .await
        .ok()?;
    if existing_candidates.len() != 1 {
        return None;
    }

    build_locally_actionable_report_context_from_request_candidate(context, &existing_candidates[0])
}

async fn resolve_report_request_candidate_slot(
    state: &(impl RequestCandidateRuntimeReader + ?Sized),
    report_context: Option<&Value>,
) -> Option<SchedulerResolvedReportRequestCandidateSlot> {
    let metadata = parse_request_candidate_report_context(report_context)?;
    if metadata
        .request_id
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
        && metadata
            .candidate_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|value| !value.is_empty())
    {
        return resolve_report_request_candidate_slot_from_candidates(
            &[],
            metadata,
            current_unix_ms(),
            Uuid::new_v4().to_string(),
        );
    }

    let request_id = metadata.request_id.clone()?;
    let existing_candidates = state
        .read_request_candidates_by_request_id(request_id.as_str())
        .await
        .ok()
        .unwrap_or_default();
    resolve_report_request_candidate_slot_from_candidates(
        &existing_candidates,
        metadata,
        current_unix_ms(),
        Uuid::new_v4().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};

    use aether_contracts::{ExecutionPlan, RequestBody};
    use aether_data::repository::auth::{
        InMemoryAuthApiKeySnapshotRepository, StoredAuthApiKeyExportRecord,
    };
    use aether_data::repository::candidates::InMemoryRequestCandidateRepository;
    use aether_data::repository::usage::InMemoryUsageReadRepository;
    use aether_data_contracts::repository::candidates::{
        RequestCandidateReadRepository, RequestCandidateStatus, StoredRequestCandidate,
        UpsertRequestCandidateRecord,
    };
    use aether_data_contracts::repository::usage::StoredRequestUsageAudit;
    use aether_scheduler_core::SchedulerMinimalCandidateSelectionCandidate;
    use serde_json::json;

    use super::{
        ensure_execution_request_candidate_slot, persist_available_local_candidate,
        record_report_request_candidate_status, report_candidate_terminal_usage_is_durable,
        request_usage_is_durable_terminal, resolve_request_candidate_required_capabilities,
        select_requested_model_capabilities, snapshot_local_request_candidate_status,
        terminal_candidate_matches_usage, terminal_usage_matches_candidate,
        try_enqueue_local_request_candidate_status_snapshot, RequestCandidateRuntimeWriter,
        SchedulerRequestCandidateStatusUpdate,
    };
    use crate::data::GatewayDataState;
    use crate::AppState;

    fn build_test_state(repository: Arc<InMemoryRequestCandidateRepository>) -> AppState {
        AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    repository,
                    Arc::new(InMemoryUsageReadRepository::default()),
                ),
            )
    }

    fn build_test_state_with_auth(
        repository: Arc<InMemoryRequestCandidateRepository>,
        auth_repository: Arc<InMemoryAuthApiKeySnapshotRepository>,
    ) -> AppState {
        AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    repository,
                    Arc::new(InMemoryUsageReadRepository::default()),
                )
                .with_auth_api_key_reader(auth_repository),
            )
    }

    #[derive(Default)]
    struct SynchronousStatusWriter {
        records: Mutex<Vec<UpsertRequestCandidateRecord>>,
    }

    #[async_trait::async_trait]
    impl RequestCandidateRuntimeWriter for SynchronousStatusWriter {
        fn has_request_candidate_data_writer(&self) -> bool {
            true
        }

        async fn upsert_request_candidate(
            &self,
            _candidate: UpsertRequestCandidateRecord,
        ) -> Result<Option<StoredRequestCandidate>, crate::GatewayError> {
            panic!("synchronous status fast path must not call the async writer")
        }

        fn try_enqueue_request_candidate_status(
            &self,
            candidate: UpsertRequestCandidateRecord,
        ) -> Result<(), UpsertRequestCandidateRecord> {
            self.records
                .lock()
                .expect("synchronous status records lock")
                .push(candidate);
            Ok(())
        }
    }

    fn sample_plan() -> ExecutionPlan {
        ExecutionPlan {
            request_id: "req-request-candidate-seed-123".to_string(),
            candidate_id: None,
            provider_name: Some("openai".to_string()),
            provider_id: "provider-request-candidate-seed-123".to_string(),
            endpoint_id: "endpoint-request-candidate-seed-123".to_string(),
            key_id: "key-request-candidate-seed-123".to_string(),
            method: "POST".to_string(),
            url: "https://api.openai.example/v1/chat/completions".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: RequestBody::from_json(json!({"model": "gpt-5", "messages": []})),
            stream: false,
            client_api_format: "openai:chat".to_string(),
            provider_api_format: "openai:chat".to_string(),
            model_name: Some("gpt-5".to_string()),
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    #[test]
    fn terminal_candidate_reconciliation_requires_finalized_usage() {
        assert!(!request_usage_is_durable_terminal("completed", None));
        assert!(!request_usage_is_durable_terminal(
            "pending",
            Some(1_700_000_000)
        ));
        assert!(request_usage_is_durable_terminal(
            " COMPLETED ",
            Some(1_700_000_000)
        ));
        assert!(request_usage_is_durable_terminal(
            "failed",
            Some(1_700_000_001)
        ));
        assert!(request_usage_is_durable_terminal(
            "cancelled",
            Some(1_700_000_002)
        ));
    }

    #[test]
    fn terminal_candidate_reconciliation_requires_matching_usage_outcome() {
        assert!(terminal_candidate_matches_usage(
            RequestCandidateStatus::Success,
            " COMPLETED "
        ));
        assert!(terminal_candidate_matches_usage(
            RequestCandidateStatus::Failed,
            "failed"
        ));
        assert!(terminal_candidate_matches_usage(
            RequestCandidateStatus::Cancelled,
            "cancelled"
        ));

        assert!(!terminal_candidate_matches_usage(
            RequestCandidateStatus::Failed,
            "completed"
        ));
        assert!(!terminal_candidate_matches_usage(
            RequestCandidateStatus::Success,
            "cancelled"
        ));
        assert!(!terminal_candidate_matches_usage(
            RequestCandidateStatus::Cancelled,
            "failed"
        ));
        assert!(!terminal_candidate_matches_usage(
            RequestCandidateStatus::Streaming,
            "completed"
        ));
    }

    #[test]
    fn terminal_candidate_reconciliation_requires_matching_candidate_identity() {
        assert!(terminal_usage_matches_candidate(
            Some("candidate-a"),
            Some(" candidate-a "),
        ));
        assert!(terminal_usage_matches_candidate(None, None));
        assert!(!terminal_usage_matches_candidate(
            Some("candidate-a"),
            Some("candidate-b"),
        ));
        assert!(!terminal_usage_matches_candidate(Some("candidate-a"), None));
        assert!(!terminal_usage_matches_candidate(None, Some("candidate-a")));
    }

    #[tokio::test]
    async fn durable_report_gate_does_not_fallback_to_metadata_candidate_identity() {
        let mut usage = StoredRequestUsageAudit::new(
            "usage-report-typed-identity".to_string(),
            "req-report-typed-identity".to_string(),
            Some("user-1".to_string()),
            Some("api-key-1".to_string()),
            None,
            None,
            "provider-1".to_string(),
            "model-1".to_string(),
            None,
            Some("provider-1".to_string()),
            Some("endpoint-1".to_string()),
            Some("key-1".to_string()),
            Some("chat".to_string()),
            Some("openai:chat".to_string()),
            Some("openai".to_string()),
            Some("chat".to_string()),
            Some("openai:chat".to_string()),
            Some("openai".to_string()),
            Some("chat".to_string()),
            false,
            false,
            1,
            1,
            2,
            0.0,
            0.0,
            Some(200),
            None,
            None,
            Some(10),
            Some(2),
            "completed".to_string(),
            "settled".to_string(),
            1_000,
            1,
            Some(1),
        )
        .expect("terminal usage should build");
        usage.candidate_id = None;
        usage.candidate_index = None;
        usage.request_metadata = Some(json!({
            "candidate_id": "candidate-metadata-only",
            "candidate_index": 0,
        }));

        let state = AppState::new()
            .expect("gateway state should build")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::new(InMemoryRequestCandidateRepository::default()),
                    Arc::new(InMemoryUsageReadRepository::seed([usage])),
                ),
            );
        let report_context = json!({
            "request_id": "req-report-typed-identity",
            "candidate_id": "candidate-metadata-only",
        });

        assert!(
            !report_candidate_terminal_usage_is_durable(
                &state,
                Some(&report_context),
                RequestCandidateStatus::Success,
            )
            .await
        );
    }

    #[test]
    fn streaming_snapshot_uses_synchronous_status_enqueue_fast_path() {
        let mut plan = sample_plan();
        plan.candidate_id = Some("candidate-streaming-fast-path".to_string());
        let snapshot = snapshot_local_request_candidate_status(&plan, None)
            .expect("candidate snapshot should build");
        let writer = SynchronousStatusWriter::default();

        try_enqueue_local_request_candidate_status_snapshot(
            &writer,
            &snapshot,
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Streaming,
                status_code: Some(200),
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(123),
                finished_at_unix_ms: None,
            },
        )
        .expect("streaming status should use the synchronous enqueue path");

        let records = writer
            .records
            .lock()
            .expect("synchronous status records lock");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].status, RequestCandidateStatus::Streaming);
        assert_eq!(records[0].status_code, Some(200));
    }

    fn sample_minimal_candidate() -> SchedulerMinimalCandidateSelectionCandidate {
        SchedulerMinimalCandidateSelectionCandidate {
            provider_id: "provider-1".to_string(),
            provider_name: "Provider".to_string(),
            provider_type: "custom".to_string(),
            provider_priority: 0,
            endpoint_id: "endpoint-1".to_string(),
            endpoint_api_format: "openai:chat".to_string(),
            key_id: "provider-key-1".to_string(),
            key_name: "provider-key-1".to_string(),
            key_auth_type: "api_key".to_string(),
            key_internal_priority: 0,
            key_global_priority_for_format: Some(0),
            key_capabilities: Some(json!({"provider_only_capability": true})),
            model_id: "model-1".to_string(),
            global_model_id: "global-model-1".to_string(),
            global_model_name: "gpt-5".to_string(),
            selected_provider_model_name: "gpt-5".to_string(),
            supports_streaming: true,
            mapping_matched_model: None,
        }
    }

    #[tokio::test]
    async fn seeds_execution_request_candidate_slot_for_plan_without_candidate_id() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = build_test_state(Arc::clone(&repository));
        let mut plan = sample_plan();
        let mut report_context = Some(json!({
            "request_id": "req-request-candidate-seed-123",
            "client_api_format": "openai:chat"
        }));

        ensure_execution_request_candidate_slot(&state, &mut plan, &mut report_context).await;

        let candidate_id = plan
            .candidate_id
            .clone()
            .expect("candidate id should be seeded");
        let report_context = report_context.expect("report context should be populated");
        assert_eq!(
            report_context
                .get("candidate_id")
                .and_then(|value| value.as_str()),
            Some(candidate_id.as_str())
        );
        assert_eq!(
            report_context
                .get("candidate_index")
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert_eq!(
            report_context
                .get("provider_id")
                .and_then(|value| value.as_str()),
            Some("provider-request-candidate-seed-123")
        );

        let stored = repository
            .list_by_request_id("req-request-candidate-seed-123")
            .await
            .expect("request candidates should read");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, candidate_id);
        assert_eq!(stored[0].status, RequestCandidateStatus::Pending);
        assert_eq!(
            stored[0].provider_id.as_deref(),
            Some("provider-request-candidate-seed-123")
        );
        assert_eq!(
            stored[0].endpoint_id.as_deref(),
            Some("endpoint-request-candidate-seed-123")
        );
        assert_eq!(
            stored[0].key_id.as_deref(),
            Some("key-request-candidate-seed-123")
        );
    }

    #[tokio::test]
    async fn does_not_reseed_execution_request_candidate_slot_when_report_context_matches_plan_candidate_id(
    ) {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = build_test_state(Arc::clone(&repository));
        let mut plan = sample_plan();
        plan.candidate_id = Some("cand-existing-123".to_string());
        let mut report_context = Some(json!({
            "request_id": "req-request-candidate-seed-123",
            "candidate_id": "cand-existing-123"
        }));

        ensure_execution_request_candidate_slot(&state, &mut plan, &mut report_context).await;

        assert_eq!(plan.candidate_id.as_deref(), Some("cand-existing-123"));
        let stored = repository
            .list_by_request_id("req-request-candidate-seed-123")
            .await
            .expect("request candidates should read");
        assert!(stored.is_empty());
        assert_eq!(
            report_context
                .as_ref()
                .and_then(|value| value.get("candidate_id"))
                .and_then(|value| value.as_str()),
            Some("cand-existing-123")
        );
    }

    #[tokio::test]
    async fn seeds_execution_request_candidate_slot_when_plan_candidate_id_lacks_report_context() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = build_test_state(Arc::clone(&repository));
        let mut plan = sample_plan();
        plan.candidate_id = Some("cand-existing-123".to_string());
        let mut report_context = None;

        ensure_execution_request_candidate_slot(&state, &mut plan, &mut report_context).await;

        assert_eq!(plan.candidate_id.as_deref(), Some("cand-existing-123"));
        let report_context = report_context.expect("report context should be populated");
        assert_eq!(
            report_context
                .get("candidate_id")
                .and_then(|value| value.as_str()),
            Some("cand-existing-123")
        );
        let stored = repository
            .list_by_request_id("req-request-candidate-seed-123")
            .await
            .expect("request candidates should read");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, "cand-existing-123");
        assert_eq!(stored[0].status, RequestCandidateStatus::Pending);
    }

    #[tokio::test]
    async fn defers_report_request_candidate_status_until_usage_is_visible() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::seed(vec![
            StoredRequestCandidate::new(
                "cand-report-123".to_string(),
                "req-report-123".to_string(),
                Some("user-1".to_string()),
                Some("api-key-1".to_string()),
                None,
                None,
                0,
                0,
                Some("provider-report-123".to_string()),
                Some("endpoint-report-123".to_string()),
                Some("key-report-123".to_string()),
                RequestCandidateStatus::Pending,
                None,
                false,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                100_000,
                Some(100_000),
                None,
            )
            .expect("request candidate should build"),
        ]));
        let state = build_test_state(Arc::clone(&repository));
        let report_context = json!({
            "request_id": "req-report-123",
            "candidate_id": "cand-report-123",
            "candidate_index": 0,
            "retry_index": 0,
            "provider_id": "provider-report-123",
            "endpoint_id": "endpoint-report-123",
            "key_id": "key-report-123"
        });

        record_report_request_candidate_status(
            &state,
            Some(&report_context),
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Success,
                status_code: Some(200),
                error_type: None,
                error_message: None,
                latency_ms: Some(25),
                started_at_unix_ms: Some(101),
                finished_at_unix_ms: Some(102),
            },
        )
        .await;

        let stored = repository
            .list_by_request_id("req-report-123")
            .await
            .expect("request candidates should read");
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].id, "cand-report-123");
        assert_eq!(stored[0].status, RequestCandidateStatus::Pending);
        assert_eq!(stored[0].status_code, None);
        assert_eq!(stored[0].latency_ms, None);
        assert_eq!(stored[0].started_at_unix_ms, Some(100_000));
        assert_eq!(stored[0].finished_at_unix_ms, None);
    }

    #[tokio::test]
    async fn resolves_request_candidate_required_capabilities_from_user_model_and_api_key() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let auth_repository = Arc::new(
            InMemoryAuthApiKeySnapshotRepository::default().with_export_records(vec![
                StoredAuthApiKeyExportRecord::new(
                    "user-1".to_string(),
                    "api-key-1".to_string(),
                    "hash-1".to_string(),
                    None,
                    Some("default".to_string()),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(json!({"cache_1h": false, "context_1m": true})),
                    true,
                    None,
                    false,
                    0,
                    0,
                    0.0,
                    false,
                )
                .expect("export record should build"),
            ]),
        );
        let state = build_test_state_with_auth(repository, auth_repository)
            .with_auth_user_model_capability_settings_for_tests(
                "user-1",
                json!({
                    "gpt-5": {
                        "cache_1h": true,
                        "context_1m": false
                    }
                }),
            );
        let explicit_required_capabilities = json!({"gemini_files": true});

        let required_capabilities = resolve_request_candidate_required_capabilities(
            &state,
            "user-1",
            "api-key-1",
            Some("gpt-5"),
            Some(&explicit_required_capabilities),
            None,
        )
        .await
        .expect("required capabilities should resolve");

        assert_eq!(required_capabilities["cache_1h"], json!(false));
        assert_eq!(required_capabilities["context_1m"], json!(true));
        assert_eq!(required_capabilities["gemini_files"], json!(true));
    }

    #[test]
    fn requested_model_capabilities_use_the_policy_resolved_base_model() {
        let base_only = json!({
            "deployment-alias": {
                "context_1m": true
            }
        });
        assert_eq!(
            select_requested_model_capabilities(
                Some(&base_only),
                Some("deployment-alias-VendorFuture"),
                Some("deployment-alias"),
            ),
            Some(&base_only["deployment-alias"])
        );
        assert_eq!(
            select_requested_model_capabilities(
                Some(&base_only),
                Some("deployment-alias-VendorFuture"),
                None,
            ),
            None
        );

        let exact_and_base = json!({
            "deployment-alias-VendorFuture": {
                "cache_1h": true
            },
            "deployment-alias": {
                "context_1m": true
            }
        });
        assert_eq!(
            select_requested_model_capabilities(
                Some(&exact_and_base),
                Some("deployment-alias-VendorFuture"),
                Some("deployment-alias"),
            ),
            Some(&exact_and_base["deployment-alias-VendorFuture"])
        );
    }

    #[tokio::test]
    async fn persists_request_required_capabilities_instead_of_provider_key_capabilities() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = build_test_state(Arc::clone(&repository));
        let required_capabilities = json!({"cache_1h": true});

        persist_available_local_candidate(
            &state,
            "req-runtime-cap-123",
            "user-1",
            "api-key-1",
            &sample_minimal_candidate(),
            0,
            0,
            "cand-runtime-cap-123",
            Some(&required_capabilities),
            None,
            100_000,
            "request candidate persist should succeed",
        )
        .await;

        let stored = repository
            .list_by_request_id("req-runtime-cap-123")
            .await
            .expect("request candidates should read");
        assert_eq!(stored.len(), 1);
        assert_eq!(
            stored[0].required_capabilities,
            Some(required_capabilities.clone())
        );
        assert_ne!(
            stored[0].required_capabilities,
            sample_minimal_candidate().key_capabilities
        );
    }
}
