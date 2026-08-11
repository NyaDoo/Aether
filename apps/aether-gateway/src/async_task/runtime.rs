use std::time::Duration;

use aether_billing::enrich_usage_event_with_billing;
use aether_contracts::{ExecutionErrorKind, ExecutionResult};
use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, UpsertVideoTask, VideoTaskStatus,
};
use aether_usage_runtime::{build_upsert_usage_record_from_event, settle_usage_if_needed};
use serde_json::{Map, Value};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use crate::log_ids::short_request_id;
use crate::usage::{UsageEvent, UsageEventData, UsageEventType};
use crate::video_tasks::{LocalVideoTaskReadRefreshPlan, LocalVideoTaskSnapshot};
use crate::{AppState, GatewayError};

const MAX_VIDEO_TASK_POLL_BACKOFF_SECONDS: u64 = 300;
const VIDEO_TASK_POLL_CLAIM_SECONDS: u64 = 30;

#[derive(Debug, Clone)]
struct VideoTaskRefreshError {
    message: String,
    permanent: bool,
}

enum VideoTaskRefreshAttempt {
    Success { provider_body: Map<String, Value> },
    Error(VideoTaskRefreshError),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoTaskPollerConfig {
    pub(crate) interval: Duration,
    pub(crate) batch_size: usize,
}

pub(crate) async fn execute_video_task_refresh_plan(
    state: &AppState,
    refresh_plan: &LocalVideoTaskReadRefreshPlan,
) -> Result<bool, GatewayError> {
    let Some(runtime_snapshot) = state.video_tasks.snapshot_for_refresh_plan(refresh_plan) else {
        return Ok(false);
    };
    match fetch_video_task_refresh_attempt(state, refresh_plan).await? {
        VideoTaskRefreshAttempt::Success { provider_body } => {
            let task_id = runtime_snapshot.to_upsert_record().id;
            if let Some(task) = state.find_video_task_by_id(&task_id).await? {
                let Some(updated) = build_successful_poll_update(
                    &task,
                    &runtime_snapshot,
                    &provider_body,
                    now_unix_secs(),
                    false,
                )?
                else {
                    return Ok(false);
                };
                if let Some(stored) = state.update_active_video_task(updated).await? {
                    if let Some(snapshot) =
                        snapshot_with_runtime_transport(&stored, &runtime_snapshot)
                    {
                        state.video_tasks.record_snapshot(snapshot);
                    }
                    finalize_video_task_if_terminal(state, &stored).await;
                    return Ok(true);
                }

                // A concurrent refresh may already have made the row terminal.
                // Restore that database truth instead of regressing the memory
                // snapshot with a late active-state response.
                if let Some(stored) = state.find_video_task_by_id(&task_id).await? {
                    if let Some(snapshot) =
                        snapshot_with_runtime_transport(&stored, &runtime_snapshot)
                    {
                        state.video_tasks.record_snapshot(snapshot);
                    }
                    // The winning concurrent refresh normally finalizes usage,
                    // but settlement/upsert can fail transiently after its CAS
                    // succeeds. Re-running the idempotent terminal finalizer on
                    // the losing reader closes that reliability gap.
                    if stored.status != VideoTaskStatus::Deleted {
                        finalize_video_task_if_terminal(state, &stored).await;
                    }
                }
                return Ok(false);
            }

            // Freshly-created in-memory tasks are normally already persisted,
            // but retain the previous behavior if a custom repository has no
            // row yet. The upsert result still drives terminal settlement.
            let mut snapshot = runtime_snapshot;
            snapshot.apply_provider_body(&provider_body);
            let stored = state.upsert_video_task_snapshot(&snapshot).await?;
            state.video_tasks.record_snapshot(snapshot);
            if let Some(stored) = stored.as_ref() {
                finalize_video_task_if_terminal(state, stored).await;
            }
            Ok(true)
        }
        VideoTaskRefreshAttempt::Error(err) => {
            warn!(
                event_name = "video_task_refresh_failed",
                log_type = "event",
                error = %err.message,
                permanent = err.permanent,
                "gateway video task refresh failed"
            );
            Ok(false)
        }
    }
}

async fn poll_video_tasks_once(state: &AppState, batch_size: usize) -> Result<usize, GatewayError> {
    if !state.video_tasks.is_rust_authoritative() {
        return Ok(0);
    }
    let now_unix_secs = now_unix_secs();
    let tasks = state
        .claim_due_video_tasks(
            now_unix_secs,
            now_unix_secs.saturating_add(VIDEO_TASK_POLL_CLAIM_SECONDS),
            batch_size,
        )
        .await?;
    let mut refreshed = 0usize;
    for (index, task) in tasks.into_iter().enumerate() {
        let trace_id = format!("video-task-poller-{index}");
        let runtime_snapshot = state.reconstruct_video_task_snapshot(&task).await?;
        let Some(runtime_snapshot) = runtime_snapshot else {
            warn!(
                event_name = "video_task_transport_unavailable",
                log_type = "event",
                task_id = %task.id,
                "gateway could not reconstruct provider credentials for video task polling"
            );
            // `Ok(None)` means the provider/key no longer defines a usable
            // authenticated transport (deleted, disabled, expired, or otherwise
            // unsupported). Merely releasing the lease would retry forever
            // without consuming poll budget and leave the original usage record
            // pending indefinitely. Fail the active row and settle it exactly as
            // other permanent poll failures are handled.
            let updated = build_credential_unavailable_poll_update(&task, now_unix_secs);
            if let Some(stored) = state.update_active_video_task(updated).await? {
                // This path exists precisely because no current credential can
                // be reconstructed. Never repopulate runtime state directly
                // from a legacy DB snapshot that may still contain an old SK.
                state.video_tasks.hydrate_from_stored_task(&stored);
                finalize_video_task_if_terminal(state, &stored).await;
                refreshed += 1;
            }
            continue;
        };
        let Some(refresh_plan) = state
            .video_tasks
            .prepare_poll_refresh_plan_for_snapshot(&runtime_snapshot, &trace_id)
        else {
            continue;
        };

        match fetch_video_task_refresh_attempt(state, &refresh_plan).await? {
            VideoTaskRefreshAttempt::Success { provider_body } => {
                let Some(updated) = build_successful_poll_update(
                    &task,
                    &runtime_snapshot,
                    &provider_body,
                    now_unix_secs,
                    true,
                )?
                else {
                    continue;
                };
                match state.update_active_video_task(updated).await? {
                    Some(stored) => {
                        if let Some(snapshot) =
                            snapshot_with_runtime_transport(&stored, &runtime_snapshot)
                        {
                            state.video_tasks.record_snapshot(snapshot);
                        }
                        info!(
                            event_name = "video_task_status_updated",
                            log_type = "event",
                            request_id = %short_request_id(stored.request_id.as_str()),
                            task_id = %stored.id,
                            status = ?stored.status,
                            "gateway updated video task status from poll refresh"
                        );
                        finalize_video_task_if_terminal(state, &stored).await;
                        refreshed += 1;
                    }
                    None => continue,
                }
            }
            VideoTaskRefreshAttempt::Error(err) => {
                let updated = build_failed_poll_update(&task, &err, now_unix_secs);
                match state.update_active_video_task(updated).await? {
                    Some(stored) => {
                        if let Some(snapshot) =
                            snapshot_with_runtime_transport(&stored, &runtime_snapshot)
                        {
                            state.video_tasks.record_snapshot(snapshot);
                        }
                        info!(
                            event_name = "video_task_status_updated",
                            log_type = "event",
                            request_id = %short_request_id(stored.request_id.as_str()),
                            task_id = %stored.id,
                            status = ?stored.status,
                            "gateway updated video task status from poll refresh"
                        );
                        finalize_video_task_if_terminal(state, &stored).await;
                        refreshed += 1;
                    }
                    None => continue,
                }
            }
        }
    }
    Ok(refreshed)
}

fn snapshot_with_runtime_transport(
    task: &StoredVideoTask,
    runtime_snapshot: &LocalVideoTaskSnapshot,
) -> Option<LocalVideoTaskSnapshot> {
    let transport = match runtime_snapshot {
        LocalVideoTaskSnapshot::OpenAi(seed) => seed.transport.clone(),
        LocalVideoTaskSnapshot::Gemini(seed) => seed.transport.clone(),
        LocalVideoTaskSnapshot::Doubao(seed) => seed.transport.clone(),
    };
    LocalVideoTaskSnapshot::from_stored_task_with_transport(task, transport)
}

pub(crate) fn spawn_video_task_poller(state: AppState) -> Option<JoinHandle<()>> {
    let config = state.video_task_poller?;
    if !state.video_tasks.is_rust_authoritative() {
        return None;
    }

    Some(crate::task_runtime::spawn_singleton_worker(
        state,
        crate::task_runtime::TASK_KEY_VIDEO_TASK_POLLER,
        move |state| async move {
            let mut interval = tokio::time::interval(config.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            interval.tick().await;
            let mut deferred_since = None;
            loop {
                interval.tick().await;
                if state
                    .data
                    .should_defer_maintenance_for_database_pool_pressure(&mut deferred_since)
                {
                    debug!(
                        event_name = "video_task_poller_deferred",
                        log_type = "event",
                        "gateway video task poller deferred because database pool has no idle reserve"
                    );
                    continue;
                }
                if let Err(err) = poll_video_tasks_once(&state, config.batch_size).await {
                    warn!(
                        event_name = "video_task_poller_tick_failed",
                        log_type = "event",
                        error = ?err,
                        "gateway video task poller tick failed"
                    );
                }
            }
        },
    ))
}

async fn fetch_video_task_refresh_attempt(
    state: &AppState,
    refresh_plan: &LocalVideoTaskReadRefreshPlan,
) -> Result<VideoTaskRefreshAttempt, GatewayError> {
    let result = match crate::execution_runtime::execute_execution_runtime_sync_plan(
        state,
        None,
        &refresh_plan.plan,
    )
    .await
    {
        Ok(result) => result,
        Err(err) => {
            return Ok(VideoTaskRefreshAttempt::Error(VideoTaskRefreshError {
                message: format!("{err:?}"),
                permanent: false,
            }));
        }
    };
    if result.status_code >= 400 {
        return Ok(VideoTaskRefreshAttempt::Error(
            classify_refresh_result_error(&result),
        ));
    }

    let Some(provider_body) = result
        .body
        .and_then(|body| body.json_body)
        .and_then(|body| body.as_object().cloned())
    else {
        return Ok(VideoTaskRefreshAttempt::Error(VideoTaskRefreshError {
            message: "video task refresh missing json provider body".to_string(),
            permanent: false,
        }));
    };

    Ok(VideoTaskRefreshAttempt::Success { provider_body })
}

fn classify_refresh_result_error(result: &ExecutionResult) -> VideoTaskRefreshError {
    let status_code = result
        .error
        .as_ref()
        .and_then(|error| error.upstream_status)
        .unwrap_or(result.status_code);
    let message = result
        .error
        .as_ref()
        .map(|error| error.message.clone())
        .or_else(|| {
            result
                .body
                .as_ref()
                .and_then(|body| body.json_body.as_ref())
                .and_then(|value| value.get("error"))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| format!("upstream returned {status_code}"));
    let permanent = result.error.as_ref().map_or(
        matches!(status_code, 400 | 401 | 403 | 404 | 422),
        |error| match error.kind {
            ExecutionErrorKind::Upstream4xx => !matches!(status_code, 408 | 409 | 429),
            ExecutionErrorKind::Upstream5xx
            | ExecutionErrorKind::ConnectTimeout
            | ExecutionErrorKind::FirstByteTimeout
            | ExecutionErrorKind::ReadTimeout
            | ExecutionErrorKind::TlsError
            | ExecutionErrorKind::ProxyError
            | ExecutionErrorKind::ProtocolError
            | ExecutionErrorKind::Internal => false,
            ExecutionErrorKind::Cancelled => true,
        },
    );

    VideoTaskRefreshError { message, permanent }
}

fn build_successful_poll_update(
    task: &StoredVideoTask,
    runtime_snapshot: &LocalVideoTaskSnapshot,
    provider_body: &Map<String, Value>,
    now_unix_secs: u64,
    consume_poll_budget: bool,
) -> Result<Option<UpsertVideoTask>, GatewayError> {
    // The runtime snapshot may have been reconstructed entirely from database
    // columns for an older row that has no `rust_local_snapshot`.  Reuse it so
    // a successful upstream poll is never fetched and then silently discarded.
    let mut snapshot = runtime_snapshot.clone();
    snapshot.apply_provider_body(provider_body);

    let mut record = snapshot.to_upsert_record();
    record.id = task.id.clone();
    record.short_id = task.short_id.clone().or(record.short_id);
    record.request_id = task.request_id.clone();
    record.user_id = task.user_id.clone();
    record.api_key_id = task.api_key_id.clone();
    record.username = task.username.clone();
    record.api_key_name = task.api_key_name.clone();
    record.external_task_id = task.external_task_id.clone().or(record.external_task_id);
    record.provider_id = task.provider_id.clone();
    record.endpoint_id = task.endpoint_id.clone();
    record.key_id = task.key_id.clone();
    record.client_api_format = task.client_api_format.clone();
    record.provider_api_format = task.provider_api_format.clone();
    record.format_converted = task.format_converted;
    // The indexed model column intentionally keeps Ark's request-side model or
    // endpoint ID for exact list filtering. The provider-resolved response
    // model remains in the persisted local snapshot.
    record.model = task.model.clone().or(record.model);
    record.prompt = record.prompt.or_else(|| task.prompt.clone());
    record.original_request_body = task
        .original_request_body
        .clone()
        .or(record.original_request_body);
    // Provider-resolved generation dimensions are authoritative when present.
    // Ark's `frames` and `duration` are mutually exclusive, so do not revive a
    // stale duration column after a poll resolves the task to an exact frame
    // count stored in the Doubao snapshot.
    record.duration_seconds = if matches!(
        &snapshot,
        LocalVideoTaskSnapshot::Doubao(seed) if seed.frames.is_some()
    ) {
        None
    } else {
        record.duration_seconds.or(task.duration_seconds)
    };
    record.resolution = record.resolution.or_else(|| task.resolution.clone());
    record.aspect_ratio = record.aspect_ratio.or_else(|| task.aspect_ratio.clone());
    record.size = task.size.clone().or(record.size);
    record.created_at_unix_ms = task.created_at_unix_ms;
    record.submitted_at_unix_secs = task.submitted_at_unix_secs;
    // `updated_at` is part of Ark's public task contract. Preserve the
    // provider timestamp when it was supplied; otherwise use our poll time.
    if provider_body.get("updated_at").is_none() {
        record.updated_at_unix_secs = now_unix_secs;
    }
    record.retry_count = task.retry_count;
    record.poll_interval_seconds = task.poll_interval_seconds.max(1);
    record.poll_count = if consume_poll_budget {
        task.poll_count.saturating_add(1)
    } else {
        task.poll_count
    };
    record.max_poll_count = task.max_poll_count.max(1);
    record.next_poll_at_unix_secs = if record.status.is_active() {
        if consume_poll_budget {
            Some(now_unix_secs.saturating_add(u64::from(record.poll_interval_seconds)))
        } else {
            task.next_poll_at_unix_secs
        }
    } else {
        None
    };
    if !record.status.is_active() && record.completed_at_unix_secs.is_none() {
        record.completed_at_unix_secs = Some(now_unix_secs);
    }
    if consume_poll_budget
        && record.status.is_active()
        && record.poll_count >= record.max_poll_count
    {
        record.status = VideoTaskStatus::Failed;
        record.error_code = Some("poll_timeout".to_string());
        record.error_message = Some(format!("Task timed out after {} polls", record.poll_count));
        record.completed_at_unix_secs = Some(now_unix_secs);
        record.next_poll_at_unix_secs = None;
    }
    record.request_metadata = merge_video_task_request_metadata(
        task.request_metadata.clone(),
        &snapshot,
        Some(provider_body),
        None,
    )
    .map_err(|err| GatewayError::Internal(err.to_string()))?;

    Ok(Some(record))
}

fn build_failed_poll_update(
    task: &StoredVideoTask,
    err: &VideoTaskRefreshError,
    now_unix_secs: u64,
) -> UpsertVideoTask {
    let mut record = stored_task_to_upsert(task);
    record.updated_at_unix_secs = now_unix_secs;
    record.poll_count = task.poll_count.saturating_add(1);
    record.progress_message = Some(format!("Poll error: {}", err.message));
    if err.permanent {
        record.status = VideoTaskStatus::Failed;
        record.error_code = Some("poll_permanent_error".to_string());
        record.error_message = Some(err.message.clone());
        record.completed_at_unix_secs = Some(now_unix_secs);
        record.next_poll_at_unix_secs = None;
    } else {
        let backoff =
            compute_poll_backoff_seconds(task.poll_interval_seconds.max(1), task.retry_count);
        record.retry_count = task.retry_count.saturating_add(1);
        record.next_poll_at_unix_secs = Some(now_unix_secs.saturating_add(backoff));
    }
    if record.status.is_active() && record.poll_count >= record.max_poll_count {
        record.status = VideoTaskStatus::Failed;
        record.error_code = Some("poll_timeout".to_string());
        record.error_message = Some(format!("Task timed out after {} polls", record.poll_count));
        record.completed_at_unix_secs = Some(now_unix_secs);
        record.next_poll_at_unix_secs = None;
    }
    record.request_metadata = LocalVideoTaskSnapshot::from_stored_task(task)
        .and_then(|snapshot| {
            merge_video_task_request_metadata(
                task.request_metadata.clone(),
                &snapshot,
                None,
                Some(err),
            )
            .ok()
            .flatten()
        })
        .or(task.request_metadata.clone());
    record
}

fn build_credential_unavailable_poll_update(
    task: &StoredVideoTask,
    now_unix_secs: u64,
) -> UpsertVideoTask {
    let unavailable = VideoTaskRefreshError {
        message: "video task provider credentials or transport are unavailable".to_string(),
        permanent: true,
    };
    let mut record = build_failed_poll_update(task, &unavailable, now_unix_secs);
    // Keep credential revocation/deletion distinguishable from an upstream
    // permanent 4xx. Operators and clients should not have to infer this
    // security failure from a generic poll error code.
    record.error_code = Some("credential_unavailable".to_string());
    record
}

fn stored_task_to_upsert(task: &StoredVideoTask) -> UpsertVideoTask {
    let snapshot_record =
        LocalVideoTaskSnapshot::from_stored_task(task).map(|snapshot| snapshot.to_upsert_record());
    UpsertVideoTask {
        id: task.id.clone(),
        short_id: task.short_id.clone(),
        request_id: task.request_id.clone(),
        user_id: task.user_id.clone(),
        api_key_id: task.api_key_id.clone(),
        username: task.username.clone(),
        api_key_name: task.api_key_name.clone(),
        external_task_id: task.external_task_id.clone(),
        provider_id: task.provider_id.clone(),
        endpoint_id: task.endpoint_id.clone(),
        key_id: task.key_id.clone(),
        client_api_format: task.client_api_format.clone(),
        provider_api_format: task.provider_api_format.clone(),
        format_converted: task.format_converted,
        model: task.model.clone(),
        prompt: task.prompt.clone().or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.prompt.clone())
        }),
        original_request_body: task.original_request_body.clone().or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.original_request_body.clone())
        }),
        duration_seconds: task.duration_seconds.or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.duration_seconds)
        }),
        resolution: task.resolution.clone().or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.resolution.clone())
        }),
        aspect_ratio: task.aspect_ratio.clone().or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.aspect_ratio.clone())
        }),
        size: task.size.clone().or_else(|| {
            snapshot_record
                .as_ref()
                .and_then(|record| record.size.clone())
        }),
        status: task.status,
        progress_percent: task.progress_percent,
        progress_message: task.progress_message.clone(),
        retry_count: task.retry_count,
        poll_interval_seconds: task.poll_interval_seconds.max(1),
        next_poll_at_unix_secs: task.next_poll_at_unix_secs,
        poll_count: task.poll_count,
        max_poll_count: task.max_poll_count.max(1),
        created_at_unix_ms: task.created_at_unix_ms,
        submitted_at_unix_secs: task.submitted_at_unix_secs,
        completed_at_unix_secs: task.completed_at_unix_secs,
        updated_at_unix_secs: task.updated_at_unix_secs,
        error_code: task.error_code.clone(),
        error_message: task.error_message.clone(),
        video_url: task.video_url.clone(),
        request_metadata: task.request_metadata.clone(),
    }
}

fn compute_poll_backoff_seconds(poll_interval_seconds: u32, retry_count: u32) -> u64 {
    let exponent = retry_count.min(5);
    let multiplier = 1u64 << exponent;
    u64::from(poll_interval_seconds)
        .saturating_mul(multiplier)
        .min(MAX_VIDEO_TASK_POLL_BACKOFF_SECONDS)
}

fn merge_video_task_request_metadata(
    existing: Option<Value>,
    snapshot: &LocalVideoTaskSnapshot,
    provider_body: Option<&Map<String, Value>>,
    poll_error: Option<&VideoTaskRefreshError>,
) -> Result<Option<Value>, serde_json::Error> {
    let mut metadata = match existing {
        Some(Value::Object(object)) => object,
        _ => Map::new(),
    };
    // Older rows may already contain an unfiltered poll response.  Clean it
    // whenever the row is touched so a transient poll error cannot preserve a
    // legacy signed URL indefinitely.
    if let Some(raw) = metadata
        .get("poll_raw_response")
        .and_then(Value::as_object)
        .cloned()
    {
        metadata.insert(
            "poll_raw_response".to_string(),
            Value::Object(sanitize_poll_raw_response(&raw)),
        );
    } else {
        metadata.remove("poll_raw_response");
    }
    metadata.insert(
        "rust_owner".to_string(),
        Value::String("async_task".to_string()),
    );
    metadata.insert(
        "rust_local_snapshot".to_string(),
        serde_json::to_value(snapshot.redacted_for_persistence())?,
    );
    if let Some(provider_body) = provider_body {
        metadata.insert(
            "poll_raw_response".to_string(),
            Value::Object(sanitize_poll_raw_response(provider_body)),
        );
        metadata.remove("poll_error");
    }
    if let Some(poll_error) = poll_error {
        metadata.insert(
            "poll_error".to_string(),
            serde_json::json!({
                "message": poll_error.message,
                "permanent": poll_error.permanent,
                "observed_at_unix_secs": now_unix_secs(),
            }),
        );
    }
    Ok(Some(Value::Object(metadata)))
}

/// Keep only provider fields needed to recover a video task after restart.
/// Poll responses are not an API response cache: retaining arbitrary nested
/// data here can persist pre-signed asset URLs, provider request headers, or
/// future secret-bearing fields.  The live snapshot has already extracted the
/// supported content/usage fields before this function is called.
fn sanitize_poll_raw_response(provider_body: &Map<String, Value>) -> Map<String, Value> {
    const SAFE_SCALAR_FIELDS: &[&str] = &[
        "id",
        "object",
        "status",
        "done",
        "model",
        "created_at",
        "updated_at",
        "completed_at",
        "expires_at",
        "progress",
        "resolution",
        "ratio",
        "seed",
        "frames",
        "framespersecond",
        "duration",
    ];

    let mut sanitized = Map::new();
    for field in SAFE_SCALAR_FIELDS {
        if let Some(value) = provider_body.get(*field) {
            // Scalar-only fields are safe to retain.  A malicious provider
            // cannot smuggle a nested URL/object through an allowlisted key.
            if value.is_null() || value.is_string() || value.is_number() || value.is_boolean() {
                sanitized.insert((*field).to_string(), value.clone());
            }
        }
    }

    if let Some(usage) = provider_body.get("usage").and_then(Value::as_object) {
        let mut safe_usage = Map::new();
        for field in [
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        ] {
            if let Some(value) = usage.get(field).filter(|value| value.is_number()) {
                safe_usage.insert(field.to_string(), value.clone());
            }
        }
        if !safe_usage.is_empty() {
            sanitized.insert("usage".to_string(), Value::Object(safe_usage));
        }
    }

    if let Some(error) = provider_body.get("error").and_then(Value::as_object) {
        let mut safe_error = Map::new();
        for field in ["code", "message", "type", "param"] {
            if let Some(value) = error.get(field).filter(|value| {
                value.is_null() || value.is_string() || value.is_number() || value.is_boolean()
            }) {
                safe_error.insert(field.to_string(), value.clone());
            }
        }
        if !safe_error.is_empty() {
            sanitized.insert("error".to_string(), Value::Object(safe_error));
        }
    }

    sanitized
}

pub(crate) async fn finalize_video_task_if_terminal(state: &AppState, task: &StoredVideoTask) {
    let Some(event) = build_video_task_terminal_usage_event(task) else {
        return;
    };
    let mut event = event;
    if let Err(err) = enrich_usage_event_with_billing(state.data.as_ref(), &mut event).await {
        warn!(
            event_name = "video_task_finalize_billing_enrichment_failed",
            log_type = "event",
            request_id = %short_request_id(task.request_id.as_str()),
            error = %err,
            "gateway video task finalize failed to enrich billing"
        );
    }
    let billing_resolved = terminal_video_billing_is_resolved(&event);
    if !billing_resolved {
        warn!(
            event_name = "video_task_finalize_billing_unresolved",
            log_type = "event",
            request_id = %short_request_id(task.request_id.as_str()),
            provider_id = ?event.data.provider_id,
            model = %event.data.model,
            target_model = ?event.data.target_model,
            input_tokens = ?event.data.input_tokens,
            output_tokens = ?event.data.output_tokens,
            "gateway left completed video usage pending because billing could not resolve a price"
        );
    }
    match build_upsert_usage_record_from_event(&event) {
        Ok(record) => match state.data.upsert_usage(record).await {
            Ok(Some(stored)) => {
                if billing_resolved {
                    if let Err(err) = settle_usage_if_needed(state.data.as_ref(), &stored).await {
                        warn!(
                            event_name = "video_task_finalize_settlement_failed",
                            log_type = "event",
                            request_id = %short_request_id(task.request_id.as_str()),
                            error = %err,
                            "gateway video task finalize failed to settle usage"
                        );
                    }
                }
            }
            Ok(None) => {}
            Err(err) => {
                warn!(
                    event_name = "video_task_finalize_usage_upsert_failed",
                    log_type = "event",
                    request_id = %short_request_id(task.request_id.as_str()),
                    error = %err,
                    "gateway video task finalize failed to upsert usage"
                );
            }
        },
        Err(err) => {
            warn!(
                event_name = "video_task_finalize_usage_build_failed",
                log_type = "event",
                request_id = %short_request_id(task.request_id.as_str()),
                error = %err,
                "gateway video task finalize failed to build usage record"
            );
        }
    }
}

fn terminal_video_billing_is_resolved(event: &UsageEvent) -> bool {
    !matches!(event.event_type, UsageEventType::Completed)
        || (event.data.total_cost_usd.is_some() && event.data.actual_total_cost_usd.is_some())
}

/// Publishes the billing dimensions a finished video task priced by.
///
/// Billing reads dimensions out of `request_metadata.dimensions` (the same path
/// image pricing uses), so resolution, duration and whether the request supplied
/// a reference video have to be surfaced there. The task's existing metadata is
/// preserved; only the dimension bag is merged in.
fn build_video_task_billing_dimensions(task: &StoredVideoTask) -> Value {
    let mut metadata = match task.request_metadata.clone() {
        Some(Value::Object(object)) => object,
        _ => Map::new(),
    };
    let mut dimensions = match metadata.get("dimensions") {
        Some(Value::Object(object)) => object.clone(),
        _ => Map::new(),
    };

    if let Some(resolution) = task
        .resolution
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        dimensions.insert("video_resolution".to_string(), Value::from(resolution));
    }
    if let Some(duration_seconds) = task.duration_seconds {
        dimensions.insert(
            "video_duration_seconds".to_string(),
            Value::from(duration_seconds),
        );
    }
    let has_video_input = task
        .original_request_body
        .as_ref()
        .is_some_and(aether_video_tasks_core::doubao_content_has_video_input);
    dimensions.insert(
        "video_has_video_input".to_string(),
        Value::from(has_video_input),
    );

    metadata.insert("dimensions".to_string(), Value::Object(dimensions));
    Value::Object(metadata)
}

fn build_video_task_terminal_usage_event(task: &StoredVideoTask) -> Option<UsageEvent> {
    let event_type = match task.status {
        VideoTaskStatus::Completed => UsageEventType::Completed,
        VideoTaskStatus::Failed | VideoTaskStatus::Expired => UsageEventType::Failed,
        VideoTaskStatus::Cancelled | VideoTaskStatus::Deleted => UsageEventType::Cancelled,
        VideoTaskStatus::Pending
        | VideoTaskStatus::Submitted
        | VideoTaskStatus::Queued
        | VideoTaskStatus::Processing => {
            return None;
        }
    };
    let snapshot = LocalVideoTaskSnapshot::from_stored_task(task);
    let provider_name = snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.provider_name().map(str::to_string))
        .or_else(|| task.provider_id.clone())
        .unwrap_or_else(|| "unknown".to_string());
    let provider_model = snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.provider_model_name().map(str::to_string))
        .or_else(|| task.model.clone())
        .unwrap_or_else(|| "unknown".to_string());
    // Provider task responses may report a canonical model family that differs
    // from the configured global/provider model used during dispatch. Keep the
    // response-reported value as the observed model, but carry the dispatch
    // model as `target_model` so billing can still resolve the exact pricing
    // context (and fall back to the observed model when both names coincide).
    let target_model = snapshot
        .as_ref()
        .and_then(|snapshot| match snapshot {
            LocalVideoTaskSnapshot::OpenAi(seed) => seed.transport.model_name.as_deref(),
            LocalVideoTaskSnapshot::Gemini(seed) => seed.transport.model_name.as_deref(),
            LocalVideoTaskSnapshot::Doubao(seed) => seed.transport.model_name.as_deref(),
        })
        .or_else(|| task.model.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    // Doubao bills video generation by tokens rather than by duration, so the
    // reported usage has to reach the billing pipeline like a chat completion.
    let usage_tokens = snapshot.and_then(|snapshot| snapshot.usage_tokens());
    let response_time_ms = task
        .submitted_at_unix_secs
        .zip(
            task.completed_at_unix_secs
                .or(Some(task.updated_at_unix_secs)),
        )
        .map(|(submitted, completed)| completed.saturating_sub(submitted).saturating_mul(1_000));
    let status_code = match event_type {
        UsageEventType::Completed => Some(200),
        UsageEventType::Cancelled => Some(499),
        UsageEventType::Failed => Some(500),
        UsageEventType::Pending | UsageEventType::Streaming => None,
    };

    Some(UsageEvent::new(
        event_type,
        task.request_id.clone(),
        UsageEventData {
            user_id: task.user_id.clone(),
            api_key_id: task.api_key_id.clone(),
            username: task.username.clone(),
            api_key_name: task.api_key_name.clone(),
            provider_name,
            model: provider_model,
            target_model,
            provider_id: task.provider_id.clone(),
            provider_endpoint_id: task.endpoint_id.clone(),
            provider_api_key_id: task.key_id.clone(),
            request_type: Some("video".to_string()),
            api_format: task.client_api_format.clone(),
            endpoint_api_format: task.provider_api_format.clone(),
            has_format_conversion: Some(task.format_converted),
            is_stream: Some(false),
            status_code,
            error_message: task.error_message.clone().or(task.error_code.clone()),
            response_time_ms,
            request_body: task.original_request_body.clone(),
            request_metadata: Some(build_video_task_billing_dimensions(task)),
            input_tokens: usage_tokens.map(|(completion_tokens, total_tokens)| {
                total_tokens.saturating_sub(completion_tokens)
            }),
            output_tokens: usage_tokens.map(|(completion_tokens, _)| completion_tokens),
            total_tokens: usage_tokens.map(|(_, total_tokens)| total_tokens),
            ..UsageEventData::default()
        },
    ))
}

fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{
        build_credential_unavailable_poll_update, build_failed_poll_update,
        build_successful_poll_update, build_video_task_billing_dimensions,
        build_video_task_terminal_usage_event, sanitize_poll_raw_response,
        snapshot_with_runtime_transport, stored_task_to_upsert, terminal_video_billing_is_resolved,
        VideoTaskRefreshError,
    };
    use crate::video_tasks::{
        LocalVideoTaskPersistence, LocalVideoTaskSnapshot, LocalVideoTaskStatus,
        LocalVideoTaskTransport, OpenAiVideoTaskSeed,
    };
    use aether_data_contracts::repository::video_tasks::{StoredVideoTask, VideoTaskStatus};
    use aether_video_tasks_core::DoubaoVideoTaskSeed;
    use serde_json::json;
    use std::collections::BTreeMap;

    fn sample_sparse_stored_task() -> StoredVideoTask {
        let snapshot = LocalVideoTaskSnapshot::OpenAi(OpenAiVideoTaskSeed {
            local_task_id: "task-1".to_string(),
            upstream_task_id: "ext-1".to_string(),
            created_at_unix_ms: 1,
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            model: Some("sora-2".to_string()),
            prompt: Some("hello".to_string()),
            size: Some("1280x720".to_string()),
            seconds: Some("4".to_string()),
            remixed_from_video_id: None,
            status: LocalVideoTaskStatus::Processing,
            progress_percent: 50,
            completed_at_unix_secs: None,
            expires_at_unix_secs: None,
            error_code: None,
            error_message: None,
            video_url: None,
            persistence: LocalVideoTaskPersistence {
                request_id: "request-1".to_string(),
                username: Some("user".to_string()),
                api_key_name: Some("primary".to_string()),
                client_api_format: "openai:video".to_string(),
                provider_api_format: "openai:video".to_string(),
                original_request_body: json!({
                    "prompt": "hello",
                    "seconds": "4",
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "size": "1280x720"
                }),
                format_converted: false,
            },
            transport: LocalVideoTaskTransport {
                upstream_base_url: "https://example.com".to_string(),
                provider_name: Some("provider".to_string()),
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "key-1".to_string(),
                headers: BTreeMap::new(),
                content_type: Some("application/json".to_string()),
                model_name: Some("sora-2".to_string()),
                proxy: None,
                transport_profile: None,
                timeouts: None,
            },
        });

        StoredVideoTask {
            id: "task-1".to_string(),
            short_id: Some("short-task-1".to_string()),
            request_id: "request-1".to_string(),
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            username: Some("user".to_string()),
            api_key_name: Some("primary".to_string()),
            external_task_id: Some("ext-1".to_string()),
            provider_id: Some("provider-1".to_string()),
            endpoint_id: Some("endpoint-1".to_string()),
            key_id: Some("key-1".to_string()),
            client_api_format: Some("openai:video".to_string()),
            provider_api_format: Some("openai:video".to_string()),
            format_converted: false,
            model: Some("sora-2".to_string()),
            prompt: None,
            original_request_body: None,
            duration_seconds: None,
            resolution: None,
            aspect_ratio: None,
            size: None,
            status: VideoTaskStatus::Processing,
            progress_percent: 50,
            progress_message: Some("polling".to_string()),
            retry_count: 1,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: Some(20),
            poll_count: 2,
            max_poll_count: 360,
            created_at_unix_ms: 1,
            submitted_at_unix_secs: Some(1),
            completed_at_unix_secs: None,
            updated_at_unix_secs: 20,
            error_code: None,
            error_message: None,
            video_url: None,
            request_metadata: Some(json!({
                "rust_local_snapshot": serde_json::to_value(snapshot)
                    .expect("snapshot should serialize")
            })),
        }
    }

    #[test]
    fn poll_raw_response_allowlist_drops_signed_assets_and_credentials() {
        let provider_body = json!({
            "id": "cgt-1",
            "status": "succeeded",
            "updated_at": 123,
            "frames": 121,
            "framespersecond": 24,
            "usage": {"completion_tokens": 8, "total_tokens": 10},
            "content": {
                "video_url": "https://cdn.example/video.mp4?X-Amz-Signature=secret",
                "last_frame_url": "https://cdn.example/frame.jpg?token=secret"
            },
            "authorization": "Bearer provider-secret",
            "future_secret": {"token": "secret"}
        });
        let sanitized = sanitize_poll_raw_response(provider_body.as_object().unwrap());

        assert_eq!(sanitized["status"], "succeeded");
        assert_eq!(sanitized["frames"], 121);
        assert_eq!(sanitized["usage"]["total_tokens"], 10);
        assert!(sanitized.get("content").is_none());
        assert!(sanitized.get("authorization").is_none());
        assert!(sanitized.get("future_secret").is_none());
        assert!(!serde_json::to_string(&sanitized)
            .unwrap()
            .contains("X-Amz-Signature"));
    }

    #[test]
    fn touching_legacy_poll_metadata_replaces_unfiltered_payload() {
        let task = sample_sparse_stored_task();
        let snapshot = LocalVideoTaskSnapshot::from_stored_task(&task).unwrap();
        let metadata = super::merge_video_task_request_metadata(
            Some(json!({
                "poll_raw_response": {
                    "content": {"video_url": "https://cdn.example/video?sig=old"},
                    "status": "processing"
                }
            })),
            &snapshot,
            None,
            Some(&VideoTaskRefreshError {
                message: "temporary".to_string(),
                permanent: false,
            }),
        )
        .unwrap()
        .unwrap();

        assert!(metadata["poll_raw_response"].get("content").is_none());
        assert_eq!(metadata["poll_raw_response"]["status"], "processing");
        assert!(!metadata.to_string().contains("sig=old"));
    }

    #[test]
    fn stored_task_to_upsert_restores_sparse_fields_from_snapshot() {
        let record = stored_task_to_upsert(&sample_sparse_stored_task());

        assert_eq!(record.prompt.as_deref(), Some("hello"));
        assert_eq!(
            record.original_request_body,
            Some(json!({
                "prompt": "hello",
                "seconds": "4",
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "size": "1280x720"
            }))
        );
        assert_eq!(record.duration_seconds, Some(4));
        assert_eq!(record.resolution.as_deref(), Some("720p"));
        assert_eq!(record.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(record.size.as_deref(), Some("1280x720"));
    }

    #[test]
    fn failed_poll_update_keeps_snapshot_backed_request_body() {
        let record = build_failed_poll_update(
            &sample_sparse_stored_task(),
            &VideoTaskRefreshError {
                message: "temporary failure".to_string(),
                permanent: false,
            },
            100,
        );

        assert_eq!(
            record.original_request_body,
            Some(json!({
                "prompt": "hello",
                "seconds": "4",
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "size": "1280x720"
            }))
        );
        assert_eq!(record.prompt.as_deref(), Some("hello"));
        assert_eq!(record.resolution.as_deref(), Some("720p"));
    }

    #[test]
    fn unavailable_transport_is_a_terminal_poll_failure() {
        let task = sample_sparse_stored_task();
        let record = build_credential_unavailable_poll_update(&task, 100);

        assert_eq!(record.request_id, task.request_id);
        assert_eq!(record.status, VideoTaskStatus::Failed);
        assert_eq!(record.error_code.as_deref(), Some("credential_unavailable"));
        assert_eq!(record.completed_at_unix_secs, Some(100));
        assert_eq!(record.next_poll_at_unix_secs, None);
    }

    #[test]
    fn successful_poll_uses_reconstructed_snapshot_when_metadata_is_missing() {
        let task_with_snapshot = sample_sparse_stored_task();
        let runtime_snapshot = LocalVideoTaskSnapshot::from_stored_task(&task_with_snapshot)
            .expect("runtime snapshot should reconstruct");
        let mut legacy_task = task_with_snapshot;
        legacy_task.request_metadata = None;

        let provider_body = json!({
            "status": "completed",
            "progress": 100,
            "video_url": "https://cdn.example.com/video.mp4"
        });
        let record = build_successful_poll_update(
            &legacy_task,
            &runtime_snapshot,
            provider_body
                .as_object()
                .expect("provider body should be an object"),
            100,
            false,
        )
        .expect("poll projection should succeed")
        .expect("poll projection should produce an update");

        assert_eq!(record.status, VideoTaskStatus::Completed);
        assert_eq!(
            record.video_url.as_deref(),
            Some("https://cdn.example.com/video.mp4")
        );
        assert!(record
            .request_metadata
            .as_ref()
            .and_then(|value| value.get("rust_local_snapshot"))
            .is_some());
        assert_eq!(record.poll_count, legacy_task.poll_count);
    }

    #[test]
    fn doubao_frame_poll_does_not_restore_a_stale_duration_column() {
        let mut task = sample_sparse_stored_task();
        let transport =
            match LocalVideoTaskSnapshot::from_stored_task(&task).expect("fixture snapshot") {
                LocalVideoTaskSnapshot::OpenAi(seed) => seed.transport,
                _ => panic!("expected OpenAI fixture"),
            };
        task.client_api_format = Some("doubao:video".to_string());
        task.provider_api_format = Some("doubao:video".to_string());
        task.duration_seconds = Some(4);
        task.original_request_body = Some(json!({
            "model": "doubao-seedance-2-0",
            "duration": 4
        }));

        let runtime_snapshot = LocalVideoTaskSnapshot::Doubao(DoubaoVideoTaskSeed {
            local_task_id: task.id.clone(),
            upstream_task_id: task.external_task_id.clone().expect("upstream id"),
            created_at_unix_secs: 1,
            updated_at_unix_secs: Some(20),
            user_id: task.user_id.clone(),
            api_key_id: task.api_key_id.clone(),
            model: Some("doubao-seedance-2-0".to_string()),
            prompt: Some("hello".to_string()),
            resolution: Some("720p".to_string()),
            ratio: Some("16:9".to_string()),
            duration_seconds: Some(4),
            seed: None,
            frames: None,
            frames_per_second: None,
            status: LocalVideoTaskStatus::Processing,
            progress_percent: 50,
            completed_at_unix_secs: None,
            error_code: None,
            error_message: None,
            video_url: None,
            last_frame_url: None,
            completion_tokens: None,
            total_tokens: None,
            persistence: LocalVideoTaskPersistence {
                request_id: task.request_id.clone(),
                username: task.username.clone(),
                api_key_name: task.api_key_name.clone(),
                client_api_format: "doubao:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                original_request_body: task.original_request_body.clone().expect("request body"),
                format_converted: false,
            },
            transport,
        });
        let provider_body = json!({
            "status": "succeeded",
            "frames": 121,
            "framespersecond": 24
        });

        let record = build_successful_poll_update(
            &task,
            &runtime_snapshot,
            provider_body.as_object().expect("provider body"),
            30,
            true,
        )
        .expect("poll projection")
        .expect("record");

        assert_eq!(record.duration_seconds, None);
        let persisted =
            &record.request_metadata.as_ref().expect("metadata")["rust_local_snapshot"]["Doubao"];
        assert_eq!(persisted["frames"].as_i64(), Some(121));
        assert!(persisted["duration_seconds"].is_null());
    }

    #[test]
    fn reconstructed_stored_snapshot_keeps_runtime_credentials_and_proxy() {
        let task = sample_sparse_stored_task();
        let mut runtime_snapshot = LocalVideoTaskSnapshot::from_stored_task(&task)
            .expect("runtime snapshot should reconstruct");
        let LocalVideoTaskSnapshot::OpenAi(seed) = &mut runtime_snapshot else {
            panic!("expected OpenAI snapshot");
        };
        seed.transport.headers.insert(
            "authorization".to_string(),
            "Bearer runtime-secret".to_string(),
        );
        seed.transport.proxy = Some(aether_contracts::ProxySnapshot {
            enabled: Some(true),
            url: Some("http://proxy.example:8080".to_string()),
            ..aether_contracts::ProxySnapshot::default()
        });

        let restored = snapshot_with_runtime_transport(&task, &runtime_snapshot)
            .expect("stored snapshot should reconcile with runtime transport");
        let LocalVideoTaskSnapshot::OpenAi(seed) = restored else {
            panic!("expected OpenAI snapshot");
        };
        assert_eq!(
            seed.transport
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer runtime-secret")
        );
        assert_eq!(
            seed.transport
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.url.as_deref()),
            Some("http://proxy.example:8080")
        );
    }

    #[test]
    fn terminal_doubao_usage_uses_provider_model_and_splits_input_tokens() {
        let mut task = sample_sparse_stored_task();
        task.status = VideoTaskStatus::Completed;
        task.completed_at_unix_secs = Some(30);
        task.client_api_format = Some("openai:video".to_string());
        task.provider_api_format = Some("doubao:video".to_string());
        task.format_converted = true;
        task.model = Some("sora-client-alias".to_string());

        let base_transport = match LocalVideoTaskSnapshot::from_stored_task(&task)
            .expect("base snapshot should reconstruct")
        {
            LocalVideoTaskSnapshot::OpenAi(seed) => seed.transport,
            _ => panic!("expected OpenAI transport fixture"),
        };
        let snapshot = LocalVideoTaskSnapshot::Doubao(DoubaoVideoTaskSeed {
            local_task_id: task.id.clone(),
            upstream_task_id: task
                .external_task_id
                .clone()
                .expect("external task id should exist"),
            created_at_unix_secs: 1,
            updated_at_unix_secs: Some(30),
            user_id: task.user_id.clone(),
            api_key_id: task.api_key_id.clone(),
            model: Some("doubao-provider-reported-model".to_string()),
            prompt: Some("hello".to_string()),
            resolution: Some("720p".to_string()),
            ratio: Some("16:9".to_string()),
            duration_seconds: Some(4),
            seed: None,
            frames: None,
            frames_per_second: None,
            status: LocalVideoTaskStatus::Completed,
            progress_percent: 100,
            completed_at_unix_secs: Some(30),
            error_code: None,
            error_message: None,
            video_url: Some("https://cdn.example.com/video.mp4".to_string()),
            last_frame_url: None,
            completion_tokens: Some(80),
            total_tokens: Some(100),
            persistence: LocalVideoTaskPersistence {
                request_id: task.request_id.clone(),
                username: task.username.clone(),
                api_key_name: task.api_key_name.clone(),
                client_api_format: "openai:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                original_request_body: json!({"model": "sora-client-alias"}),
                format_converted: true,
            },
            transport: LocalVideoTaskTransport {
                model_name: Some("doubao-configured-model".to_string()),
                ..base_transport
            },
        });
        task.request_metadata = Some(json!({
            "rust_local_snapshot": snapshot.redacted_for_persistence()
        }));

        let event = build_video_task_terminal_usage_event(&task)
            .expect("completed task should produce a usage event");
        assert_eq!(event.data.model, "doubao-provider-reported-model");
        assert_eq!(
            event.data.target_model.as_deref(),
            Some("doubao-configured-model")
        );
        assert_eq!(event.data.input_tokens, Some(20));
        assert_eq!(event.data.output_tokens, Some(80));
        assert_eq!(event.data.total_tokens, Some(100));
        assert_eq!(
            event.data.endpoint_api_format.as_deref(),
            Some("doubao:video")
        );
    }

    #[test]
    fn completed_video_billing_requires_resolved_costs_before_settlement() {
        let mut event = build_video_task_terminal_usage_event(&{
            let mut task = sample_sparse_stored_task();
            task.status = VideoTaskStatus::Completed;
            task
        })
        .expect("completed task should produce a usage event");

        assert!(!terminal_video_billing_is_resolved(&event));

        event.data.total_cost_usd = Some(0.0);
        event.data.actual_total_cost_usd = Some(0.0);
        assert!(terminal_video_billing_is_resolved(&event));
    }

    #[test]
    fn billing_dimensions_carry_resolution_duration_and_input_kind() {
        let mut task = sample_sparse_stored_task();
        task.resolution = Some(" 720p ".to_string());
        task.duration_seconds = Some(5);
        task.original_request_body = Some(json!({
            "content": [
                {"type": "text", "text": "a cat"},
                {"type": "video_url", "video_url": {"url": "https://e/a.mp4"}}
            ]
        }));

        let metadata = build_video_task_billing_dimensions(&task);
        let dimensions = metadata
            .get("dimensions")
            .expect("dimensions should be present");

        assert_eq!(dimensions.get("video_resolution"), Some(&json!("720p")));
        assert_eq!(dimensions.get("video_duration_seconds"), Some(&json!(5)));
        assert_eq!(dimensions.get("video_has_video_input"), Some(&json!(true)));
        // Existing metadata must survive the merge.
        assert!(metadata.get("rust_local_snapshot").is_some());
    }

    #[test]
    fn billing_dimensions_omit_absent_resolution_and_default_to_no_video_input() {
        let task = sample_sparse_stored_task();

        let metadata = build_video_task_billing_dimensions(&task);
        let dimensions = metadata
            .get("dimensions")
            .expect("dimensions should be present");

        assert!(dimensions.get("video_resolution").is_none());
        assert!(dimensions.get("video_duration_seconds").is_none());
        assert_eq!(dimensions.get("video_has_video_input"), Some(&json!(false)));
    }

    #[test]
    fn billing_dimensions_preserve_unrelated_existing_dimensions() {
        let mut task = sample_sparse_stored_task();
        task.request_metadata = Some(json!({
            "dimensions": { "image_size": "1024x1024" }
        }));
        task.resolution = Some("1080p".to_string());

        let metadata = build_video_task_billing_dimensions(&task);
        let dimensions = metadata
            .get("dimensions")
            .expect("dimensions should be present");

        assert_eq!(dimensions.get("image_size"), Some(&json!("1024x1024")));
        assert_eq!(dimensions.get("video_resolution"), Some(&json!("1080p")));
    }
}
