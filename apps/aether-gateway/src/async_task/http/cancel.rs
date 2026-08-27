use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use aether_contracts::{ExecutionPlan, ExecutionResult, ExecutionTelemetry};
use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, UpsertVideoTask, VideoTaskStatus,
};
use aether_usage_runtime::{
    build_sync_terminal_usage_outcome, build_terminal_usage_event_from_outcome,
    GatewaySyncReportRequest, UsageTerminalState,
};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Map, Value};

use crate::video_tasks::{LocalVideoTaskFollowUpPlan, LocalVideoTaskSnapshot};
use crate::{AppState, GatewayError};

use super::super::finalize_video_task_if_terminal;
use super::super::read_video_task_detail;
use super::current_unix_secs;

#[derive(Debug)]
pub(crate) enum CancelVideoTaskError {
    NotFound,
    InvalidStatus(VideoTaskStatus),
    Response(axum::response::Response),
    Gateway(GatewayError),
}

impl From<GatewayError> for CancelVideoTaskError {
    fn from(value: GatewayError) -> Self {
        Self::Gateway(value)
    }
}

pub(crate) async fn cancel_video_task_record(
    state: &AppState,
    task_id: &str,
) -> Result<StoredVideoTask, CancelVideoTaskError> {
    cancel_video_task_record_for_owner(state, task_id, None).await
}

/// Cancels a video task while enforcing an optional immutable owner boundary.
///
/// Admin callers use `None`; self-service callers must pass the authenticated
/// user id. Ownership is checked both before any provider credential hydration
/// or upstream request and again before the active-only persistence CAS.
pub(crate) async fn cancel_video_task_record_for_owner(
    state: &AppState,
    task_id: &str,
    expected_user_id: Option<&str>,
) -> Result<StoredVideoTask, CancelVideoTaskError> {
    let Some(task) = read_video_task_detail(state, task_id).await? else {
        return Err(CancelVideoTaskError::NotFound);
    };
    if !video_task_owner_matches(&task, expected_user_id) {
        return Err(CancelVideoTaskError::NotFound);
    }
    if expected_user_id.is_some() && task.status == VideoTaskStatus::Deleted {
        return Err(CancelVideoTaskError::NotFound);
    }

    if matches!(
        task.status,
        VideoTaskStatus::Completed
            | VideoTaskStatus::Failed
            | VideoTaskStatus::Cancelled
            | VideoTaskStatus::Expired
            | VideoTaskStatus::Deleted
    ) {
        return Err(CancelVideoTaskError::InvalidStatus(task.status));
    }

    let trace_id = build_async_cancel_trace_id();
    let cancel_plan = build_video_task_cancel_plan(&task).ok_or_else(|| {
        CancelVideoTaskError::Gateway(GatewayError::Internal(
            "video task provider cancellation contract is unavailable".to_string(),
        ))
    })?;
    let hydrated = match expected_user_id {
        Some(expected_user_id) => {
            state
                .hydrate_video_task_for_route_for_user(
                    Some(cancel_plan.route_family),
                    &cancel_plan.request_path,
                    expected_user_id,
                )
                .await?
        }
        None => {
            let _ = state
                .hydrate_video_task_for_route(
                    Some(cancel_plan.route_family),
                    &cancel_plan.request_path,
                )
                .await?;
            true
        }
    };
    if !hydrated {
        return Err(CancelVideoTaskError::NotFound);
    }

    let body_json = json!({});
    let follow_up = match expected_user_id {
        Some(expected_user_id) => state.video_tasks.prepare_follow_up_sync_plan_for_owner(
            cancel_plan.plan_kind,
            &cancel_plan.request_path,
            Some(&body_json),
            expected_user_id,
            &trace_id,
        ),
        None => state.video_tasks.prepare_follow_up_sync_plan(
            cancel_plan.plan_kind,
            &cancel_plan.request_path,
            Some(&body_json),
            None,
            &trace_id,
        ),
    }
    .ok_or_else(|| {
        CancelVideoTaskError::Gateway(GatewayError::Internal(
            "video task provider credentials are unavailable for cancellation".to_string(),
        ))
    })?;

    execute_video_task_cancel_plan(state, &trace_id, task.request_id.as_str(), follow_up)
        .await
        .map_err(CancelVideoTaskError::Response)?;

    // The upstream DELETE races the background poller.  Re-read immediately
    // before the local mutation and use the repository's active-only CAS so a
    // late admin cancel cannot resurrect a task that has already completed or
    // failed (and cannot erase its asset/billing fields).
    let Some(current_task) = state.find_video_task_by_id(task_id).await? else {
        return Err(CancelVideoTaskError::NotFound);
    };
    if !video_task_owner_matches(&current_task, expected_user_id) {
        return Err(CancelVideoTaskError::NotFound);
    }
    if expected_user_id.is_some() && current_task.status == VideoTaskStatus::Deleted {
        return Err(CancelVideoTaskError::NotFound);
    }
    if !current_task.status.is_active() {
        return Err(CancelVideoTaskError::InvalidStatus(current_task.status));
    }
    let request_metadata = build_cancelled_request_metadata(state, &current_task).await?;
    let stored = match persist_cancelled_video_task(state, &current_task, request_metadata).await? {
        Some(stored) => stored,
        None => {
            // A concurrent poll/cancel won the CAS.  Restore the latest
            // database truth in memory and surface the conflict to the caller;
            // never claim local cancellation succeeded without an atomic write.
            if let Some(latest) = state.find_video_task_by_id(task_id).await? {
                state.video_tasks.hydrate_from_stored_task(&latest);
                return Err(CancelVideoTaskError::InvalidStatus(latest.status));
            }
            return Err(CancelVideoTaskError::NotFound);
        }
    };
    state
        .video_tasks
        .apply_finalize_mutation(&cancel_plan.request_path, cancel_plan.report_kind);
    // The Doubao delete transport mutates the in-memory registry to `Deleted`,
    // while this admin operation is represented to callers as `Cancelled`.
    // Rehydrate from the just-persisted snapshot so reads are consistent both
    // before and after a process restart.
    state.video_tasks.hydrate_from_stored_task(&stored);
    finalize_video_task_if_terminal(state, &stored).await;
    Ok(stored)
}

fn video_task_owner_matches(task: &StoredVideoTask, expected_user_id: Option<&str>) -> bool {
    expected_user_id.is_none_or(|expected| {
        let expected = expected.trim();
        !expected.is_empty() && task.user_id.as_deref().map(str::trim) == Some(expected)
    })
}

#[derive(Debug, Clone)]
struct VideoTaskCancelPlan<'a> {
    route_family: &'a str,
    plan_kind: &'a str,
    report_kind: &'a str,
    request_path: String,
}

fn build_async_cancel_trace_id() -> String {
    format!("async-task-cancel-{}", uuid::Uuid::now_v7())
}

struct CancelUsageLifecycleGuard {
    state: AppState,
    plan: ExecutionPlan,
    report_kind: String,
    report_context: Option<Value>,
    started_at: Instant,
    armed: bool,
}

#[derive(Clone)]
struct CancelUsageTerminalOutcome {
    terminal_state: UsageTerminalState,
    status_code: u16,
    error_category: Option<String>,
    error_message: Option<String>,
    provider_headers: BTreeMap<String, String>,
    provider_body_json: Option<Value>,
    provider_body_base64: Option<String>,
    client_body_json: Option<Value>,
    telemetry: Option<ExecutionTelemetry>,
}

impl CancelUsageTerminalOutcome {
    fn from_execution_result(result: &ExecutionResult) -> Self {
        let terminal_state = if (200..300).contains(&result.status_code) {
            UsageTerminalState::Completed
        } else {
            UsageTerminalState::Failed
        };
        Self {
            terminal_state,
            status_code: result.status_code,
            error_category: result
                .error
                .as_ref()
                .map(|error| format!("{:?}", error.kind)),
            error_message: result.error.as_ref().map(|error| error.message.clone()),
            provider_headers: result.headers.clone(),
            provider_body_json: result.body.as_ref().and_then(|body| body.json_body.clone()),
            provider_body_base64: result
                .body
                .as_ref()
                .and_then(|body| body.body_bytes_b64.clone()),
            client_body_json: None,
            telemetry: result.telemetry.clone(),
        }
    }

    fn failed(status_code: u16, category: &str, message: String) -> Self {
        Self {
            terminal_state: UsageTerminalState::Failed,
            status_code,
            error_category: Some(category.to_string()),
            error_message: Some(message),
            provider_headers: BTreeMap::new(),
            provider_body_json: None,
            provider_body_base64: None,
            client_body_json: None,
            telemetry: None,
        }
    }

    fn cancelled() -> Self {
        let message = "Async video cancellation was interrupted before terminal finalization";
        Self {
            terminal_state: UsageTerminalState::Cancelled,
            status_code: 499,
            error_category: Some("video_cancel_request_cancelled".to_string()),
            error_message: Some(message.to_string()),
            provider_headers: BTreeMap::new(),
            provider_body_json: None,
            provider_body_base64: None,
            client_body_json: None,
            telemetry: None,
        }
    }
}

impl CancelUsageLifecycleGuard {
    fn new(
        state: &AppState,
        parent_request_id: &str,
        follow_up: LocalVideoTaskFollowUpPlan,
    ) -> Self {
        let LocalVideoTaskFollowUpPlan {
            plan,
            report_kind,
            mut report_context,
        } = follow_up;
        ensure_cancel_parent_request_id(&mut report_context, parent_request_id);
        let report_kind = report_kind.unwrap_or_else(|| "video_cancel_sync_finalize".to_string());
        let mut lifecycle_seed =
            aether_usage_runtime::build_lifecycle_usage_seed(&plan, report_context.as_ref());
        lifecycle_seed.request_type = "video".to_string();
        state
            .usage_runtime
            .record_pending(state.usage_lifecycle_data_state().as_ref(), lifecycle_seed);
        Self {
            state: state.clone(),
            plan,
            report_kind,
            report_context,
            started_at: Instant::now(),
            armed: true,
        }
    }

    async fn finish(&mut self, outcome: CancelUsageTerminalOutcome) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let handoff = spawn_cancel_usage_terminal(
            self.state.clone(),
            self.plan.clone(),
            self.report_kind.clone(),
            self.report_context.clone(),
            self.started_at,
            outcome,
        );
        let _ = tokio::time::timeout(Duration::from_secs(5), handoff).await;
    }
}

fn ensure_cancel_parent_request_id(report_context: &mut Option<Value>, parent_request_id: &str) {
    let context = report_context.get_or_insert_with(|| json!({}));
    if !context.is_object() {
        *context = json!({ "report_context": context.take() });
    }
    context
        .as_object_mut()
        .expect("video cancel report context should be an object")
        .entry("parent_request_id".to_string())
        .or_insert_with(|| Value::String(parent_request_id.to_string()));
    // This direct control path has no separate client HTTP transport capture.
    // An explicit empty object suppresses the generic terminal builder's legacy
    // fallback from provider response headers into client response headers.
    context
        .as_object_mut()
        .expect("video cancel report context should be an object")
        .entry("client_response_headers".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    context
        .as_object_mut()
        .expect("video cancel report context should be an object")
        .entry("usage_scope".to_string())
        .or_insert_with(|| Value::String("video_upstream_control_action".to_string()));
    context
        .as_object_mut()
        .expect("video cancel report context should be an object")
        .entry("client_capture_scope".to_string())
        .or_insert_with(|| Value::String("none".to_string()));
}

impl Drop for CancelUsageLifecycleGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        spawn_cancel_usage_terminal(
            self.state.clone(),
            self.plan.clone(),
            self.report_kind.clone(),
            self.report_context.clone(),
            self.started_at,
            CancelUsageTerminalOutcome::cancelled(),
        );
    }
}

fn spawn_cancel_usage_terminal(
    state: AppState,
    plan: ExecutionPlan,
    report_kind: String,
    report_context: Option<Value>,
    started_at: Instant,
    outcome: CancelUsageTerminalOutcome,
) -> tokio::task::JoinHandle<()> {
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        record_cancel_usage_terminal(
            &state,
            &plan,
            report_kind,
            report_context,
            started_at,
            outcome,
        )
        .await;
    })
}

async fn record_cancel_usage_terminal(
    state: &AppState,
    plan: &ExecutionPlan,
    report_kind: String,
    report_context: Option<Value>,
    started_at: Instant,
    outcome: CancelUsageTerminalOutcome,
) {
    match build_cancel_usage_terminal_event(plan, report_kind, report_context, started_at, outcome)
    {
        Ok(event) => {
            let _ = state
                .usage_runtime
                .record_terminal_event_direct_with_handoff(
                    state.usage_lifecycle_data_state().as_ref(),
                    event,
                )
                .await;
        }
        Err(error) => {
            tracing::warn!(
                event_name = "video_cancel_usage_terminal_event_build_failed",
                log_type = "event",
                request_id = %plan.request_id,
                error = %error,
                "gateway could not build the independent video cancel usage event"
            );
        }
    }
}

fn build_cancel_usage_terminal_event(
    plan: &ExecutionPlan,
    report_kind: String,
    report_context: Option<Value>,
    started_at: Instant,
    outcome: CancelUsageTerminalOutcome,
) -> Result<aether_usage_runtime::UsageEvent, aether_data_contracts::DataLayerError> {
    let mut telemetry = outcome.telemetry.unwrap_or(ExecutionTelemetry {
        ttfb_ms: None,
        elapsed_ms: None,
        upstream_bytes: None,
    });
    telemetry.elapsed_ms = telemetry
        .elapsed_ms
        .or(Some(started_at.elapsed().as_millis() as u64));
    let payload = GatewaySyncReportRequest {
        trace_id: plan.request_id.clone(),
        report_kind,
        report_context: report_context.clone(),
        status_code: outcome.status_code,
        headers: outcome.provider_headers,
        body_json: outcome.provider_body_json,
        client_body_json: outcome.client_body_json,
        body_base64: outcome.provider_body_base64,
        telemetry: Some(telemetry),
    };
    let mut terminal = build_sync_terminal_usage_outcome(plan, report_context.as_ref(), &payload);
    terminal.terminal_state = outcome.terminal_state;
    terminal.request_type = "video".to_string();
    terminal.terminal_error_message = outcome.error_message;
    terminal.terminal_failure_category = outcome.error_category;
    terminal.defer_settlement = false;
    terminal.billing_treat_as_completed = false;
    // The empty client-header object in report_context is a sentinel that only
    // prevents the generic builder from copying provider headers. This direct
    // upstream child has no client transport capture of its own.
    terminal.client_response_headers = None;
    terminal.client_response = None;
    // Cancelling is a control operation. It must be auditable on both success
    // and failure, but must never charge independently from video generation.
    terminal.billing_treat_as_void = true;
    build_terminal_usage_event_from_outcome(terminal)
}

fn build_video_task_cancel_plan(task: &StoredVideoTask) -> Option<VideoTaskCancelPlan<'_>> {
    let client_api_format = task
        .client_api_format
        .as_deref()
        .or(task.provider_api_format.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())?;
    let provider_api_format = task
        .provider_api_format
        .as_deref()
        .or(task.client_api_format.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())?;

    match (client_api_format, provider_api_format) {
        ("openai:video", "openai:video" | "doubao:video") => Some(VideoTaskCancelPlan {
            route_family: "openai",
            plan_kind: "openai_video_cancel_sync",
            report_kind: "openai_video_cancel_sync_finalize",
            request_path: format!("/v1/videos/{}/cancel", task.id),
        }),
        ("gemini:video", "gemini:video") => {
            let short_id = task.short_id.as_deref().unwrap_or(task.id.as_str()).trim();
            let model = task
                .model
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            Some(VideoTaskCancelPlan {
                route_family: "gemini",
                plan_kind: "gemini_video_cancel_sync",
                report_kind: "gemini_video_cancel_sync_finalize",
                request_path: format!("/v1beta/models/{model}/operations/{short_id}:cancel"),
            })
        }
        // Ark folds cancel into delete, so an admin-initiated cancel retires the task.
        ("doubao:video", "doubao:video") => Some(VideoTaskCancelPlan {
            route_family: "doubao",
            plan_kind: "doubao_video_delete_sync",
            report_kind: "doubao_video_cancel_sync_finalize",
            request_path: format!(
                "{}/{}",
                aether_video_tasks_core::DOUBAO_VIDEO_TASKS_PATH,
                task.id
            ),
        }),
        _ => None,
    }
}

async fn execute_video_task_cancel_plan(
    state: &AppState,
    trace_id: &str,
    parent_request_id: &str,
    follow_up: LocalVideoTaskFollowUpPlan,
) -> Result<(), axum::response::Response> {
    let mut usage_guard = CancelUsageLifecycleGuard::new(state, parent_request_id, follow_up);
    let result =
        match crate::execution_runtime::execute_execution_runtime_sync_plan_with_report_context(
            state,
            Some(trace_id),
            &usage_guard.plan,
            usage_guard.report_context.as_ref(),
        )
        .await
        {
            Ok(result) => result,
            Err(err) => {
                let message = err.into_message();
                usage_guard
                    .finish(CancelUsageTerminalOutcome::failed(
                        axum::http::StatusCode::BAD_GATEWAY.as_u16(),
                        "execution_runtime_unavailable",
                        message.clone(),
                    ))
                    .await;
                return Err(GatewayError::UpstreamUnavailable {
                    trace_id: trace_id.to_string(),
                    message,
                }
                .into_response());
            }
        };

    let body_json = cancel_client_body(&result);
    usage_guard
        .finish(CancelUsageTerminalOutcome::from_execution_result(&result))
        .await;

    if result.status_code >= 400 {
        let status = axum::http::StatusCode::from_u16(result.status_code)
            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
        return Err((status, Json(body_json)).into_response());
    }

    Ok(())
}

fn cancel_client_body(result: &ExecutionResult) -> Value {
    if (200..300).contains(&result.status_code) {
        return json!({});
    }
    let provider_error = result
        .body
        .as_ref()
        .and_then(|body| body.json_body.clone())
        .and_then(|body| body.get("error").cloned());
    let message = result
        .error
        .as_ref()
        .map(|error| error.message.clone())
        .or_else(|| {
            provider_error
                .as_ref()
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| format!("execution runtime returned {}", result.status_code));
    json!({
        "error": {
            "type": "video_cancel_upstream_error",
            "message": message,
        }
    })
}

async fn build_cancelled_request_metadata(
    state: &AppState,
    task: &StoredVideoTask,
) -> Result<Option<Value>, GatewayError> {
    let mut metadata = match task.request_metadata.clone() {
        Some(Value::Object(object)) => object,
        _ => Map::new(),
    };
    let snapshot = match LocalVideoTaskSnapshot::from_stored_task(task) {
        Some(snapshot) => Some(snapshot),
        None => state.reconstruct_video_task_snapshot(task).await?,
    };
    let mut snapshot_value = snapshot
        .map(|snapshot| snapshot.redacted_for_persistence())
        .map(serde_json::to_value)
        .transpose()
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    if let Some(snapshot_value_ref) = snapshot_value.as_mut() {
        mark_snapshot_value_cancelled(snapshot_value_ref);
        metadata.insert(
            "rust_owner".to_string(),
            Value::String("async_task".to_string()),
        );
        metadata.insert(
            "rust_local_snapshot".to_string(),
            snapshot_value_ref.clone(),
        );
        return Ok(Some(Value::Object(metadata)));
    }

    // A malformed legacy snapshot cannot be safely inspected for credentials.
    // Drop only that internal field while preserving unrelated audit metadata.
    metadata.remove("rust_local_snapshot");
    Ok((!metadata.is_empty()).then_some(Value::Object(metadata)))
}

fn mark_snapshot_value_cancelled(snapshot_value: &mut Value) {
    for variant in ["OpenAi", "Gemini", "Doubao"] {
        if let Some(object) = snapshot_value
            .get_mut(variant)
            .and_then(Value::as_object_mut)
        {
            object.insert("status".to_string(), Value::String("Cancelled".to_string()));
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Instant;

    use aether_contracts::{ExecutionPlan, ExecutionResult, RequestBody, ResponseBody};
    use aether_usage_runtime::{UsageEventType, UsageTerminalState};

    use super::{
        build_async_cancel_trace_id, build_cancel_usage_terminal_event, cancel_client_body,
        ensure_cancel_parent_request_id, mark_snapshot_value_cancelled, CancelUsageTerminalOutcome,
    };
    use serde_json::json;

    fn cancel_plan() -> ExecutionPlan {
        ExecutionPlan {
            request_id: "cancel-child-request-1".to_string(),
            candidate_id: None,
            provider_name: Some("OpenAI".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            method: "DELETE".to_string(),
            url: "https://api.openai.example/videos/task-1".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: RequestBody::from_json(json!({})),
            stream: false,
            client_api_format: "openai:video".to_string(),
            provider_api_format: "openai:video".to_string(),
            model_name: Some("sora-2".to_string()),
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    fn terminal_outcome(
        terminal_state: UsageTerminalState,
        status_code: u16,
    ) -> CancelUsageTerminalOutcome {
        CancelUsageTerminalOutcome {
            terminal_state,
            status_code,
            error_category: None,
            error_message: None,
            provider_headers: BTreeMap::from([(
                "x-provider-request-id".to_string(),
                "provider-request-1".to_string(),
            )]),
            provider_body_json: Some(json!({"provider": "capture"})),
            provider_body_base64: None,
            client_body_json: None,
            telemetry: None,
        }
    }

    #[test]
    fn async_cancel_child_request_ids_are_unique() {
        let first = build_async_cancel_trace_id();
        let second = build_async_cancel_trace_id();

        assert_ne!(first, second);
        assert!(first.starts_with("async-task-cancel-"));
        assert!(second.starts_with("async-task-cancel-"));
    }

    #[test]
    fn cancel_context_keeps_operation_and_links_parent_without_client_header_fallback() {
        let mut context = Some(json!({
            "operation": "video.cancel"
        }));

        ensure_cancel_parent_request_id(&mut context, "create-parent-request-1");

        let context = context.expect("context should exist");
        assert_eq!(context["operation"], "video.cancel");
        assert_eq!(context["parent_request_id"], "create-parent-request-1");
        assert_eq!(context["client_response_headers"], json!({}));
        assert_eq!(context["usage_scope"], "video_upstream_control_action");
        assert_eq!(context["client_capture_scope"], "none");
    }

    #[test]
    fn cancel_terminal_events_are_independent_void_lifecycles() {
        for (terminal_state, event_type, status_code) in [
            (
                UsageTerminalState::Completed,
                UsageEventType::Completed,
                204,
            ),
            (UsageTerminalState::Failed, UsageEventType::Failed, 502),
            (
                UsageTerminalState::Cancelled,
                UsageEventType::Cancelled,
                499,
            ),
        ] {
            let event = build_cancel_usage_terminal_event(
                &cancel_plan(),
                "openai_video_cancel_sync_finalize".to_string(),
                Some(json!({
                    "operation": "video.cancel",
                    "parent_request_id": "create-parent-request-1",
                    "client_response_headers": {}
                })),
                Instant::now(),
                terminal_outcome(terminal_state, status_code),
            )
            .expect("cancel terminal event should build");

            assert_eq!(event.request_id, "cancel-child-request-1");
            assert_eq!(event.event_type, event_type);
            assert_eq!(event.data.request_type.as_deref(), Some("video"));
            assert_eq!(event.data.billing_treat_as_void, Some(true));
            assert_eq!(event.data.status_code, Some(status_code));
            assert_eq!(
                event
                    .data
                    .request_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("operation")),
                Some(&json!("video.cancel"))
            );
            assert_eq!(
                event
                    .data
                    .request_metadata
                    .as_ref()
                    .and_then(|metadata| metadata.get("parent_request_id")),
                Some(&json!("create-parent-request-1"))
            );
            assert_eq!(
                event.data.response_headers,
                Some(json!({
                    "x-provider-request-id": "provider-request-1"
                }))
            );
            assert_eq!(event.data.client_response_headers, None);
            assert_eq!(event.data.client_response_body, None);
        }
    }

    #[test]
    fn successful_empty_cancel_response_does_not_invent_an_error_body() {
        let result = ExecutionResult {
            request_id: "cancel-child-request-1".to_string(),
            candidate_id: None,
            status_code: 204,
            headers: BTreeMap::new(),
            response_observation: None,
            body: Some(ResponseBody {
                json_body: Some(json!({"provider": "success-payload"})),
                body_bytes_b64: None,
            }),
            telemetry: None,
            error: None,
        };

        assert_eq!(cancel_client_body(&result), json!({}));
    }

    #[test]
    fn failed_cancel_uses_a_control_error_projection_not_the_provider_body() {
        let result = ExecutionResult {
            request_id: "cancel-child-request-1".to_string(),
            candidate_id: None,
            status_code: 502,
            headers: BTreeMap::new(),
            response_observation: None,
            body: Some(ResponseBody {
                json_body: Some(json!({
                    "error": {
                        "message": "provider rejected cancel",
                        "provider_private": "must-not-be-projected"
                    },
                    "provider_only": true
                })),
                body_bytes_b64: None,
            }),
            telemetry: None,
            error: None,
        };

        assert_eq!(
            cancel_client_body(&result),
            json!({
                "error": {
                    "type": "video_cancel_upstream_error",
                    "message": "provider rejected cancel"
                }
            })
        );
    }

    #[test]
    fn marks_doubao_snapshot_cancelled_without_dropping_other_fields() {
        let mut snapshot = json!({
            "Doubao": {
                "local_task_id": "cgt-local-1",
                "status": "Processing",
                "progress_percent": 50
            }
        });

        mark_snapshot_value_cancelled(&mut snapshot);

        assert_eq!(snapshot["Doubao"]["status"], json!("Cancelled"));
        assert_eq!(snapshot["Doubao"]["local_task_id"], json!("cgt-local-1"));
        assert_eq!(snapshot["Doubao"]["progress_percent"], json!(50));
    }
}

async fn persist_cancelled_video_task(
    state: &AppState,
    task: &StoredVideoTask,
    request_metadata: Option<Value>,
) -> Result<Option<StoredVideoTask>, GatewayError> {
    let now_unix_secs = current_unix_secs();
    state
        .data
        .update_active_video_task(UpsertVideoTask {
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
            prompt: task.prompt.clone(),
            original_request_body: task.original_request_body.clone(),
            duration_seconds: task.duration_seconds,
            resolution: task.resolution.clone(),
            aspect_ratio: task.aspect_ratio.clone(),
            size: task.size.clone(),
            status: VideoTaskStatus::Cancelled,
            progress_percent: task.progress_percent,
            progress_message: task.progress_message.clone(),
            retry_count: task.retry_count,
            poll_interval_seconds: task.poll_interval_seconds,
            next_poll_at_unix_secs: None,
            poll_count: task.poll_count,
            max_poll_count: task.max_poll_count,
            created_at_unix_ms: task.created_at_unix_ms,
            submitted_at_unix_secs: task.submitted_at_unix_secs,
            completed_at_unix_secs: Some(now_unix_secs),
            updated_at_unix_secs: now_unix_secs,
            error_code: task.error_code.clone(),
            error_message: task.error_message.clone(),
            video_url: task.video_url.clone(),
            request_metadata,
        })
        .await
        .map_err(|err| GatewayError::Internal(err.to_string()))
}
