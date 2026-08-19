use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, UpsertVideoTask, VideoTaskStatus,
};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Map, Value};

use crate::video_tasks::LocalVideoTaskSnapshot;
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

    let trace_id = format!("async-task-admin-cancel-{task_id}");
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

    execute_video_task_cancel_plan(state, &trace_id, follow_up.plan)
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
    plan: aether_contracts::ExecutionPlan,
) -> Result<(), axum::response::Response> {
    let result =
        crate::execution_runtime::execute_execution_runtime_sync_plan(state, Some(trace_id), &plan)
            .await
            .map_err(|err| {
                GatewayError::UpstreamUnavailable {
                    trace_id: trace_id.to_string(),
                    message: err.into_message(),
                }
                .into_response()
            })?;

    if result.status_code >= 400 {
        let status = axum::http::StatusCode::from_u16(result.status_code)
            .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
        let body_json = result
            .body
            .and_then(|body| body.json_body)
            .unwrap_or_else(|| {
                json!({
                    "error": {
                        "message": result
                            .error
                            .as_ref()
                            .map(|error| error.message.clone())
                            .unwrap_or_else(|| {
                                format!("execution runtime returned {}", result.status_code)
                            }),
                    }
                })
            });
        return Err((status, Json(body_json)).into_response());
    }

    Ok(())
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
    use super::mark_snapshot_value_cancelled;
    use serde_json::json;

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
