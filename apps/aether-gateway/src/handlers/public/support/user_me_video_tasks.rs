use std::collections::{BTreeMap, BTreeSet};
use std::time::{SystemTime, UNIX_EPOCH};

use aether_contracts::{ExecutionPlan, RequestBody, EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER};
use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, VideoTaskLookupKey, VideoTaskQueryFilter, VideoTaskStatus,
};
use axum::{
    body::Body,
    http,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use url::form_urlencoded;

use crate::async_task::{
    build_video_task_video_response, cancel_video_task_record_for_owner, read_video_task_detail,
    read_video_task_page, read_video_task_stats, read_video_task_video_source_for_owner,
    CancelVideoTaskError, VideoTaskVideoSource,
};
use crate::GatewayError;

use super::{
    build_auth_error_response, query_param_value, resolve_authenticated_local_user,
    unix_secs_to_rfc3339, AppState, GatewayPublicRequestContext,
};

const USERS_ME_VIDEO_TASKS_ROOT: &str = "/api/users/me/video-tasks";

#[derive(Debug, Clone, Default)]
struct UsersMeVideoTaskUsageSummary {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cost: f64,
}

pub(super) async fn handle_users_me_video_tasks_list(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(auth) => auth,
        Err(response) => return response,
    };
    let query = request_context.request_query_string.as_deref();
    if query_has_parameter(query, "user_id") {
        return users_me_video_tasks_bad_request(
            "user_id is not allowed; task ownership comes from the authenticated session",
        );
    }

    let status = match parse_status(query) {
        Ok(status) => status,
        Err(response) => return response,
    };
    let page = match parse_positive_usize(query, "page", 1, 1_000) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let page_size = match parse_positive_usize(query, "page_size", 20, 100) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let filter =
        users_me_video_task_filter(&auth.user.id, status, query_param_value(query, "model"));
    // The DTO id depends on the original client contract. Summary rows omit
    // `client_api_format`, so use the full row here instead of guessing from
    // the internal id or the presence of an upstream id.
    let page_response = match read_video_task_page(state, &filter, page, page_size).await {
        Ok(response) => response,
        Err(err) => {
            return users_me_video_tasks_internal_error(request_context, "list_video_tasks", &err)
        }
    };
    let usage =
        match build_users_me_video_task_usage_summaries(state, &page_response.items, &auth.user.id)
            .await
        {
            Ok(usage) => usage,
            Err(err) => {
                return users_me_video_tasks_internal_error(
                    request_context,
                    "list_video_task_usage",
                    &err,
                )
            }
        };
    let Some(items) = page_response
        .items
        .iter()
        .map(|task| build_users_me_video_task_item(task, &usage))
        .collect::<Option<Vec<_>>>()
    else {
        let err = GatewayError::Internal(
            "native Doubao video task is missing its upstream task id".to_string(),
        );
        return users_me_video_tasks_internal_error(
            request_context,
            "project_video_task_public_id",
            &err,
        );
    };

    Json(json!({
        "items": items,
        "total": page_response.total,
        "page": page_response.page,
        "page_size": page_response.page_size,
        "pages": page_response.pages,
    }))
    .into_response()
}

pub(super) async fn handle_users_me_video_tasks_stats(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(auth) => auth,
        Err(response) => return response,
    };
    let query = request_context.request_query_string.as_deref();
    if query_has_parameter(query, "user_id") {
        return users_me_video_tasks_bad_request(
            "user_id is not allowed; task ownership comes from the authenticated session",
        );
    }
    let filter = users_me_video_task_filter(&auth.user.id, None, None);
    match read_video_task_stats(state, &filter, current_unix_secs()).await {
        Ok(stats) => Json(json!({
            "total": stats.total,
            "by_status": stats.by_status,
            "by_model": stats.by_model,
            "today_count": stats.today_count,
            "processing_count": stats.processing_count,
        }))
        .into_response(),
        Err(err) => {
            users_me_video_tasks_internal_error(request_context, "read_video_task_stats", &err)
        }
    }
}

pub(super) async fn handle_users_me_video_task_detail(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(auth) => auth,
        Err(response) => return response,
    };
    let Some(task_id) = users_me_video_task_detail_id(&request_context.request_path) else {
        return users_me_video_task_not_found();
    };
    let task = match read_owned_video_task(state, task_id, &auth.user.id).await {
        Ok(Some(task)) => task,
        Ok(None) => return users_me_video_task_not_found(),
        Err(err) => {
            return users_me_video_tasks_internal_error(
                request_context,
                "read_video_task_detail",
                &err,
            )
        }
    };
    let usage = match build_users_me_video_task_usage_summaries(
        state,
        std::slice::from_ref(&task),
        &auth.user.id,
    )
    .await
    {
        Ok(usage) => usage,
        Err(err) => {
            return users_me_video_tasks_internal_error(
                request_context,
                "read_video_task_usage",
                &err,
            )
        }
    };
    let Some(item) = build_users_me_video_task_item(&task, &usage) else {
        return users_me_video_task_not_found();
    };
    Json(item).into_response()
}

pub(super) async fn handle_users_me_video_task_cancel(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(auth) => auth,
        Err(response) => return response,
    };
    let Some(task_id) = users_me_video_task_nested_id(&request_context.request_path, "/cancel")
    else {
        return users_me_video_task_not_found();
    };
    let task = match read_owned_video_task(state, task_id, &auth.user.id).await {
        Ok(Some(task)) => task,
        Ok(None) => return users_me_video_task_not_found(),
        Err(err) => {
            return users_me_video_tasks_internal_error(
                request_context,
                "authorize_video_task_cancel",
                &err,
            )
        }
    };
    let Some(public_task_id) = users_me_video_task_public_id(&task).map(ToOwned::to_owned) else {
        return users_me_video_task_not_found();
    };

    match cancel_video_task_record_for_owner(state, &task.id, Some(&auth.user.id)).await {
        Ok(_stored) => Json(json!({
            "id": public_task_id,
            "status": "cancelled",
            "message": "Task cancelled successfully",
        }))
        .into_response(),
        Err(CancelVideoTaskError::NotFound) => users_me_video_task_not_found(),
        Err(CancelVideoTaskError::InvalidStatus(status)) => build_auth_error_response(
            http::StatusCode::BAD_REQUEST,
            format!(
                "Cannot cancel task with status: {}",
                video_task_status_name(status)
            ),
            false,
        ),
        Err(CancelVideoTaskError::Response(response)) => {
            tracing::warn!(
                trace_id = %request_context.trace_id,
                upstream_status = %response.status(),
                "self-service video task cancellation failed upstream"
            );
            build_auth_error_response(
                http::StatusCode::BAD_GATEWAY,
                "Unable to cancel video task upstream",
                false,
            )
        }
        Err(CancelVideoTaskError::Gateway(err)) => {
            users_me_video_tasks_internal_error(request_context, "cancel_video_task", &err)
        }
    }
}

pub(super) async fn handle_users_me_video_task_video(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(auth) => auth,
        Err(response) => return protect_users_me_video_response(response),
    };
    let Some(task_id) = users_me_video_task_nested_id(&request_context.request_path, "/video")
    else {
        return protect_users_me_video_response(users_me_video_task_not_found());
    };
    let task = match read_owned_video_task(state, task_id, &auth.user.id).await {
        Ok(Some(task)) => task,
        Ok(None) => return protect_users_me_video_response(users_me_video_task_not_found()),
        Err(err) => {
            return protect_users_me_video_response(users_me_video_tasks_internal_error(
                request_context,
                "authorize_video_task_media",
                &err,
            ))
        }
    };
    let internal_task_id = task.id.clone();
    let source =
        match read_video_task_video_source_for_owner(state, &internal_task_id, &auth.user.id).await
        {
            Ok(Some(source)) => source,
            Ok(None) => return protect_users_me_video_response(users_me_video_task_not_found()),
            Err(err) => {
                return protect_users_me_video_response(users_me_video_tasks_internal_error(
                    request_context,
                    "resolve_video_task_media",
                    &err,
                ))
            }
        };
    let response = match source {
        VideoTaskVideoSource::Redirect { url } => {
            proxy_users_me_direct_video_asset(request_context, headers, &task, url).await
        }
        source @ VideoTaskVideoSource::Proxy { .. } => {
            build_video_task_video_response(state, &internal_task_id, source).await
        }
    };
    match response {
        Ok(response) => protect_users_me_video_response(response),
        Err(err) => protect_users_me_video_response(users_me_video_tasks_internal_error(
            request_context,
            "stream_video_task_media",
            &err,
        )),
    }
}

fn protect_users_me_video_response(mut response: Response<Body>) -> Response<Body> {
    // This route is owner-authenticated. Never allow an upstream CDN cache
    // policy to turn it into a shared response at Aether's public origin.
    response.headers_mut().insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("private, no-store"),
    );
    response.headers_mut().insert(
        http::header::VARY,
        http::HeaderValue::from_static("Authorization"),
    );
    response
}

async fn proxy_users_me_direct_video_asset(
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
    task: &StoredVideoTask,
    url: String,
) -> Result<Response<Body>, GatewayError> {
    let plan = build_users_me_direct_video_asset_plan(request_context, headers, task, url)?;
    crate::execution_runtime::execute_direct_asset_response(&plan)
        .await
        .map_err(|err| GatewayError::UpstreamUnavailable {
            trace_id: request_context.trace_id.clone(),
            message: err.to_string(),
        })
}

fn build_users_me_direct_video_asset_plan(
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
    task: &StoredVideoTask,
    url: String,
) -> Result<ExecutionPlan, GatewayError> {
    let provider_api_format = task
        .provider_api_format
        .as_deref()
        .or(task.client_api_format.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| GatewayError::Internal("video task is missing api format".to_string()))?;
    if !matches!(
        provider_api_format.to_ascii_lowercase().as_str(),
        "openai:video" | "doubao:video" | "gemini:video"
    ) {
        return Err(GatewayError::Internal(
            "video task has an unsupported media api format".to_string(),
        ));
    }
    let mut request_headers = BTreeMap::from([(
        EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER.to_string(),
        "1".to_string(),
    )]);
    if let Some(range) = headers
        .get(http::header::RANGE)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        request_headers.insert(http::header::RANGE.as_str().to_string(), range.to_string());
    }
    Ok(ExecutionPlan {
        request_id: request_context.trace_id.clone(),
        candidate_id: None,
        provider_name: None,
        provider_id: task
            .provider_id
            .clone()
            .unwrap_or_else(|| "video-task-provider".to_string()),
        endpoint_id: task
            .endpoint_id
            .clone()
            .unwrap_or_else(|| "video-task-endpoint".to_string()),
        key_id: task
            .key_id
            .clone()
            .unwrap_or_else(|| "video-task-key".to_string()),
        method: http::Method::GET.to_string(),
        url,
        headers: request_headers,
        content_type: None,
        content_encoding: None,
        body: RequestBody {
            json_body: None,
            body_bytes_b64: None,
            body_ref: None,
        },
        stream: true,
        client_api_format: task
            .client_api_format
            .clone()
            .unwrap_or_else(|| provider_api_format.to_string()),
        provider_api_format: provider_api_format.to_string(),
        model_name: task.model.clone(),
        // Direct asset egress intentionally bypasses provider proxies/tunnels;
        // the runtime performs its own public-IP validation and DNS pinning.
        proxy: None,
        transport_profile: None,
        timeouts: None,
    })
}

fn users_me_video_task_filter(
    user_id: &str,
    status: Option<VideoTaskStatus>,
    model_substring: Option<String>,
) -> VideoTaskQueryFilter {
    VideoTaskQueryFilter {
        user_id: Some(user_id.to_string()),
        status,
        model_exact: None,
        model_substring,
        client_api_format: None,
        exclude_deleted: true,
    }
}

async fn read_owned_video_task(
    state: &AppState,
    public_task_id: &str,
    user_id: &str,
) -> Result<Option<StoredVideoTask>, GatewayError> {
    let public_task_id = public_task_id.trim();
    let user_id = user_id.trim();
    if public_task_id.is_empty() || user_id.is_empty() {
        return Ok(None);
    }

    // Native Ark clients see the upstream cgt id. Resolve it together with
    // the immutable authenticated owner before exposing or acting on a row.
    let by_external = state
        .data
        .find_video_task(VideoTaskLookupKey::UserExternal {
            user_id,
            external_task_id: public_task_id,
        })
        .await
        .map_err(|err| GatewayError::Internal(err.to_string()))?;
    if let Some(task) = by_external.filter(|task| {
        is_native_doubao_video_task(task) && users_me_video_task_owner_matches(task, user_id)
    }) {
        return Ok(Some(task));
    }

    // OpenAI and Gemini keep their original client-facing local ids. Native
    // Doubao rows are deliberately excluded from this public fallback; after
    // resolution, downstream provider operations use the returned `task.id`.
    Ok(read_video_task_detail(state, public_task_id)
        .await?
        .filter(|task| {
            !is_native_doubao_video_task(task) && users_me_video_task_owner_matches(task, user_id)
        }))
}

fn users_me_video_task_owner_matches(task: &StoredVideoTask, user_id: &str) -> bool {
    task.status != VideoTaskStatus::Deleted
        && task.user_id.as_deref().map(str::trim) == Some(user_id.trim())
}

async fn build_users_me_video_task_usage_summaries(
    state: &AppState,
    tasks: &[StoredVideoTask],
    expected_user_id: &str,
) -> Result<BTreeMap<String, UsersMeVideoTaskUsageSummary>, GatewayError> {
    let request_ids = tasks
        .iter()
        .map(|task| task.request_id.trim())
        .filter(|request_id| !request_id.is_empty())
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>();
    let mut summaries = BTreeMap::new();
    for request_id in request_ids {
        let usage = state
            .data
            .read_request_usage_audit_shallow(&request_id)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))?;
        let Some(usage) = usage.filter(|usage| usage.user_id.as_deref() == Some(expected_user_id))
        else {
            continue;
        };
        summaries.insert(
            request_id,
            UsersMeVideoTaskUsageSummary {
                input_tokens: usage.input_tokens,
                output_tokens: usage.output_tokens,
                total_tokens: usage.total_tokens,
                // User surfaces expose the charged/sale price only. Provider
                // actual cost is an internal margin field.
                cost: usage.total_cost_usd,
            },
        );
    }
    Ok(summaries)
}

fn build_users_me_video_task_item(
    task: &StoredVideoTask,
    usage_summaries: &BTreeMap<String, UsersMeVideoTaskUsageSummary>,
) -> Option<Value> {
    let public_task_id = users_me_video_task_public_id(task)?;
    let usage = usage_summaries.get(task.request_id.trim());
    Some(json!({
        "id": public_task_id,
        "request_id": task.request_id,
        "global_model_name": users_me_video_task_global_model_name(task),
        "model": task.model,
        "prompt": task.prompt,
        "status": video_task_status_name(task.status),
        "progress_percent": task.progress_percent,
        "progress_message": task.progress_message,
        "duration_seconds": task.duration_seconds,
        "resolution": task.resolution,
        "aspect_ratio": task.aspect_ratio,
        "video_available": task.video_url.as_deref().is_some_and(|url| !url.trim().is_empty()),
        "error_code": task.error_code,
        "error_message": task.error_message,
        "input_tokens": usage.map(|usage| usage.input_tokens),
        "output_tokens": usage.map(|usage| usage.output_tokens),
        "total_tokens": usage.map(|usage| usage.total_tokens),
        "cost": usage.map(|usage| usage.cost),
        "created_at": unix_secs_to_rfc3339(task.created_at_unix_ms),
        "updated_at": unix_secs_to_rfc3339(task.updated_at_unix_secs),
        "submitted_at": task.submitted_at_unix_secs.and_then(unix_secs_to_rfc3339),
        "completed_at": task.completed_at_unix_secs.and_then(unix_secs_to_rfc3339),
    }))
}

fn users_me_video_task_public_id(task: &StoredVideoTask) -> Option<&str> {
    if is_native_doubao_video_task(task) {
        task.external_task_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    } else {
        let id = task.id.trim();
        (!id.is_empty()).then_some(id)
    }
}

fn is_native_doubao_video_task(task: &StoredVideoTask) -> bool {
    task.client_api_format
        .as_deref()
        .map(str::trim)
        .is_some_and(|format| format.eq_ignore_ascii_case("doubao:video"))
}

fn users_me_video_task_global_model_name(task: &StoredVideoTask) -> Option<String> {
    let metadata = task.request_metadata.as_ref().and_then(Value::as_object);
    metadata
        .and_then(|metadata| metadata.get("global_model_name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| task.model.clone())
}

fn parse_status(query: Option<&str>) -> Result<Option<VideoTaskStatus>, Response<Body>> {
    query_param_value(query, "status")
        .map(|status| {
            VideoTaskStatus::from_database(&status)
                .map_err(|err| users_me_video_tasks_bad_request(&err.to_string()))
        })
        .transpose()
}

fn parse_positive_usize(
    query: Option<&str>,
    key: &str,
    default: usize,
    maximum: usize,
) -> Result<usize, Response<Body>> {
    let Some(value) = query_param_value(query, key) else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>().map_err(|_| {
        users_me_video_tasks_bad_request(&format!("{key} must be a positive integer"))
    })?;
    if parsed == 0 || parsed > maximum {
        return Err(users_me_video_tasks_bad_request(&format!(
            "{key} must be between 1 and {maximum}"
        )));
    }
    Ok(parsed)
}

fn query_has_parameter(query: Option<&str>, key: &str) -> bool {
    query.is_some_and(|query| {
        form_urlencoded::parse(query.as_bytes()).any(|(entry_key, _)| entry_key == key)
    })
}

pub(super) fn users_me_video_task_detail_id(request_path: &str) -> Option<&str> {
    let normalized = request_path.trim_end_matches('/');
    let task_id = normalized.strip_prefix(&format!("{USERS_ME_VIDEO_TASKS_ROOT}/"))?;
    if task_id.is_empty() || task_id.contains('/') || task_id == "stats" {
        return None;
    }
    Some(task_id)
}

pub(super) fn users_me_video_task_nested_id<'a>(
    request_path: &'a str,
    suffix: &str,
) -> Option<&'a str> {
    let normalized = request_path.trim_end_matches('/');
    let task_id = normalized
        .strip_prefix(&format!("{USERS_ME_VIDEO_TASKS_ROOT}/"))?
        .strip_suffix(suffix)?;
    if task_id.is_empty() || task_id.contains('/') {
        return None;
    }
    Some(task_id)
}

fn users_me_video_task_not_found() -> Response<Body> {
    build_auth_error_response(http::StatusCode::NOT_FOUND, "Video task not found", false)
}

fn users_me_video_tasks_bad_request(detail: &str) -> Response<Body> {
    build_auth_error_response(http::StatusCode::BAD_REQUEST, detail, false)
}

fn users_me_video_tasks_internal_error(
    request_context: &GatewayPublicRequestContext,
    operation: &str,
    err: &GatewayError,
) -> Response<Body> {
    tracing::error!(
        trace_id = %request_context.trace_id,
        operation,
        error = ?err,
        "self-service video task operation failed"
    );
    build_auth_error_response(
        http::StatusCode::SERVICE_UNAVAILABLE,
        "Video task service is temporarily unavailable",
        false,
    )
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn video_task_status_name(status: VideoTaskStatus) -> &'static str {
    match status {
        VideoTaskStatus::Pending => "pending",
        VideoTaskStatus::Submitted => "submitted",
        VideoTaskStatus::Queued => "queued",
        VideoTaskStatus::Processing => "processing",
        VideoTaskStatus::Completed => "completed",
        VideoTaskStatus::Failed => "failed",
        VideoTaskStatus::Cancelled => "cancelled",
        VideoTaskStatus::Expired => "expired",
        VideoTaskStatus::Deleted => "deleted",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task(user_id: &str) -> StoredVideoTask {
        StoredVideoTask {
            id: "task-secret".to_string(),
            short_id: Some("short-secret".to_string()),
            request_id: "request-1".to_string(),
            user_id: Some(user_id.to_string()),
            api_key_id: Some("api-key-secret".to_string()),
            username: Some("alice".to_string()),
            api_key_name: Some("primary".to_string()),
            external_task_id: Some("upstream-secret".to_string()),
            provider_id: Some("provider-secret".to_string()),
            endpoint_id: Some("endpoint-secret".to_string()),
            key_id: Some("key-secret".to_string()),
            client_api_format: Some("openai:video".to_string()),
            provider_api_format: Some("doubao:video".to_string()),
            format_converted: true,
            model: Some("seedance-requested".to_string()),
            prompt: Some("hello".to_string()),
            original_request_body: Some(json!({"secret": "request-body"})),
            duration_seconds: Some(5),
            resolution: Some("720p".to_string()),
            aspect_ratio: Some("16:9".to_string()),
            size: Some("1280x720".to_string()),
            status: VideoTaskStatus::Completed,
            progress_percent: 100,
            progress_message: Some("done".to_string()),
            retry_count: 1,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: None,
            poll_count: 2,
            max_poll_count: 100,
            created_at_unix_ms: 1_710_000_000,
            submitted_at_unix_secs: Some(1_710_000_001),
            completed_at_unix_secs: Some(1_710_000_010),
            updated_at_unix_secs: 1_710_000_010,
            error_code: None,
            error_message: None,
            video_url: Some("https://provider.invalid/signed-secret.mp4".to_string()),
            request_metadata: Some(json!({
                "global_model_name": "seedance-global",
                "provider_token": "metadata-secret"
            })),
        }
    }

    #[test]
    fn user_video_task_dto_exposes_only_safe_fields() {
        let payload = build_users_me_video_task_item(&sample_task("user-1"), &BTreeMap::new())
            .expect("OpenAI task should have a public id");
        assert_eq!(payload["id"], "task-secret");
        assert_eq!(payload["global_model_name"], "seedance-global");
        assert_eq!(payload["video_available"], true);
        for forbidden in [
            "user_id",
            "username",
            "api_key_id",
            "api_key_name",
            "external_task_id",
            "provider_id",
            "provider_name",
            "endpoint_id",
            "key_id",
            "provider_api_format",
            "mapped_model",
            "observed_model",
            "original_request_body",
            "request_metadata",
            "video_url",
            "actual_cost",
        ] {
            assert!(
                payload.get(forbidden).is_none(),
                "must not expose {forbidden}"
            );
        }
    }

    #[test]
    fn native_doubao_user_video_task_dto_uses_only_upstream_id() {
        let mut task = sample_task("user-1");
        task.client_api_format = Some("doubao:video".to_string());
        task.provider_api_format = Some("doubao:video".to_string());
        task.format_converted = false;
        task.external_task_id = Some(" cgt-upstream-public ".to_string());

        let payload = build_users_me_video_task_item(&task, &BTreeMap::new())
            .expect("native Doubao task should have an upstream id");
        assert_eq!(payload["id"], "cgt-upstream-public");
        assert_ne!(payload["id"], task.id);

        task.external_task_id = Some("  ".to_string());
        assert!(build_users_me_video_task_item(&task, &BTreeMap::new()).is_none());
    }

    #[test]
    fn list_contract_rejects_even_an_empty_user_id_hint() {
        assert!(query_has_parameter(Some("page=1&user_id="), "user_id"));
        assert!(query_has_parameter(Some("user_id=someone-else"), "user_id"));
        assert!(!query_has_parameter(Some("page=1"), "user_id"));
        assert!(parse_positive_usize(Some("page=1000"), "page", 1, 1_000).is_ok());
        assert!(parse_positive_usize(Some("page=1001"), "page", 1, 1_000).is_err());
    }

    #[test]
    fn path_extractors_do_not_accept_nested_or_reserved_ids() {
        assert_eq!(
            users_me_video_task_detail_id("/api/users/me/video-tasks/task-1"),
            Some("task-1")
        );
        assert_eq!(
            users_me_video_task_nested_id("/api/users/me/video-tasks/task-1/video", "/video"),
            Some("task-1")
        );
        assert_eq!(
            users_me_video_task_detail_id("/api/users/me/video-tasks/stats"),
            None
        );
        assert_eq!(
            users_me_video_task_detail_id("/api/users/me/video-tasks/task-1/video"),
            None
        );
    }

    #[test]
    fn direct_user_media_plan_uses_the_dns_pinned_asset_transport_without_credentials() {
        let uri: http::Uri = "/api/users/me/video-tasks/task-secret/video"
            .parse()
            .expect("uri should parse");
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::RANGE, "bytes=0-1023".parse().unwrap());
        let request_context = GatewayPublicRequestContext::from_request_parts(
            "trace-media",
            &http::Method::GET,
            &uri,
            &headers,
            None,
        );
        let task = sample_task("user-1");
        let plan = build_users_me_direct_video_asset_plan(
            &request_context,
            &headers,
            &task,
            "https://cdn.example.com/video.mp4".to_string(),
        )
        .expect("direct media plan should build");

        assert_eq!(plan.provider_api_format, "doubao:video");
        assert_eq!(
            plan.headers.get("range").map(String::as_str),
            Some("bytes=0-1023")
        );
        assert_eq!(
            plan.headers
                .get(EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER)
                .map(String::as_str),
            Some("1")
        );
        assert!(!plan.headers.keys().any(|name| {
            name.eq_ignore_ascii_case("authorization")
                || name.eq_ignore_ascii_case("x-api-key")
                || name.eq_ignore_ascii_case("x-goog-api-key")
        }));
        assert!(plan.proxy.is_none());
        assert!(plan.body.json_body.is_none());
        assert!(plan.body.body_bytes_b64.is_none());
        assert!(plan.body.body_ref.is_none());
    }

    #[test]
    fn user_media_response_overrides_shared_upstream_cache_policy() {
        let response = Response::builder()
            .header(http::header::CACHE_CONTROL, "public, max-age=86400")
            .body(Body::empty())
            .expect("response should build");
        let response = protect_users_me_video_response(response);
        assert_eq!(
            response
                .headers()
                .get(http::header::CACHE_CONTROL)
                .and_then(|value| value.to_str().ok()),
            Some("private, no-store")
        );
        assert_eq!(
            response
                .headers()
                .get(http::header::VARY)
                .and_then(|value| value.to_str().ok()),
            Some("Authorization")
        );
    }
}
