use std::collections::BTreeMap;

use aether_contracts::{ExecutionPlan, RequestBody, EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER};
use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, UpsertVideoTask, VideoTaskStatus,
};
use serde_json::{json, Map, Value};

use crate::util::{normalize_video_task_error_code, safe_external_http_url, value_i32, value_u64};
use crate::{
    build_video_follow_up_report_context, current_unix_timestamp_secs, doubao_video_tasks_url,
    map_openai_task_status, parse_doubao_video_content_variant, parse_video_content_variant,
    request_body_string, resolve_follow_up_auth, DoubaoVideoTaskSeed, LocalVideoTaskContentAction,
    LocalVideoTaskFollowUpPlan, LocalVideoTaskReadResponse, LocalVideoTaskSnapshot,
    LocalVideoTaskStatus, VideoFollowUpReportContextInput, DEFAULT_VIDEO_TASK_MAX_POLL_COUNT,
    DEFAULT_VIDEO_TASK_POLL_INTERVAL_SECONDS,
};

/// Ark keeps cancelled content-generation tasks queryable for roughly one day.
pub const DOUBAO_CANCELLED_TASK_RETENTION_SECONDS: u64 = 24 * 60 * 60;

pub fn doubao_cancelled_task_is_visible_at(
    cancelled_at_unix_secs: u64,
    now_unix_secs: u64,
) -> bool {
    now_unix_secs.saturating_sub(cancelled_at_unix_secs) < DOUBAO_CANCELLED_TASK_RETENTION_SECONDS
}

pub fn doubao_stored_task_is_visible_at(task: &StoredVideoTask, now_unix_secs: u64) -> bool {
    match task.status {
        VideoTaskStatus::Deleted => false,
        VideoTaskStatus::Cancelled => {
            let cancelled_at_unix_secs = task
                .completed_at_unix_secs
                .or((task.updated_at_unix_secs > 0).then_some(task.updated_at_unix_secs))
                .or(task.submitted_at_unix_secs)
                // Despite the legacy field name, all video-task repositories
                // store this column as Unix seconds (Postgres reads
                // `EXTRACT(EPOCH ...)`, and the other adapters mirror it).
                // Dividing again makes a sparse legacy row look decades old.
                .unwrap_or(task.created_at_unix_ms);
            doubao_cancelled_task_is_visible_at(cancelled_at_unix_secs, now_unix_secs)
        }
        _ => true,
    }
}

pub fn map_doubao_stored_task_to_read_response(
    task: StoredVideoTask,
) -> LocalVideoTaskReadResponse {
    map_doubao_stored_task_to_read_response_at(task, current_unix_timestamp_secs())
}

pub fn map_doubao_stored_task_to_read_response_at(
    task: StoredVideoTask,
    now_unix_secs: u64,
) -> LocalVideoTaskReadResponse {
    if !doubao_stored_task_is_visible_at(&task, now_unix_secs) {
        return LocalVideoTaskReadResponse {
            status_code: 404,
            body_json: json!({
                "error": {
                    "code": "NotFound",
                    "message": "The requested generation task was not found.",
                }
            }),
        };
    }

    let status = task.status;
    LocalVideoTaskReadResponse {
        status_code: 200,
        body_json: build_doubao_stored_task_body(task, status),
    }
}

#[derive(Default)]
struct DoubaoStoredSnapshotFields {
    global_model_name: Option<String>,
    model: Option<String>,
    last_frame_url: Option<String>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    duration_seconds: Option<u32>,
    seed: Option<i32>,
    frames: Option<i32>,
    frames_per_second: Option<i32>,
}

fn build_doubao_stored_task_body(task: StoredVideoTask, status: VideoTaskStatus) -> Value {
    let metadata_global_model = task
        .request_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("global_model_name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let snapshot_fields = task
        .request_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("rust_local_snapshot"))
        .and_then(|value| serde_json::from_value::<LocalVideoTaskSnapshot>(value.clone()).ok())
        .and_then(|snapshot| match snapshot {
            LocalVideoTaskSnapshot::Doubao(seed) => Some(DoubaoStoredSnapshotFields {
                global_model_name: seed.persistence.global_model_name,
                model: seed.model,
                last_frame_url: seed
                    .last_frame_url
                    .and_then(|value| safe_external_http_url(&value)),
                completion_tokens: seed.completion_tokens,
                total_tokens: seed.total_tokens,
                duration_seconds: seed.duration_seconds,
                seed: seed.seed,
                frames: seed.frames,
                frames_per_second: seed.frames_per_second,
            }),
            _ => None,
        })
        .unwrap_or_default();
    let poll_body = task
        .request_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("poll_raw_response"))
        .and_then(Value::as_object);
    let request_body = task.original_request_body.as_ref();
    let projected_seed = snapshot_fields
        .seed
        .or_else(|| {
            poll_body
                .and_then(|body| body.get("seed"))
                .and_then(value_i32)
        })
        .or_else(|| {
            request_body
                .and_then(|body| body.get("seed"))
                .and_then(value_i32)
        });
    let projected_frames = snapshot_fields
        .frames
        .or_else(|| {
            poll_body
                .and_then(|body| body.get("frames"))
                .and_then(value_i32)
        })
        .or_else(|| {
            request_body
                .and_then(|body| body.get("frames"))
                .and_then(value_i32)
        });
    let projected_frames_per_second = snapshot_fields
        .frames_per_second
        .or_else(|| {
            poll_body
                .and_then(|body| body.get("framespersecond"))
                .and_then(value_i32)
        })
        .or_else(|| {
            request_body
                .and_then(|body| body.get("framespersecond"))
                .and_then(value_i32)
        });
    let projected_duration = if projected_frames.is_some() {
        None
    } else {
        snapshot_fields
            .duration_seconds
            .or_else(|| {
                poll_body
                    .and_then(|body| body.get("duration"))
                    .and_then(value_u64)
                    .and_then(|value| u32::try_from(value).ok())
            })
            .or(task.duration_seconds)
    };
    let mut body = json!({
        "id": task.id,
        "status": map_doubao_stored_task_status(status),
        "created_at": task.created_at_unix_ms,
        "updated_at": task.updated_at_unix_secs,
    });

    if let Some(model) = metadata_global_model
        .or(snapshot_fields.global_model_name)
        .or(snapshot_fields.model)
        .or(task.model)
    {
        body["model"] = Value::String(model);
    }
    let video_url = (status == VideoTaskStatus::Completed)
        .then(|| {
            task.video_url
                .and_then(|value| safe_external_http_url(&value))
        })
        .flatten();
    let last_frame_url = (status == VideoTaskStatus::Completed)
        .then(|| {
            snapshot_fields.last_frame_url.or_else(|| {
                poll_body
                    .and_then(|body| body.get("content"))
                    .and_then(Value::as_object)
                    .and_then(|content| content.get("last_frame_url"))
                    .and_then(Value::as_str)
                    .and_then(safe_external_http_url)
            })
        })
        .flatten();
    if video_url.is_some() || last_frame_url.is_some() {
        let mut content = Map::new();
        if let Some(video_url) = video_url {
            content.insert("video_url".to_string(), Value::String(video_url));
        }
        if let Some(last_frame_url) = last_frame_url {
            content.insert("last_frame_url".to_string(), Value::String(last_frame_url));
        }
        body["content"] = Value::Object(content);
    }
    if let Some(resolution) = task.resolution {
        body["resolution"] = Value::String(resolution);
    }
    if let Some(ratio) = task.aspect_ratio {
        body["ratio"] = Value::String(ratio);
    }
    if let Some(seed) = projected_seed {
        body["seed"] = Value::Number(seed.into());
    }
    if let Some(frames) = projected_frames {
        body["frames"] = Value::Number(frames.into());
    } else if let Some(duration) = projected_duration {
        body["duration"] = Value::String(duration.to_string());
    }
    if let Some(frames_per_second) = projected_frames_per_second {
        body["framespersecond"] = Value::Number(frames_per_second.into());
    }
    let completion_tokens = snapshot_fields.completion_tokens.or_else(|| {
        poll_body
            .and_then(|body| body.get("usage"))
            .and_then(Value::as_object)
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(value_u64)
    });
    let total_tokens = snapshot_fields.total_tokens.or_else(|| {
        poll_body
            .and_then(|body| body.get("usage"))
            .and_then(Value::as_object)
            .and_then(|usage| usage.get("total_tokens"))
            .and_then(value_u64)
    });
    if let Some(completion_tokens) = completion_tokens {
        body["usage"] = json!({
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens.unwrap_or(completion_tokens),
        });
    }
    if matches!(status, VideoTaskStatus::Failed | VideoTaskStatus::Expired) {
        body["error"] = json!({
            "code": task.error_code.unwrap_or_else(|| "InternalServiceError".to_string()),
            "message": task
                .error_message
                .unwrap_or_else(|| "Video generation failed".to_string()),
        });
    }

    body
}

fn map_doubao_stored_task_status(status: VideoTaskStatus) -> &'static str {
    match status {
        VideoTaskStatus::Pending | VideoTaskStatus::Submitted | VideoTaskStatus::Queued => "queued",
        VideoTaskStatus::Processing => "running",
        VideoTaskStatus::Completed => "succeeded",
        VideoTaskStatus::Failed | VideoTaskStatus::Expired => "failed",
        VideoTaskStatus::Cancelled | VideoTaskStatus::Deleted => "cancelled",
    }
}

pub fn map_doubao_task_status(status: LocalVideoTaskStatus) -> &'static str {
    match status {
        LocalVideoTaskStatus::Submitted | LocalVideoTaskStatus::Queued => "queued",
        LocalVideoTaskStatus::Processing => "running",
        LocalVideoTaskStatus::Completed => "succeeded",
        LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Expired => "failed",
        LocalVideoTaskStatus::Cancelled | LocalVideoTaskStatus::Deleted => "cancelled",
    }
}

impl DoubaoVideoTaskSeed {
    pub fn is_visible_at(&self, now_unix_secs: u64) -> bool {
        match self.status {
            LocalVideoTaskStatus::Deleted => false,
            LocalVideoTaskStatus::Cancelled => doubao_cancelled_task_is_visible_at(
                self.completed_at_unix_secs
                    .or(self.updated_at_unix_secs)
                    .unwrap_or(self.created_at_unix_secs),
                now_unix_secs,
            ),
            _ => true,
        }
    }

    fn effective_client_api_format(&self) -> &'static str {
        if self
            .persistence
            .client_api_format
            .trim()
            .eq_ignore_ascii_case("openai:video")
        {
            "openai:video"
        } else {
            "doubao:video"
        }
    }

    pub fn apply_provider_body(&mut self, provider_body: &Map<String, Value>) {
        let raw_status = provider_body
            .get("status")
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default();
        let provider_status = match raw_status {
            "queued" => Some(LocalVideoTaskStatus::Queued),
            "running" => Some(LocalVideoTaskStatus::Processing),
            "succeeded" => Some(LocalVideoTaskStatus::Completed),
            "failed" => Some(LocalVideoTaskStatus::Failed),
            "cancelled" => Some(LocalVideoTaskStatus::Cancelled),
            _ => None,
        };
        let current_is_terminal = matches!(
            self.status,
            LocalVideoTaskStatus::Completed
                | LocalVideoTaskStatus::Failed
                | LocalVideoTaskStatus::Cancelled
                | LocalVideoTaskStatus::Expired
                | LocalVideoTaskStatus::Deleted
        );
        if current_is_terminal && provider_status.is_some_and(|status| status != self.status) {
            // Provider task states are terminal once completed/failed/cancelled.
            // Ignore a late or out-of-order response instead of resurrecting a
            // task and potentially billing it twice.
            return;
        }
        let regresses_active_status = matches!(
            (self.status, provider_status),
            (
                LocalVideoTaskStatus::Processing,
                Some(LocalVideoTaskStatus::Queued)
            )
        );
        if let Some(provider_status) = provider_status.filter(|_| !regresses_active_status) {
            self.status = provider_status;
        }
        // Ark reports no progress percentage, so it is derived from the status.
        self.progress_percent = match self.status {
            LocalVideoTaskStatus::Completed => 100,
            LocalVideoTaskStatus::Processing => 50,
            LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Cancelled => 100,
            _ => self.progress_percent,
        };

        if let Some(model) = provider_body
            .get("model")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            // Ark's response model is an observed provider/version identity.
            // Never overwrite the request/global model used by the public
            // contract, list filtering, or billing dimensions.
            self.observed_model = Some(model.to_string());
        }

        if let Some(created_at) = provider_body.get("created_at").and_then(value_u64) {
            self.created_at_unix_secs = created_at;
        }

        let content = provider_body.get("content").and_then(Value::as_object);
        if let Some(video_url) = content
            .and_then(|content| content.get("video_url"))
            .and_then(Value::as_str)
            .and_then(safe_external_http_url)
        {
            self.video_url = Some(video_url);
        }
        if let Some(last_frame_url) = content
            .and_then(|content| content.get("last_frame_url"))
            .and_then(Value::as_str)
            .and_then(safe_external_http_url)
        {
            self.last_frame_url = Some(last_frame_url);
        }

        let usage = provider_body.get("usage").and_then(Value::as_object);
        if let Some(completion_tokens) = usage
            .and_then(|usage| usage.get("completion_tokens"))
            .and_then(value_u64)
        {
            self.completion_tokens = Some(completion_tokens);
        }
        if let Some(total_tokens) = usage
            .and_then(|usage| usage.get("total_tokens"))
            .and_then(value_u64)
        {
            self.total_tokens = Some(total_tokens);
        }

        // The provider echoes the resolved generation parameters, which are more
        // authoritative than whatever the client asked for.
        if let Some(resolution) = provider_body
            .get("resolution")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.resolution = Some(resolution.to_string());
        }
        if let Some(ratio) = provider_body
            .get("ratio")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            self.ratio = Some(ratio.to_string());
        }
        if let Some(seed) = provider_body.get("seed").and_then(value_i32) {
            self.seed = Some(seed);
        }
        let provider_frames = provider_body.get("frames").and_then(value_i32);
        let provider_duration = provider_body
            .get("duration")
            .and_then(value_u64)
            .and_then(|value| u32::try_from(value).ok());
        // Ark defines `frames` and `duration` as mutually exclusive. Prefer
        // the exact frame count if a non-conforming provider sends both.
        if let Some(frames) = provider_frames {
            self.frames = Some(frames);
            self.duration_seconds = None;
        } else if let Some(duration) = provider_duration {
            self.duration_seconds = Some(duration);
            self.frames = None;
        }
        if let Some(frames_per_second) = provider_body.get("framespersecond").and_then(value_i32) {
            self.frames_per_second = Some(frames_per_second);
        }

        if let Some(updated_at) = provider_body.get("updated_at").and_then(value_u64) {
            let updated_at = self
                .updated_at_unix_secs
                .map_or(updated_at, |current| current.max(updated_at));
            self.updated_at_unix_secs = Some(updated_at);
            if matches!(
                self.status,
                LocalVideoTaskStatus::Completed
                    | LocalVideoTaskStatus::Failed
                    | LocalVideoTaskStatus::Cancelled
            ) {
                self.completed_at_unix_secs = Some(updated_at);
            }
        }

        let error = provider_body.get("error").and_then(Value::as_object);
        self.error_code = normalize_video_task_error_code(
            error
                .and_then(|error| error.get("code"))
                .and_then(Value::as_str),
        );
        self.error_message = error
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .map(str::to_string);
    }

    pub fn client_body_json(&self) -> Value {
        let mut body = json!({
            "id": self.local_task_id,
            "status": map_doubao_task_status(self.status),
            "created_at": self.created_at_unix_secs,
            "updated_at": self
                .updated_at_unix_secs
                .or(self.completed_at_unix_secs)
                .unwrap_or(self.created_at_unix_secs),
        });

        // The request/global identity is intentionally projected here. The
        // provider-observed value remains available internally as
        // `observed_model` and is not allowed to cause response drift.
        if let Some(model) = self
            .persistence
            .global_model_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                self.persistence
                    .original_request_body
                    .get("model")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
            })
            .or_else(|| self.model.as_deref())
        {
            body["model"] = Value::String(model.to_string());
        }
        if let Some(resolution) = &self.resolution {
            body["resolution"] = Value::String(resolution.clone());
        }
        if let Some(ratio) = &self.ratio {
            body["ratio"] = Value::String(ratio.clone());
        }
        if let Some(seed) = self.seed {
            body["seed"] = Value::Number(seed.into());
        }
        if let Some(frames) = self.frames {
            body["frames"] = Value::Number(frames.into());
        } else if let Some(duration) = self.duration_seconds {
            // Ark's Get/List response schema declares duration as a string,
            // even though create accepts an integer request field.
            body["duration"] = Value::String(duration.to_string());
        }
        if let Some(frames_per_second) = self.frames_per_second {
            body["framespersecond"] = Value::Number(frames_per_second.into());
        }
        if self.status == LocalVideoTaskStatus::Completed {
            let mut content = Map::new();
            if let Some(video_url) = self.video_url.as_deref().and_then(safe_external_http_url) {
                content.insert("video_url".to_string(), Value::String(video_url));
            }
            if let Some(last_frame_url) = self
                .last_frame_url
                .as_deref()
                .and_then(safe_external_http_url)
            {
                content.insert("last_frame_url".to_string(), Value::String(last_frame_url));
            }
            if !content.is_empty() {
                body["content"] = Value::Object(content);
            }
        }
        if let Some(completion_tokens) = self.completion_tokens {
            body["usage"] = json!({
                "completion_tokens": completion_tokens,
                "total_tokens": self.total_tokens.unwrap_or(completion_tokens),
            });
        }
        if matches!(
            self.status,
            LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Expired
        ) {
            body["error"] = json!({
                "code": self
                    .error_code
                    .clone()
                    .unwrap_or_else(|| "InternalServiceError".to_string()),
                "message": self
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Video generation failed".to_string()),
            });
        }

        body
    }

    /// Projects an Ark-backed task onto the OpenAI video task contract.  The
    /// provider's resolved model stays in the Ark snapshot, while fields shown
    /// to the OpenAI caller are taken from the original client request first.
    pub fn openai_client_body_json(&self) -> Value {
        let request_body = &self.persistence.original_request_body;
        let mut body = json!({
            "id": self.local_task_id,
            "object": "video",
            "status": map_openai_task_status(self.status),
            "progress": self.progress_percent,
            "created_at": self.created_at_unix_secs,
        });

        if let Some(model) =
            request_body_string(request_body, "model").or_else(|| self.model.clone())
        {
            body["model"] = Value::String(model);
        }
        if let Some(prompt) =
            request_body_string(request_body, "prompt").or_else(|| self.prompt.clone())
        {
            body["prompt"] = Value::String(prompt);
        }
        if let Some(size) = request_body_string(request_body, "size") {
            body["size"] = Value::String(size);
        }
        if let Some(seconds) = request_body_string(request_body, "seconds")
            .or_else(|| self.duration_seconds.map(|value| value.to_string()))
        {
            body["seconds"] = Value::String(seconds);
        }
        if let Some(completed_at) = self.completed_at_unix_secs {
            body["completed_at"] = Value::Number(completed_at.into());
        }
        if matches!(
            self.status,
            LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Expired
        ) {
            body["error"] = json!({
                "code": self
                    .error_code
                    .clone()
                    .unwrap_or_else(|| "unknown".to_string()),
                "message": self
                    .error_message
                    .clone()
                    .unwrap_or_else(|| "Video generation failed".to_string()),
            });
        }

        body
    }

    pub fn build_get_follow_up_plan(&self, trace_id: &str) -> Option<ExecutionPlan> {
        if !matches!(
            self.status,
            LocalVideoTaskStatus::Submitted
                | LocalVideoTaskStatus::Queued
                | LocalVideoTaskStatus::Processing
        ) {
            return None;
        }

        let mut headers = self.transport.headers.clone();
        headers.remove("content-type");
        headers.remove("content-length");

        Some(ExecutionPlan {
            request_id: trace_id.to_string(),
            candidate_id: None,
            provider_name: self.transport.provider_name.clone(),
            provider_id: self.transport.provider_id.clone(),
            endpoint_id: self.transport.endpoint_id.clone(),
            key_id: self.transport.key_id.clone(),
            method: "GET".to_string(),
            url: doubao_video_tasks_url(
                &self.transport.upstream_base_url,
                Some(&self.upstream_task_id),
            ),
            headers,
            content_type: None,
            content_encoding: None,
            body: RequestBody {
                json_body: None,
                body_bytes_b64: None,
                body_ref: None,
            },
            stream: false,
            client_api_format: self.effective_client_api_format().to_string(),
            provider_api_format: "doubao:video".to_string(),
            model_name: self
                .model
                .clone()
                .or_else(|| self.transport.model_name.clone()),
            proxy: self.transport.proxy.clone(),
            transport_profile: self.transport.transport_profile.clone(),
            timeouts: self.transport.timeouts.clone(),
        })
    }

    /// Ark exposes a single DELETE that cancels an in-flight task and removes a
    /// finished one, so both client intents map onto this plan.
    pub fn build_delete_follow_up_plan(
        &self,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        if matches!(self.status, LocalVideoTaskStatus::Deleted) {
            return None;
        }
        let report_kind = if matches!(
            self.status,
            LocalVideoTaskStatus::Submitted
                | LocalVideoTaskStatus::Queued
                | LocalVideoTaskStatus::Processing
        ) {
            "doubao_video_cancel_sync_finalize"
        } else {
            "doubao_video_delete_sync_finalize"
        };
        self.build_delete_follow_up_plan_for_contract(
            fallback_user_id,
            fallback_api_key_id,
            trace_id,
            "doubao:video",
            report_kind,
        )
    }

    pub fn build_openai_delete_follow_up_plan(
        &self,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        if !matches!(
            self.status,
            LocalVideoTaskStatus::Completed | LocalVideoTaskStatus::Failed
        ) {
            return None;
        }
        self.build_delete_follow_up_plan_for_contract(
            fallback_user_id,
            fallback_api_key_id,
            trace_id,
            "openai:video",
            "openai_video_delete_sync_finalize",
        )
    }

    pub fn build_openai_cancel_follow_up_plan(
        &self,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        if !matches!(
            self.status,
            LocalVideoTaskStatus::Submitted
                | LocalVideoTaskStatus::Queued
                | LocalVideoTaskStatus::Processing
        ) {
            return None;
        }
        self.build_delete_follow_up_plan_for_contract(
            fallback_user_id,
            fallback_api_key_id,
            trace_id,
            "openai:video",
            "openai_video_cancel_sync_finalize",
        )
    }

    fn build_delete_follow_up_plan_for_contract(
        &self,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
        client_api_format: &'static str,
        report_kind: &'static str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        let (user_id, api_key_id) = resolve_follow_up_auth(
            self.user_id.as_deref(),
            self.api_key_id.as_deref(),
            fallback_user_id,
            fallback_api_key_id,
        )?;
        let model_name = self
            .model
            .clone()
            .or_else(|| self.transport.model_name.clone());

        let mut headers = self.transport.headers.clone();
        headers.remove("content-type");
        headers.remove("content-length");

        Some(LocalVideoTaskFollowUpPlan {
            plan: ExecutionPlan {
                request_id: trace_id.to_string(),
                candidate_id: None,
                provider_name: self.transport.provider_name.clone(),
                provider_id: self.transport.provider_id.clone(),
                endpoint_id: self.transport.endpoint_id.clone(),
                key_id: self.transport.key_id.clone(),
                method: "DELETE".to_string(),
                url: doubao_video_tasks_url(
                    &self.transport.upstream_base_url,
                    Some(&self.upstream_task_id),
                ),
                headers,
                content_type: None,
                content_encoding: None,
                body: RequestBody {
                    json_body: None,
                    body_bytes_b64: None,
                    body_ref: None,
                },
                stream: false,
                client_api_format: client_api_format.to_string(),
                provider_api_format: "doubao:video".to_string(),
                model_name: model_name.clone(),
                proxy: self.transport.proxy.clone(),
                transport_profile: self.transport.transport_profile.clone(),
                timeouts: self.transport.timeouts.clone(),
            },
            report_kind: Some(report_kind.to_string()),
            report_context: Some(build_video_follow_up_report_context(
                VideoFollowUpReportContextInput {
                    request_id: &self.persistence.request_id,
                    user_id: &user_id,
                    api_key_id: &api_key_id,
                    task_id: &self.local_task_id,
                    provider_id: &self.transport.provider_id,
                    endpoint_id: &self.transport.endpoint_id,
                    key_id: &self.transport.key_id,
                    provider_name: self.transport.provider_name.as_deref(),
                    model_name: model_name.as_deref(),
                    global_model_name: self.persistence.global_model_name.as_deref(),
                    mapped_model: self.persistence.mapped_model.as_deref(),
                    model_id: self.persistence.model_id.as_deref(),
                    global_model_id: self.persistence.global_model_id.as_deref(),
                    client_api_format,
                    provider_api_format: "doubao:video",
                },
            )),
        })
    }

    /// Proxies the generated asset so clients never see the signed upstream URL,
    /// which also avoids handing out a link that expires within a day.
    pub fn build_content_stream_action(
        &self,
        query_string: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        let Some(variant) = parse_doubao_video_content_variant(query_string) else {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 400,
                body_json: json!({
                    "error": {
                        "code": "InvalidParameter",
                        "message": "Unsupported content variant. Expected 'video' or 'last_frame'.",
                    }
                }),
            });
        };

        match self.status {
            LocalVideoTaskStatus::Submitted
            | LocalVideoTaskStatus::Queued
            | LocalVideoTaskStatus::Processing => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 202,
                    body_json: json!({
                        "error": {
                            "code": "TaskNotCompleted",
                            "message": format!(
                                "Video is still processing (status: {})",
                                map_doubao_task_status(self.status)
                            ),
                        }
                    }),
                });
            }
            LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Expired => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 422,
                    body_json: json!({
                        "error": {
                            "code": self
                                .error_code
                                .clone()
                                .unwrap_or_else(|| "InternalServiceError".to_string()),
                            "message": self
                                .error_message
                                .clone()
                                .unwrap_or_else(|| "Video generation failed".to_string()),
                        }
                    }),
                });
            }
            // Ark keeps cancelled tasks readable for its retention window. A
            // missing asset is therefore a state conflict, not a missing task.
            LocalVideoTaskStatus::Cancelled => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 409,
                    body_json: json!({
                        "error": {
                            "code": "ContentNotAvailable",
                            "message": "Content is not available for a cancelled generation task.",
                        }
                    }),
                });
            }
            LocalVideoTaskStatus::Deleted => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 404,
                    body_json: json!({
                        "error": {
                            "code": "NotFound",
                            "message": "The requested generation task was not found.",
                        }
                    }),
                });
            }
            LocalVideoTaskStatus::Completed => {}
        }

        let requested_url = match variant {
            "last_frame" => self.last_frame_url.clone(),
            _ => self.video_url.clone(),
        };
        let Some(url) = requested_url.and_then(|value| safe_external_http_url(&value)) else {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 409,
                body_json: json!({
                    "error": {
                        "code": "ContentNotAvailable",
                        "message": format!(
                            "The requested '{variant}' content is not available for this generation task."
                        ),
                    }
                }),
            });
        };

        Some(LocalVideoTaskContentAction::StreamPlan(Box::new(
            ExecutionPlan {
                request_id: trace_id.to_string(),
                candidate_id: None,
                provider_name: self.transport.provider_name.clone(),
                provider_id: self.transport.provider_id.clone(),
                endpoint_id: self.transport.endpoint_id.clone(),
                key_id: self.transport.key_id.clone(),
                method: "GET".to_string(),
                url,
                // The asset URL is pre-signed, so upstream credentials must not ride along.
                headers: BTreeMap::from([(
                    EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER.to_string(),
                    "1".to_string(),
                )]),
                content_type: None,
                content_encoding: None,
                body: RequestBody {
                    json_body: None,
                    body_bytes_b64: None,
                    body_ref: None,
                },
                stream: true,
                client_api_format: "doubao:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                model_name: self
                    .model
                    .clone()
                    .or_else(|| self.transport.model_name.clone()),
                proxy: self.transport.proxy.clone(),
                transport_profile: self.transport.transport_profile.clone(),
                timeouts: self.transport.timeouts.clone(),
            },
        )))
    }

    /// Serves an Ark-generated asset through the OpenAI content route.  Ark's
    /// optional last frame is the closest lossless projection for OpenAI's
    /// thumbnail variant; spritesheets are not synthesized.
    pub fn build_openai_content_stream_action(
        &self,
        query_string: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        let Some(variant) = parse_video_content_variant(query_string) else {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 400,
                body_json: json!({
                    "error": {
                        "message": "Invalid content variant. Expected 'video', 'thumbnail', or 'spritesheet'.",
                        "type": "invalid_request_error",
                        "param": "variant",
                        "code": "invalid_variant",
                    }
                }),
            });
        };
        if variant == "spritesheet" {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 400,
                body_json: json!({
                    "error": {
                        "message": "The 'spritesheet' content variant is not supported for this video.",
                        "type": "invalid_request_error",
                        "param": "variant",
                        "code": "unsupported_variant",
                    }
                }),
            });
        }

        match self.status {
            LocalVideoTaskStatus::Submitted
            | LocalVideoTaskStatus::Queued
            | LocalVideoTaskStatus::Processing => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 202,
                    body_json: json!({
                        "detail": format!(
                            "Video is still processing (status: {})",
                            map_openai_task_status(self.status)
                        )
                    }),
                });
            }
            LocalVideoTaskStatus::Failed | LocalVideoTaskStatus::Expired => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 422,
                    body_json: json!({
                        "detail": format!(
                            "Video generation failed: {}",
                            self.error_message
                                .clone()
                                .unwrap_or_else(|| "Unknown error".to_string())
                        )
                    }),
                });
            }
            LocalVideoTaskStatus::Cancelled => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 404,
                    body_json: json!({"detail": "Video task was cancelled"}),
                });
            }
            LocalVideoTaskStatus::Deleted => {
                return Some(LocalVideoTaskContentAction::Immediate {
                    status_code: 404,
                    body_json: json!({"detail": "Video task not found"}),
                });
            }
            LocalVideoTaskStatus::Completed => {}
        }

        let requested_url = match variant {
            "video" => self.video_url.clone(),
            "thumbnail" => self.last_frame_url.clone(),
            _ => None,
        };
        let Some(url) = requested_url.and_then(|value| safe_external_http_url(&value)) else {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 409,
                body_json: json!({
                    "error": {
                        "message": format!(
                            "The requested '{variant}' content is not available for this video."
                        ),
                        "type": "invalid_request_error",
                        "param": "variant",
                        "code": "content_not_available",
                    }
                }),
            });
        };

        Some(LocalVideoTaskContentAction::StreamPlan(Box::new(
            ExecutionPlan {
                request_id: trace_id.to_string(),
                candidate_id: None,
                provider_name: self.transport.provider_name.clone(),
                provider_id: self.transport.provider_id.clone(),
                endpoint_id: self.transport.endpoint_id.clone(),
                key_id: self.transport.key_id.clone(),
                method: "GET".to_string(),
                url,
                // Ark content URLs are pre-signed; never attach the provider SK.
                headers: BTreeMap::from([(
                    EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER.to_string(),
                    "1".to_string(),
                )]),
                content_type: None,
                content_encoding: None,
                body: RequestBody {
                    json_body: None,
                    body_bytes_b64: None,
                    body_ref: None,
                },
                stream: true,
                client_api_format: "openai:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                model_name: self
                    .model
                    .clone()
                    .or_else(|| self.transport.model_name.clone()),
                proxy: self.transport.proxy.clone(),
                transport_profile: self.transport.transport_profile.clone(),
                timeouts: self.transport.timeouts.clone(),
            },
        )))
    }

    pub fn to_upsert_record(&self) -> UpsertVideoTask {
        let now_unix_secs = current_unix_timestamp_secs();
        let next_poll_at_unix_secs = match self.status {
            LocalVideoTaskStatus::Submitted
            | LocalVideoTaskStatus::Queued
            | LocalVideoTaskStatus::Processing => Some(
                now_unix_secs.saturating_add(u64::from(DEFAULT_VIDEO_TASK_POLL_INTERVAL_SECONDS)),
            ),
            _ => None,
        };
        UpsertVideoTask {
            id: self.local_task_id.clone(),
            // The column is NOT NULL; Doubao has no short-id concept of its own.
            short_id: Some(crate::derive_video_task_short_id(&self.local_task_id)),
            request_id: self.persistence.request_id.clone(),
            user_id: self.user_id.clone(),
            api_key_id: self.api_key_id.clone(),
            username: self.persistence.username.clone(),
            api_key_name: self.persistence.api_key_name.clone(),
            external_task_id: Some(self.upstream_task_id.clone()),
            provider_id: Some(self.transport.provider_id.clone()),
            endpoint_id: Some(self.transport.endpoint_id.clone()),
            key_id: Some(self.transport.key_id.clone()),
            client_api_format: Some(self.persistence.client_api_format.clone()),
            provider_api_format: Some(self.persistence.provider_api_format.clone()),
            format_converted: self.persistence.format_converted,
            // Ark's list `filter.model` remains an exact request-side
            // model/endpoint-ID filter. Keep that raw value in the indexed
            // column, while the public response projects the stable global
            // identity and the provider echo remains diagnostic metadata.
            model: crate::request_body_string(&self.persistence.original_request_body, "model")
                .or_else(|| self.model.clone())
                .or_else(|| Some(String::new())),
            prompt: self.prompt.clone().or_else(|| Some(String::new())),
            original_request_body: Some(self.persistence.original_request_body.clone()),
            duration_seconds: self.duration_seconds,
            resolution: self.resolution.clone(),
            aspect_ratio: self.ratio.clone(),
            size: request_body_string(&self.persistence.original_request_body, "size"),
            status: self.status.as_database_status(),
            progress_percent: self.progress_percent,
            progress_message: None,
            retry_count: 0,
            poll_interval_seconds: DEFAULT_VIDEO_TASK_POLL_INTERVAL_SECONDS,
            next_poll_at_unix_secs,
            poll_count: 0,
            max_poll_count: DEFAULT_VIDEO_TASK_MAX_POLL_COUNT,
            created_at_unix_ms: self.created_at_unix_secs,
            submitted_at_unix_secs: Some(self.created_at_unix_secs),
            completed_at_unix_secs: self.completed_at_unix_secs,
            updated_at_unix_secs: self
                .updated_at_unix_secs
                .or(self.completed_at_unix_secs)
                .unwrap_or(now_unix_secs),
            error_code: normalize_video_task_error_code(self.error_code.as_deref()),
            error_message: self.error_message.clone(),
            video_url: self.video_url.clone(),
            request_metadata: Some({
                let mut metadata = Map::new();
                metadata.insert(
                    "rust_owner".to_string(),
                    Value::String("async_task".to_string()),
                );
                metadata.insert(
                    "rust_local_snapshot".to_string(),
                    serde_json::to_value(
                        LocalVideoTaskSnapshot::Doubao(self.clone()).redacted_for_persistence(),
                    )
                    .expect("video snapshot should serialize"),
                );
                self.persistence.append_identity_metadata(&mut metadata);
                if let Some(observed_model) = self.observed_model.as_deref() {
                    metadata.insert(
                        "observed_model".to_string(),
                        Value::String(observed_model.to_string()),
                    );
                }
                Value::Object(metadata)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use aether_contracts::EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER;
    use aether_data_contracts::repository::video_tasks::{StoredVideoTask, VideoTaskStatus};
    use serde_json::{json, Value};

    use super::{
        map_doubao_stored_task_to_read_response, map_doubao_stored_task_to_read_response_at,
        DOUBAO_CANCELLED_TASK_RETENTION_SECONDS,
    };
    use crate::{
        DoubaoVideoTaskSeed, LocalVideoTaskPersistence, LocalVideoTaskSnapshot,
        LocalVideoTaskStatus, LocalVideoTaskTransport,
    };

    fn sample_seed() -> DoubaoVideoTaskSeed {
        DoubaoVideoTaskSeed {
            local_task_id: "cgt-local-123".to_string(),
            upstream_task_id: "cgt-upstream-123".to_string(),
            created_at_unix_secs: 1_768_294_532,
            updated_at_unix_secs: None,
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            model: Some("doubao-seedance-2-0-260128".to_string()),
            observed_model: None,
            prompt: Some("a cat yawning".to_string()),
            resolution: None,
            ratio: Some("16:9".to_string()),
            duration_seconds: Some(11),
            seed: None,
            frames: None,
            frames_per_second: None,
            status: LocalVideoTaskStatus::Submitted,
            progress_percent: 0,
            completed_at_unix_secs: None,
            error_code: None,
            error_message: None,
            video_url: None,
            last_frame_url: None,
            completion_tokens: None,
            total_tokens: None,
            persistence: LocalVideoTaskPersistence {
                request_id: "req-1".to_string(),
                username: None,
                api_key_name: None,
                client_api_format: "doubao:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                original_request_body: json!({}),
                format_converted: false,
                global_model_name: None,
                mapped_model: None,
                model_id: None,
                global_model_id: None,
            },
            transport: LocalVideoTaskTransport {
                upstream_base_url: "https://ark.cn-beijing.volces.com/api".to_string(),
                provider_name: Some("Ark".to_string()),
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "key-1".to_string(),
                headers: [(
                    "authorization".to_string(),
                    "Bearer provider-sk".to_string(),
                )]
                .into_iter()
                .collect(),
                content_type: Some("application/json".to_string()),
                model_name: Some("doubao-seedance-2-0-260128".to_string()),
                proxy: None,
                transport_profile: None,
                timeouts: None,
            },
        }
    }

    fn sample_stored_task(status: VideoTaskStatus) -> StoredVideoTask {
        StoredVideoTask {
            id: "cgt-local-123".to_string(),
            short_id: None,
            request_id: "req-1".to_string(),
            user_id: None,
            api_key_id: None,
            username: None,
            api_key_name: None,
            external_task_id: Some("cgt-upstream-123".to_string()),
            provider_id: None,
            endpoint_id: None,
            key_id: None,
            client_api_format: Some("doubao:video".to_string()),
            provider_api_format: Some("doubao:video".to_string()),
            format_converted: false,
            model: Some("doubao-seedance-2-0-260128".to_string()),
            prompt: Some("a cat yawning".to_string()),
            original_request_body: None,
            duration_seconds: Some(11),
            resolution: None,
            aspect_ratio: Some("16:9".to_string()),
            size: None,
            status,
            progress_percent: 100,
            progress_message: None,
            retry_count: 0,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: None,
            poll_count: 0,
            max_poll_count: 360,
            created_at_unix_ms: 1_768_294_532,
            submitted_at_unix_secs: Some(1_768_294_532),
            completed_at_unix_secs: Some(1_768_294_581),
            updated_at_unix_secs: 1_768_294_581,
            error_code: None,
            error_message: None,
            video_url: Some("https://tos.example.com/video.mp4?X-Sig=abc".to_string()),
            request_metadata: None,
        }
    }

    #[test]
    fn applies_succeeded_provider_body() {
        let mut seed = sample_seed();
        let body = json!({
            "id": "cgt-upstream-123",
            "model": "doubao-seedance-2-0-260128-resolved",
            "status": "succeeded",
            "content": {"video_url": "https://tos.example.com/v.mp4?X-Sig=abc"},
            "usage": {"completion_tokens": 295_800, "total_tokens": 295_800},
            "created_at": "1768294533",
            "updated_at": "1768294581",
            "resolution": "1080p",
            "ratio": "16:9",
            "duration": "11",
            "seed": -1,
            "framespersecond": 24
        });

        seed.apply_provider_body(body.as_object().expect("object"));

        assert_eq!(seed.status, LocalVideoTaskStatus::Completed);
        assert_eq!(seed.progress_percent, 100);
        assert_eq!(
            seed.video_url.as_deref(),
            Some("https://tos.example.com/v.mp4?X-Sig=abc")
        );
        assert_eq!(seed.completion_tokens, Some(295_800));
        assert_eq!(seed.total_tokens, Some(295_800));
        assert_eq!(seed.resolution.as_deref(), Some("1080p"));
        assert_eq!(seed.duration_seconds, Some(11));
        assert_eq!(seed.seed, Some(-1));
        assert_eq!(seed.frames, None);
        assert_eq!(seed.frames_per_second, Some(24));
        assert_eq!(seed.created_at_unix_secs, 1_768_294_533);
        assert_eq!(seed.updated_at_unix_secs, Some(1_768_294_581));
        assert_eq!(seed.completed_at_unix_secs, Some(1_768_294_581));
        assert_eq!(seed.model.as_deref(), Some("doubao-seedance-2-0-260128"));
        assert_eq!(
            seed.observed_model.as_deref(),
            Some("doubao-seedance-2-0-260128-resolved")
        );
    }

    #[test]
    fn provider_poll_model_does_not_drift_public_model_identity() {
        let mut seed = sample_seed();
        seed.persistence.global_model_name = Some("Doubao-Seedance-2.0".to_string());
        seed.persistence.mapped_model = Some("doubao-seedance-2-0-260128".to_string());
        seed.persistence.original_request_body = json!({
            "model": "ep-requested-model",
            "prompt": "a cat yawning"
        });

        seed.apply_provider_body(
            json!({
                "status": "succeeded",
                "model": "doubao-seedance-2-0-260128-resolved",
                "content": {"video_url": "https://tos.example.com/v.mp4?X-Sig=abc"},
                "usage": {"completion_tokens": 10, "total_tokens": 15}
            })
            .as_object()
            .expect("object"),
        );

        assert_eq!(seed.client_body_json()["model"], "Doubao-Seedance-2.0");
        assert_eq!(
            seed.observed_model.as_deref(),
            Some("doubao-seedance-2-0-260128-resolved")
        );
        let record = seed.to_upsert_record();
        assert_eq!(record.model.as_deref(), Some("ep-requested-model"));
        let metadata = record
            .request_metadata
            .as_ref()
            .and_then(Value::as_object)
            .expect("identity metadata");
        assert_eq!(metadata["global_model_name"], "Doubao-Seedance-2.0");
        assert_eq!(metadata["mapped_model"], "doubao-seedance-2-0-260128");
        assert_eq!(
            metadata["observed_model"],
            "doubao-seedance-2-0-260128-resolved"
        );
    }

    #[test]
    fn provider_frames_replace_duration_and_project_official_field_names() {
        let mut seed = sample_seed();
        seed.apply_provider_body(
            json!({
                "status": "running",
                "seed": 42,
                "frames": 121,
                "framespersecond": 24
            })
            .as_object()
            .expect("object"),
        );

        assert_eq!(seed.seed, Some(42));
        assert_eq!(seed.frames, Some(121));
        assert_eq!(seed.frames_per_second, Some(24));
        assert_eq!(seed.duration_seconds, None);

        let body = seed.client_body_json();
        assert_eq!(body["seed"].as_i64(), Some(42));
        assert_eq!(body["frames"].as_i64(), Some(121));
        assert_eq!(body["framespersecond"].as_i64(), Some(24));
        assert!(body.get("duration").is_none());
    }

    #[test]
    fn maps_running_status_to_processing() {
        let mut seed = sample_seed();
        seed.apply_provider_body(
            json!({"status": "running", "updated_at": "1768294550"})
                .as_object()
                .expect("object"),
        );

        assert_eq!(seed.status, LocalVideoTaskStatus::Processing);
        assert_eq!(seed.progress_percent, 50);
        assert_eq!(seed.updated_at_unix_secs, Some(1_768_294_550));
        assert_eq!(seed.completed_at_unix_secs, None);
        assert_eq!(seed.client_body_json()["updated_at"], 1_768_294_550u64);
    }

    #[test]
    fn applies_failed_provider_body_with_error() {
        let mut seed = sample_seed();
        seed.apply_provider_body(
            json!({
                "status": "failed",
                "error": {"code": "InputImageSensitiveContentDetected", "message": "blocked"}
            })
            .as_object()
            .expect("object"),
        );

        assert_eq!(seed.status, LocalVideoTaskStatus::Failed);
        assert_eq!(
            seed.error_code.as_deref(),
            Some("InputImageSensitiveContentDetected")
        );
        assert_eq!(seed.error_message.as_deref(), Some("blocked"));
    }

    #[test]
    fn unknown_status_keeps_previous_state() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Processing;
        seed.apply_provider_body(json!({"status": "weird"}).as_object().expect("object"));

        assert_eq!(seed.status, LocalVideoTaskStatus::Processing);
    }

    #[test]
    fn late_running_response_cannot_resurrect_a_terminal_task() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.video_url = Some("https://tos.example.com/final.mp4?sig=1".to_string());
        seed.completed_at_unix_secs = Some(1_768_294_581);

        seed.apply_provider_body(
            json!({
                "status": "running",
                "updated_at": "1768294599",
                "content": {"video_url": "https://tos.example.com/stale.mp4?sig=2"}
            })
            .as_object()
            .expect("object"),
        );

        assert_eq!(seed.status, LocalVideoTaskStatus::Completed);
        assert_eq!(
            seed.video_url.as_deref(),
            Some("https://tos.example.com/final.mp4?sig=1")
        );
        assert_eq!(seed.completed_at_unix_secs, Some(1_768_294_581));
    }

    #[test]
    fn late_queued_response_cannot_regress_a_running_task() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Processing;
        seed.updated_at_unix_secs = Some(1_768_294_600);

        seed.apply_provider_body(
            json!({
                "status": "queued",
                "updated_at": "1768294599"
            })
            .as_object()
            .expect("object"),
        );

        assert_eq!(seed.status, LocalVideoTaskStatus::Processing);
        assert_eq!(seed.progress_percent, 50);
        assert_eq!(seed.updated_at_unix_secs, Some(1_768_294_600));
    }

    #[test]
    fn client_body_uses_local_id_and_doubao_shape() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.video_url = Some("https://tos.example.com/v.mp4".to_string());
        seed.completion_tokens = Some(1_000);
        seed.seed = Some(-1);
        seed.frames_per_second = Some(24);

        let body = seed.client_body_json();

        assert_eq!(body["id"], "cgt-local-123");
        assert_eq!(body["status"], "succeeded");
        assert_eq!(
            body["content"]["video_url"],
            "https://tos.example.com/v.mp4"
        );
        assert_eq!(body["usage"]["completion_tokens"], 1_000);
        assert_eq!(body["ratio"], "16:9");
        assert_eq!(body["duration"].as_str(), Some("11"));
        assert_eq!(body["seed"].as_i64(), Some(-1));
        assert_eq!(body["framespersecond"].as_i64(), Some(24));
    }

    #[test]
    fn get_follow_up_plan_targets_task_resource() {
        let mut seed = sample_seed();
        seed.transport.headers.insert(
            "authorization".to_string(),
            "Bearer provider-sk".to_string(),
        );
        let plan = seed.build_get_follow_up_plan("trace-1").expect("plan");

        assert_eq!(plan.method, "GET");
        assert_eq!(
            plan.url,
            "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-upstream-123"
        );
        assert_eq!(plan.provider_api_format, "doubao:video");
        assert_eq!(
            plan.headers.get("authorization").map(String::as_str),
            Some("Bearer provider-sk")
        );
    }

    #[test]
    fn terminal_task_has_no_get_follow_up_plan() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;

        assert!(seed.build_get_follow_up_plan("trace-1").is_none());
    }

    #[test]
    fn content_stream_action_drops_upstream_auth_for_signed_urls() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.video_url = Some("https://tos.example.com/v.mp4?X-Sig=abc".to_string());
        seed.transport.headers.insert(
            "authorization".to_string(),
            "Bearer upstream-secret".to_string(),
        );

        let action = seed
            .build_content_stream_action(None, "trace-1")
            .expect("action");

        match action {
            crate::LocalVideoTaskContentAction::StreamPlan(plan) => {
                assert_eq!(plan.url, "https://tos.example.com/v.mp4?X-Sig=abc");
                assert!(!plan.headers.contains_key("authorization"));
                assert_eq!(
                    plan.headers
                        .get(EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER)
                        .map(String::as_str),
                    Some("1")
                );
                assert!(plan.stream);
            }
            other => panic!("expected stream plan, got {other:?}"),
        }
    }

    #[test]
    fn content_stream_action_reports_pending_task() {
        let seed = sample_seed();
        let action = seed
            .build_content_stream_action(None, "trace-1")
            .expect("action");

        match action {
            crate::LocalVideoTaskContentAction::Immediate {
                status_code,
                body_json,
            } => {
                assert_eq!(status_code, 202);
                assert_eq!(body_json["error"]["code"], "TaskNotCompleted");
            }
            other => panic!("expected immediate response, got {other:?}"),
        }
    }

    #[test]
    fn native_content_variants_return_precise_errors_instead_of_false_not_found() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.video_url = Some("https://tos.example.com/video.mp4?X-Sig=secret".to_string());

        let invalid = seed
            .build_content_stream_action(Some("variant=spritesheet"), "trace-invalid")
            .expect("invalid variant should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = invalid
        else {
            panic!("invalid variant should not stream");
        };
        assert_eq!(status_code, 400);
        assert_eq!(body_json["error"]["code"], "InvalidParameter");
        assert!(!body_json.to_string().contains("X-Sig=secret"));

        let unavailable = seed
            .build_content_stream_action(Some("variant=last_frame"), "trace-missing")
            .expect("missing last frame should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = unavailable
        else {
            panic!("missing last frame should not stream");
        };
        assert_eq!(status_code, 409);
        assert_eq!(body_json["error"]["code"], "ContentNotAvailable");
        assert!(!body_json.to_string().contains("X-Sig=secret"));

        seed.status = LocalVideoTaskStatus::Cancelled;
        let cancelled = seed
            .build_content_stream_action(None, "trace-cancelled")
            .expect("cancelled task should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = cancelled
        else {
            panic!("cancelled task should not stream");
        };
        assert_eq!(status_code, 409);
        assert_eq!(body_json["error"]["code"], "ContentNotAvailable");
    }

    #[test]
    fn native_last_frame_stream_uses_signed_url_without_provider_credentials() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.last_frame_url =
            Some("https://tos.example.com/frame.jpg?X-Sig=frame-secret".to_string());
        seed.transport.headers.insert(
            "authorization".to_string(),
            "Bearer upstream-secret".to_string(),
        );

        let action = seed
            .build_content_stream_action(Some("variant=last_frame"), "trace-frame")
            .expect("last frame should stream");
        let crate::LocalVideoTaskContentAction::StreamPlan(plan) = action else {
            panic!("last frame should produce a stream plan");
        };
        assert_eq!(
            plan.url,
            "https://tos.example.com/frame.jpg?X-Sig=frame-secret"
        );
        assert!(!plan.headers.contains_key("authorization"));
        assert_eq!(
            plan.headers
                .get(EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER)
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(plan.client_api_format, "doubao:video");
        assert_eq!(plan.provider_api_format, "doubao:video");
    }

    #[test]
    fn openai_projection_maps_thumbnail_and_rejects_unsynthesized_spritesheet() {
        let mut seed = sample_seed();
        seed.persistence.client_api_format = "openai:video".to_string();
        seed.persistence.format_converted = true;
        seed.status = LocalVideoTaskStatus::Completed;
        seed.video_url = Some("https://tos.example.com/video.mp4?X-Sig=video-secret".to_string());
        seed.last_frame_url =
            Some("https://tos.example.com/frame.jpg?X-Sig=frame-secret".to_string());
        seed.transport.headers.insert(
            "authorization".to_string(),
            "Bearer upstream-secret".to_string(),
        );

        let client_body = seed.openai_client_body_json();
        assert!(client_body.get("content").is_none());
        assert!(client_body.get("video_url").is_none());
        assert!(!client_body.to_string().contains("X-Sig="));

        let thumbnail = seed
            .build_openai_content_stream_action(Some("variant=thumbnail"), "trace-thumbnail")
            .expect("thumbnail should map to the Ark last frame");
        let crate::LocalVideoTaskContentAction::StreamPlan(plan) = thumbnail else {
            panic!("thumbnail should stream");
        };
        assert_eq!(
            plan.url,
            "https://tos.example.com/frame.jpg?X-Sig=frame-secret"
        );
        assert!(!plan.headers.contains_key("authorization"));
        assert_eq!(
            plan.headers
                .get(EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER)
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(plan.client_api_format, "openai:video");
        assert_eq!(plan.provider_api_format, "doubao:video");

        let spritesheet = seed
            .build_openai_content_stream_action(Some("variant=spritesheet"), "trace-spritesheet")
            .expect("unsupported spritesheet should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = spritesheet
        else {
            panic!("spritesheet should not stream");
        };
        assert_eq!(status_code, 400);
        assert_eq!(body_json["error"]["type"], "invalid_request_error");
        assert_eq!(body_json["error"]["param"], "variant");
        assert_eq!(body_json["error"]["code"], "unsupported_variant");
        assert!(!body_json.to_string().contains("X-Sig="));

        let invalid = seed
            .build_openai_content_stream_action(Some("variant=bogus"), "trace-invalid")
            .expect("invalid variant should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = invalid
        else {
            panic!("invalid variant should not stream");
        };
        assert_eq!(status_code, 400);
        assert_eq!(body_json["error"]["code"], "invalid_variant");
    }

    #[test]
    fn openai_projection_reports_missing_or_unsafe_thumbnail_as_conflict() {
        let mut seed = sample_seed();
        seed.persistence.client_api_format = "openai:video".to_string();
        seed.persistence.format_converted = true;
        seed.status = LocalVideoTaskStatus::Completed;
        seed.last_frame_url = Some("https://127.0.0.1/internal-frame.jpg".to_string());

        let action = seed
            .build_openai_content_stream_action(Some("variant=thumbnail"), "trace-thumbnail")
            .expect("unsafe thumbnail should produce an error action");
        let crate::LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = action
        else {
            panic!("unsafe thumbnail should not stream");
        };
        assert_eq!(status_code, 409);
        assert_eq!(body_json["error"]["type"], "invalid_request_error");
        assert_eq!(body_json["error"]["code"], "content_not_available");
        assert!(!body_json.to_string().contains("127.0.0.1"));
    }

    #[test]
    fn active_delete_follow_up_uses_cancel_report_semantics() {
        let seed = sample_seed();
        let plan = seed
            .build_delete_follow_up_plan(None, None, "trace-1")
            .expect("plan");

        assert_eq!(plan.plan.method, "DELETE");
        assert_eq!(
            plan.plan.url,
            "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/cgt-upstream-123"
        );
        assert_eq!(
            plan.report_kind.as_deref(),
            Some("doubao_video_cancel_sync_finalize")
        );
    }

    #[test]
    fn upsert_record_carries_billing_dimensions_and_snapshot() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.updated_at_unix_secs = Some(1_768_294_581);
        seed.completion_tokens = Some(295_800);
        seed.seed = Some(7);
        seed.frames_per_second = Some(24);

        let record = seed.to_upsert_record();

        assert_eq!(record.id, "cgt-local-123");
        assert_eq!(record.external_task_id.as_deref(), Some("cgt-upstream-123"));
        assert_eq!(record.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(record.duration_seconds, Some(11));
        assert_eq!(record.updated_at_unix_secs, 1_768_294_581);
        assert_eq!(record.next_poll_at_unix_secs, None);
        assert!(record
            .request_metadata
            .as_ref()
            .and_then(|value| value.get("rust_local_snapshot"))
            .is_some());
        let persisted_snapshot =
            &record.request_metadata.as_ref().expect("metadata")["rust_local_snapshot"]["Doubao"];
        assert_eq!(persisted_snapshot["seed"].as_i64(), Some(7));
        assert_eq!(persisted_snapshot["frames_per_second"].as_i64(), Some(24));
        assert!(
            record
                .request_metadata
                .as_ref()
                .is_some_and(|value| value["rust_local_snapshot"]["Doubao"]["transport"]
                    ["headers"]
                    .as_object()
                    .is_some_and(serde_json::Map::is_empty)),
            "provider SK must not be persisted in task metadata"
        );
    }

    #[test]
    fn persists_request_model_for_filter_but_projects_stable_global_model() {
        let mut seed = sample_seed();
        seed.persistence.original_request_body = json!({"model": "ep-requested-123"});
        seed.persistence.global_model_name = Some("Doubao-Seedance-2.0".to_string());
        seed.persistence.mapped_model = Some("doubao-seedance-2-0-260128".to_string());
        seed.model = Some("Doubao-Seedance-2.0".to_string());
        seed.observed_model = Some("doubao-seedance-2-0-260128-resolved".to_string());

        let record = seed.to_upsert_record();
        assert_eq!(record.model.as_deref(), Some("ep-requested-123"));

        let mut task = sample_stored_task(VideoTaskStatus::Completed);
        task.model = record.model;
        task.request_metadata = record.request_metadata;
        let response = map_doubao_stored_task_to_read_response(task);
        assert_eq!(response.body_json["model"], "Doubao-Seedance-2.0");
    }

    #[test]
    fn reconstruction_prefers_snapshot_fields_and_refreshes_transport() {
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Completed;
        seed.updated_at_unix_secs = Some(1_768_294_581);
        seed.last_frame_url = Some("https://tos.example.com/last.jpg".to_string());
        seed.completion_tokens = Some(295_800);
        seed.total_tokens = Some(295_900);
        seed.duration_seconds = None;
        seed.seed = Some(77);
        seed.frames = Some(121);
        seed.frames_per_second = Some(24);

        let mut task = sample_stored_task(VideoTaskStatus::Completed);
        task.request_metadata = Some(json!({
            "rust_local_snapshot": LocalVideoTaskSnapshot::Doubao(seed.clone())
        }));

        let mut refreshed_transport = seed.transport.clone();
        refreshed_transport.key_id = "replacement-key".to_string();
        let snapshot =
            LocalVideoTaskSnapshot::from_stored_task_with_transport(&task, refreshed_transport)
                .expect("snapshot");

        let LocalVideoTaskSnapshot::Doubao(reconstructed) = snapshot else {
            panic!("expected Doubao snapshot");
        };
        assert_eq!(
            reconstructed.last_frame_url.as_deref(),
            Some("https://tos.example.com/last.jpg")
        );
        assert_eq!(reconstructed.completion_tokens, Some(295_800));
        assert_eq!(reconstructed.total_tokens, Some(295_900));
        assert_eq!(reconstructed.seed, Some(77));
        assert_eq!(reconstructed.frames, Some(121));
        assert_eq!(reconstructed.frames_per_second, Some(24));
        assert_eq!(reconstructed.duration_seconds, None);
        assert_eq!(reconstructed.updated_at_unix_secs, Some(1_768_294_581));
        assert_eq!(reconstructed.transport.key_id, "replacement-key");
    }

    #[test]
    fn old_snapshot_without_new_optional_fields_still_deserializes() {
        let mut value = serde_json::to_value(LocalVideoTaskSnapshot::Doubao(sample_seed()))
            .expect("serialize snapshot");
        let seed = value["Doubao"].as_object_mut().expect("Doubao seed");
        for field in [
            "updated_at_unix_secs",
            "seed",
            "frames",
            "frames_per_second",
        ] {
            seed.remove(field);
        }

        let snapshot = serde_json::from_value::<LocalVideoTaskSnapshot>(value)
            .expect("old snapshot remains compatible");
        let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
            panic!("expected Doubao snapshot");
        };
        assert_eq!(seed.updated_at_unix_secs, None);
        assert_eq!(seed.seed, None);
        assert_eq!(seed.frames, None);
        assert_eq!(seed.frames_per_second, None);
    }

    #[test]
    fn old_snapshot_recovers_provider_fields_from_last_poll_after_restart() {
        let mut value = serde_json::to_value(LocalVideoTaskSnapshot::Doubao(sample_seed()))
            .expect("serialize snapshot");
        let snapshot_seed = value["Doubao"].as_object_mut().expect("Doubao seed");
        for field in ["seed", "frames", "frames_per_second"] {
            snapshot_seed.remove(field);
        }

        let mut task = sample_stored_task(VideoTaskStatus::Completed);
        task.request_metadata = Some(json!({
            "rust_local_snapshot": value,
            "poll_raw_response": {
                "seed": -1,
                "frames": 121,
                "framespersecond": 24,
                "duration": "11"
            }
        }));

        let restored = LocalVideoTaskSnapshot::from_stored_task(&task).expect("snapshot");
        let LocalVideoTaskSnapshot::Doubao(seed) = restored else {
            panic!("expected Doubao snapshot");
        };
        assert_eq!(seed.seed, Some(-1));
        assert_eq!(seed.frames, Some(121));
        assert_eq!(seed.frames_per_second, Some(24));
        assert_eq!(seed.duration_seconds, None);

        let response = map_doubao_stored_task_to_read_response(task);
        assert_eq!(response.body_json["seed"].as_i64(), Some(-1));
        assert_eq!(response.body_json["frames"].as_i64(), Some(121));
        assert_eq!(response.body_json["framespersecond"].as_i64(), Some(24));
        assert!(response.body_json.get("duration").is_none());
    }

    #[test]
    fn snapshotless_legacy_row_recovers_fields_from_request_and_poll_metadata() {
        let mut task = sample_stored_task(VideoTaskStatus::Completed);
        task.duration_seconds = Some(11);
        task.original_request_body = Some(json!({
            "model": "ep-requested-123",
            "content": [{"type": "text", "text": "a cat yawning"}],
            "seed": 7,
            "frames": 117,
            "framespersecond": 24
        }));
        task.request_metadata = Some(json!({
            "poll_raw_response": {
                "seed": 9,
                "frames": 121,
                "framespersecond": 25
            }
        }));

        let snapshot =
            LocalVideoTaskSnapshot::from_stored_task_with_transport(&task, sample_seed().transport)
                .expect("legacy row should reconstruct");
        let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
            panic!("expected Doubao snapshot");
        };
        assert_eq!(seed.seed, Some(9));
        assert_eq!(seed.frames, Some(121));
        assert_eq!(seed.frames_per_second, Some(25));
        assert_eq!(seed.duration_seconds, None);
    }

    #[test]
    fn create_poll_persist_restart_and_read_preserve_official_field_types() {
        let mut seed = sample_seed();
        seed.persistence.original_request_body = json!({
            "model": "ep-requested-123",
            "content": [{"type": "text", "text": "a cat yawning"}],
            "seed": 7,
            "frames": 117,
            "framespersecond": 24
        });
        seed.duration_seconds = None;
        seed.seed = Some(7);
        seed.frames = Some(117);
        seed.frames_per_second = Some(24);

        seed.apply_provider_body(
            json!({
                "status": "succeeded",
                "updated_at": "1768294581",
                "seed": 9,
                "frames": 121,
                "framespersecond": 25,
                "content": {"video_url": "https://tos.example.com/final.mp4"}
            })
            .as_object()
            .expect("poll body"),
        );

        let record = seed.to_upsert_record();
        assert_eq!(record.duration_seconds, None);
        let persisted =
            &record.request_metadata.as_ref().expect("metadata")["rust_local_snapshot"]["Doubao"];
        assert_eq!(persisted["seed"].as_i64(), Some(9));
        assert_eq!(persisted["frames"].as_i64(), Some(121));
        assert_eq!(persisted["frames_per_second"].as_i64(), Some(25));

        let mut stored = sample_stored_task(VideoTaskStatus::Completed);
        stored.model = record.model.clone();
        stored.original_request_body = record.original_request_body.clone();
        stored.duration_seconds = record.duration_seconds;
        stored.resolution = record.resolution.clone();
        stored.aspect_ratio = record.aspect_ratio.clone();
        stored.updated_at_unix_secs = record.updated_at_unix_secs;
        stored.video_url = record.video_url.clone();
        stored.request_metadata = record.request_metadata.clone();

        let restored = LocalVideoTaskSnapshot::from_stored_task(&stored).expect("restart snapshot");
        let LocalVideoTaskSnapshot::Doubao(restored_seed) = &restored else {
            panic!("expected Doubao snapshot");
        };
        assert_eq!(restored_seed.seed, Some(9));
        assert_eq!(restored_seed.frames, Some(121));
        assert_eq!(restored_seed.frames_per_second, Some(25));
        assert_eq!(restored_seed.duration_seconds, None);
        let immediate_body = restored.read_response().body_json;
        assert_eq!(immediate_body["seed"].as_i64(), Some(9));
        assert_eq!(immediate_body["frames"].as_i64(), Some(121));
        assert_eq!(immediate_body["framespersecond"].as_i64(), Some(25));
        assert!(immediate_body.get("duration").is_none());

        let database_body = map_doubao_stored_task_to_read_response(stored).body_json;
        assert_eq!(database_body["seed"].as_i64(), Some(9));
        assert_eq!(database_body["frames"].as_i64(), Some(121));
        assert_eq!(database_body["framespersecond"].as_i64(), Some(25));
        assert!(database_body.get("duration").is_none());
    }

    #[test]
    fn stored_terminal_status_overrides_a_stale_snapshot_status() {
        let mut task = sample_stored_task(VideoTaskStatus::Deleted);
        let mut snapshot = sample_seed();
        snapshot.status = LocalVideoTaskStatus::Completed;
        task.request_metadata = Some(json!({
            "rust_local_snapshot": LocalVideoTaskSnapshot::Doubao(snapshot)
        }));

        let restored = LocalVideoTaskSnapshot::from_stored_task(&task).expect("snapshot");
        assert!(
            matches!(&restored, LocalVideoTaskSnapshot::Doubao(seed) if seed.status == LocalVideoTaskStatus::Deleted)
        );
        assert_eq!(restored.read_response().status_code, 404);
    }

    #[test]
    fn upsert_record_satisfies_the_short_id_column_constraint() {
        // `video_tasks.short_id` is NOT NULL and capped at 16 characters; a
        // missing or oversized value fails the insert at runtime only.
        let seed = DoubaoVideoTaskSeed {
            local_task_id: format!("cgt-{}", "a".repeat(32)),
            ..sample_seed()
        };

        let short_id = seed.to_upsert_record().short_id.expect("short id required");

        assert!(!short_id.is_empty());
        assert!(short_id.len() <= 16, "short_id must fit the column");
    }

    #[test]
    fn maps_deleted_stored_task_to_not_found() {
        let response =
            map_doubao_stored_task_to_read_response(sample_stored_task(VideoTaskStatus::Deleted));

        assert_eq!(response.status_code, 404);
        assert_eq!(response.body_json["error"]["code"], "NotFound");
    }

    #[test]
    fn maps_cancelled_stored_task_to_visible_cancelled_state() {
        let now_unix_secs = 2_000_000_000;
        let mut task = sample_stored_task(VideoTaskStatus::Cancelled);
        task.completed_at_unix_secs = Some(now_unix_secs - 60);
        task.updated_at_unix_secs = now_unix_secs - 60;
        let response = map_doubao_stored_task_to_read_response_at(task, now_unix_secs);

        assert_eq!(response.status_code, 200);
        assert_eq!(response.body_json["status"], "cancelled");
        assert!(response.body_json.get("content").is_none());
    }

    #[test]
    fn hides_cancelled_stored_task_after_retention_window() {
        let now_unix_secs = 2_000_000_000;
        let mut task = sample_stored_task(VideoTaskStatus::Cancelled);
        task.completed_at_unix_secs = Some(now_unix_secs - DOUBAO_CANCELLED_TASK_RETENTION_SECONDS);
        task.updated_at_unix_secs = now_unix_secs - DOUBAO_CANCELLED_TASK_RETENTION_SECONDS;

        let response = map_doubao_stored_task_to_read_response_at(task, now_unix_secs);

        assert_eq!(response.status_code, 404);
        assert_eq!(response.body_json["error"]["code"], "NotFound");
    }

    #[test]
    fn cancelled_retention_fallback_treats_created_at_column_as_unix_seconds() {
        let now_unix_secs = 2_000_000_000;
        let mut task = sample_stored_task(VideoTaskStatus::Cancelled);
        task.completed_at_unix_secs = None;
        task.updated_at_unix_secs = 0;
        task.submitted_at_unix_secs = None;
        task.created_at_unix_ms = now_unix_secs - 60;

        let response = map_doubao_stored_task_to_read_response_at(task, now_unix_secs);

        assert_eq!(response.status_code, 200);
        assert_eq!(response.body_json["status"], "cancelled");
    }

    #[test]
    fn only_completed_stored_tasks_expose_content_urls() {
        for status in [
            VideoTaskStatus::Pending,
            VideoTaskStatus::Submitted,
            VideoTaskStatus::Queued,
            VideoTaskStatus::Processing,
            VideoTaskStatus::Failed,
            VideoTaskStatus::Expired,
        ] {
            let response = map_doubao_stored_task_to_read_response_at(
                sample_stored_task(status),
                1_768_294_600,
            );
            assert_eq!(
                response.status_code, 200,
                "unexpected status for {status:?}"
            );
            assert!(
                response.body_json.get("content").is_none(),
                "{status:?} leaked a signed content URL"
            );
        }
    }

    #[test]
    fn snapshot_cancelled_retention_matches_stored_task_and_hides_content() {
        let now_unix_secs = 2_000_000_000;
        let mut seed = sample_seed();
        seed.status = LocalVideoTaskStatus::Cancelled;
        seed.video_url = Some("https://tos.example.com/video.mp4?X-Sig=abc".to_string());
        seed.last_frame_url = Some("https://tos.example.com/frame.jpg?X-Sig=abc".to_string());
        seed.completed_at_unix_secs = Some(now_unix_secs - 60);
        seed.updated_at_unix_secs = Some(now_unix_secs - 60);
        let snapshot = LocalVideoTaskSnapshot::Doubao(seed.clone());

        let fresh = snapshot.read_response_at(now_unix_secs);
        assert_eq!(fresh.status_code, 200);
        assert_eq!(fresh.body_json["status"], "cancelled");
        assert!(fresh.body_json.get("content").is_none());

        seed.completed_at_unix_secs = Some(now_unix_secs - DOUBAO_CANCELLED_TASK_RETENTION_SECONDS);
        seed.updated_at_unix_secs = Some(now_unix_secs - DOUBAO_CANCELLED_TASK_RETENTION_SECONDS);
        let expired = LocalVideoTaskSnapshot::Doubao(seed).read_response_at(now_unix_secs);
        assert_eq!(expired.status_code, 404);
        assert_eq!(expired.body_json["error"]["code"], "NotFound");
    }

    #[test]
    fn maps_completed_stored_task_to_doubao_body() {
        let response =
            map_doubao_stored_task_to_read_response(sample_stored_task(VideoTaskStatus::Completed));

        assert_eq!(response.status_code, 200);
        assert_eq!(response.body_json["id"], "cgt-local-123");
        assert_eq!(response.body_json["status"], "succeeded");
        assert_eq!(response.body_json["duration"].as_str(), Some("11"));
        assert_eq!(
            response.body_json["content"]["video_url"],
            "https://tos.example.com/video.mp4?X-Sig=abc"
        );
    }
}
