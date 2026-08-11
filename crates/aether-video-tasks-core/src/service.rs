use std::path::PathBuf;
use std::sync::Arc;

use aether_contracts::ExecutionPlan;
use aether_data_contracts::repository::video_tasks::StoredVideoTask;
use serde_json::{Map, Value};

use crate::{
    extract_doubao_task_id_from_content_path, extract_doubao_task_id_from_path,
    extract_gemini_short_id_from_cancel_path, extract_gemini_short_id_from_path,
    extract_openai_task_id_from_cancel_path, extract_openai_task_id_from_content_path,
    extract_openai_task_id_from_path, extract_openai_task_id_from_remix_path,
    resolve_local_video_registry_mutation, FileVideoTaskStore, InMemoryVideoTaskStore,
    LocalVideoTaskContentAction, LocalVideoTaskFollowUpPlan, LocalVideoTaskProjectionTarget,
    LocalVideoTaskReadRefreshPlan, LocalVideoTaskReadResponse, LocalVideoTaskSnapshot,
    LocalVideoTaskSuccessPlan, VideoTaskStore, VideoTaskTruthSourceMode,
};

#[derive(Debug)]
pub struct VideoTaskService {
    truth_source_mode: VideoTaskTruthSourceMode,
    store: Arc<dyn VideoTaskStore>,
}

#[cfg(test)]
mod tests {
    use aether_contracts::EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER;
    use serde_json::json;

    use super::VideoTaskService;
    use crate::{
        DoubaoVideoTaskSeed, LocalVideoTaskContentAction, LocalVideoTaskPersistence,
        LocalVideoTaskProjectionTarget, LocalVideoTaskSnapshot, LocalVideoTaskStatus,
        LocalVideoTaskTransport, VideoTaskTruthSourceMode,
    };

    fn cross_format_seed(status: LocalVideoTaskStatus) -> DoubaoVideoTaskSeed {
        DoubaoVideoTaskSeed {
            local_task_id: "task-cross-1".to_string(),
            upstream_task_id: "cgt-upstream-1".to_string(),
            created_at_unix_secs: 1_768_294_532,
            updated_at_unix_secs: None,
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            model: Some("doubao-seedance-resolved".to_string()),
            prompt: Some("a cat yawning".to_string()),
            resolution: Some("720p".to_string()),
            ratio: Some("16:9".to_string()),
            duration_seconds: Some(5),
            seed: None,
            frames: None,
            frames_per_second: None,
            status,
            progress_percent: if status == LocalVideoTaskStatus::Processing {
                50
            } else {
                0
            },
            completed_at_unix_secs: None,
            error_code: None,
            error_message: None,
            video_url: None,
            last_frame_url: None,
            completion_tokens: None,
            total_tokens: None,
            persistence: LocalVideoTaskPersistence {
                request_id: "req-cross-1".to_string(),
                username: None,
                api_key_name: None,
                client_api_format: "openai:video".to_string(),
                provider_api_format: "doubao:video".to_string(),
                original_request_body: json!({
                    "model": "sora-compatible-alias",
                    "prompt": "a cat yawning",
                    "size": "1280x720",
                    "seconds": "5"
                }),
                format_converted: true,
            },
            transport: LocalVideoTaskTransport {
                upstream_base_url: "https://ark.example.com/api".to_string(),
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
                model_name: Some("doubao-seedance-resolved".to_string()),
                proxy: None,
                transport_profile: None,
                timeouts: None,
            },
        }
    }

    #[test]
    fn openai_route_drives_doubao_refresh_read_content_and_delete() {
        let service = VideoTaskService::new(VideoTaskTruthSourceMode::RustAuthoritative);
        service.record_snapshot(LocalVideoTaskSnapshot::Doubao(cross_format_seed(
            LocalVideoTaskStatus::Processing,
        )));

        let response = service
            .read_response_for_user(Some("openai"), "/v1/videos/task-cross-1", "user-1")
            .expect("OpenAI client route should own the converted task");
        assert_eq!(response.body_json["object"], "video");
        assert_eq!(response.body_json["status"], "processing");
        assert!(service
            .read_response_for_user(Some("openai"), "/v1/videos/task-cross-1", "user-2")
            .is_none());
        assert!(service
            .read_response(
                Some("doubao"),
                "/v3/contents/generations/tasks/task-cross-1"
            )
            .is_none());

        let refresh = service
            .prepare_read_refresh_sync_plan_for_user(
                Some("openai"),
                "/v1/videos/task-cross-1",
                "user-1",
                "trace-refresh",
            )
            .expect("converted task should refresh through Ark");
        assert_eq!(refresh.plan.method, "GET");
        assert_eq!(
            refresh.plan.url,
            "https://ark.example.com/api/v3/contents/generations/tasks/cgt-upstream-1"
        );
        assert_eq!(refresh.plan.client_api_format, "openai:video");
        assert_eq!(refresh.plan.provider_api_format, "doubao:video");
        assert_eq!(
            refresh
                .plan
                .headers
                .get("authorization")
                .map(String::as_str),
            Some("Bearer provider-sk")
        );
        assert!(matches!(
            &refresh.projection_target,
            LocalVideoTaskProjectionTarget::OpenAi { .. }
        ));

        assert!(service.apply_read_refresh_projection(
            &refresh,
            json!({
                "status": "succeeded",
                "updated_at": "1768294581",
                "content": {
                    "video_url": "https://cdn.example.com/video.mp4?sig=1",
                    "last_frame_url": "https://cdn.example.com/frame.jpg?sig=1"
                }
            })
            .as_object()
            .expect("object")
        ));
        let completed = service
            .read_response_for_user(Some("openai"), "/v1/videos/task-cross-1", "user-1")
            .expect("projected task should remain readable");
        assert_eq!(completed.body_json["status"], "completed");
        assert_eq!(completed.body_json["model"], "sora-compatible-alias");

        let action = service
            .prepare_openai_content_stream_action_for_user(
                "/v1/videos/task-cross-1/content",
                None,
                "trace-content",
                "user-1",
            )
            .expect("completed asset should stream");
        let LocalVideoTaskContentAction::StreamPlan(plan) = action else {
            panic!("expected a stream plan");
        };
        assert_eq!(plan.url, "https://cdn.example.com/video.mp4?sig=1");
        assert!(!plan.headers.contains_key("authorization"));
        assert_eq!(
            plan.headers
                .get(EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER)
                .map(String::as_str),
            Some("1")
        );
        assert_eq!(plan.client_api_format, "openai:video");
        assert_eq!(plan.provider_api_format, "doubao:video");

        let delete = service
            .prepare_follow_up_sync_plan_for_user(
                "openai_video_delete_sync",
                "/v1/videos/task-cross-1",
                None,
                Some("user-1"),
                Some("api-key-1"),
                "trace-delete",
            )
            .expect("OpenAI delete should map to Ark DELETE");
        assert_eq!(delete.plan.method, "DELETE");
        assert_eq!(delete.plan.provider_api_format, "doubao:video");
        assert_eq!(delete.plan.client_api_format, "openai:video");
        assert_eq!(
            delete.report_kind.as_deref(),
            Some("openai_video_delete_sync_finalize")
        );

        service.apply_finalize_mutation(
            "/v1/videos/task-cross-1",
            "openai_video_delete_sync_finalize",
        );
        assert_eq!(
            service
                .read_response(Some("openai"), "/v1/videos/task-cross-1")
                .expect("deleted task response")
                .status_code,
            404
        );
    }

    #[test]
    fn converted_openai_cancel_requires_owner_and_rehydrated_provider_auth() {
        let service = VideoTaskService::new(VideoTaskTruthSourceMode::RustAuthoritative);
        let seed = cross_format_seed(LocalVideoTaskStatus::Queued);
        service.record_snapshot(LocalVideoTaskSnapshot::Doubao(seed.clone()));

        assert!(service
            .prepare_follow_up_sync_plan_for_user(
                "openai_video_cancel_sync",
                "/v1/videos/task-cross-1/cancel",
                None,
                Some("user-2"),
                Some("api-key-2"),
                "trace-foreign",
            )
            .is_none());

        let cancel = service
            .prepare_follow_up_sync_plan_for_user(
                "openai_video_cancel_sync",
                "/v1/videos/task-cross-1/cancel",
                None,
                Some("user-1"),
                Some("api-key-1"),
                "trace-cancel",
            )
            .expect("owner should cancel through Ark");
        assert_eq!(cancel.plan.method, "DELETE");
        assert_eq!(cancel.plan.provider_api_format, "doubao:video");
        assert_eq!(
            cancel.report_kind.as_deref(),
            Some("openai_video_cancel_sync_finalize")
        );

        service.apply_finalize_mutation(
            "/v1/videos/task-cross-1/cancel",
            "openai_video_cancel_sync_finalize",
        );
        assert!(service
            .snapshot_for_route(Some("openai"), "/v1/videos/task-cross-1/cancel")
            .is_some());
        assert_eq!(
            service
                .read_response(Some("openai"), "/v1/videos/task-cross-1")
                .expect("cancelled task response")
                .status_code,
            404
        );

        let redacted = LocalVideoTaskSnapshot::Doubao(seed).redacted_for_persistence();
        let redacted_service = VideoTaskService::new(VideoTaskTruthSourceMode::RustAuthoritative);
        redacted_service.record_snapshot(redacted);
        assert!(redacted_service
            .prepare_read_refresh_sync_plan_for_user(
                Some("openai"),
                "/v1/videos/task-cross-1",
                "user-1",
                "trace-no-auth",
            )
            .is_none());
        assert!(redacted_service
            .prepare_follow_up_sync_plan_for_user(
                "openai_video_cancel_sync",
                "/v1/videos/task-cross-1/cancel",
                None,
                Some("user-1"),
                Some("api-key-1"),
                "trace-no-auth-cancel",
            )
            .is_none());
    }

    #[test]
    fn completed_asset_fails_closed_when_runtime_transport_cannot_be_reconstructed() {
        let mut seed = cross_format_seed(LocalVideoTaskStatus::Completed);
        seed.progress_percent = 100;
        seed.completed_at_unix_secs = Some(1_768_294_581);
        seed.video_url = Some("https://cdn.example.com/video.mp4?sig=1".to_string());
        seed.transport.proxy = Some(aether_contracts::ProxySnapshot {
            enabled: Some(true),
            url: Some("http://egress-proxy.example:8080".to_string()),
            ..aether_contracts::ProxySnapshot::default()
        });

        let redacted = LocalVideoTaskSnapshot::Doubao(seed).redacted_for_persistence();
        assert!(!redacted.has_runtime_auth_headers());
        let LocalVideoTaskSnapshot::Doubao(redacted_seed) = &redacted else {
            panic!("expected Doubao snapshot");
        };
        assert!(redacted_seed.transport.proxy.is_none());

        let service = VideoTaskService::new(VideoTaskTruthSourceMode::RustAuthoritative);
        service.record_snapshot(redacted);
        let action = service
            .prepare_openai_content_stream_action_for_user(
                "/v1/videos/task-cross-1/content",
                None,
                "trace-content-no-transport",
                "user-1",
            )
            .expect("owned completed task should return an explicit transport error");
        let LocalVideoTaskContentAction::Immediate {
            status_code,
            body_json,
        } = action
        else {
            panic!("redacted transport must not produce a direct stream plan");
        };
        assert_eq!(status_code, 503);
        assert_eq!(body_json["detail"], "Video asset transport is unavailable");
    }

    #[test]
    fn native_doubao_delete_cancels_active_tasks_but_deletes_terminal_tasks() {
        let service = VideoTaskService::new(VideoTaskTruthSourceMode::RustAuthoritative);
        let mut active = cross_format_seed(LocalVideoTaskStatus::Processing);
        active.local_task_id = "cgt-active".to_string();
        active.persistence.client_api_format = "doubao:video".to_string();
        active.persistence.format_converted = false;
        service.record_snapshot(LocalVideoTaskSnapshot::Doubao(active));

        let cancelled_after_unix_secs = crate::current_unix_timestamp_secs();
        service.apply_finalize_mutation(
            "/v3/contents/generations/tasks/cgt-active",
            "doubao_video_delete_sync_finalize",
        );
        let LocalVideoTaskSnapshot::Doubao(cancelled_snapshot) = service
            .snapshot_for_route(Some("doubao"), "/v3/contents/generations/tasks/cgt-active")
            .expect("cancelled snapshot")
        else {
            panic!("expected Doubao snapshot");
        };
        let cancellation_time = cancelled_snapshot
            .completed_at_unix_secs
            .expect("cancellation time");
        assert!(cancellation_time >= cancelled_after_unix_secs);
        assert_eq!(
            cancelled_snapshot.updated_at_unix_secs,
            Some(cancellation_time)
        );
        let cancelled = service
            .read_response(Some("doubao"), "/v3/contents/generations/tasks/cgt-active")
            .expect("cancelled Ark task should remain queryable");
        assert_eq!(cancelled.status_code, 200);
        assert_eq!(cancelled.body_json["status"], "cancelled");

        let mut terminal = cross_format_seed(LocalVideoTaskStatus::Completed);
        terminal.local_task_id = "cgt-terminal".to_string();
        terminal.persistence.client_api_format = "doubao:video".to_string();
        terminal.persistence.format_converted = false;
        service.record_snapshot(LocalVideoTaskSnapshot::Doubao(terminal));
        service.apply_finalize_mutation(
            "/v3/contents/generations/tasks/cgt-terminal",
            "doubao_video_delete_sync_finalize",
        );
        assert_eq!(
            service
                .read_response(
                    Some("doubao"),
                    "/v3/contents/generations/tasks/cgt-terminal",
                )
                .expect("deleted Ark task response")
                .status_code,
            404
        );
    }
}

impl VideoTaskService {
    pub fn new(mode: VideoTaskTruthSourceMode) -> Self {
        Self::with_store(mode, Arc::new(InMemoryVideoTaskStore::default()))
    }

    pub fn with_file_store(
        mode: VideoTaskTruthSourceMode,
        path: impl Into<PathBuf>,
    ) -> std::io::Result<Self> {
        Ok(Self::with_store(
            mode,
            Arc::new(FileVideoTaskStore::new(path)?),
        ))
    }

    fn with_store(mode: VideoTaskTruthSourceMode, store: Arc<dyn VideoTaskStore>) -> Self {
        Self {
            truth_source_mode: mode,
            store,
        }
    }

    pub fn with_truth_source_mode(&self, mode: VideoTaskTruthSourceMode) -> Self {
        Self {
            truth_source_mode: mode,
            store: self.store.clone(),
        }
    }

    pub fn is_rust_authoritative(&self) -> bool {
        self.truth_source_mode == VideoTaskTruthSourceMode::RustAuthoritative
    }

    pub fn truth_source_mode(&self) -> VideoTaskTruthSourceMode {
        self.truth_source_mode
    }

    pub fn prepare_sync_success(
        &self,
        report_kind: &str,
        provider_body: &Map<String, Value>,
        report_context: &Map<String, Value>,
        plan: &ExecutionPlan,
    ) -> Option<LocalVideoTaskSuccessPlan> {
        self.truth_source_mode.prepare_sync_success(
            report_kind,
            provider_body,
            report_context,
            plan,
        )
    }

    pub fn record_snapshot(&self, snapshot: LocalVideoTaskSnapshot) {
        self.store.insert(snapshot);
    }

    pub fn hydrate_from_stored_task(&self, task: &StoredVideoTask) -> bool {
        let Some(snapshot) = LocalVideoTaskSnapshot::from_stored_task(task) else {
            return false;
        };
        // Persisted snapshots are never an authority for live credentials.
        // Redact even legacy rows before they enter the runtime registry so a
        // deleted/revoked provider key cannot be reused after hydration.
        self.store.insert(snapshot.redacted_for_persistence());
        true
    }

    pub fn apply_finalize_mutation(&self, request_path: &str, report_kind: &str) {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return;
        }
        let mutation = if report_kind == "doubao_video_delete_sync_finalize" {
            let Some(task_id) = extract_doubao_task_id_from_path(request_path) else {
                return;
            };
            let is_active = self
                .store
                .clone_doubao_snapshot(task_id)
                .is_some_and(|snapshot| match snapshot {
                    LocalVideoTaskSnapshot::Doubao(seed) => matches!(
                        seed.status,
                        crate::LocalVideoTaskStatus::Submitted
                            | crate::LocalVideoTaskStatus::Queued
                            | crate::LocalVideoTaskStatus::Processing
                    ),
                    _ => false,
                });
            if is_active {
                crate::LocalVideoTaskRegistryMutation::DoubaoCancelled {
                    task_id: task_id.to_string(),
                }
            } else {
                crate::LocalVideoTaskRegistryMutation::DoubaoDeleted {
                    task_id: task_id.to_string(),
                }
            }
        } else {
            let Some(mutation) = resolve_local_video_registry_mutation(
                self.truth_source_mode,
                request_path,
                report_kind,
            ) else {
                return;
            };
            mutation
        };
        self.store.apply_mutation(mutation);
    }

    pub fn read_response(
        &self,
        route_family: Option<&str>,
        request_path: &str,
    ) -> Option<LocalVideoTaskReadResponse> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        match route_family {
            Some("openai") => extract_openai_task_id_from_path(request_path)
                .and_then(|task_id| self.store.read_openai(task_id)),
            Some("gemini") => extract_gemini_short_id_from_path(request_path)
                .and_then(|short_id| self.store.read_gemini(short_id)),
            Some("doubao") => extract_doubao_task_id_from_path(request_path)
                .and_then(|task_id| self.store.read_doubao(task_id)),
            _ => None,
        }
    }

    pub fn read_response_for_user(
        &self,
        route_family: Option<&str>,
        request_path: &str,
        user_id: &str,
    ) -> Option<LocalVideoTaskReadResponse> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = self.snapshot_for_route(route_family, request_path)?;
        if !snapshot.belongs_to_user(user_id) {
            return None;
        }
        Some(snapshot.read_response())
    }

    pub fn snapshot_for_route(
        &self,
        route_family: Option<&str>,
        request_path: &str,
    ) -> Option<LocalVideoTaskSnapshot> {
        match route_family {
            Some("openai") => extract_openai_task_id_from_path(request_path)
                .or_else(|| extract_openai_task_id_from_cancel_path(request_path))
                .or_else(|| extract_openai_task_id_from_remix_path(request_path))
                .or_else(|| extract_openai_task_id_from_content_path(request_path))
                .and_then(|task_id| self.store.clone_openai_snapshot(task_id)),
            Some("gemini") => extract_gemini_short_id_from_path(request_path)
                .or_else(|| extract_gemini_short_id_from_cancel_path(request_path))
                .and_then(|short_id| self.store.clone_gemini_snapshot(short_id)),
            Some("doubao") => extract_doubao_task_id_from_path(request_path)
                .or_else(|| extract_doubao_task_id_from_content_path(request_path))
                .and_then(|task_id| self.store.clone_doubao_snapshot(task_id)),
            _ => None,
        }
    }

    pub fn prepare_openai_content_stream_action(
        &self,
        request_path: &str,
        query_string: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let task_id = extract_openai_task_id_from_content_path(request_path)?;
        match self.store.clone_openai_snapshot(task_id)? {
            LocalVideoTaskSnapshot::OpenAi(seed) => {
                seed.build_content_stream_action(query_string, trace_id)
            }
            LocalVideoTaskSnapshot::Doubao(seed) => {
                seed.build_openai_content_stream_action(query_string, trace_id)
            }
            LocalVideoTaskSnapshot::Gemini(_) => None,
        }
    }

    pub fn prepare_openai_content_stream_action_for_user(
        &self,
        request_path: &str,
        query_string: Option<&str>,
        trace_id: &str,
        user_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let task_id = extract_openai_task_id_from_content_path(request_path)?;
        let snapshot = self.store.clone_openai_snapshot(task_id)?;
        if !snapshot.belongs_to_user(user_id) {
            return None;
        }
        let has_runtime_transport = snapshot.has_runtime_auth_headers();
        let action = match snapshot {
            LocalVideoTaskSnapshot::OpenAi(seed) => {
                seed.build_content_stream_action(query_string, trace_id)
            }
            LocalVideoTaskSnapshot::Doubao(seed) => {
                seed.build_openai_content_stream_action(query_string, trace_id)
            }
            LocalVideoTaskSnapshot::Gemini(_) => None,
        };
        if !has_runtime_transport
            && matches!(
                action.as_ref(),
                Some(LocalVideoTaskContentAction::StreamPlan(_))
            )
        {
            // Persisted snapshots deliberately drop proxy/profile credentials.
            // If the current provider/key can no longer be reconstructed, a
            // completed pre-signed asset must not silently downgrade to direct
            // egress using the redacted transport.
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 503,
                body_json: serde_json::json!({
                    "detail": "Video asset transport is unavailable"
                }),
            });
        }
        action
    }

    pub fn prepare_doubao_content_stream_action(
        &self,
        request_path: &str,
        query_string: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let task_id = extract_doubao_task_id_from_content_path(request_path)?;
        let LocalVideoTaskSnapshot::Doubao(seed) = self.store.clone_doubao_snapshot(task_id)?
        else {
            return None;
        };
        seed.build_content_stream_action(query_string, trace_id)
    }

    pub fn prepare_doubao_content_stream_action_for_user(
        &self,
        request_path: &str,
        query_string: Option<&str>,
        trace_id: &str,
        user_id: &str,
    ) -> Option<LocalVideoTaskContentAction> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let task_id = extract_doubao_task_id_from_content_path(request_path)?;
        let snapshot = self.store.clone_doubao_snapshot(task_id)?;
        if !snapshot.belongs_to_user(user_id) {
            return None;
        }
        let has_runtime_transport = snapshot.has_runtime_auth_headers();
        let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
            return None;
        };
        let action = seed.build_content_stream_action(query_string, trace_id);
        if !has_runtime_transport
            && matches!(
                action.as_ref(),
                Some(LocalVideoTaskContentAction::StreamPlan(_))
            )
        {
            return Some(LocalVideoTaskContentAction::Immediate {
                status_code: 503,
                body_json: serde_json::json!({
                    "error": {
                        "code": "InternalServiceError",
                        "message": "Video asset transport is unavailable"
                    }
                }),
            });
        }
        action
    }

    pub fn snapshot_for_refresh_plan(
        &self,
        refresh_plan: &LocalVideoTaskReadRefreshPlan,
    ) -> Option<LocalVideoTaskSnapshot> {
        match &refresh_plan.projection_target {
            LocalVideoTaskProjectionTarget::OpenAi { task_id } => {
                self.store.clone_openai_snapshot(task_id)
            }
            LocalVideoTaskProjectionTarget::Gemini { short_id } => {
                self.store.clone_gemini_snapshot(short_id)
            }
            LocalVideoTaskProjectionTarget::Doubao { task_id } => {
                self.store.clone_doubao_snapshot(task_id)
            }
        }
    }

    pub fn project_openai_task_response(
        &self,
        task_id: &str,
        provider_body: &Map<String, Value>,
    ) -> bool {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return false;
        }
        self.store.project_openai(task_id, provider_body)
    }

    pub fn project_gemini_task_response(
        &self,
        short_id: &str,
        provider_body: &Map<String, Value>,
    ) -> bool {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return false;
        }
        self.store.project_gemini(short_id, provider_body)
    }

    pub fn project_doubao_task_response(
        &self,
        task_id: &str,
        provider_body: &Map<String, Value>,
    ) -> bool {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return false;
        }
        self.store.project_doubao(task_id, provider_body)
    }

    pub fn prepare_read_refresh_sync_plan(
        &self,
        route_family: Option<&str>,
        request_path: &str,
        trace_id: &str,
    ) -> Option<LocalVideoTaskReadRefreshPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = self.snapshot_for_route(route_family, request_path)?;
        if !snapshot.has_runtime_auth_headers() {
            return None;
        }
        Self::build_read_refresh_plan_for_snapshot(&snapshot, route_family, request_path, trace_id)
    }

    fn build_read_refresh_plan_for_snapshot(
        snapshot: &LocalVideoTaskSnapshot,
        route_family: Option<&str>,
        request_path: &str,
        trace_id: &str,
    ) -> Option<LocalVideoTaskReadRefreshPlan> {
        let plan = match &snapshot {
            LocalVideoTaskSnapshot::OpenAi(seed) => seed.build_get_follow_up_plan(trace_id)?,
            LocalVideoTaskSnapshot::Gemini(seed) => seed.build_get_follow_up_plan(trace_id)?,
            LocalVideoTaskSnapshot::Doubao(seed) => seed.build_get_follow_up_plan(trace_id)?,
        };
        let projection_target = match route_family {
            Some("openai") => LocalVideoTaskProjectionTarget::OpenAi {
                task_id: extract_openai_task_id_from_path(request_path)?.to_string(),
            },
            Some("gemini") => LocalVideoTaskProjectionTarget::Gemini {
                short_id: extract_gemini_short_id_from_path(request_path)?.to_string(),
            },
            Some("doubao") => LocalVideoTaskProjectionTarget::Doubao {
                task_id: extract_doubao_task_id_from_path(request_path)?.to_string(),
            },
            _ => return None,
        };
        Some(LocalVideoTaskReadRefreshPlan {
            plan,
            projection_target,
        })
    }

    pub fn prepare_read_refresh_sync_plan_for_user(
        &self,
        route_family: Option<&str>,
        request_path: &str,
        user_id: &str,
        trace_id: &str,
    ) -> Option<LocalVideoTaskReadRefreshPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = self.snapshot_for_route(route_family, request_path)?;
        if !snapshot.belongs_to_user(user_id) {
            return None;
        }
        // A persisted snapshot is deliberately credential-redacted.  Do not
        // turn it into an upstream refresh plan unless the provider transport
        // was reconstructed for this request; otherwise a missing provider
        // configuration would result in an unauthenticated GET.
        if !snapshot.has_runtime_auth_headers() {
            return None;
        }
        Self::build_read_refresh_plan_for_snapshot(&snapshot, route_family, request_path, trace_id)
    }

    pub fn prepare_poll_refresh_batch(
        &self,
        limit: usize,
        trace_prefix: &str,
    ) -> Vec<LocalVideoTaskReadRefreshPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative || limit == 0 {
            return Vec::new();
        }

        self.store
            .list_active_snapshots(limit)
            .into_iter()
            .enumerate()
            .filter_map(|(index, snapshot)| {
                let trace_id = format!("{trace_prefix}-{index}");
                self.prepare_poll_refresh_plan_for_snapshot(&snapshot, &trace_id)
            })
            .collect()
    }

    pub fn prepare_poll_refresh_plan_for_stored_task(
        &self,
        task: &StoredVideoTask,
        trace_id: &str,
    ) -> Option<LocalVideoTaskReadRefreshPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = LocalVideoTaskSnapshot::from_stored_task(task)?.redacted_for_persistence();
        self.prepare_poll_refresh_plan_for_snapshot(&snapshot, trace_id)
    }

    pub fn prepare_poll_refresh_plan_for_snapshot(
        &self,
        snapshot: &LocalVideoTaskSnapshot,
        trace_id: &str,
    ) -> Option<LocalVideoTaskReadRefreshPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        if !snapshot.has_runtime_auth_headers() {
            return None;
        }

        let plan = match snapshot {
            LocalVideoTaskSnapshot::OpenAi(seed) => seed.build_get_follow_up_plan(trace_id)?,
            LocalVideoTaskSnapshot::Gemini(seed) => seed.build_get_follow_up_plan(trace_id)?,
            LocalVideoTaskSnapshot::Doubao(seed) => seed.build_get_follow_up_plan(trace_id)?,
        };
        let projection_target = match snapshot.client_api_format().trim() {
            "openai:video" => {
                let task_id = match snapshot {
                    LocalVideoTaskSnapshot::OpenAi(seed) => seed.local_task_id.clone(),
                    LocalVideoTaskSnapshot::Doubao(seed) => seed.local_task_id.clone(),
                    LocalVideoTaskSnapshot::Gemini(_) => return None,
                };
                LocalVideoTaskProjectionTarget::OpenAi { task_id }
            }
            "gemini:video" => {
                let LocalVideoTaskSnapshot::Gemini(seed) = snapshot else {
                    return None;
                };
                LocalVideoTaskProjectionTarget::Gemini {
                    short_id: seed.local_short_id.clone(),
                }
            }
            "doubao:video" => {
                let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
                    return None;
                };
                LocalVideoTaskProjectionTarget::Doubao {
                    task_id: seed.local_task_id.clone(),
                }
            }
            _ => return None,
        };
        Some(LocalVideoTaskReadRefreshPlan {
            plan,
            projection_target,
        })
    }

    pub fn apply_read_refresh_projection(
        &self,
        refresh_plan: &LocalVideoTaskReadRefreshPlan,
        provider_body: &Map<String, Value>,
    ) -> bool {
        match &refresh_plan.projection_target {
            LocalVideoTaskProjectionTarget::OpenAi { task_id } => {
                self.project_openai_task_response(task_id, provider_body)
            }
            LocalVideoTaskProjectionTarget::Gemini { short_id } => {
                self.project_gemini_task_response(short_id, provider_body)
            }
            LocalVideoTaskProjectionTarget::Doubao { task_id } => {
                self.project_doubao_task_response(task_id, provider_body)
            }
        }
    }

    pub fn prepare_follow_up_sync_plan(
        &self,
        plan_kind: &str,
        request_path: &str,
        body_json: Option<&Value>,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = self.snapshot_for_follow_up_route(plan_kind, request_path)?;
        if !snapshot.has_runtime_auth_headers() {
            return None;
        }
        Self::build_follow_up_plan_from_snapshot(
            snapshot,
            plan_kind,
            body_json,
            fallback_user_id,
            fallback_api_key_id,
            trace_id,
        )
    }

    fn build_follow_up_plan_from_snapshot(
        snapshot: LocalVideoTaskSnapshot,
        plan_kind: &str,
        body_json: Option<&Value>,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        match (plan_kind, snapshot) {
            ("openai_video_remix_sync", LocalVideoTaskSnapshot::OpenAi(seed)) => seed
                .build_remix_follow_up_plan(
                    body_json?,
                    fallback_user_id,
                    fallback_api_key_id,
                    trace_id,
                ),
            ("openai_video_delete_sync", LocalVideoTaskSnapshot::OpenAi(seed)) => {
                seed.build_delete_follow_up_plan(fallback_user_id, fallback_api_key_id, trace_id)
            }
            ("openai_video_delete_sync", LocalVideoTaskSnapshot::Doubao(seed)) => seed
                .build_openai_delete_follow_up_plan(
                    fallback_user_id,
                    fallback_api_key_id,
                    trace_id,
                ),
            ("openai_video_cancel_sync", LocalVideoTaskSnapshot::OpenAi(seed)) => {
                seed.build_cancel_follow_up_plan(fallback_user_id, fallback_api_key_id, trace_id)
            }
            ("openai_video_cancel_sync", LocalVideoTaskSnapshot::Doubao(seed)) => seed
                .build_openai_cancel_follow_up_plan(
                    fallback_user_id,
                    fallback_api_key_id,
                    trace_id,
                ),
            ("gemini_video_cancel_sync", LocalVideoTaskSnapshot::Gemini(seed)) => {
                seed.build_cancel_follow_up_plan(fallback_user_id, fallback_api_key_id, trace_id)
            }
            ("doubao_video_delete_sync", LocalVideoTaskSnapshot::Doubao(seed)) => {
                seed.build_delete_follow_up_plan(fallback_user_id, fallback_api_key_id, trace_id)
            }
            _ => None,
        }
    }

    pub fn prepare_follow_up_sync_plan_for_user(
        &self,
        plan_kind: &str,
        request_path: &str,
        body_json: Option<&Value>,
        fallback_user_id: Option<&str>,
        fallback_api_key_id: Option<&str>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        if self.truth_source_mode != VideoTaskTruthSourceMode::RustAuthoritative {
            return None;
        }
        let snapshot = self.snapshot_for_follow_up_route(plan_kind, request_path)?;
        let user_id = fallback_user_id?.trim();
        if !snapshot.belongs_to_user(user_id) {
            return None;
        }
        // DELETE/cancel/remix follow-ups also require the provider credential;
        // ownership alone must not authorize an empty-header upstream call.
        if !snapshot.has_runtime_auth_headers() {
            return None;
        }
        Self::build_follow_up_plan_from_snapshot(
            snapshot,
            plan_kind,
            body_json,
            Some(user_id),
            fallback_api_key_id,
            trace_id,
        )
    }

    fn snapshot_for_follow_up_route(
        &self,
        plan_kind: &str,
        request_path: &str,
    ) -> Option<LocalVideoTaskSnapshot> {
        match plan_kind {
            "openai_video_remix_sync" => extract_openai_task_id_from_remix_path(request_path)
                .and_then(|task_id| self.store.clone_openai_snapshot(task_id)),
            "openai_video_delete_sync" => extract_openai_task_id_from_path(request_path)
                .and_then(|task_id| self.store.clone_openai_snapshot(task_id)),
            "openai_video_cancel_sync" => extract_openai_task_id_from_cancel_path(request_path)
                .and_then(|task_id| self.store.clone_openai_snapshot(task_id)),
            "gemini_video_cancel_sync" => extract_gemini_short_id_from_cancel_path(request_path)
                .and_then(|short_id| self.store.clone_gemini_snapshot(short_id)),
            "doubao_video_delete_sync" => self.snapshot_for_route(Some("doubao"), request_path),
            _ => None,
        }
    }
}
