use std::ops::Deref;
use std::path::PathBuf;

use crate::control::GatewayControlAuthContext;

use super::{LocalVideoTaskFollowUpPlan, VideoTaskTruthSourceMode};

#[derive(Debug)]
pub(crate) struct VideoTaskService(aether_video_tasks_core::VideoTaskService);

impl VideoTaskService {
    pub(crate) fn new(mode: VideoTaskTruthSourceMode) -> Self {
        Self(aether_video_tasks_core::VideoTaskService::new(mode))
    }

    pub(crate) fn with_file_store(
        mode: VideoTaskTruthSourceMode,
        path: impl Into<PathBuf>,
    ) -> std::io::Result<Self> {
        Ok(Self(
            aether_video_tasks_core::VideoTaskService::with_file_store(mode, path)?,
        ))
    }

    pub(crate) fn with_truth_source_mode(&self, mode: VideoTaskTruthSourceMode) -> Self {
        Self(self.0.with_truth_source_mode(mode))
    }

    pub(crate) fn prepare_follow_up_sync_plan(
        &self,
        plan_kind: &str,
        request_path: &str,
        body_json: Option<&serde_json::Value>,
        auth_context: Option<&GatewayControlAuthContext>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        self.0
            .prepare_follow_up_sync_plan(
                plan_kind,
                request_path,
                body_json,
                auth_context.map(|value| value.user_id.as_str()),
                auth_context.map(|value| value.api_key_id.as_str()),
                trace_id,
            )
            .map(|plan| annotate_video_follow_up_usage_operation(plan, plan_kind))
    }

    pub(crate) fn prepare_follow_up_sync_plan_for_user(
        &self,
        plan_kind: &str,
        request_path: &str,
        body_json: Option<&serde_json::Value>,
        auth_context: Option<&GatewayControlAuthContext>,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        self.0
            .prepare_follow_up_sync_plan_for_user(
                plan_kind,
                request_path,
                body_json,
                auth_context.map(|value| value.user_id.as_str()),
                auth_context.map(|value| value.api_key_id.as_str()),
                trace_id,
            )
            .map(|plan| annotate_video_follow_up_usage_operation(plan, plan_kind))
    }

    pub(crate) fn prepare_follow_up_sync_plan_for_owner(
        &self,
        plan_kind: &str,
        request_path: &str,
        body_json: Option<&serde_json::Value>,
        user_id: &str,
        trace_id: &str,
    ) -> Option<LocalVideoTaskFollowUpPlan> {
        self.0
            .prepare_follow_up_sync_plan_for_user(
                plan_kind,
                request_path,
                body_json,
                Some(user_id),
                None,
                trace_id,
            )
            .map(|plan| annotate_video_follow_up_usage_operation(plan, plan_kind))
    }
}

fn annotate_video_follow_up_usage_operation(
    mut follow_up: LocalVideoTaskFollowUpPlan,
    plan_kind: &str,
) -> LocalVideoTaskFollowUpPlan {
    let discriminator = follow_up.report_kind.as_deref().unwrap_or(plan_kind);
    let operation = if discriminator.contains("_remix_") {
        Some("video.remix")
    } else if discriminator.contains("_cancel_") {
        Some("video.cancel")
    } else if discriminator.contains("_delete_") {
        Some("video.delete")
    } else {
        None
    };
    let Some(operation) = operation else {
        return follow_up;
    };

    let context = follow_up
        .report_context
        .get_or_insert_with(|| serde_json::json!({}));
    if !context.is_object() {
        *context = serde_json::json!({ "report_context": context.take() });
    }
    context
        .as_object_mut()
        .expect("video follow-up report context should be an object")
        .insert(
            "operation".to_string(),
            serde_json::Value::String(operation.to_string()),
        );
    follow_up
}

impl Deref for VideoTaskService {
    type Target = aether_video_tasks_core::VideoTaskService;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
