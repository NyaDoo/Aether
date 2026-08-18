use super::{AppState, GatewayError};

use crate::{async_task, video_tasks};
use aether_data_contracts::repository::video_tasks::{
    StoredVideoTask, UpsertVideoTask, VideoTaskLookupKey, VideoTaskModelCount,
    VideoTaskQueryFilter, VideoTaskStatus, VideoTaskStatusCount,
};

impl AppState {
    pub(crate) async fn read_data_backed_video_task_response(
        &self,
        route_family: Option<&str>,
        request_path: &str,
        user_id: &str,
    ) -> Result<Option<video_tasks::LocalVideoTaskReadResponse>, GatewayError> {
        self.data
            .read_video_task_response_for_user(route_family, request_path, user_id)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn find_video_task_by_id(
        &self,
        task_id: &str,
    ) -> Result<Option<StoredVideoTask>, GatewayError> {
        self.data
            .find_video_task(VideoTaskLookupKey::Id(task_id))
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn find_video_task_by_short_id(
        &self,
        short_id: &str,
    ) -> Result<Option<StoredVideoTask>, GatewayError> {
        self.data
            .find_video_task(VideoTaskLookupKey::ShortId(short_id))
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn upsert_video_task_snapshot(
        &self,
        snapshot: &video_tasks::LocalVideoTaskSnapshot,
    ) -> Result<Option<StoredVideoTask>, GatewayError> {
        // Keep this boundary defensive even though the current seed-specific
        // serializers redact their embedded snapshot.  This method is the
        // gateway's generic persistence entry point and must never make a
        // future provider variant's runtime Authorization/proxy/profile data
        // durable by accident.
        let persisted_snapshot = snapshot.redacted_for_persistence();
        let record = persisted_snapshot.to_upsert_record();

        // Cancellation is a state transition, not an unconditional upsert.
        // A poll can complete the task after the DELETE was sent but before
        // this write; using UPSERT here would resurrect the row as Cancelled,
        // erase the completed asset, and misclassify billing.  Use the
        // repository CAS whenever a row already exists, and restore the
        // winning database snapshot when the CAS loses.
        if record.status == VideoTaskStatus::Cancelled {
            if let Some(current) = self.find_video_task_by_id(&record.id).await? {
                if !current.status.is_active() {
                    self.video_tasks.hydrate_from_stored_task(&current);
                    return Ok(Some(current));
                }
                let task_id = record.id.clone();
                let updated = self.update_active_video_task(record).await?;
                if updated.is_none() {
                    if let Some(latest) = self.find_video_task_by_id(&task_id).await? {
                        if !latest.status.is_active() {
                            self.video_tasks.hydrate_from_stored_task(&latest);
                            return Ok(Some(latest));
                        }
                        // The writer may be unavailable, or another active
                        // update may still be in flight. Do not report a
                        // cancellation success for an unchanged active row.
                        if let Some(runtime) = self.reconstruct_video_task_snapshot(&latest).await?
                        {
                            self.video_tasks.record_snapshot(runtime);
                        } else {
                            self.video_tasks.hydrate_from_stored_task(&latest);
                        }
                        return Ok(None);
                    }
                }
                return Ok(updated);
            }
        }

        self.data
            .upsert_video_task(record)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn hydrate_video_task_for_route(
        &self,
        route_family: Option<&str>,
        request_path: &str,
    ) -> Result<bool, GatewayError> {
        let lookup =
            video_tasks::resolve_video_task_hydration_lookup_key(route_family, request_path);
        let Some(lookup) = lookup else {
            return Ok(false);
        };
        let Some(task) = self
            .data
            .find_video_task(lookup)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))?
        else {
            return Ok(false);
        };
        if let Some(snapshot) = self.reconstruct_video_task_snapshot(&task).await? {
            self.video_tasks.record_snapshot(snapshot);
            return Ok(true);
        }
        // Terminal reads can still be projected from a redacted snapshot when
        // the provider/key was removed. Active follow-ups require reconstructed
        // runtime credentials and therefore will not produce a plan.
        Ok(self.video_tasks.hydrate_from_stored_task(&task))
    }

    pub(crate) async fn hydrate_video_task_for_route_for_user(
        &self,
        route_family: Option<&str>,
        request_path: &str,
        user_id: &str,
    ) -> Result<bool, GatewayError> {
        let Some(lookup) =
            video_tasks::resolve_video_task_hydration_lookup_key(route_family, request_path)
        else {
            return Ok(false);
        };
        let Some(task) = self
            .data
            .find_video_task(lookup)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))?
        else {
            return Ok(false);
        };
        if task.user_id.as_deref().map(str::trim) != Some(user_id.trim()) {
            return Ok(false);
        }
        let hydrated = if let Some(snapshot) = self.reconstruct_video_task_snapshot(&task).await? {
            self.video_tasks.record_snapshot(snapshot);
            true
        } else {
            self.video_tasks.hydrate_from_stored_task(&task)
        };
        if matches!(
            task.status,
            VideoTaskStatus::Completed
                | VideoTaskStatus::Failed
                | VideoTaskStatus::Expired
                | VideoTaskStatus::Cancelled
        ) {
            // A terminal row can outlive a transient usage-upsert/settlement
            // failure. Authenticated owner reads are a safe compensation point:
            // terminal finalization is idempotent and foreign users never reach
            // this branch. Deleted is deliberately excluded because deleting an
            // already-settled task must not rewrite its original outcome.
            async_task::finalize_video_task_if_terminal(self, &task).await;
        }
        Ok(hydrated)
    }

    pub(crate) async fn reconstruct_video_task_snapshot(
        &self,
        task: &StoredVideoTask,
    ) -> Result<Option<video_tasks::LocalVideoTaskSnapshot>, GatewayError> {
        crate::provider_transport::reconstruct_local_video_task_snapshot(self, task)
            .await
            .map_err(GatewayError::Internal)
    }

    pub(crate) async fn claim_due_video_tasks(
        &self,
        now_unix_secs: u64,
        claim_until_unix_secs: u64,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, GatewayError> {
        self.data
            .claim_due_video_tasks(now_unix_secs, claim_until_unix_secs, limit)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn update_active_video_task(
        &self,
        task: UpsertVideoTask,
    ) -> Result<Option<StoredVideoTask>, GatewayError> {
        self.data
            .update_active_video_task(task)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn list_video_task_page(
        &self,
        filter: &VideoTaskQueryFilter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, GatewayError> {
        self.data
            .list_video_task_page(filter, offset, limit)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn list_video_task_page_summary(
        &self,
        filter: &VideoTaskQueryFilter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, GatewayError> {
        self.data
            .list_video_task_page_summary(filter, offset, limit)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn count_video_tasks(
        &self,
        filter: &VideoTaskQueryFilter,
    ) -> Result<u64, GatewayError> {
        self.data
            .count_video_tasks(filter)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn count_video_tasks_by_status(
        &self,
        filter: &VideoTaskQueryFilter,
    ) -> Result<Vec<VideoTaskStatusCount>, GatewayError> {
        self.data
            .count_video_tasks_by_status(filter)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn count_distinct_video_task_users(
        &self,
        filter: &VideoTaskQueryFilter,
    ) -> Result<u64, GatewayError> {
        self.data
            .count_distinct_video_task_users(filter)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn top_video_task_models(
        &self,
        filter: &VideoTaskQueryFilter,
        limit: usize,
    ) -> Result<Vec<VideoTaskModelCount>, GatewayError> {
        self.data
            .top_video_task_models(filter, limit)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn count_video_tasks_created_since(
        &self,
        filter: &VideoTaskQueryFilter,
        created_since_unix_secs: u64,
    ) -> Result<u64, GatewayError> {
        self.data
            .count_video_tasks_created_since(filter, created_since_unix_secs)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))
    }

    pub(crate) async fn execute_video_task_refresh_plan(
        &self,
        refresh_plan: &video_tasks::LocalVideoTaskReadRefreshPlan,
    ) -> Result<bool, GatewayError> {
        async_task::execute_video_task_refresh_plan(self, refresh_plan).await
    }
}
