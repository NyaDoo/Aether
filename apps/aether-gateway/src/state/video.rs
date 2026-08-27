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

        // Every update of an existing task is a state transition, not a blind
        // UPSERT. A delayed/replayed create response can arrive after the
        // poller has already completed or failed the task; replacing that row
        // would erase its asset/token/error metadata and resurrect billing as
        // active. Only an explicit administrative Deleted tombstone may
        // replace a non-deleted terminal, and the repositories preserve the
        // terminal payload while applying that tombstone.
        if let Some(current) = self.find_video_task_by_id(&record.id).await? {
            if !current
                .status
                .allows_snapshot_replacement_with(record.status)
            {
                self.record_video_task_snapshot_with_stored_truth(&current, snapshot);
                return Ok(Some(current));
            }

            if current.status.is_active() {
                let task_id = record.id.clone();
                if let Some(updated) = self.update_active_video_task(record).await? {
                    self.record_video_task_snapshot_with_stored_truth(&updated, snapshot);
                    return Ok(Some(updated));
                }

                // A concurrent poll/cancel/delete won the active-row CAS.
                // Restore its immutable database truth rather than publishing
                // the stale input snapshot to the in-memory registry.
                if let Some(latest) = self.find_video_task_by_id(&task_id).await? {
                    self.record_video_task_snapshot_with_stored_truth(&latest, snapshot);
                    return Ok((!latest.status.is_active()).then_some(latest));
                }
                self.video_tasks.record_snapshot(snapshot.clone());
                return Ok(None);
            }

            // The only allowed non-active transition is an explicit delete.
            // Adapter upserts apply only the tombstone status/timestamp and
            // preserve the generation terminal's asset and billing metadata.
            let stored = self
                .data
                .upsert_video_task(record)
                .await
                .map_err(|err| GatewayError::Internal(err.to_string()))?;
            if let Some(stored) = stored.as_ref() {
                self.record_video_task_snapshot_with_stored_truth(stored, snapshot);
            }
            return Ok(stored);
        }

        let stored = self
            .data
            .upsert_video_task(record)
            .await
            .map_err(|err| GatewayError::Internal(err.to_string()))?;
        if let Some(stored) = stored.as_ref() {
            // SQL adapters independently enforce the same monotonic rule, so
            // this also restores a concurrent insert's terminal truth.
            self.record_video_task_snapshot_with_stored_truth(stored, snapshot);
        } else {
            // Preserve the existing in-memory-only fallback when no task
            // writer is configured.
            self.video_tasks.record_snapshot(snapshot.clone());
        }
        Ok(stored)
    }

    fn record_video_task_snapshot_with_stored_truth(
        &self,
        stored: &StoredVideoTask,
        runtime_source: &video_tasks::LocalVideoTaskSnapshot,
    ) {
        let transport = match runtime_source {
            video_tasks::LocalVideoTaskSnapshot::OpenAi(seed) => seed.transport.clone(),
            video_tasks::LocalVideoTaskSnapshot::Gemini(seed) => seed.transport.clone(),
            video_tasks::LocalVideoTaskSnapshot::Doubao(seed) => seed.transport.clone(),
        };
        if let Some(snapshot) =
            video_tasks::LocalVideoTaskSnapshot::from_stored_task_with_transport(stored, transport)
        {
            self.video_tasks.record_snapshot(snapshot);
        } else {
            self.video_tasks.hydrate_from_stored_task(stored);
        }
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
