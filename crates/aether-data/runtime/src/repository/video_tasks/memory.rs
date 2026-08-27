use std::collections::{BTreeMap, BTreeSet};
use std::sync::RwLock;

use async_trait::async_trait;

use super::{
    StoredVideoTask, UpsertVideoTask, VideoTaskLookupKey, VideoTaskModelCount,
    VideoTaskQueryFilter, VideoTaskReadRepository, VideoTaskStatus, VideoTaskStatusCount,
    VideoTaskWriteRepository,
};
use crate::DataLayerError;

#[derive(Debug, Default)]
struct MemoryVideoTaskIndex {
    by_id: BTreeMap<String, StoredVideoTask>,
    short_to_id: BTreeMap<String, String>,
    user_external_to_ids: BTreeMap<(String, String), BTreeSet<String>>,
}

#[derive(Debug, Default)]
pub struct InMemoryVideoTaskRepository {
    index: RwLock<MemoryVideoTaskIndex>,
}

impl InMemoryVideoTaskRepository {
    fn apply_snapshot_upsert(
        existing: Option<&StoredVideoTask>,
        incoming: StoredVideoTask,
    ) -> StoredVideoTask {
        let Some(existing) = existing else {
            return incoming;
        };
        if !existing
            .status
            .allows_snapshot_replacement_with(incoming.status)
        {
            return existing.clone();
        }
        if !existing.status.is_active() && incoming.status == VideoTaskStatus::Deleted {
            let mut deleted = existing.clone();
            deleted.status = VideoTaskStatus::Deleted;
            deleted.updated_at_unix_secs = incoming.updated_at_unix_secs;
            return deleted;
        }
        incoming
    }

    fn store_locked(index: &mut MemoryVideoTaskIndex, task: StoredVideoTask) -> StoredVideoTask {
        if let Some(previous) = index.by_id.insert(task.id.clone(), task.clone()) {
            if let Some(short_id) = previous.short_id {
                index.short_to_id.remove(&short_id);
            }
            if let (Some(user_id), Some(external_task_id)) =
                (&previous.user_id, &previous.external_task_id)
            {
                let key = (user_id.clone(), external_task_id.clone());
                if let std::collections::btree_map::Entry::Occupied(mut entry) =
                    index.user_external_to_ids.entry(key)
                {
                    entry.get_mut().remove(&previous.id);
                    if entry.get().is_empty() {
                        entry.remove();
                    }
                }
            }
        }

        if let Some(short_id) = &task.short_id {
            index.short_to_id.insert(short_id.clone(), task.id.clone());
        }
        if let (Some(user_id), Some(external_task_id)) = (&task.user_id, &task.external_task_id) {
            index
                .user_external_to_ids
                .entry((user_id.clone(), external_task_id.clone()))
                .or_default()
                .insert(task.id.clone());
        }

        task
    }

    fn matches_filter(task: &StoredVideoTask, filter: &VideoTaskQueryFilter) -> bool {
        if let Some(user_id) = filter.user_id.as_deref() {
            if task.user_id.as_deref() != Some(user_id) {
                return false;
            }
        }
        if let Some(status) = filter.status {
            if task.status != status {
                return false;
            }
        }
        if filter.exclude_deleted && task.status == VideoTaskStatus::Deleted {
            return false;
        }
        if let Some(model_exact) = filter.model_exact.as_deref() {
            if task.model.as_deref().map(str::trim) != Some(model_exact.trim()) {
                return false;
            }
        }
        if let Some(model_substring) = filter.model_substring.as_deref() {
            let needle = model_substring.trim().to_ascii_lowercase();
            let Some(model) = task.model.as_deref() else {
                return false;
            };
            if !model.to_ascii_lowercase().contains(&needle) {
                return false;
            }
        }
        if let Some(client_api_format) = filter.client_api_format.as_deref() {
            if task.client_api_format.as_deref() != Some(client_api_format) {
                return false;
            }
        }
        true
    }
}

#[async_trait]
impl VideoTaskReadRepository for InMemoryVideoTaskRepository {
    async fn find(
        &self,
        key: VideoTaskLookupKey<'_>,
    ) -> Result<Option<StoredVideoTask>, DataLayerError> {
        let index = self.index.read().expect("video task repository lock");
        match key {
            VideoTaskLookupKey::Id(id) => Ok(index.by_id.get(id).cloned()),
            VideoTaskLookupKey::ShortId(short_id) => Ok(index
                .short_to_id
                .get(short_id)
                .and_then(|id| index.by_id.get(id))
                .cloned()),
            VideoTaskLookupKey::UserExternal {
                user_id,
                external_task_id,
            } => {
                let ids = index
                    .user_external_to_ids
                    .get(&(user_id.to_string(), external_task_id.to_string()));
                match ids.map(BTreeSet::len).unwrap_or_default() {
                    0 => Ok(None),
                    1 => Ok(ids
                        .and_then(|ids| ids.first())
                        .and_then(|id| index.by_id.get(id))
                        .cloned()),
                    _ => Err(DataLayerError::UnexpectedValue(
                        "ambiguous video task user/external lookup".to_string(),
                    )),
                }
            }
        }
    }

    async fn list_active(&self, limit: usize) -> Result<Vec<StoredVideoTask>, DataLayerError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut tasks = self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| task.status.is_active())
            .cloned()
            .collect::<Vec<_>>();
        tasks.sort_by_key(|entry| std::cmp::Reverse(entry.updated_at_unix_secs));
        tasks.truncate(limit);
        Ok(tasks)
    }

    async fn list_due(
        &self,
        now_unix_secs: u64,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, DataLayerError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut tasks = self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| {
                matches!(
                    task.status,
                    super::VideoTaskStatus::Submitted
                        | super::VideoTaskStatus::Queued
                        | super::VideoTaskStatus::Processing
                ) && task.poll_count < task.max_poll_count
                    && task
                        .next_poll_at_unix_secs
                        .is_some_and(|value| value <= now_unix_secs)
            })
            .cloned()
            .collect::<Vec<_>>();
        tasks.sort_by(|left, right| {
            left.next_poll_at_unix_secs
                .cmp(&right.next_poll_at_unix_secs)
                .then_with(|| left.updated_at_unix_secs.cmp(&right.updated_at_unix_secs))
        });
        tasks.truncate(limit);
        Ok(tasks)
    }

    async fn list_page(
        &self,
        filter: &VideoTaskQueryFilter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, DataLayerError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut tasks = self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| Self::matches_filter(task, filter))
            .cloned()
            .collect::<Vec<_>>();
        tasks.sort_by(|left, right| {
            right
                .created_at_unix_ms
                .cmp(&left.created_at_unix_ms)
                .then_with(|| right.updated_at_unix_secs.cmp(&left.updated_at_unix_secs))
        });
        Ok(tasks.into_iter().skip(offset).take(limit).collect())
    }

    async fn list_page_summary(
        &self,
        filter: &VideoTaskQueryFilter,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, DataLayerError> {
        Self::list_page(self, filter, offset, limit).await
    }

    async fn count(&self, filter: &VideoTaskQueryFilter) -> Result<u64, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| Self::matches_filter(task, filter))
            .count() as u64)
    }

    async fn count_by_status(
        &self,
        filter: &VideoTaskQueryFilter,
    ) -> Result<Vec<VideoTaskStatusCount>, DataLayerError> {
        let mut counts = BTreeMap::<VideoTaskStatus, u64>::new();
        for task in self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| Self::matches_filter(task, filter))
        {
            *counts.entry(task.status).or_default() += 1;
        }
        Ok(counts
            .into_iter()
            .map(|(status, count)| VideoTaskStatusCount { status, count })
            .collect())
    }

    async fn count_distinct_users(
        &self,
        filter: &VideoTaskQueryFilter,
    ) -> Result<u64, DataLayerError> {
        let index = self.index.read().expect("video task repository lock");
        let users = index
            .by_id
            .values()
            .filter(|task| Self::matches_filter(task, filter))
            .filter_map(|task| task.user_id.as_deref())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .collect::<std::collections::BTreeSet<_>>();
        Ok(users.len() as u64)
    }

    async fn top_models(
        &self,
        filter: &VideoTaskQueryFilter,
        limit: usize,
    ) -> Result<Vec<VideoTaskModelCount>, DataLayerError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut counts = BTreeMap::<String, u64>::new();
        for task in self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| Self::matches_filter(task, filter))
        {
            let Some(model) = task.model.as_deref() else {
                continue;
            };
            if model.trim().is_empty() {
                continue;
            }
            *counts.entry(model.to_string()).or_default() += 1;
        }

        let mut models = counts
            .into_iter()
            .map(|(model, count)| VideoTaskModelCount { model, count })
            .collect::<Vec<_>>();
        models.sort_by(|left, right| {
            right
                .count
                .cmp(&left.count)
                .then_with(|| left.model.cmp(&right.model))
        });
        models.truncate(limit);
        Ok(models)
    }

    async fn count_created_since(
        &self,
        filter: &VideoTaskQueryFilter,
        created_since_unix_secs: u64,
    ) -> Result<u64, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("video task repository lock")
            .by_id
            .values()
            .filter(|task| {
                Self::matches_filter(task, filter)
                    && task.created_at_unix_ms >= created_since_unix_secs
            })
            .count() as u64)
    }
}

#[async_trait]
impl VideoTaskWriteRepository for InMemoryVideoTaskRepository {
    async fn upsert(&self, task: UpsertVideoTask) -> Result<StoredVideoTask, DataLayerError> {
        let mut index = self.index.write().expect("video task repository lock");
        let incoming = task.into_stored();
        let stored = Self::apply_snapshot_upsert(index.by_id.get(&incoming.id), incoming);
        Ok(Self::store_locked(&mut index, stored))
    }

    async fn update_if_active(
        &self,
        task: UpsertVideoTask,
    ) -> Result<Option<StoredVideoTask>, DataLayerError> {
        let mut index = self.index.write().expect("video task repository lock");
        let Some(existing) = index.by_id.get(&task.id) else {
            return Ok(None);
        };
        if !existing.status.is_active() {
            return Ok(None);
        }
        Ok(Some(Self::store_locked(&mut index, task.into_stored())))
    }

    async fn claim_due(
        &self,
        now_unix_secs: u64,
        claim_until_unix_secs: u64,
        limit: usize,
    ) -> Result<Vec<StoredVideoTask>, DataLayerError> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut index = self.index.write().expect("video task repository lock");
        let mut due_ids = index
            .by_id
            .values()
            .filter(|task| {
                matches!(
                    task.status,
                    VideoTaskStatus::Submitted
                        | VideoTaskStatus::Queued
                        | VideoTaskStatus::Processing
                ) && task.poll_count < task.max_poll_count
                    && task
                        .next_poll_at_unix_secs
                        .is_some_and(|value| value <= now_unix_secs)
            })
            .map(|task| task.id.clone())
            .collect::<Vec<_>>();
        due_ids.sort_by(|left_id, right_id| {
            let left = index.by_id.get(left_id).expect("task should exist");
            let right = index.by_id.get(right_id).expect("task should exist");
            left.next_poll_at_unix_secs
                .cmp(&right.next_poll_at_unix_secs)
                .then_with(|| left.updated_at_unix_secs.cmp(&right.updated_at_unix_secs))
        });
        due_ids.truncate(limit);

        let mut claimed = Vec::with_capacity(due_ids.len());
        for id in due_ids {
            let Some(task) = index.by_id.get_mut(&id) else {
                continue;
            };
            task.next_poll_at_unix_secs = Some(claim_until_unix_secs);
            task.updated_at_unix_secs = now_unix_secs.max(task.updated_at_unix_secs);
            claimed.push(task.clone());
        }
        Ok(claimed)
    }
}

#[cfg(test)]
mod tests {
    use super::InMemoryVideoTaskRepository;
    use crate::repository::video_tasks::{
        UpsertVideoTask, VideoTaskLookupKey, VideoTaskQueryFilter, VideoTaskReadRepository,
        VideoTaskStatus, VideoTaskWriteRepository,
    };
    use crate::DataLayerError;

    fn sample_task(
        id: &str,
        status: VideoTaskStatus,
        updated_at_unix_secs: u64,
    ) -> UpsertVideoTask {
        UpsertVideoTask {
            id: id.to_string(),
            short_id: Some(format!("short-{id}")),
            request_id: format!("request-{id}"),
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            username: Some("user".to_string()),
            api_key_name: Some("primary".to_string()),
            external_task_id: Some(format!("ext-{id}")),
            provider_id: Some("provider-1".to_string()),
            endpoint_id: Some("endpoint-1".to_string()),
            key_id: Some("provider-key-1".to_string()),
            client_api_format: Some("openai:video".to_string()),
            provider_api_format: Some("openai:video".to_string()),
            format_converted: false,
            model: Some("sora-2".to_string()),
            prompt: Some("hello".to_string()),
            original_request_body: Some(serde_json::json!({"prompt": "hello"})),
            duration_seconds: Some(4),
            resolution: Some("720p".to_string()),
            aspect_ratio: Some("16:9".to_string()),
            size: Some("1280x720".to_string()),
            status,
            progress_percent: 0,
            progress_message: None,
            retry_count: 0,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: Some(updated_at_unix_secs),
            poll_count: 0,
            max_poll_count: 360,
            created_at_unix_ms: updated_at_unix_secs.saturating_sub(10),
            submitted_at_unix_secs: Some(updated_at_unix_secs.saturating_sub(10)),
            completed_at_unix_secs: None,
            updated_at_unix_secs,
            error_code: None,
            error_message: None,
            video_url: None,
            request_metadata: None,
        }
    }

    #[tokio::test]
    async fn reads_task_by_all_supported_lookup_keys() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("upsert should succeed");

        assert!(repo
            .find(VideoTaskLookupKey::Id("task-1"))
            .await
            .expect("find by id should succeed")
            .is_some());
        assert!(repo
            .find(VideoTaskLookupKey::ShortId("short-task-1"))
            .await
            .expect("find by short id should succeed")
            .is_some());
        assert!(repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-1",
                external_task_id: "ext-task-1",
            })
            .await
            .expect("find by user/external should succeed")
            .is_some());
    }

    #[tokio::test]
    async fn user_external_lookup_fails_closed_when_multiple_tasks_match() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("first task should insert");
        repo.upsert(UpsertVideoTask {
            external_task_id: Some("ext-task-1".to_string()),
            ..sample_task("task-2", VideoTaskStatus::Submitted, 200)
        })
        .await
        .expect("second task should insert");

        let error = repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-1",
                external_task_id: "ext-task-1",
            })
            .await
            .expect_err("ambiguous lookup must fail closed");

        assert!(matches!(
            error,
            DataLayerError::UnexpectedValue(message)
                if message == "ambiguous video task user/external lookup"
        ));
    }

    #[tokio::test]
    async fn same_external_id_remains_unambiguous_across_different_users() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("first task should insert");
        repo.upsert(UpsertVideoTask {
            user_id: Some("user-2".to_string()),
            external_task_id: Some("ext-task-1".to_string()),
            ..sample_task("task-2", VideoTaskStatus::Submitted, 200)
        })
        .await
        .expect("second task should insert");

        let first = repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-1",
                external_task_id: "ext-task-1",
            })
            .await
            .expect("first user lookup should succeed")
            .expect("first user task should exist");
        let second = repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-2",
                external_task_id: "ext-task-1",
            })
            .await
            .expect("second user lookup should succeed")
            .expect("second user task should exist");

        assert_eq!(first.id, "task-1");
        assert_eq!(second.id, "task-2");
    }

    #[tokio::test]
    async fn list_active_only_returns_active_tasks_in_descending_update_order() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Completed, 100))
            .await
            .expect("upsert should succeed");
        repo.upsert(sample_task("task-2", VideoTaskStatus::Processing, 200))
            .await
            .expect("upsert should succeed");
        repo.upsert(sample_task("task-3", VideoTaskStatus::Queued, 150))
            .await
            .expect("upsert should succeed");

        let active = repo
            .list_active(10)
            .await
            .expect("list active should succeed");
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].id, "task-2");
        assert_eq!(active[1].id, "task-3");
    }

    #[tokio::test]
    async fn upsert_replaces_secondary_indexes() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("upsert should succeed");

        repo.upsert(UpsertVideoTask {
            id: "task-1".to_string(),
            short_id: Some("short-task-1b".to_string()),
            request_id: "request-task-1b".to_string(),
            user_id: Some("user-2".to_string()),
            api_key_id: Some("api-key-2".to_string()),
            username: Some("user-2".to_string()),
            api_key_name: Some("secondary".to_string()),
            external_task_id: Some("ext-task-1b".to_string()),
            provider_id: Some("provider-2".to_string()),
            endpoint_id: Some("endpoint-2".to_string()),
            key_id: Some("provider-key-2".to_string()),
            client_api_format: Some("gemini:video".to_string()),
            provider_api_format: Some("gemini:video".to_string()),
            format_converted: false,
            model: Some("veo-3".to_string()),
            prompt: Some("remix".to_string()),
            original_request_body: Some(serde_json::json!({"prompt": "remix"})),
            duration_seconds: Some(8),
            resolution: Some("1080p".to_string()),
            aspect_ratio: Some("16:9".to_string()),
            size: Some("720p".to_string()),
            status: VideoTaskStatus::Processing,
            progress_percent: 50,
            progress_message: Some("processing".to_string()),
            retry_count: 1,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: Some(200),
            poll_count: 2,
            max_poll_count: 360,
            created_at_unix_ms: 150,
            submitted_at_unix_secs: Some(150),
            completed_at_unix_secs: None,
            updated_at_unix_secs: 200,
            error_code: None,
            error_message: None,
            video_url: None,
            request_metadata: None,
        })
        .await
        .expect("upsert should succeed");

        assert!(repo
            .find(VideoTaskLookupKey::ShortId("short-task-1"))
            .await
            .expect("find should succeed")
            .is_none());
        assert!(repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-1",
                external_task_id: "ext-task-1",
            })
            .await
            .expect("find should succeed")
            .is_none());
        assert!(repo
            .find(VideoTaskLookupKey::ShortId("short-task-1b"))
            .await
            .expect("find should succeed")
            .is_some());
        assert!(repo
            .find(VideoTaskLookupKey::UserExternal {
                user_id: "user-2",
                external_task_id: "ext-task-1b",
            })
            .await
            .expect("find should succeed")
            .is_some());
    }

    #[tokio::test]
    async fn list_due_returns_due_active_tasks_in_next_poll_order() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 300))
            .await
            .expect("upsert should succeed");
        repo.upsert(sample_task("task-2", VideoTaskStatus::Processing, 100))
            .await
            .expect("upsert should succeed");
        repo.upsert(UpsertVideoTask {
            next_poll_at_unix_secs: Some(500),
            ..sample_task("task-3", VideoTaskStatus::Queued, 200)
        })
        .await
        .expect("upsert should succeed");

        let due = repo
            .list_due(300, 10)
            .await
            .expect("list due should succeed");
        assert_eq!(due.len(), 2);
        assert_eq!(due[0].id, "task-2");
        assert_eq!(due[1].id, "task-1");
    }

    #[tokio::test]
    async fn update_if_active_skips_terminal_tasks() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Completed, 100))
            .await
            .expect("upsert should succeed");

        let updated = repo
            .update_if_active(UpsertVideoTask {
                progress_percent: 100,
                ..sample_task("task-1", VideoTaskStatus::Completed, 200)
            })
            .await
            .expect("update should succeed");

        assert!(updated.is_none());
    }

    #[tokio::test]
    async fn snapshot_upsert_preserves_billing_terminals_and_allows_metadata_safe_delete() {
        for (index, terminal_status) in [
            VideoTaskStatus::Completed,
            VideoTaskStatus::Failed,
            VideoTaskStatus::Cancelled,
        ]
        .into_iter()
        .enumerate()
        {
            let repo = InMemoryVideoTaskRepository::default();
            let task_id = format!("terminal-{index}");
            let terminal = UpsertVideoTask {
                completed_at_unix_secs: Some(100),
                video_url: Some("https://assets.example/final.mp4".to_string()),
                error_code: Some("terminal-code".to_string()),
                error_message: Some("terminal-message".to_string()),
                request_metadata: Some(serde_json::json!({
                    "completion_tokens": 41,
                    "total_tokens": 42,
                })),
                ..sample_task(&task_id, terminal_status, 100)
            };
            repo.upsert(terminal.clone())
                .await
                .expect("terminal task should insert");

            let blocked = repo
                .upsert(UpsertVideoTask {
                    video_url: None,
                    error_code: None,
                    error_message: None,
                    request_metadata: None,
                    ..sample_task(&task_id, VideoTaskStatus::Processing, 200)
                })
                .await
                .expect("stale active upsert should return stored truth");
            assert_eq!(blocked.status, terminal_status);
            assert_eq!(blocked.video_url, terminal.video_url);
            assert_eq!(blocked.error_code, terminal.error_code);
            assert_eq!(blocked.error_message, terminal.error_message);
            assert_eq!(blocked.request_metadata, terminal.request_metadata);
            assert_eq!(blocked.completed_at_unix_secs, Some(100));

            let conflicting_terminal_status = if terminal_status == VideoTaskStatus::Failed {
                VideoTaskStatus::Completed
            } else {
                VideoTaskStatus::Failed
            };
            let conflicting_terminal = repo
                .upsert(UpsertVideoTask {
                    video_url: None,
                    error_code: None,
                    error_message: None,
                    request_metadata: None,
                    ..sample_task(&task_id, conflicting_terminal_status, 250)
                })
                .await
                .expect("conflicting terminal upsert should return stored truth");
            assert_eq!(conflicting_terminal, blocked);

            let deleted = repo
                .upsert(UpsertVideoTask {
                    video_url: None,
                    error_code: None,
                    error_message: None,
                    request_metadata: None,
                    ..sample_task(&task_id, VideoTaskStatus::Deleted, 300)
                })
                .await
                .expect("explicit delete should tombstone terminal task");
            assert_eq!(deleted.status, VideoTaskStatus::Deleted);
            assert_eq!(deleted.updated_at_unix_secs, 300);
            assert_eq!(deleted.video_url, terminal.video_url);
            assert_eq!(deleted.error_code, terminal.error_code);
            assert_eq!(deleted.error_message, terminal.error_message);
            assert_eq!(deleted.request_metadata, terminal.request_metadata);
            assert_eq!(deleted.completed_at_unix_secs, Some(100));

            let still_deleted = repo
                .upsert(sample_task(&task_id, VideoTaskStatus::Completed, 400))
                .await
                .expect("deleted task should remain immutable");
            assert_eq!(still_deleted, deleted);
        }
    }

    #[tokio::test]
    async fn list_page_and_stats_apply_filters() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("upsert should succeed");
        repo.upsert(UpsertVideoTask {
            model: Some("veo-3-fast".to_string()),
            user_id: Some("user-2".to_string()),
            client_api_format: Some("gemini:video".to_string()),
            created_at_unix_ms: 250,
            updated_at_unix_secs: 250,
            ..sample_task("task-2", VideoTaskStatus::Completed, 250)
        })
        .await
        .expect("upsert should succeed");
        repo.upsert(UpsertVideoTask {
            model: Some("veo-3-fast".to_string()),
            user_id: Some("user-2".to_string()),
            client_api_format: Some("gemini:video".to_string()),
            created_at_unix_ms: 260,
            updated_at_unix_secs: 260,
            ..sample_task("task-3", VideoTaskStatus::Completed, 260)
        })
        .await
        .expect("upsert should succeed");

        let filter = VideoTaskQueryFilter {
            user_id: Some("user-2".to_string()),
            status: Some(VideoTaskStatus::Completed),
            model_exact: None,
            model_substring: Some("veo".to_string()),
            client_api_format: Some("gemini:video".to_string()),
            exclude_deleted: false,
        };

        let page = repo
            .list_page(&filter, 0, 10)
            .await
            .expect("list page should succeed");
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].id, "task-3");
        assert_eq!(page[1].id, "task-2");

        let count = repo.count(&filter).await.expect("count should succeed");
        assert_eq!(count, 2);

        let by_status = repo
            .count_by_status(&filter)
            .await
            .expect("status count should succeed");
        assert_eq!(by_status.len(), 1);
        assert_eq!(by_status[0].status, VideoTaskStatus::Completed);
        assert_eq!(by_status[0].count, 2);

        let top_models = repo
            .top_models(&filter, 10)
            .await
            .expect("top models should succeed");
        assert_eq!(top_models.len(), 1);
        assert_eq!(top_models[0].model, "veo-3-fast");
        assert_eq!(top_models[0].count, 2);

        let today_count = repo
            .count_created_since(&filter, 255)
            .await
            .expect("today count should succeed");
        assert_eq!(today_count, 1);
    }

    #[tokio::test]
    async fn exact_model_filter_excludes_prefix_matches_and_deleted_tombstones() {
        let repo = InMemoryVideoTaskRepository::default();
        for (id, model, status) in [
            ("task-exact", "ep-video-1", VideoTaskStatus::Completed),
            (
                "task-prefix",
                "ep-video-1-canary",
                VideoTaskStatus::Completed,
            ),
            ("task-deleted", "ep-video-1", VideoTaskStatus::Deleted),
        ] {
            repo.upsert(UpsertVideoTask {
                model: Some(model.to_string()),
                user_id: Some("user-1".to_string()),
                client_api_format: Some("doubao:video".to_string()),
                ..sample_task(id, status, 100)
            })
            .await
            .expect("upsert should succeed");
        }

        let filter = VideoTaskQueryFilter {
            user_id: Some("user-1".to_string()),
            model_exact: Some("ep-video-1".to_string()),
            client_api_format: Some("doubao:video".to_string()),
            exclude_deleted: true,
            ..VideoTaskQueryFilter::default()
        };

        let page = repo
            .list_page(&filter, 0, 10)
            .await
            .expect("list page should succeed");
        assert_eq!(page.len(), 1);
        assert_eq!(page[0].id, "task-exact");
        assert_eq!(repo.count(&filter).await.expect("count"), 1);
    }

    #[tokio::test]
    async fn claim_due_advances_claimed_tasks_until_claim_deadline() {
        let repo = InMemoryVideoTaskRepository::default();
        repo.upsert(sample_task("task-1", VideoTaskStatus::Submitted, 100))
            .await
            .expect("upsert should succeed");
        repo.upsert(sample_task("task-2", VideoTaskStatus::Processing, 90))
            .await
            .expect("upsert should succeed");

        let claimed = repo
            .claim_due(100, 130, 1)
            .await
            .expect("claim should succeed");
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].id, "task-2");
        assert_eq!(claimed[0].next_poll_at_unix_secs, Some(130));

        let remaining = repo
            .list_due(100, 10)
            .await
            .expect("list due should succeed");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "task-1");
    }
}
