use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    current_unix_timestamp_secs, DoubaoVideoTaskSeed, GeminiVideoTaskSeed,
    LocalVideoTaskReadResponse, LocalVideoTaskRegistryMutation, LocalVideoTaskSnapshot,
    LocalVideoTaskStatus, OpenAiVideoTaskSeed,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoTaskRegistry {
    openai: BTreeMap<String, LocalVideoTaskSnapshot>,
    gemini: BTreeMap<String, LocalVideoTaskSnapshot>,
    #[serde(default)]
    doubao: BTreeMap<String, LocalVideoTaskSnapshot>,
}

impl VideoTaskRegistry {
    pub fn redacted_for_persistence(&self) -> Self {
        Self {
            openai: self
                .openai
                .iter()
                .map(|(id, snapshot)| (id.clone(), snapshot.redacted_for_persistence()))
                .collect(),
            gemini: self
                .gemini
                .iter()
                .map(|(id, snapshot)| (id.clone(), snapshot.redacted_for_persistence()))
                .collect(),
            doubao: self
                .doubao
                .iter()
                .map(|(id, snapshot)| (id.clone(), snapshot.redacted_for_persistence()))
                .collect(),
        }
    }

    pub fn insert(&mut self, snapshot: LocalVideoTaskSnapshot) {
        // A snapshot's enum variant describes the upstream provider contract;
        // the registry map describes the client route that owns the internal
        // persistence ID.
        // Those differ for an OpenAI request served by a Doubao endpoint.
        match snapshot
            .client_api_format()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "openai:video" => {
                let task_id = match &snapshot {
                    LocalVideoTaskSnapshot::OpenAi(seed) => Some(seed.local_task_id.clone()),
                    LocalVideoTaskSnapshot::Doubao(seed) => Some(seed.local_task_id.clone()),
                    LocalVideoTaskSnapshot::Gemini(_) => None,
                };
                if let Some(task_id) = task_id {
                    self.openai.insert(task_id, snapshot);
                }
            }
            "gemini:video" => {
                if let LocalVideoTaskSnapshot::Gemini(seed) = &snapshot {
                    self.gemini.insert(seed.local_short_id.clone(), snapshot);
                }
            }
            "doubao:video" => {
                if let LocalVideoTaskSnapshot::Doubao(seed) = &snapshot {
                    self.doubao.insert(seed.local_task_id.clone(), snapshot);
                }
            }
            _ => match &snapshot {
                LocalVideoTaskSnapshot::OpenAi(seed) => {
                    self.openai.insert(seed.local_task_id.clone(), snapshot);
                }
                LocalVideoTaskSnapshot::Gemini(seed) => {
                    self.gemini.insert(seed.local_short_id.clone(), snapshot);
                }
                LocalVideoTaskSnapshot::Doubao(seed) => {
                    self.doubao.insert(seed.local_task_id.clone(), snapshot);
                }
            },
        }
    }

    pub fn read_openai(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        self.openai
            .get(task_id)
            .map(LocalVideoTaskSnapshot::read_response)
    }

    pub fn read_gemini(&self, short_id: &str) -> Option<LocalVideoTaskReadResponse> {
        self.gemini
            .get(short_id)
            .map(LocalVideoTaskSnapshot::read_response)
    }

    pub fn read_doubao(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        self.doubao
            .get(task_id)
            .map(LocalVideoTaskSnapshot::read_response)
    }

    pub fn clone_openai(&self, task_id: &str) -> Option<OpenAiVideoTaskSeed> {
        let LocalVideoTaskSnapshot::OpenAi(seed) = self.openai.get(task_id)?.clone() else {
            return None;
        };
        Some(seed)
    }

    pub fn clone_gemini(&self, short_id: &str) -> Option<GeminiVideoTaskSeed> {
        let LocalVideoTaskSnapshot::Gemini(seed) = self.gemini.get(short_id)?.clone() else {
            return None;
        };
        Some(seed)
    }

    pub fn clone_doubao(&self, task_id: &str) -> Option<DoubaoVideoTaskSeed> {
        let LocalVideoTaskSnapshot::Doubao(seed) = self.doubao.get(task_id)?.clone() else {
            return None;
        };
        Some(seed)
    }

    pub fn clone_openai_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        self.openai.get(task_id).cloned()
    }

    pub fn clone_gemini_snapshot(&self, short_id: &str) -> Option<LocalVideoTaskSnapshot> {
        self.gemini.get(short_id).cloned()
    }

    pub fn clone_doubao_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        self.doubao.get(task_id).cloned()
    }

    /// Resolves a native Ark task by the provider-visible ID without making
    /// that external ID a registry key. Ownership is part of the lookup so an
    /// upstream ID can never be used to discover another user's snapshot.
    /// Ambiguous legacy rows fail closed instead of selecting an arbitrary
    /// internal task.
    pub fn clone_doubao_snapshot_for_user_by_upstream_task_id(
        &self,
        user_id: &str,
        upstream_task_id: &str,
    ) -> Option<LocalVideoTaskSnapshot> {
        let user_id = user_id.trim();
        let upstream_task_id = upstream_task_id.trim();
        if user_id.is_empty() || upstream_task_id.is_empty() {
            return None;
        }

        let mut matches = self.doubao.values().filter(|snapshot| {
            let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
                return false;
            };
            seed.persistence
                .client_api_format
                .trim()
                .eq_ignore_ascii_case("doubao:video")
                && snapshot.belongs_to_user(user_id)
                && seed.upstream_task_id.trim() == upstream_task_id
        });
        let snapshot = matches.next()?.clone();
        if matches.next().is_some() {
            return None;
        }
        Some(snapshot)
    }

    pub fn list_active_snapshots(&self, limit: usize) -> Vec<LocalVideoTaskSnapshot> {
        self.openai
            .values()
            .chain(self.gemini.values())
            .chain(self.doubao.values())
            .filter(|snapshot| snapshot.is_active_for_refresh())
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn apply_mutation(&mut self, mutation: LocalVideoTaskRegistryMutation) {
        match mutation {
            LocalVideoTaskRegistryMutation::OpenAiCancelled { task_id } => {
                if let Some(snapshot) = self.openai.get_mut(&task_id) {
                    if snapshot_is_active(snapshot) {
                        mark_snapshot_cancelled(snapshot, current_unix_timestamp_secs());
                    }
                }
            }
            LocalVideoTaskRegistryMutation::OpenAiDeleted { task_id } => {
                if let Some(snapshot) = self.openai.get_mut(&task_id) {
                    set_snapshot_status(snapshot, LocalVideoTaskStatus::Deleted);
                }
            }
            LocalVideoTaskRegistryMutation::GeminiCancelled { short_id } => {
                if let Some(LocalVideoTaskSnapshot::Gemini(seed)) = self.gemini.get_mut(&short_id) {
                    if matches!(
                        seed.status,
                        LocalVideoTaskStatus::Submitted
                            | LocalVideoTaskStatus::Queued
                            | LocalVideoTaskStatus::Processing
                    ) {
                        seed.status = LocalVideoTaskStatus::Cancelled;
                    }
                }
            }
            LocalVideoTaskRegistryMutation::DoubaoCancelled { task_id } => {
                if let Some(snapshot) = self.doubao.get_mut(&task_id) {
                    if snapshot_is_active(snapshot) {
                        mark_snapshot_cancelled(snapshot, current_unix_timestamp_secs());
                    }
                }
            }
            LocalVideoTaskRegistryMutation::DoubaoDeleted { task_id } => {
                if let Some(LocalVideoTaskSnapshot::Doubao(seed)) = self.doubao.get_mut(&task_id) {
                    seed.status = LocalVideoTaskStatus::Deleted;
                }
            }
        }
    }

    pub fn project_openai(&mut self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Some(snapshot) = self.openai.get_mut(task_id) else {
            return false;
        };
        snapshot.apply_provider_body(provider_body);
        true
    }

    pub fn project_gemini(&mut self, short_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Some(snapshot) = self.gemini.get_mut(short_id) else {
            return false;
        };
        snapshot.apply_provider_body(provider_body);
        true
    }

    pub fn project_doubao(&mut self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Some(snapshot) = self.doubao.get_mut(task_id) else {
            return false;
        };
        snapshot.apply_provider_body(provider_body);
        true
    }
}

fn set_snapshot_status(snapshot: &mut LocalVideoTaskSnapshot, status: LocalVideoTaskStatus) {
    match snapshot {
        LocalVideoTaskSnapshot::OpenAi(seed) => seed.status = status,
        LocalVideoTaskSnapshot::Gemini(seed) => seed.status = status,
        LocalVideoTaskSnapshot::Doubao(seed) => seed.status = status,
    }
}

fn snapshot_is_active(snapshot: &LocalVideoTaskSnapshot) -> bool {
    let status = match snapshot {
        LocalVideoTaskSnapshot::OpenAi(seed) => seed.status,
        LocalVideoTaskSnapshot::Gemini(seed) => seed.status,
        LocalVideoTaskSnapshot::Doubao(seed) => seed.status,
    };
    matches!(
        status,
        LocalVideoTaskStatus::Submitted
            | LocalVideoTaskStatus::Queued
            | LocalVideoTaskStatus::Processing
    )
}

fn mark_snapshot_cancelled(snapshot: &mut LocalVideoTaskSnapshot, cancelled_at_unix_secs: u64) {
    set_snapshot_status(snapshot, LocalVideoTaskStatus::Cancelled);
    match snapshot {
        LocalVideoTaskSnapshot::OpenAi(seed) => {
            seed.completed_at_unix_secs = Some(cancelled_at_unix_secs);
        }
        LocalVideoTaskSnapshot::Doubao(seed) => {
            seed.updated_at_unix_secs = Some(cancelled_at_unix_secs);
            seed.completed_at_unix_secs = Some(cancelled_at_unix_secs);
        }
        LocalVideoTaskSnapshot::Gemini(_) => {}
    }
}
