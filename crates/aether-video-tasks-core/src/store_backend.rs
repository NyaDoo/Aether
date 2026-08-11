use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::{
    DoubaoVideoTaskSeed, GeminiVideoTaskSeed, LocalVideoTaskReadResponse,
    LocalVideoTaskRegistryMutation, LocalVideoTaskSnapshot, OpenAiVideoTaskSeed, VideoTaskRegistry,
    VideoTaskStore,
};

#[derive(Debug, Default)]
pub struct InMemoryVideoTaskStore {
    registry: Mutex<VideoTaskRegistry>,
}

#[derive(Debug)]
pub struct FileVideoTaskStore {
    path: PathBuf,
    registry: Mutex<VideoTaskRegistry>,
}

impl VideoTaskStore for InMemoryVideoTaskStore {
    fn insert(&self, snapshot: LocalVideoTaskSnapshot) {
        if let Ok(mut registry) = self.registry.lock() {
            registry.insert(snapshot);
        }
    }

    fn read_openai(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_openai(task_id)
    }

    fn read_gemini(&self, short_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_gemini(short_id)
    }

    fn read_doubao(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_doubao(task_id)
    }

    fn clone_openai(&self, task_id: &str) -> Option<OpenAiVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_openai(task_id)
    }

    fn clone_gemini(&self, short_id: &str) -> Option<GeminiVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_gemini(short_id)
    }

    fn clone_doubao(&self, task_id: &str) -> Option<DoubaoVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_doubao(task_id)
    }

    fn clone_openai_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_openai_snapshot(task_id)
    }

    fn clone_gemini_snapshot(&self, short_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_gemini_snapshot(short_id)
    }

    fn clone_doubao_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_doubao_snapshot(task_id)
    }

    fn list_active_snapshots(&self, limit: usize) -> Vec<LocalVideoTaskSnapshot> {
        let Ok(registry) = self.registry.lock() else {
            return Vec::new();
        };
        registry.list_active_snapshots(limit)
    }

    fn apply_mutation(&self, mutation: LocalVideoTaskRegistryMutation) {
        if let Ok(mut registry) = self.registry.lock() {
            registry.apply_mutation(mutation);
        }
    }

    fn project_openai(&self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Ok(mut registry) = self.registry.lock() else {
            return false;
        };
        registry.project_openai(task_id, provider_body)
    }

    fn project_gemini(&self, short_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Ok(mut registry) = self.registry.lock() else {
            return false;
        };
        registry.project_gemini(short_id, provider_body)
    }

    fn project_doubao(&self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        let Ok(mut registry) = self.registry.lock() else {
            return false;
        };
        registry.project_doubao(task_id, provider_body)
    }
}

impl FileVideoTaskStore {
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let had_persisted_registry = path.exists();
        let registry = Self::load_registry(&path)?;
        let store = Self {
            path,
            registry: Mutex::new(registry),
        };
        // Loading already strips credentials from the in-memory copy. Rewrite
        // an existing legacy file immediately as a one-time on-disk cleanup so
        // revoked secrets are not left behind until some unrelated mutation.
        if had_persisted_registry {
            let registry = store
                .registry
                .lock()
                .map_err(|_| std::io::Error::other("video task registry lock is poisoned"))?;
            store.persist_registry(&registry)?;
        }
        Ok(store)
    }

    fn load_registry(path: &Path) -> std::io::Result<VideoTaskRegistry> {
        if !path.exists() {
            return Ok(VideoTaskRegistry::default());
        }
        let bytes = std::fs::read(path)?;
        if bytes.is_empty() {
            return Ok(VideoTaskRegistry::default());
        }
        let registry: VideoTaskRegistry = serde_json::from_slice(&bytes)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        // Older gateway releases serialized live transport snapshots, including
        // provider and proxy credentials. Never make those historical values
        // authoritative again after restart: current provider configuration is
        // the only source allowed to restore runtime credentials.
        Ok(registry.redacted_for_persistence())
    }

    fn persist_registry(&self, registry: &VideoTaskRegistry) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // The in-memory registry needs live provider credentials, but the file
        // store is task state rather than a secret store.  Serialize a redacted
        // copy so Authorization/API-key and proxy credentials never reach disk.
        let persisted_registry = registry.redacted_for_persistence();
        let bytes = serde_json::to_vec_pretty(&persisted_registry)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        let temp_path = self.path.with_extension("tmp");
        std::fs::write(&temp_path, bytes)?;
        std::fs::rename(temp_path, &self.path)?;
        Ok(())
    }

    fn mutate_registry(&self, mutator: impl FnOnce(&mut VideoTaskRegistry) -> bool) -> bool {
        let Ok(mut registry) = self.registry.lock() else {
            return false;
        };
        if !mutator(&mut registry) {
            return false;
        }
        self.persist_registry(&registry).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    use aether_contracts::ProxySnapshot;
    use serde_json::json;

    use super::*;
    use crate::{LocalVideoTaskPersistence, LocalVideoTaskStatus, LocalVideoTaskTransport};

    #[test]
    fn loading_legacy_file_snapshot_never_restores_transport_credentials() {
        let mut registry = VideoTaskRegistry::default();
        registry.insert(LocalVideoTaskSnapshot::Doubao(DoubaoVideoTaskSeed {
            local_task_id: "cgt-local-1".to_string(),
            upstream_task_id: "cgt-upstream-1".to_string(),
            created_at_unix_secs: 1,
            updated_at_unix_secs: None,
            user_id: Some("user-1".to_string()),
            api_key_id: Some("caller-key-1".to_string()),
            model: Some("doubao-seedance-1-0-pro-250528".to_string()),
            prompt: Some("test".to_string()),
            resolution: None,
            ratio: None,
            duration_seconds: Some(5),
            seed: None,
            frames: None,
            frames_per_second: None,
            status: LocalVideoTaskStatus::Queued,
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
                original_request_body: json!({"model": "endpoint-1"}),
                format_converted: false,
            },
            transport: LocalVideoTaskTransport {
                upstream_base_url:
                    "https://legacy-user:legacy-password@ark.example.com?token=legacy-secret"
                        .to_string(),
                provider_name: Some("ark".to_string()),
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "provider-key-1".to_string(),
                headers: BTreeMap::from([(
                    "authorization".to_string(),
                    "Bearer revoked-provider-secret".to_string(),
                )]),
                content_type: Some("application/json".to_string()),
                model_name: None,
                proxy: Some(ProxySnapshot {
                    enabled: Some(true),
                    mode: Some("manual".to_string()),
                    node_id: None,
                    label: None,
                    url: Some("http://proxy-user:proxy-password@proxy.example.com".to_string()),
                    extra: Some(json!({"password": "legacy-proxy-secret"})),
                }),
                transport_profile: None,
                timeouts: None,
            },
        }));

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "aether-video-task-legacy-{}-{unique}.json",
            std::process::id()
        ));
        std::fs::write(
            &path,
            serde_json::to_vec(&registry).expect("serialize legacy registry"),
        )
        .expect("write legacy registry");

        let store = FileVideoTaskStore::new(&path).expect("load legacy registry");
        let persisted = std::fs::read_to_string(&path).expect("read cleaned registry");
        assert!(!persisted.contains("revoked-provider-secret"));
        assert!(!persisted.contains("legacy-password"));
        assert!(!persisted.contains("legacy-proxy-secret"));
        let snapshot = store
            .clone_doubao_snapshot("cgt-local-1")
            .expect("legacy task should remain readable");
        let LocalVideoTaskSnapshot::Doubao(seed) = snapshot else {
            panic!("expected Doubao snapshot");
        };
        assert!(seed.transport.headers.is_empty());
        assert_eq!(seed.transport.upstream_base_url, "https://ark.example.com");
        assert!(seed.transport.proxy.is_none());
        assert!(seed.transport.transport_profile.is_none());
        assert!(!LocalVideoTaskSnapshot::Doubao(seed).has_runtime_auth_headers());

        let _ = std::fs::remove_file(path);
    }
}

impl VideoTaskStore for FileVideoTaskStore {
    fn insert(&self, snapshot: LocalVideoTaskSnapshot) {
        let _ = self.mutate_registry(|registry| {
            registry.insert(snapshot);
            true
        });
    }

    fn read_openai(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_openai(task_id)
    }

    fn read_gemini(&self, short_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_gemini(short_id)
    }

    fn read_doubao(&self, task_id: &str) -> Option<LocalVideoTaskReadResponse> {
        let registry = self.registry.lock().ok()?;
        registry.read_doubao(task_id)
    }

    fn clone_openai(&self, task_id: &str) -> Option<OpenAiVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_openai(task_id)
    }

    fn clone_gemini(&self, short_id: &str) -> Option<GeminiVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_gemini(short_id)
    }

    fn clone_doubao(&self, task_id: &str) -> Option<DoubaoVideoTaskSeed> {
        let registry = self.registry.lock().ok()?;
        registry.clone_doubao(task_id)
    }

    fn clone_openai_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_openai_snapshot(task_id)
    }

    fn clone_gemini_snapshot(&self, short_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_gemini_snapshot(short_id)
    }

    fn clone_doubao_snapshot(&self, task_id: &str) -> Option<LocalVideoTaskSnapshot> {
        let registry = self.registry.lock().ok()?;
        registry.clone_doubao_snapshot(task_id)
    }

    fn list_active_snapshots(&self, limit: usize) -> Vec<LocalVideoTaskSnapshot> {
        let Ok(registry) = self.registry.lock() else {
            return Vec::new();
        };
        registry.list_active_snapshots(limit)
    }

    fn apply_mutation(&self, mutation: LocalVideoTaskRegistryMutation) {
        let _ = self.mutate_registry(|registry| {
            registry.apply_mutation(mutation);
            true
        });
    }

    fn project_openai(&self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        self.mutate_registry(|registry| registry.project_openai(task_id, provider_body))
    }

    fn project_gemini(&self, short_id: &str, provider_body: &Map<String, Value>) -> bool {
        self.mutate_registry(|registry| registry.project_gemini(short_id, provider_body))
    }

    fn project_doubao(&self, task_id: &str, provider_body: &Map<String, Value>) -> bool {
        self.mutate_registry(|registry| registry.project_doubao(task_id, provider_body))
    }
}
