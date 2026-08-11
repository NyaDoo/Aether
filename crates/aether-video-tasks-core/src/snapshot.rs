use aether_data_contracts::repository::video_tasks::{StoredVideoTask, UpsertVideoTask};
use serde_json::{json, Map, Value};

use crate::{
    current_unix_timestamp_secs, doubao_content_prompt, doubao_prompt_text,
    doubao_string_parameter, doubao_u32_parameter, local_status_from_stored, non_empty_owned,
    request_body_string, DoubaoVideoTaskSeed, GeminiVideoTaskSeed, LocalVideoTaskPersistence,
    LocalVideoTaskReadResponse, LocalVideoTaskSnapshot, LocalVideoTaskStatus,
    LocalVideoTaskTransport, OpenAiVideoTaskSeed,
};

impl LocalVideoTaskSnapshot {
    pub fn client_api_format(&self) -> &str {
        match self {
            Self::OpenAi(seed) => seed.persistence.client_api_format.as_str(),
            Self::Gemini(seed) => seed.persistence.client_api_format.as_str(),
            Self::Doubao(seed) => seed.persistence.client_api_format.as_str(),
        }
    }

    pub fn provider_api_format(&self) -> &str {
        match self {
            Self::OpenAi(seed) => seed.persistence.provider_api_format.as_str(),
            Self::Gemini(seed) => seed.persistence.provider_api_format.as_str(),
            Self::Doubao(seed) => seed.persistence.provider_api_format.as_str(),
        }
    }

    /// Snapshot persisted with a task record. Provider credentials are runtime
    /// configuration, not task data; they are rehydrated from the provider key
    /// store before any follow-up request.
    pub fn redacted_for_persistence(&self) -> Self {
        let mut snapshot = self.clone();
        let transport = match &mut snapshot {
            Self::OpenAi(seed) => &mut seed.transport,
            Self::Gemini(seed) => &mut seed.transport,
            Self::Doubao(seed) => &mut seed.transport,
        };
        transport.headers.clear();
        transport.upstream_base_url = redacted_transport_base_url(&transport.upstream_base_url);
        // Proxy URLs can contain userinfo and profile extras can contain
        // transport-specific credentials. Both are rebuilt from provider
        // configuration before a follow-up request.
        transport.proxy = None;
        transport.transport_profile = None;
        snapshot
    }

    pub fn has_runtime_auth_headers(&self) -> bool {
        let (provider_api_format, headers) = match self {
            Self::OpenAi(seed) => (
                seed.persistence.provider_api_format.as_str(),
                &seed.transport.headers,
            ),
            Self::Gemini(seed) => (
                seed.persistence.provider_api_format.as_str(),
                &seed.transport.headers,
            ),
            Self::Doubao(seed) => (
                seed.persistence.provider_api_format.as_str(),
                &seed.transport.headers,
            ),
        };
        match provider_api_format.trim().to_ascii_lowercase().as_str() {
            "openai:video" | "doubao:video" => bearer_header_is_present(headers),
            "gemini:video" => {
                non_empty_header(headers, "x-goog-api-key").is_some()
                    || bearer_header_is_present(headers)
            }
            _ => false,
        }
    }

    pub fn belongs_to_user(&self, user_id: &str) -> bool {
        let user_id = user_id.trim();
        if user_id.is_empty() {
            return false;
        }
        let owner = match self {
            Self::OpenAi(seed) => seed.user_id.as_deref(),
            Self::Gemini(seed) => seed.user_id.as_deref(),
            Self::Doubao(seed) => seed.user_id.as_deref(),
        };
        owner.map(str::trim) == Some(user_id)
    }

    pub fn to_upsert_record(&self) -> UpsertVideoTask {
        match self {
            Self::OpenAi(seed) => seed.to_upsert_record(),
            Self::Gemini(seed) => seed.to_upsert_record(),
            Self::Doubao(seed) => seed.to_upsert_record(),
        }
    }

    pub fn from_stored_task(task: &StoredVideoTask) -> Option<Self> {
        let snapshot = task
            .request_metadata
            .as_ref()
            .and_then(|metadata| metadata.get("rust_local_snapshot"))
            .cloned()
            .and_then(|value| serde_json::from_value::<LocalVideoTaskSnapshot>(value).ok())?;
        // Database metadata may have been written by a release that persisted
        // live Authorization or proxy credentials. A stored row is never a
        // credential source: callers that need a follow-up transport must use
        // `from_stored_task_with_transport` with values rebuilt from the
        // current provider configuration.
        Some(reconcile_snapshot_with_stored_task(snapshot, task).redacted_for_persistence())
    }

    pub fn from_stored_task_with_transport(
        task: &StoredVideoTask,
        transport: LocalVideoTaskTransport,
    ) -> Option<Self> {
        let provider_api_format = task.provider_api_format.as_deref()?.trim();

        // Prefer the persisted snapshot because it carries provider-only
        // fields that have no dedicated database column (for example Doubao's
        // last-frame URL and token usage). Refresh only the transport, whose
        // credentials/configuration may have changed since the snapshot was
        // stored. A mismatched snapshot variant is ignored and reconstructed
        // from columns below instead.
        if let Some(snapshot) = Self::from_stored_task(task) {
            match (provider_api_format, snapshot) {
                ("openai:video", Self::OpenAi(mut seed)) => {
                    seed.transport = transport;
                    return Some(Self::OpenAi(seed));
                }
                ("gemini:video", Self::Gemini(mut seed)) => {
                    seed.transport = transport;
                    return Some(Self::Gemini(seed));
                }
                ("doubao:video", Self::Doubao(mut seed)) => {
                    seed.transport = transport;
                    return Some(Self::Doubao(seed));
                }
                _ => {}
            }
        }

        let persistence = LocalVideoTaskPersistence::from_stored_task(task)?;

        match provider_api_format {
            "openai:video" => {
                let upstream_task_id = non_empty_owned(task.external_task_id.as_ref())?;
                Some(Self::OpenAi(OpenAiVideoTaskSeed {
                    local_task_id: task.id.clone(),
                    upstream_task_id,
                    created_at_unix_ms: task.created_at_unix_ms,
                    user_id: task.user_id.clone(),
                    api_key_id: task.api_key_id.clone(),
                    model: non_empty_owned(task.model.as_ref()),
                    prompt: non_empty_owned(task.prompt.as_ref()).or_else(|| {
                        request_body_string(&persistence.original_request_body, "prompt")
                    }),
                    size: non_empty_owned(task.size.as_ref()).or_else(|| {
                        request_body_string(&persistence.original_request_body, "size")
                    }),
                    seconds: task
                        .duration_seconds
                        .map(|value| value.to_string())
                        .or_else(|| {
                            request_body_string(&persistence.original_request_body, "seconds")
                        }),
                    remixed_from_video_id: request_body_string(
                        &persistence.original_request_body,
                        "remix_video_id",
                    )
                    .or_else(|| {
                        request_body_string(
                            &persistence.original_request_body,
                            "remixed_from_video_id",
                        )
                    }),
                    status: local_status_from_stored(task.status),
                    progress_percent: task.progress_percent,
                    completed_at_unix_secs: task.completed_at_unix_secs,
                    expires_at_unix_secs: None,
                    error_code: task.error_code.clone(),
                    error_message: task.error_message.clone(),
                    video_url: non_empty_owned(task.video_url.as_ref()),
                    persistence,
                    transport,
                }))
            }
            "gemini:video" => {
                let local_short_id =
                    non_empty_owned(task.short_id.as_ref()).unwrap_or_else(|| task.id.clone());
                let upstream_operation_name = non_empty_owned(task.external_task_id.as_ref())?;
                let model = non_empty_owned(task.model.as_ref())?;
                Some(Self::Gemini(GeminiVideoTaskSeed {
                    local_short_id,
                    upstream_operation_name,
                    user_id: task.user_id.clone(),
                    api_key_id: task.api_key_id.clone(),
                    model,
                    status: local_status_from_stored(task.status),
                    progress_percent: task.progress_percent,
                    error_code: task.error_code.clone(),
                    error_message: task.error_message.clone(),
                    metadata: Value::Object(Map::new()),
                    persistence,
                    transport,
                }))
            }
            "doubao:video" => {
                let upstream_task_id = non_empty_owned(task.external_task_id.as_ref())?;
                let request_body = &persistence.original_request_body;
                let seed = stored_doubao_i32(task, request_body, "seed");
                let frames = stored_doubao_i32(task, request_body, "frames");
                let frames_per_second = stored_doubao_i32(task, request_body, "framespersecond");
                let duration_seconds = if frames.is_some() {
                    None
                } else {
                    stored_doubao_u32(task, "duration")
                        .or(task.duration_seconds)
                        .or_else(|| {
                            doubao_u32_parameter(request_body, "duration", &["dur", "duration"])
                        })
                        .or_else(|| crate::request_body_u32(request_body, "seconds"))
                };
                Some(Self::Doubao(DoubaoVideoTaskSeed {
                    local_task_id: task.id.clone(),
                    upstream_task_id,
                    created_at_unix_secs: task.created_at_unix_ms,
                    updated_at_unix_secs: Some(task.updated_at_unix_secs),
                    user_id: task.user_id.clone(),
                    api_key_id: task.api_key_id.clone(),
                    model: non_empty_owned(task.model.as_ref()),
                    prompt: non_empty_owned(task.prompt.as_ref())
                        .or_else(|| {
                            doubao_content_prompt(request_body)
                                .map(|prompt| doubao_prompt_text(&prompt))
                        })
                        .or_else(|| request_body_string(request_body, "prompt")),
                    resolution: non_empty_owned(task.resolution.as_ref()).or_else(|| {
                        doubao_string_parameter(request_body, "resolution", &["rs", "resolution"])
                    }),
                    ratio: non_empty_owned(task.aspect_ratio.as_ref()).or_else(|| {
                        doubao_string_parameter(request_body, "ratio", &["rt", "ratio"])
                    }),
                    duration_seconds,
                    seed,
                    frames,
                    frames_per_second,
                    status: local_status_from_stored(task.status),
                    progress_percent: task.progress_percent,
                    completed_at_unix_secs: task.completed_at_unix_secs,
                    error_code: task.error_code.clone(),
                    error_message: task.error_message.clone(),
                    video_url: non_empty_owned(task.video_url.as_ref()),
                    // Only the video URL is persisted as a column; the last frame
                    // and token usage are recovered from the snapshot when present.
                    last_frame_url: None,
                    completion_tokens: None,
                    total_tokens: None,
                    persistence,
                    transport,
                }))
            }
            _ => None,
        }
    }

    pub fn read_response(&self) -> LocalVideoTaskReadResponse {
        self.read_response_at(current_unix_timestamp_secs())
    }

    pub fn read_response_at(&self, now_unix_secs: u64) -> LocalVideoTaskReadResponse {
        match self {
            Self::OpenAi(seed) => match seed.status {
                LocalVideoTaskStatus::Cancelled => LocalVideoTaskReadResponse {
                    status_code: 404,
                    body_json: json!({"detail": "Video task was cancelled"}),
                },
                LocalVideoTaskStatus::Deleted => LocalVideoTaskReadResponse {
                    status_code: 404,
                    body_json: json!({"detail": "Video task not found"}),
                },
                _ => LocalVideoTaskReadResponse {
                    status_code: 200,
                    body_json: seed.client_body_json(),
                },
            },
            Self::Gemini(seed) => match seed.status {
                LocalVideoTaskStatus::Cancelled => LocalVideoTaskReadResponse {
                    status_code: 404,
                    body_json: json!({"detail": "Video task was cancelled"}),
                },
                LocalVideoTaskStatus::Deleted => LocalVideoTaskReadResponse {
                    status_code: 404,
                    body_json: json!({"detail": "Video task not found"}),
                },
                _ => LocalVideoTaskReadResponse {
                    status_code: 200,
                    body_json: seed.client_body_json(),
                },
            },
            Self::Doubao(seed)
                if seed
                    .persistence
                    .client_api_format
                    .trim()
                    .eq_ignore_ascii_case("openai:video") =>
            {
                match seed.status {
                    LocalVideoTaskStatus::Cancelled => LocalVideoTaskReadResponse {
                        status_code: 404,
                        body_json: json!({"detail": "Video task was cancelled"}),
                    },
                    LocalVideoTaskStatus::Deleted => LocalVideoTaskReadResponse {
                        status_code: 404,
                        body_json: json!({"detail": "Video task not found"}),
                    },
                    _ => LocalVideoTaskReadResponse {
                        status_code: 200,
                        body_json: seed.openai_client_body_json(),
                    },
                }
            }
            Self::Doubao(seed) => match seed.status {
                // Deleted tasks are gone; cancelled tasks remain queryable for
                // the provider's retention window and are represented below.
                LocalVideoTaskStatus::Deleted => LocalVideoTaskReadResponse {
                    status_code: 404,
                    body_json: json!({
                        "error": {
                            "code": "NotFound",
                            "message": "The requested generation task was not found.",
                        }
                    }),
                },
                LocalVideoTaskStatus::Cancelled if !seed.is_visible_at(now_unix_secs) => {
                    LocalVideoTaskReadResponse {
                        status_code: 404,
                        body_json: json!({
                            "error": {
                                "code": "NotFound",
                                "message": "The requested generation task was not found.",
                            }
                        }),
                    }
                }
                _ => LocalVideoTaskReadResponse {
                    status_code: 200,
                    body_json: seed.client_body_json(),
                },
            },
        }
    }

    pub fn is_active_for_refresh(&self) -> bool {
        let status = match self {
            Self::OpenAi(seed) => seed.status,
            Self::Gemini(seed) => seed.status,
            Self::Doubao(seed) => seed.status,
        };
        matches!(
            status,
            LocalVideoTaskStatus::Submitted
                | LocalVideoTaskStatus::Queued
                | LocalVideoTaskStatus::Processing
        )
    }

    pub fn apply_provider_body(&mut self, provider_body: &Map<String, Value>) {
        match self {
            Self::OpenAi(seed) => seed.apply_provider_body(provider_body),
            Self::Gemini(seed) => seed.apply_provider_body(provider_body),
            Self::Doubao(seed) => seed.apply_provider_body(provider_body),
        }
    }

    pub fn provider_name(&self) -> Option<&str> {
        match self {
            Self::OpenAi(seed) => seed.transport.provider_name.as_deref(),
            Self::Gemini(seed) => seed.transport.provider_name.as_deref(),
            Self::Doubao(seed) => seed.transport.provider_name.as_deref(),
        }
    }

    pub fn provider_model_name(&self) -> Option<&str> {
        match self {
            Self::OpenAi(seed) => seed
                .model
                .as_deref()
                .or(seed.transport.model_name.as_deref()),
            Self::Gemini(seed) => Some(seed.model.as_str()),
            Self::Doubao(seed) => seed
                .model
                .as_deref()
                .or(seed.transport.model_name.as_deref()),
        }
        .map(str::trim)
        .filter(|value| !value.is_empty())
    }

    /// Token usage reported by the provider, for surfaces that bill by tokens.
    pub fn usage_tokens(&self) -> Option<(u64, u64)> {
        match self {
            Self::Doubao(seed) => {
                let completion_tokens = seed.completion_tokens?;
                Some((
                    completion_tokens,
                    seed.total_tokens.unwrap_or(completion_tokens),
                ))
            }
            Self::OpenAi(_) | Self::Gemini(_) => None,
        }
    }
}

fn redacted_transport_base_url(value: &str) -> String {
    let Ok(mut url) = url::Url::parse(value.trim()) else {
        return String::new();
    };
    let _ = url.set_username("");
    let _ = url.set_password(None);
    url.set_query(None);
    url.set_fragment(None);
    url.to_string().trim_end_matches('/').to_string()
}

fn non_empty_header<'a>(
    headers: &'a std::collections::BTreeMap<String, String>,
    name: &str,
) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.trim())
        .filter(|value| !value.is_empty())
}

fn bearer_header_is_present(headers: &std::collections::BTreeMap<String, String>) -> bool {
    non_empty_header(headers, "authorization").is_some_and(|value| {
        let Some((scheme, token)) = value.split_once(char::is_whitespace) else {
            return false;
        };
        scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()
    })
}

fn reconcile_snapshot_with_stored_task(
    mut snapshot: LocalVideoTaskSnapshot,
    task: &StoredVideoTask,
) -> LocalVideoTaskSnapshot {
    let status = local_status_from_stored(task.status);
    match &mut snapshot {
        LocalVideoTaskSnapshot::OpenAi(seed) => {
            if task.user_id.is_some() {
                seed.user_id = task.user_id.clone();
            }
            if task.api_key_id.is_some() {
                seed.api_key_id = task.api_key_id.clone();
            }
            seed.status = status;
            seed.progress_percent = task.progress_percent;
            if task.completed_at_unix_secs.is_some() {
                seed.completed_at_unix_secs = task.completed_at_unix_secs;
            }
            if task.error_code.is_some() {
                seed.error_code = task.error_code.clone();
            }
            if task.error_message.is_some() {
                seed.error_message = task.error_message.clone();
            }
        }
        LocalVideoTaskSnapshot::Gemini(seed) => {
            if task.user_id.is_some() {
                seed.user_id = task.user_id.clone();
            }
            if task.api_key_id.is_some() {
                seed.api_key_id = task.api_key_id.clone();
            }
            seed.status = status;
            seed.progress_percent = task.progress_percent;
            if task.error_code.is_some() {
                seed.error_code = task.error_code.clone();
            }
            if task.error_message.is_some() {
                seed.error_message = task.error_message.clone();
            }
        }
        LocalVideoTaskSnapshot::Doubao(seed) => {
            let recovered_seed =
                stored_doubao_i32(task, &seed.persistence.original_request_body, "seed");
            let recovered_frames =
                stored_doubao_i32(task, &seed.persistence.original_request_body, "frames");
            let recovered_frames_per_second = stored_doubao_i32(
                task,
                &seed.persistence.original_request_body,
                "framespersecond",
            );
            if task.user_id.is_some() {
                seed.user_id = task.user_id.clone();
            }
            if task.api_key_id.is_some() {
                seed.api_key_id = task.api_key_id.clone();
            }
            seed.status = status;
            seed.progress_percent = task.progress_percent;
            if task.completed_at_unix_secs.is_some() {
                seed.completed_at_unix_secs = task.completed_at_unix_secs;
            }
            if task.error_code.is_some() {
                seed.error_code = task.error_code.clone();
            }
            if task.error_message.is_some() {
                seed.error_message = task.error_message.clone();
            }
            if task.updated_at_unix_secs > 0 {
                seed.updated_at_unix_secs = Some(task.updated_at_unix_secs);
            }
            seed.seed = seed.seed.or(recovered_seed);
            seed.frames = seed.frames.or(recovered_frames);
            seed.frames_per_second = seed.frames_per_second.or(recovered_frames_per_second);
            if seed.frames.is_some() {
                seed.duration_seconds = None;
            } else if seed.duration_seconds.is_none() {
                seed.duration_seconds = stored_doubao_u32(task, "duration")
                    .or(task.duration_seconds)
                    .or_else(|| {
                        doubao_u32_parameter(
                            &seed.persistence.original_request_body,
                            "duration",
                            &["dur", "duration"],
                        )
                    });
            }
        }
    }
    snapshot
}

fn stored_poll_raw_response(task: &StoredVideoTask) -> Option<&Map<String, Value>> {
    task.request_metadata
        .as_ref()
        .and_then(|metadata| metadata.get("poll_raw_response"))
        .and_then(Value::as_object)
}

fn stored_doubao_i32(task: &StoredVideoTask, request_body: &Value, key: &str) -> Option<i32> {
    stored_poll_raw_response(task)
        .and_then(|body| body.get(key))
        .and_then(crate::util::value_i32)
        .or_else(|| request_body.get(key).and_then(crate::util::value_i32))
}

fn stored_doubao_u32(task: &StoredVideoTask, key: &str) -> Option<u32> {
    stored_poll_raw_response(task)
        .and_then(|body| body.get(key))
        .and_then(crate::util::value_u64)
        .and_then(|value| u32::try_from(value).ok())
}
