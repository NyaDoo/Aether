use aether_data_contracts::repository::video_tasks::{StoredVideoTask, VideoTaskLookupKey};
use aether_data_contracts::DataLayerError;
use async_trait::async_trait;

use crate::{
    map_doubao_stored_task_to_read_response, map_gemini_stored_task_to_read_response,
    map_openai_stored_task_to_read_response, resolve_video_task_read_lookup_key,
    LocalVideoTaskReadResponse,
};

#[async_trait]
pub trait StoredVideoTaskReadSide: Send + Sync {
    async fn find_stored_video_task(
        &self,
        key: VideoTaskLookupKey<'_>,
    ) -> Result<Option<StoredVideoTask>, DataLayerError>;
}

pub async fn read_data_backed_video_task_response(
    state: &impl StoredVideoTaskReadSide,
    route_family: Option<&str>,
    request_path: &str,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    read_data_backed_video_task_response_inner(state, route_family, request_path, None).await
}

pub async fn read_data_backed_video_task_response_for_user(
    state: &impl StoredVideoTaskReadSide,
    route_family: Option<&str>,
    request_path: &str,
    user_id: &str,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    let Some(user_id) = non_empty_user_id(user_id) else {
        return Ok(None);
    };
    read_data_backed_video_task_response_inner(state, route_family, request_path, Some(user_id))
        .await
}

async fn read_data_backed_video_task_response_inner(
    state: &impl StoredVideoTaskReadSide,
    route_family: Option<&str>,
    request_path: &str,
    user_id: Option<&str>,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    match route_family {
        Some("openai") => read_openai_video_task_response(state, request_path, user_id).await,
        Some("gemini") => read_gemini_video_task_response(state, request_path, user_id).await,
        Some("doubao") => read_doubao_video_task_response(state, request_path, user_id).await,
        _ => Ok(None),
    }
}

async fn read_openai_video_task_response(
    state: &impl StoredVideoTaskReadSide,
    request_path: &str,
    user_id: Option<&str>,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    let Some(lookup) = resolve_video_task_read_lookup_key(Some("openai"), request_path) else {
        return Ok(None);
    };

    let Some(task) = state.find_stored_video_task(lookup).await? else {
        return Ok(None);
    };
    if !task_belongs_to_user(&task, user_id) {
        return Ok(None);
    }

    if !matches!(task.client_api_format.as_deref(), Some("openai:video"))
        || !matches!(
            task.provider_api_format.as_deref(),
            Some("openai:video" | "doubao:video")
        )
    {
        return Ok(None);
    }

    Ok(Some(map_openai_stored_task_to_read_response(task)))
}

async fn read_gemini_video_task_response(
    state: &impl StoredVideoTaskReadSide,
    request_path: &str,
    user_id: Option<&str>,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    let Some(lookup) = resolve_video_task_read_lookup_key(Some("gemini"), request_path) else {
        return Ok(None);
    };

    let Some(task) = state.find_stored_video_task(lookup).await? else {
        return Ok(None);
    };
    if !task_belongs_to_user(&task, user_id) {
        return Ok(None);
    }

    if !matches!(task.client_api_format.as_deref(), Some("gemini:video"))
        || !matches!(task.provider_api_format.as_deref(), Some("gemini:video"))
    {
        return Ok(None);
    }

    Ok(Some(map_gemini_stored_task_to_read_response(task)))
}

async fn read_doubao_video_task_response(
    state: &impl StoredVideoTaskReadSide,
    request_path: &str,
    user_id: Option<&str>,
) -> Result<Option<LocalVideoTaskReadResponse>, DataLayerError> {
    let Some(parsed_lookup) = resolve_video_task_read_lookup_key(Some("doubao"), request_path)
    else {
        return Ok(None);
    };
    let (lookup, fallback_lookup) = doubao_read_lookups_for_user(parsed_lookup, user_id);

    let task = match state.find_stored_video_task(lookup).await? {
        Some(task) => Some(task),
        None => match fallback_lookup {
            Some(fallback_lookup) => state.find_stored_video_task(fallback_lookup).await?,
            None => None,
        },
    };
    let Some(task) = task else {
        return Ok(None);
    };
    if !task_belongs_to_user(&task, user_id) {
        return Ok(None);
    }

    if !matches!(task.client_api_format.as_deref(), Some("doubao:video"))
        || !matches!(task.provider_api_format.as_deref(), Some("doubao:video"))
    {
        return Ok(None);
    }

    Ok(Some(map_doubao_stored_task_to_read_response(task)))
}

fn doubao_read_lookups_for_user<'a>(
    lookup: VideoTaskLookupKey<'a>,
    user_id: Option<&'a str>,
) -> (VideoTaskLookupKey<'a>, Option<VideoTaskLookupKey<'a>>) {
    match (lookup, user_id) {
        (local_fallback @ VideoTaskLookupKey::Id(external_task_id), Some(user_id)) => (
            VideoTaskLookupKey::UserExternal {
                user_id,
                external_task_id,
            },
            Some(local_fallback),
        ),
        (lookup, _) => (lookup, None),
    }
}

fn task_belongs_to_user(task: &StoredVideoTask, user_id: Option<&str>) -> bool {
    // Only the private inner call made by the legacy internal/admin wrapper is
    // unscoped. The public `_for_user` entry point cannot construct `None`.
    let Some(user_id) = user_id else {
        return true;
    };
    task.user_id.as_deref().map(str::trim) == Some(user_id)
}

fn non_empty_user_id(user_id: &str) -> Option<&str> {
    let user_id = user_id.trim();
    (!user_id.is_empty()).then_some(user_id)
}

#[cfg(test)]
mod tests {
    use super::doubao_read_lookups_for_user;
    use aether_data_contracts::repository::video_tasks::VideoTaskLookupKey;

    #[test]
    fn scopes_authenticated_doubao_reads_by_owner_and_external_id() {
        assert_eq!(
            doubao_read_lookups_for_user(
                VideoTaskLookupKey::Id("cgt-upstream-123"),
                Some("user-123"),
            ),
            (
                VideoTaskLookupKey::UserExternal {
                    user_id: "user-123",
                    external_task_id: "cgt-upstream-123",
                },
                Some(VideoTaskLookupKey::Id("cgt-upstream-123")),
            )
        );
    }

    #[test]
    fn keeps_unscoped_internal_doubao_reads_on_the_local_id() {
        assert_eq!(
            doubao_read_lookups_for_user(VideoTaskLookupKey::Id("task-local-123"), None),
            (VideoTaskLookupKey::Id("task-local-123"), None)
        );
    }
}
