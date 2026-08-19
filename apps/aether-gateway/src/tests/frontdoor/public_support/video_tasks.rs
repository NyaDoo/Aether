use super::*;

use aether_data::repository::video_tasks::InMemoryVideoTaskRepository;
use aether_data_contracts::repository::video_tasks::{
    UpsertVideoTask, VideoTaskLookupKey, VideoTaskReadRepository, VideoTaskStatus,
    VideoTaskWriteRepository,
};

fn sample_user_video_task(id: &str, user_id: &str, status: VideoTaskStatus) -> UpsertVideoTask {
    UpsertVideoTask {
        id: id.to_string(),
        short_id: Some(format!("short-{id}")),
        request_id: format!("request-{id}"),
        user_id: Some(user_id.to_string()),
        api_key_id: Some(format!("api-key-{id}")),
        username: Some(user_id.to_string()),
        api_key_name: Some("primary".to_string()),
        external_task_id: Some(format!("external-{id}")),
        provider_id: Some("provider-secret".to_string()),
        endpoint_id: Some("endpoint-secret".to_string()),
        key_id: Some("key-secret".to_string()),
        client_api_format: Some("openai:video".to_string()),
        provider_api_format: Some("openai:video".to_string()),
        format_converted: false,
        model: Some("seedance-user-model".to_string()),
        prompt: Some(format!("prompt-{id}")),
        original_request_body: Some(json!({ "provider_secret": "must-not-leak" })),
        duration_seconds: Some(5),
        resolution: Some("720p".to_string()),
        aspect_ratio: Some("16:9".to_string()),
        size: Some("1280x720".to_string()),
        status,
        progress_percent: if status == VideoTaskStatus::Completed {
            100
        } else {
            10
        },
        progress_message: None,
        retry_count: 0,
        poll_interval_seconds: 10,
        next_poll_at_unix_secs: None,
        poll_count: 1,
        max_poll_count: 100,
        created_at_unix_ms: 1_710_000_000,
        submitted_at_unix_secs: Some(1_710_000_001),
        completed_at_unix_secs: (status == VideoTaskStatus::Completed).then_some(1_710_000_010),
        updated_at_unix_secs: 1_710_000_010,
        error_code: None,
        error_message: None,
        video_url: Some(format!("https://provider.invalid/{id}-signed.mp4")),
        request_metadata: Some(json!({
            "global_model_name": "seedance-global",
            "credential": "must-not-leak"
        })),
    }
}

#[tokio::test]
async fn users_me_video_tasks_are_owner_scoped_and_deleted_tasks_stay_hidden() {
    let now = Utc::now();
    let user = sample_auth_user(now);
    let user_id = user.id.clone();
    let session_id = "session-users-me-video-tasks";
    let device_id = "device-users-me-video-tasks";
    let access_token = build_test_auth_token(
        "access",
        serde_json::Map::from_iter([
            ("user_id".to_string(), json!(user.id)),
            ("role".to_string(), json!(user.role)),
            (
                "created_at".to_string(),
                json!(user.created_at.map(|value| value.to_rfc3339())),
            ),
            ("session_id".to_string(), json!(session_id)),
        ]),
        now + chrono::Duration::hours(1),
    );
    let repository = Arc::new(InMemoryVideoTaskRepository::default());
    for task in [
        sample_user_video_task("owned", &user_id, VideoTaskStatus::Completed),
        sample_user_video_task("foreign", "user-auth-2", VideoTaskStatus::Processing),
        sample_user_video_task("deleted", &user_id, VideoTaskStatus::Deleted),
    ] {
        repository
            .upsert(task)
            .await
            .expect("video task should persist");
    }
    let user_repository = Arc::new(InMemoryUserReadRepository::seed_auth_users(vec![user]));
    let data_state =
        GatewayDataState::with_video_task_repository_for_tests(Arc::clone(&repository))
            .with_user_reader(user_repository);
    let state = AppState::new()
        .expect("gateway should build")
        .with_data_state_for_tests(data_state)
        .with_auth_sessions_for_tests([sample_auth_session(
            &user_id,
            session_id,
            device_id,
            "refresh-users-me-video-tasks",
            now,
        )]);
    let gateway = build_router_with_state(state);
    let (gateway_url, gateway_handle) = start_server(gateway).await;
    let client = reqwest::Client::new();

    let authenticated_get = |path: &str| {
        client
            .get(format!("{gateway_url}{path}"))
            .header("authorization", format!("Bearer {access_token}"))
            .header("x-client-device-id", device_id)
            .header("user-agent", "AetherTest/1.0")
    };

    let list_response = authenticated_get("/api/users/me/video-tasks")
        .send()
        .await
        .expect("list request should succeed");
    assert_eq!(list_response.status(), StatusCode::OK);
    let list_payload: serde_json::Value =
        list_response.json().await.expect("list body should parse");
    assert_eq!(list_payload["total"], 1);
    assert_eq!(list_payload["items"][0]["id"], "owned");
    assert_eq!(list_payload["items"][0]["video_available"], true);
    for forbidden in [
        "user_id",
        "api_key_id",
        "external_task_id",
        "provider_id",
        "endpoint_id",
        "key_id",
        "video_url",
        "original_request_body",
        "request_metadata",
        "actual_cost",
    ] {
        assert!(
            list_payload["items"][0].get(forbidden).is_none(),
            "self-service item must not expose {forbidden}"
        );
    }

    let stats_response = authenticated_get("/api/users/me/video-tasks/stats")
        .send()
        .await
        .expect("stats request should succeed");
    assert_eq!(stats_response.status(), StatusCode::OK);
    let stats_payload: serde_json::Value = stats_response
        .json()
        .await
        .expect("stats body should parse");
    assert_eq!(stats_payload["total"], 1);
    assert_eq!(stats_payload["by_status"]["completed"], 1);

    let detail_response = authenticated_get("/api/users/me/video-tasks/owned")
        .send()
        .await
        .expect("detail request should succeed");
    assert_eq!(detail_response.status(), StatusCode::OK);

    for path in [
        "/api/users/me/video-tasks/foreign",
        "/api/users/me/video-tasks/foreign/video",
        "/api/users/me/video-tasks/deleted",
        "/api/users/me/video-tasks/deleted/video",
    ] {
        let response = authenticated_get(path)
            .send()
            .await
            .expect("foreign/deleted request should succeed");
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "path={path}");
        if path.ends_with("/video") {
            assert_eq!(
                response
                    .headers()
                    .get(http::header::CACHE_CONTROL)
                    .and_then(|value| value.to_str().ok()),
                Some("private, no-store")
            );
        }
    }

    for task_id in ["foreign", "deleted"] {
        let cancel = client
            .post(format!(
                "{gateway_url}/api/users/me/video-tasks/{task_id}/cancel"
            ))
            .header("authorization", format!("Bearer {access_token}"))
            .header("x-client-device-id", device_id)
            .header("user-agent", "AetherTest/1.0")
            .send()
            .await
            .expect("foreign/deleted cancel request should succeed");
        assert_eq!(cancel.status(), StatusCode::NOT_FOUND, "task_id={task_id}");
    }
    let foreign = repository
        .find(VideoTaskLookupKey::Id("foreign"))
        .await
        .expect("foreign task read should succeed")
        .expect("foreign task should remain");
    assert_eq!(foreign.status, VideoTaskStatus::Processing);

    let spoofed_list = authenticated_get("/api/users/me/video-tasks?user_id=user-auth-2")
        .send()
        .await
        .expect("spoofed list request should succeed");
    assert_eq!(spoofed_list.status(), StatusCode::BAD_REQUEST);

    gateway_handle.abort();
}
