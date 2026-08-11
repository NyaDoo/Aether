use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use crate::async_task::{
    cancel_video_task, get_video_task_detail, get_video_task_stats, get_video_task_video,
    list_video_tasks,
};
use crate::audit::{get_auth_api_key_snapshot, get_decision_trace, get_request_candidate_trace};
use crate::constants::TRUSTED_ADMIN_USER_ROLE_HEADER;
use crate::hooks::{get_request_audit_bundle, get_request_usage_audit};
use crate::router::metrics;
use crate::state::AppState;

pub(crate) fn mount_operational_routes(router: Router<AppState>) -> Router<AppState> {
    let sensitive_routes = Router::<AppState>::new()
        .route("/_gateway/async-tasks/video-tasks", get(list_video_tasks))
        .route(
            "/_gateway/async-tasks/video-tasks/stats",
            get(get_video_task_stats),
        )
        .route(
            "/_gateway/async-tasks/video-tasks/{task_id}/video",
            get(get_video_task_video),
        )
        .route(
            "/_gateway/async-tasks/video-tasks/{task_id}/cancel",
            post(cancel_video_task),
        )
        .route(
            "/_gateway/async-tasks/video-tasks/{task_id}",
            get(get_video_task_detail),
        )
        .route(
            "/_gateway/audit/auth/users/{user_id}/api-keys/{api_key_id}",
            get(get_auth_api_key_snapshot),
        )
        .route(
            "/_gateway/audit/decision-trace/{request_id}",
            get(get_decision_trace),
        )
        .route(
            "/_gateway/audit/request-candidates/{request_id}",
            get(get_request_candidate_trace),
        )
        .route(
            "/_gateway/audit/request-audit/{request_id}",
            get(get_request_audit_bundle),
        )
        .route(
            "/_gateway/audit/request-usage/{request_id}",
            get(get_request_usage_audit),
        )
        .route_layer(middleware::from_fn(require_trusted_internal_admin_request));

    router
        .route("/_gateway/metrics", get(metrics))
        .merge(sensitive_routes)
}

async fn require_trusted_internal_admin_request(request: Request, next: Next) -> Response {
    let method = request.method();
    let uri = request.uri();
    let is_full_admin = request
        .headers()
        .get(TRUSTED_ADMIN_USER_ROLE_HEADER)
        .and_then(|value| value.to_str().ok())
        .is_some_and(crate::roles::is_full_admin_role);
    let has_valid_proof =
        crate::control::verify_trusted_admin_forward_headers(request.headers(), method, uri);

    if !is_full_admin
        || !has_valid_proof
        || crate::control::internal_forward_proof_is_replay(request.headers(), method, uri)
    {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({
                "error": {
                    "message": "Trusted internal administrator authentication required"
                }
            })),
        )
            .into_response();
    }

    next.run(request).await
}
