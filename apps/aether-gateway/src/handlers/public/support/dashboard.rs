pub(super) use super::{
    build_auth_error_response, build_unhandled_public_support_response, query_param_value,
    resolve_authenticated_local_user, AppState, GatewayError, GatewayPublicRequestContext,
};
use axum::{
    body::Body,
    http,
    response::{IntoResponse, Response},
};

#[path = "dashboard_filters.rs"]
mod dashboard_helpers;

use self::dashboard_helpers::{
    decision_route_kind, handle_dashboard_daily_stats_get, handle_dashboard_provider_status_get,
    handle_dashboard_recent_requests_get, handle_dashboard_stats_get,
};

pub(super) async fn maybe_build_local_dashboard_response(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    match decision_route_kind(request_context) {
        Some("stats")
            if request_context.request_method == http::Method::GET
                && request_context.request_path == "/api/dashboard/stats" =>
        {
            handle_dashboard_stats_get(state, request_context, headers).await
        }
        Some("recent_requests")
            if request_context.request_method == http::Method::GET
                && request_context.request_path == "/api/dashboard/recent-requests" =>
        {
            handle_dashboard_recent_requests_get(state, request_context, headers).await
        }
        Some("provider_status")
            if request_context.request_method == http::Method::GET
                && request_context.request_path == "/api/dashboard/provider-status" =>
        {
            handle_dashboard_provider_status_get(state, request_context, headers).await
        }
        Some("daily_stats")
            if request_context.request_method == http::Method::GET
                && request_context.request_path == "/api/dashboard/daily-stats" =>
        {
            handle_dashboard_daily_stats_get(state, request_context, headers).await
        }
        Some("realtime")
            if request_context.request_method == http::Method::GET
                && request_context.request_path == "/api/dashboard/realtime" =>
        {
            handle_dashboard_realtime_get(state, request_context, headers).await
        }
        _ => build_unhandled_public_support_response(request_context),
    }
}

async fn handle_dashboard_realtime_get(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &http::HeaderMap,
) -> Response<Body> {
    let auth = match resolve_authenticated_local_user(state, request_context, headers).await {
        Ok(value) => value,
        Err(response) => return realtime_no_cache(response),
    };
    if !auth.user.role.eq_ignore_ascii_case("admin") {
        return realtime_no_cache(build_auth_error_response(
            http::StatusCode::FORBIDDEN,
            "仅管理员可查看全站实时指标",
            false,
        ));
    }

    match crate::dashboard_realtime::snapshot(state.runtime_state()).await {
        Ok(payload) => realtime_no_cache(
            Response::builder()
                .status(http::StatusCode::OK)
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    serde_json::to_vec(&payload).unwrap_or_else(|_| b"{}".to_vec()),
                ))
                .unwrap_or_else(|_| http::StatusCode::INTERNAL_SERVER_ERROR.into_response()),
        ),
        Err(err) => realtime_no_cache(build_auth_error_response(
            http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("dashboard realtime metrics lookup failed: {err:?}"),
            false,
        )),
    }
}

fn realtime_no_cache(mut response: Response<Body>) -> Response<Body> {
    response.headers_mut().insert(
        http::header::CACHE_CONTROL,
        http::HeaderValue::from_static("no-store, no-cache, max-age=0"),
    );
    response.headers_mut().insert(
        http::header::PRAGMA,
        http::HeaderValue::from_static("no-cache"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::{maybe_build_local_dashboard_response, AppState, GatewayPublicRequestContext};
    use crate::control::GatewayControlDecision;
    use axum::body::to_bytes;
    use axum::http::{HeaderMap, Method, StatusCode, Uri};

    fn request_context(method: Method, uri: &str, route_kind: &str) -> GatewayPublicRequestContext {
        GatewayPublicRequestContext::from_request_parts(
            "trace-dashboard-unhandled",
            &method,
            &uri.parse::<Uri>().expect("uri should parse"),
            &HeaderMap::new(),
            Some(GatewayControlDecision::synthetic(
                uri,
                Some("public_support".to_string()),
                Some("dashboard".to_string()),
                Some(route_kind.to_string()),
                Some("user:dashboard".to_string()),
            )),
        )
    }

    #[tokio::test]
    async fn dashboard_unhandled_route_returns_local_not_implemented_response() {
        let state = AppState::new().expect("gateway should build");
        let request_context = request_context(Method::GET, "/api/dashboard/stats/history", "stats");
        let response =
            maybe_build_local_dashboard_response(&state, &request_context, &HeaderMap::new()).await;

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let payload: serde_json::Value =
            serde_json::from_slice(&body).expect("json body should parse");
        assert_eq!(
            payload["detail"],
            "public support route not implemented in rust frontdoor"
        );
        assert_eq!(payload["route_family"], "dashboard");
        assert_eq!(payload["route_kind"], "stats");
        assert_eq!(payload["request_path"], "/api/dashboard/stats/history");
    }
}
