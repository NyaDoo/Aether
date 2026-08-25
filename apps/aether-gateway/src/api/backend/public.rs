use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderValue, Response, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use crate::{handlers::proxy::proxy_request, state::AppState, GatewayError};

pub(crate) fn mount_public_support_routes(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/v1/models", get(proxy_request))
        .route("/v1beta/models", get(proxy_request))
        .route("/v1/health", get(proxy_request))
        .route("/health", get(proxy_request))
        .route("/v1/providers", get(proxy_request))
        .route("/v1/providers/{*provider_path}", get(proxy_request))
        .route("/v1/test-connection", get(proxy_request))
        .route("/test-connection", get(proxy_request))
        .route("/api/public/site-info", get(proxy_request))
        .route("/api/public/providers", get(proxy_request))
        .route("/api/public/models", get(proxy_request))
        .route("/api/public/search/models", get(proxy_request))
        .route("/api/public/stats", get(proxy_request))
        .route("/api/public/global-models", get(proxy_request))
        .route("/api/public/health/api-formats", get(proxy_request))
        .route("/api/public/health/models", get(proxy_request))
        .route("/api/public/health/related", get(proxy_request))
        .route("/api/modules/auth-status", get(proxy_request))
        .route("/api/capabilities", get(proxy_request))
        .route("/api/capabilities/user-configurable", get(proxy_request))
        .route("/api/capabilities/model/{*model_path}", get(proxy_request))
        .route("/install/{*install_path}", get(proxy_request))
        .route("/install-tunnel/{*install_path}", get(proxy_request))
        .route("/i/{*install_path}", get(proxy_request))
        .route("/", get(proxy_root_request))
}

pub(crate) fn root_query_targets_action(path: &str, query: Option<&str>) -> bool {
    path == "/"
        && query.is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes())
                .any(|(key, _)| key.eq_ignore_ascii_case("Action"))
        })
}

async fn proxy_root_request(
    state: State<AppState>,
    remote_addr: ConnectInfo<SocketAddr>,
    request: Request,
) -> Result<Response<Body>, GatewayError> {
    if root_query_targets_action(request.uri().path(), request.uri().query()) {
        let mut response = StatusCode::METHOD_NOT_ALLOWED.into_response();
        response
            .headers_mut()
            .insert(header::ALLOW, HeaderValue::from_static("POST"));
        return Ok(response);
    }

    proxy_request(state, remote_addr, request).await
}

#[cfg(test)]
mod tests {
    use super::root_query_targets_action;

    #[test]
    fn root_action_queries_target_the_post_only_api() {
        assert!(root_query_targets_action(
            "/",
            Some("Action=ListAssetGroups&Version=2024-01-01")
        ));
        assert!(root_query_targets_action(
            "/",
            Some("action=ListAssetGroups")
        ));
        assert!(!root_query_targets_action("/", Some("Version=2024-01-01")));
        assert!(!root_query_targets_action(
            "/guide",
            Some("Action=ListAssetGroups")
        ));
    }
}
