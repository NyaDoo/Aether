pub(super) use std::convert::Infallible;
pub(super) use std::sync::{Arc, Mutex};

pub(super) use axum::body::{to_bytes, Body, Bytes};
pub(super) use axum::response::Response;
pub(super) use axum::routing::any;
pub(super) use axum::{extract::Request, Json, Router};
pub(super) use http::header::{HeaderName, HeaderValue};
pub(super) use http::StatusCode;
pub(super) use serde_json::json;

mod ai_execute;
mod architecture;
mod async_task;
mod audit;
mod concurrency;
mod control;
mod files;
mod frontdoor;
mod proxy;
mod usage;
mod video;

pub(super) use super::async_task::VideoTaskTruthSourceMode;
pub(super) use super::constants::*;
pub(super) use super::fallback_metrics::{GatewayFallbackMetricKind, GatewayFallbackReason};
pub(super) use super::rate_limit::FrontdoorUserRpmConfig;
pub(super) use super::router::{attach_static_frontend, build_router, build_router_with_state};
pub(super) use super::state::{AppState, FrontdoorCorsConfig};
pub(super) use super::usage::UsageRuntimeConfig;

pub(super) async fn start_server(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let app = app.layer(axum::middleware::from_fn(
        sign_legacy_test_admin_forward_headers,
    ));
    let listener = crate::test_support::bind_loopback_listener()
        .await
        .expect("listener should bind");
    let addr = listener.local_addr().expect("local addr should resolve");
    let handle = tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .expect("server should run");
    });
    (format!("http://{addr}"), handle)
}

pub(super) async fn send_request(app: Router, mut request: Request) -> Response {
    use tower::ServiceExt;

    sign_legacy_test_admin_forward_request(&mut request);
    request
        .extensions_mut()
        .insert(axum::extract::ConnectInfo(std::net::SocketAddr::from((
            [127, 0, 0, 1],
            40000,
        ))));
    app.oneshot(request)
        .await
        .expect("router request should complete")
}

pub(super) fn signed_internal_admin_headers(
    method: http::Method,
    uri_text: &str,
) -> http::HeaderMap {
    let uri: http::Uri = uri_text.parse().expect("internal admin URI should parse");
    let mut headers = http::HeaderMap::new();
    headers.insert(
        HeaderName::from_static(TRUSTED_ADMIN_USER_ID_HEADER),
        HeaderValue::from_static("internal-ops-admin"),
    );
    headers.insert(
        HeaderName::from_static(TRUSTED_ADMIN_USER_ROLE_HEADER),
        HeaderValue::from_static("admin"),
    );
    headers.insert(
        HeaderName::from_static(TRUSTED_ADMIN_SESSION_ID_HEADER),
        HeaderValue::from_static("internal-ops-session"),
    );
    crate::control::sign_trusted_admin_forward_headers(&mut headers, &method, &uri)
        .expect("internal admin headers should sign");
    headers
}

// The older integration suite used raw trusted-admin headers as a shortcut for
// an internal forwarding hop. Upgrade only that exact legacy test marker into
// the same HMAC proof a real internal producer must create. Production request
// handling has no marker-only compatibility path.
async fn sign_legacy_test_admin_forward_headers(
    mut request: Request,
    next: axum::middleware::Next,
) -> Response {
    sign_legacy_test_admin_forward_request(&mut request);
    next.run(request).await
}

fn sign_legacy_test_admin_forward_request(request: &mut Request) {
    let has_legacy_marker = request
        .headers()
        .get(GATEWAY_HEADER)
        .and_then(|value| value.to_str().ok())
        == Some("rust-phase3b");
    if !has_legacy_marker || !request.headers().contains_key(TRUSTED_ADMIN_USER_ID_HEADER) {
        return;
    }

    let method = request.method().clone();
    let uri = request.uri().clone();
    let _ =
        crate::control::sign_trusted_admin_forward_headers(request.headers_mut(), &method, &uri);
}

pub(super) fn build_router_with_execution_runtime_override(
    execution_runtime_override_base_url: impl Into<String>,
) -> Router {
    let state = build_state_with_execution_runtime_override(execution_runtime_override_base_url);
    build_router_with_state(state)
}

pub(super) fn build_state_with_execution_runtime_override(
    execution_runtime_override_base_url: impl Into<String>,
) -> AppState {
    AppState::new()
        .expect("gateway should build")
        .with_execution_runtime_override_base_url(execution_runtime_override_base_url)
}

pub(super) async fn wait_until(timeout_ms: u64, mut predicate: impl FnMut() -> bool) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    loop {
        if predicate() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition not met within {}ms",
            timeout_ms
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

pub(crate) fn strip_sse_keepalive_comments(body: &str) -> String {
    body.replace(": aether-keepalive\n\n", "")
}

pub(crate) async fn next_non_keepalive_chunk(response: &mut reqwest::Response) -> Bytes {
    loop {
        let chunk = response
            .chunk()
            .await
            .expect("chunk should read")
            .expect("chunk should exist");
        if chunk.as_ref() != b": aether-keepalive\n\n" {
            return chunk;
        }
    }
}
