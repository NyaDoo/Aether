use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use futures_util::{StreamExt, TryStreamExt};
use http_body_util::{BodyExt, Full, Limited, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::{Request, Response};
use tracing::{debug, warn};
use url::Url;

use super::BoxBody;
use crate::auth;
use crate::config::Config;
use crate::proxy::target_filter::{self, DnsCache};

/// Handle delegation requests: Aether sends a full request description,
/// and the proxy issues the actual upstream HTTP call using its own TLS stack.
///
/// Endpoint: POST /_aether/delegate
///
/// Wire format: metadata in HTTP headers, upstream body sent directly
/// as HTTP body (optionally gzip-compressed via `Content-Encoding: gzip`).
///
/// Headers:
///   X-Delegate-Method: POST
///   X-Delegate-Url: https://api.anthropic.com/v1/messages
///   X-Delegate-Headers: base64-encoded JSON {"Authorization": "Bearer ...", ...}
///   X-Delegate-Timeout: 30              (accepted but not used — Aether controls timeouts)
///   Content-Encoding: gzip              (optional, indicates body is gzip-compressed)
pub async fn handle_delegate(
    req: Request<Incoming>,
    config: Arc<Config>,
    allowed_ports: &HashSet<u16>,
    timestamp_tolerance: u64,
    dns_cache: &DnsCache,
    http_client: &reqwest::Client,
) -> Response<BoxBody> {
    let total_start = Instant::now();

    // ── Auth ──
    let auth_header = req
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    if let Err(e) = auth::validate_proxy_auth(auth_header, &config, timestamp_tolerance) {
        warn!(error = %e, "delegate auth failed");
        return error_response(401, "authentication_failed", &e.to_string());
    }
    let auth_ms = total_start.elapsed().as_millis() as u64;

    // ── Parse metadata from headers ──
    let meta_start = Instant::now();

    let method_str = match req
        .headers()
        .get("x-delegate-method")
        .and_then(|v| v.to_str().ok())
    {
        Some(m) => m.to_string(),
        None => {
            warn!("delegate missing X-Delegate-Method");
            return error_response(400, "bad_request", "missing X-Delegate-Method header");
        }
    };

    let target_url = match req
        .headers()
        .get("x-delegate-url")
        .and_then(|v| v.to_str().ok())
    {
        Some(u) => u.to_string(),
        None => {
            warn!("delegate missing X-Delegate-Url");
            return error_response(400, "bad_request", "missing X-Delegate-Url header");
        }
    };

    let upstream_headers: HashMap<String, String> = match req
        .headers()
        .get("x-delegate-headers")
        .and_then(|v| v.to_str().ok())
    {
        Some(b64) => {
            match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64) {
                Ok(decoded) => match serde_json::from_slice(&decoded) {
                    Ok(h) => h,
                    Err(e) => {
                        warn!(error = %e, "delegate invalid X-Delegate-Headers JSON");
                        return error_response(
                            400,
                            "bad_request",
                            "invalid X-Delegate-Headers JSON",
                        );
                    }
                },
                Err(e) => {
                    warn!(error = %e, "delegate invalid X-Delegate-Headers base64");
                    return error_response(400, "bad_request", "invalid X-Delegate-Headers base64");
                }
            }
        }
        None => HashMap::new(),
    };

    let is_gzip = req
        .headers()
        .get("content-encoding")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("gzip"))
        .unwrap_or(false);

    let req_content_length: u64 = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let meta_ms = meta_start.elapsed().as_millis() as u64;

    // ── Target validation ──
    let parsed_url = match Url::parse(&target_url) {
        Ok(u) => u,
        Err(e) => {
            warn!(url = %target_url, error = %e, "delegate invalid target URL");
            return error_response(400, "bad_request", &format!("invalid URL: {}", e));
        }
    };

    let host = match parsed_url.host_str() {
        Some(h) => h.to_string(),
        None => {
            warn!(url = %target_url, "delegate target URL missing host");
            return error_response(400, "bad_request", "URL missing host");
        }
    };

    let port = parsed_url.port_or_known_default().unwrap_or(443);

    let dns_start = Instant::now();
    if let Err(e) = target_filter::validate_target(&host, port, allowed_ports, dns_cache).await {
        warn!(host = %host, port, error = %e, "delegate target rejected");
        return error_response(403, "target_not_allowed", &e.to_string());
    }
    let dns_ms = dns_start.elapsed().as_millis() as u64;

    debug!(method = %method_str, url = %target_url, is_gzip, "delegate request");

    // ── Build upstream request ──
    let method = match method_str.parse::<reqwest::Method>() {
        Ok(m) => m,
        Err(e) => {
            warn!(error = %e, method = %method_str, "delegate invalid HTTP method");
            return error_response(400, "bad_request", &format!("invalid method: {}", e));
        }
    };

    let mut upstream_req = http_client.request(method, &target_url);

    // Set headers (skip `host` — reqwest sets it from the URL automatically,
    // and a duplicate Host header can confuse certain upstreams)
    for (name, value) in &upstream_headers {
        if name.eq_ignore_ascii_case("host") {
            continue;
        }
        upstream_req = upstream_req.header(name.as_str(), value.as_str());
    }

    // ── Stream body passthrough ──
    // When body is gzip-compressed, forward it directly to upstream with
    // Content-Encoding: gzip header — no collect/decompress needed.
    // All major AI API providers (Anthropic, OpenAI, Google) accept gzip request bodies.
    let wire_size: u64;
    if is_gzip {
        // Passthrough: stream the gzip body directly to upstream
        upstream_req = upstream_req.header("content-encoding", "gzip");
        let body_stream = req.into_body();
        let byte_stream = http_body_util::BodyStream::new(body_stream).filter_map(|result| async {
            match result {
                Ok(frame) => frame.into_data().ok().map(Ok),
                Err(e) => Some(Err(e)),
            }
        });
        let reqwest_body = reqwest::Body::wrap_stream(byte_stream);
        upstream_req = upstream_req.body(reqwest_body);
        // wire_size will be reported from Content-Length if available, otherwise 0
        wire_size = req_content_length;
    } else {
        // Non-gzip: read body into memory (legacy path)
        const MAX_BODY: usize = 10 * 1024 * 1024;
        let body_bytes = match Limited::new(req.into_body(), MAX_BODY).collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                warn!(error = %e, "delegate failed to read request body");
                return error_response(413, "payload_too_large", "request body exceeds 10MB limit");
            }
        };
        wire_size = body_bytes.len() as u64;
        if !body_bytes.is_empty() {
            upstream_req = upstream_req.body(body_bytes.to_vec());
        }
    }

    // ── Send upstream request ──
    // NOTE: We intentionally do NOT set a per-request timeout here.
    // reqwest's `.timeout()` caps the *entire* request including body streaming,
    // which would truncate long-lived SSE streams. The delegate_client already
    // has a configured connect_timeout for connection establishment, and Aether controls
    // first-byte / idle timeouts on its own side via asyncio.
    let upstream_start = Instant::now();
    let upstream_resp = match upstream_req.send().await {
        Ok(resp) => resp,
        Err(e) => {
            warn!(url = %target_url, error = %e, "delegate upstream request failed");
            let safe_detail = sanitize_upstream_error(&e.to_string());
            if e.is_timeout() {
                return error_response(504, "upstream_timeout", &safe_detail);
            }
            return error_response(502, "upstream_connection_failed", &safe_detail);
        }
    };
    let upstream_ms = upstream_start.elapsed().as_millis() as u64;

    // ── Build response ──
    let status = upstream_resp.status().as_u16();
    let resp_headers = upstream_resp.headers().clone();
    let total_ms = total_start.elapsed().as_millis() as u64;

    debug!(
        url = %target_url,
        status,
        dns_ms,
        upstream_ms,
        total_ms,
        wire_size,
        is_gzip,
        "delegate upstream response"
    );

    let timing = serde_json::json!({
        "auth_ms": auth_ms,
        "meta_ms": meta_ms,
        "wire_size": wire_size,
        "passthrough": is_gzip,
        "dns_ms": dns_ms,
        "upstream_ms": upstream_ms,
        "total_ms": total_ms,
    });

    let body_stream = upstream_resp
        .bytes_stream()
        .map_ok(Frame::data)
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { Box::new(e) });

    let stream_body: BoxBody = BodyExt::boxed(StreamBody::new(body_stream));

    let mut builder = Response::builder().status(status);
    for (name, value) in resp_headers.iter() {
        builder = builder.header(name, value);
    }
    builder = builder.header("X-Proxy-Timing", timing.to_string());

    builder.body(stream_body).unwrap_or_else(|_| {
        Response::builder()
            .status(500)
            .body(super::empty_box_body())
            .unwrap()
    })
}

// ── Sanitisation ─────────────────────────────────────────────────────────────

/// Strip full URLs from error messages to prevent leaking upstream API keys,
/// paths, or query parameters in the delegate error response.
///
/// Replaces `https://api.example.com/v1/chat?key=xxx` with `api.example.com`.
fn sanitize_upstream_error(msg: &str) -> String {
    // Simple regex-free approach: find "https://..." or "http://..." spans and
    // replace them with just the host portion.
    let mut result = msg.to_string();
    for scheme in &["https://", "http://"] {
        while let Some(start) = result.find(scheme) {
            let after_scheme = start + scheme.len();
            // Host ends at '/', '?', '#', ' ', or end of string
            let host_end = result[after_scheme..]
                .find(['/', '?', '#', ' '])
                .map(|i| after_scheme + i)
                .unwrap_or(result.len());
            let host = &result[after_scheme..host_end];
            result = format!("{}{}{}", &result[..start], host, &result[host_end..]);
        }
    }
    result
}

// ── Error response helpers ───────────────────────────────────────────────────

fn error_response(status: u16, error: &str, detail: &str) -> Response<BoxBody> {
    let body = serde_json::json!({
        "error": error,
        "detail": detail,
    });
    let body_bytes = bytes::Bytes::from(body.to_string());

    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .header("X-Delegate-Error", "true")
        .body(
            Full::new(body_bytes)
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { match e {} })
                .boxed(),
        )
        .unwrap()
}
