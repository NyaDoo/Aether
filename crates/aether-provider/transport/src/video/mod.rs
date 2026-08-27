use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use aether_ai_formats::api::convert_openai_video_request_to_doubao;
use aether_data_contracts::repository::video_tasks::StoredVideoTask;
use aether_video_tasks_core::{
    LocalVideoTaskSnapshot, LocalVideoTaskTransport, LocalVideoTaskTransportBridgeInput,
};
use async_trait::async_trait;
use serde_json::Value;

use super::auth::{
    build_passthrough_headers_with_auth, resolve_local_gemini_auth,
    resolve_local_openai_bearer_auth,
};
use super::network::{
    resolve_transport_execution_timeouts, resolve_transport_profile,
    resolve_transport_proxy_snapshot,
};
use super::policy::{
    local_gemini_transport_unsupported_reason_with_network,
    local_standard_transport_unsupported_reason_with_network,
    supports_local_gemini_transport_with_network, supports_local_standard_transport_with_network,
};
use super::rules::{
    apply_local_body_rules_with_request_headers, apply_local_header_rules_with_request_headers,
};
use super::snapshot::GatewayProviderTransportSnapshot;
use super::url::{
    build_gemini_video_predict_long_running_url, build_passthrough_path_url,
    doubao_video_tasks_upstream_url,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderVideoCreateFamily {
    OpenAi,
    Gemini,
    Doubao,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderVideoCreateHeadersInput<'a> {
    pub headers: &'a http::HeaderMap,
    pub auth_header: &'a str,
    pub auth_value: &'a str,
    pub header_rules: Option<&'a Value>,
    pub provider_request_body: &'a Value,
    pub original_request_body: &'a Value,
}

#[async_trait]
pub trait VideoTaskTransportSnapshotLookup: Send + Sync {
    async fn read_video_task_provider_transport_snapshot(
        &self,
        provider_id: &str,
        endpoint_id: &str,
        key_id: &str,
    ) -> Result<Option<GatewayProviderTransportSnapshot>, String>;

    /// Resolve the effective runtime proxy for a follow-up request. The
    /// gateway implementation includes system proxy selection and tunnel
    /// attachment affinity; lightweight callers retain the provider-local
    /// fallback.
    async fn resolve_video_task_proxy_snapshot(
        &self,
        transport: &GatewayProviderTransportSnapshot,
    ) -> Option<aether_contracts::ProxySnapshot> {
        resolve_transport_proxy_snapshot(transport)
    }
}

pub fn resolve_local_video_task_transport(
    transport: &GatewayProviderTransportSnapshot,
    api_format: &str,
    model_name: Option<String>,
) -> Option<LocalVideoTaskTransport> {
    if video_provider_key_is_expired(transport) {
        return None;
    }
    let api_format = api_format.trim();
    let (auth_header, auth_value) = match api_format {
        // Ark authenticates with a bearer token, same as the OpenAI surface.
        "openai:video" | "doubao:video" => {
            if !supports_local_standard_transport_with_network(transport, api_format) {
                return None;
            }
            resolve_local_openai_bearer_auth(transport)?
        }
        "gemini:video" => {
            if !supports_local_gemini_transport_with_network(transport, api_format) {
                return None;
            }
            resolve_local_gemini_auth(transport)?
        }
        _ => return None,
    };
    if !video_auth_is_usable(api_format, &auth_header, &auth_value) {
        return None;
    }

    Some(LocalVideoTaskTransport::from_bridge_input(
        LocalVideoTaskTransportBridgeInput {
            upstream_base_url: transport.endpoint.base_url.clone(),
            provider_name: Some(transport.provider.name.clone()),
            provider_id: transport.provider.id.clone(),
            endpoint_id: transport.endpoint.id.clone(),
            key_id: transport.key.id.clone(),
            auth_header,
            auth_value,
            content_type: Some("application/json".to_string()),
            model_name,
            proxy: resolve_transport_proxy_snapshot(transport),
            transport_profile: resolve_transport_profile(transport),
            timeouts: resolve_transport_execution_timeouts(transport),
        },
    ))
}

fn video_auth_is_usable(api_format: &str, header: &str, value: &str) -> bool {
    let header = header.trim();
    let value = value.trim();
    if header.is_empty() || value.is_empty() {
        return false;
    }
    match api_format {
        "openai:video" | "doubao:video" => {
            header.eq_ignore_ascii_case("authorization")
                && value
                    .split_once(char::is_whitespace)
                    .is_some_and(|(scheme, token)| {
                        scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()
                    })
        }
        "gemini:video" => {
            header.eq_ignore_ascii_case("x-goog-api-key")
                || (header.eq_ignore_ascii_case("authorization")
                    && value
                        .split_once(char::is_whitespace)
                        .is_some_and(|(scheme, token)| {
                            scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()
                        }))
        }
        _ => false,
    }
}

pub fn video_create_transport_unsupported_reason(
    transport: &GatewayProviderTransportSnapshot,
    family: ProviderVideoCreateFamily,
    api_format: &str,
) -> Option<&'static str> {
    // A create-only custom path does not define how task resource URLs should
    // be formed for GET/DELETE. Accepting it would create tasks that cannot be
    // polled or cancelled reliably after the initial request.
    if transport
        .endpoint
        .custom_path
        .as_deref()
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
    {
        return Some("video_task_custom_path_follow_up_unsupported");
    }
    match family {
        ProviderVideoCreateFamily::OpenAi => {
            local_standard_transport_unsupported_reason_with_network(transport, api_format)
        }
        ProviderVideoCreateFamily::Gemini => {
            local_gemini_transport_unsupported_reason_with_network(transport, api_format)
        }
        ProviderVideoCreateFamily::Doubao => {
            local_standard_transport_unsupported_reason_with_network(transport, api_format)
        }
    }
}

pub fn resolve_video_create_auth(
    transport: &GatewayProviderTransportSnapshot,
    family: ProviderVideoCreateFamily,
) -> Option<(String, String)> {
    if video_provider_key_is_expired(transport) {
        return None;
    }
    let api_format = match family {
        ProviderVideoCreateFamily::OpenAi => "openai:video",
        ProviderVideoCreateFamily::Gemini => "gemini:video",
        ProviderVideoCreateFamily::Doubao => "doubao:video",
    };
    let (header, value) = match family {
        ProviderVideoCreateFamily::OpenAi | ProviderVideoCreateFamily::Doubao => {
            resolve_local_openai_bearer_auth(transport)
        }
        ProviderVideoCreateFamily::Gemini => resolve_local_gemini_auth(transport),
    }?;
    video_auth_is_usable(api_format, &header, &value).then_some((header, value))
}

fn video_provider_key_is_expired(transport: &GatewayProviderTransportSnapshot) -> bool {
    let now_unix_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    transport
        .key
        .expires_at_unix_secs
        // Provider catalog uses zero as the legacy "no expiry" sentinel.
        .is_some_and(|expires_at| expires_at > 0 && expires_at <= now_unix_secs)
}

pub fn build_video_create_request_body(
    body_json: &Value,
    family: ProviderVideoCreateFamily,
    mapped_model: &str,
    body_rules: Option<&Value>,
    request_headers: Option<&http::HeaderMap>,
) -> Option<Value> {
    build_video_create_request_body_for_client(
        body_json,
        family,
        family,
        mapped_model,
        body_rules,
        request_headers,
    )
}

/// Builds the upstream request body, converting first when the client surface
/// differs from the provider surface.
///
/// Only OpenAI-to-Doubao is convertible; other mismatches are rejected so a
/// request is never silently sent in a shape the provider cannot read.
pub fn build_video_create_request_body_for_client(
    body_json: &Value,
    client_family: ProviderVideoCreateFamily,
    provider_family: ProviderVideoCreateFamily,
    mapped_model: &str,
    body_rules: Option<&Value>,
    request_headers: Option<&http::HeaderMap>,
) -> Option<Value> {
    let converted;
    let body_json = if client_family == provider_family {
        body_json
    } else {
        match (client_family, provider_family) {
            (ProviderVideoCreateFamily::OpenAi, ProviderVideoCreateFamily::Doubao) => {
                converted = convert_openai_video_request_to_doubao(body_json).ok()?;
                &converted
            }
            _ => return None,
        }
    };

    let mut provider_request_body = match provider_family {
        // Ark and OpenAI both carry the model in the request body, so the mapped
        // model replaces whatever the client asked for. Every other field rides
        // through untouched, which keeps new Ark parameters working without a
        // gateway change.
        ProviderVideoCreateFamily::OpenAi | ProviderVideoCreateFamily::Doubao => {
            let mut provider_request_body = body_json.as_object().cloned().unwrap_or_default();
            provider_request_body
                .insert("model".to_string(), Value::String(mapped_model.to_string()));
            Value::Object(provider_request_body)
        }
        ProviderVideoCreateFamily::Gemini => body_json.clone(),
    };
    if !apply_local_body_rules_with_request_headers(
        &mut provider_request_body,
        body_rules,
        Some(body_json),
        request_headers,
    ) {
        return None;
    }
    Some(provider_request_body)
}

pub fn build_video_create_upstream_url(
    transport: &GatewayProviderTransportSnapshot,
    request_path: &str,
    request_query: Option<&str>,
    mapped_model: &str,
    family: ProviderVideoCreateFamily,
) -> Option<String> {
    let custom_path = transport
        .endpoint
        .custom_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());

    if let Some(path) = custom_path {
        let blocked_keys = match family {
            ProviderVideoCreateFamily::OpenAi | ProviderVideoCreateFamily::Doubao => &[][..],
            ProviderVideoCreateFamily::Gemini => &["key"][..],
        };
        return build_passthrough_path_url(
            &transport.endpoint.base_url,
            path,
            request_query,
            blocked_keys,
        );
    }

    match family {
        ProviderVideoCreateFamily::OpenAi => build_passthrough_path_url(
            &transport.endpoint.base_url,
            openai_video_api_root_request_path(request_path),
            request_query,
            &[],
        ),
        ProviderVideoCreateFamily::Gemini => build_gemini_video_predict_long_running_url(
            &transport.endpoint.base_url,
            mapped_model,
            request_query,
        ),
        // The client path is `/api/v3/...`; rebuild the provider resource path
        // from the configured API root rather than passing the client path
        // through verbatim.
        ProviderVideoCreateFamily::Doubao => {
            doubao_video_tasks_upstream_url(&transport.endpoint.base_url, None, request_query)
        }
    }
}

fn openai_video_api_root_request_path(request_path: &str) -> &str {
    if request_path.starts_with("/v1/") {
        &request_path[3..]
    } else {
        request_path
    }
}

pub fn build_video_create_headers(
    input: ProviderVideoCreateHeadersInput<'_>,
) -> Option<BTreeMap<String, String>> {
    let mut provider_request_headers = build_passthrough_headers_with_auth(
        input.headers,
        input.auth_header,
        input.auth_value,
        &BTreeMap::new(),
    );
    if !apply_local_header_rules_with_request_headers(
        &mut provider_request_headers,
        input.header_rules,
        &[input.auth_header, "content-type"],
        input.provider_request_body,
        Some(input.original_request_body),
        Some(input.headers),
    ) {
        return None;
    }
    Some(provider_request_headers)
}

pub async fn reconstruct_local_video_task_snapshot(
    lookup: &dyn VideoTaskTransportSnapshotLookup,
    task: &StoredVideoTask,
) -> Result<Option<LocalVideoTaskSnapshot>, String> {
    let provider_api_format = task
        .provider_api_format
        .as_deref()
        .unwrap_or_default()
        .trim();
    if !matches!(
        provider_api_format,
        "openai:video" | "gemini:video" | "doubao:video"
    ) {
        return Ok(None);
    }

    let Some(provider_id) = task.provider_id.as_deref() else {
        return Ok(None);
    };
    let Some(endpoint_id) = task.endpoint_id.as_deref() else {
        return Ok(None);
    };
    let Some(key_id) = task.key_id.as_deref() else {
        return Ok(None);
    };

    let Some(transport) = lookup
        .read_video_task_provider_transport_snapshot(provider_id, endpoint_id, key_id)
        .await?
    else {
        return Ok(None);
    };

    let runtime_proxy = lookup.resolve_video_task_proxy_snapshot(&transport).await;
    let Some(mut local_transport) =
        resolve_local_video_task_transport(&transport, provider_api_format, task.model.clone())
    else {
        return Ok(None);
    };
    local_transport.proxy = runtime_proxy;
    let original_body = task.original_request_body.as_ref().unwrap_or(&Value::Null);
    let request_headers = http::HeaderMap::new();
    if !apply_local_header_rules_with_request_headers(
        &mut local_transport.headers,
        transport.endpoint.header_rules.as_ref(),
        &["authorization", "x-goog-api-key", "content-type"],
        original_body,
        Some(original_body),
        Some(&request_headers),
    ) {
        return Ok(None);
    }

    Ok(LocalVideoTaskSnapshot::from_stored_task_with_transport(
        task,
        local_transport,
    ))
}

#[cfg(test)]
mod tests {
    use aether_data_contracts::repository::video_tasks::{StoredVideoTask, VideoTaskStatus};
    use aether_video_tasks_core::LocalVideoTaskSnapshot;
    use async_trait::async_trait;
    use serde_json::json;

    use super::{
        build_video_create_headers, build_video_create_request_body,
        build_video_create_request_body_for_client, build_video_create_upstream_url,
        reconstruct_local_video_task_snapshot, resolve_local_video_task_transport,
        resolve_video_create_auth, video_create_transport_unsupported_reason,
        ProviderVideoCreateFamily, ProviderVideoCreateHeadersInput,
        VideoTaskTransportSnapshotLookup,
    };
    use crate::snapshot::{
        GatewayProviderTransportEndpoint, GatewayProviderTransportKey,
        GatewayProviderTransportProvider, GatewayProviderTransportSnapshot,
    };

    fn sample_transport(api_format: &str, auth_type: &str) -> GatewayProviderTransportSnapshot {
        GatewayProviderTransportSnapshot {
            provider: GatewayProviderTransportProvider {
                id: "provider-1".to_string(),
                name: "Provider One".to_string(),
                provider_type: "openai".to_string(),
                website: None,
                is_active: true,
                keep_priority_on_conversion: false,
                enable_format_conversion: false,
                concurrent_limit: None,
                max_retries: None,
                proxy: None,
                request_timeout_secs: Some(30.0),
                stream_first_byte_timeout_secs: Some(5.0),
                config: None,
            },
            endpoint: GatewayProviderTransportEndpoint {
                id: "endpoint-1".to_string(),
                provider_id: "provider-1".to_string(),
                api_format: api_format.to_string(),
                api_family: None,
                endpoint_kind: None,
                is_active: true,
                base_url: "https://example.com".to_string(),
                header_rules: None,
                body_rules: None,
                max_retries: None,
                custom_path: None,
                config: None,
                format_acceptance_config: None,
                proxy: None,
            },
            key: GatewayProviderTransportKey {
                id: "key-1".to_string(),
                provider_id: "provider-1".to_string(),
                name: "key".to_string(),
                auth_type: auth_type.to_string(),
                is_active: true,
                api_formats: None,
                auth_type_by_format: None,
                allow_auth_channel_mismatch_formats: None,

                allowed_models: None,
                capabilities: None,
                rate_multipliers: None,
                global_priority_by_format: None,
                expires_at_unix_secs: None,
                proxy: None,
                fingerprint: None,
                upstream_metadata: None,
                decrypted_api_key: "secret".to_string(),
                decrypted_auth_config: None,
            },
        }
    }

    fn sample_stored_video_task() -> StoredVideoTask {
        StoredVideoTask {
            id: "task-1".to_string(),
            short_id: Some("short-1".to_string()),
            request_id: "request-1".to_string(),
            user_id: Some("user-1".to_string()),
            api_key_id: Some("api-key-1".to_string()),
            username: Some("user".to_string()),
            api_key_name: Some("key".to_string()),
            external_task_id: Some("upstream-task-1".to_string()),
            provider_id: Some("provider-1".to_string()),
            endpoint_id: Some("endpoint-1".to_string()),
            key_id: Some("key-1".to_string()),
            client_api_format: Some("openai:video".to_string()),
            provider_api_format: Some("openai:video".to_string()),
            format_converted: false,
            model: Some("sora".to_string()),
            prompt: Some("generate".to_string()),
            original_request_body: Some(json!({"prompt": "generate"})),
            duration_seconds: None,
            resolution: None,
            aspect_ratio: None,
            size: Some("1024x1024".to_string()),
            status: VideoTaskStatus::Submitted,
            progress_percent: 0,
            progress_message: None,
            retry_count: 0,
            poll_interval_seconds: 10,
            next_poll_at_unix_secs: None,
            poll_count: 0,
            max_poll_count: 360,
            created_at_unix_ms: 1,
            submitted_at_unix_secs: Some(1),
            completed_at_unix_secs: None,
            updated_at_unix_secs: 1,
            error_code: None,
            error_message: None,
            video_url: None,
            request_metadata: None,
        }
    }

    struct TestLookup(Option<GatewayProviderTransportSnapshot>);

    #[async_trait]
    impl VideoTaskTransportSnapshotLookup for TestLookup {
        async fn read_video_task_provider_transport_snapshot(
            &self,
            _provider_id: &str,
            _endpoint_id: &str,
            _key_id: &str,
        ) -> Result<Option<GatewayProviderTransportSnapshot>, String> {
            Ok(self.0.clone())
        }
    }

    struct TestRuntimeProxyLookup(GatewayProviderTransportSnapshot);

    #[async_trait]
    impl VideoTaskTransportSnapshotLookup for TestRuntimeProxyLookup {
        async fn read_video_task_provider_transport_snapshot(
            &self,
            _provider_id: &str,
            _endpoint_id: &str,
            _key_id: &str,
        ) -> Result<Option<GatewayProviderTransportSnapshot>, String> {
            Ok(Some(self.0.clone()))
        }

        async fn resolve_video_task_proxy_snapshot(
            &self,
            _transport: &GatewayProviderTransportSnapshot,
        ) -> Option<aether_contracts::ProxySnapshot> {
            Some(aether_contracts::ProxySnapshot {
                enabled: Some(true),
                mode: Some("tunnel".to_string()),
                node_id: Some("runtime-tunnel-node".to_string()),
                ..aether_contracts::ProxySnapshot::default()
            })
        }
    }

    #[test]
    fn resolves_openai_video_transport() {
        let transport = resolve_local_video_task_transport(
            &sample_transport("openai:video", "api_key"),
            "openai:video",
            Some("sora".to_string()),
        )
        .expect("transport");

        assert_eq!(
            transport.headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );
        assert_eq!(transport.model_name.as_deref(), Some("sora"));
        assert_eq!(transport.provider_id, "provider-1");
    }

    #[test]
    fn resolves_gemini_video_transport() {
        let transport = resolve_local_video_task_transport(
            &sample_transport("gemini:video", "api_key"),
            "gemini:video",
            Some("veo".to_string()),
        )
        .expect("transport");

        assert_eq!(
            transport.headers.get("x-goog-api-key").map(String::as_str),
            Some("secret")
        );
        assert_eq!(transport.model_name.as_deref(), Some("veo"));
        assert_eq!(transport.endpoint_id, "endpoint-1");
    }

    #[test]
    fn builds_openai_video_create_request_body_with_mapped_model() {
        let body = build_video_create_request_body(
            &json!({"prompt": "make a clip", "model": "client-model"}),
            ProviderVideoCreateFamily::OpenAi,
            "upstream-video-model",
            None,
            None,
        )
        .expect("body should build");

        assert_eq!(body.get("prompt"), Some(&json!("make a clip")));
        assert_eq!(body.get("model"), Some(&json!("upstream-video-model")));
    }

    #[test]
    fn builds_openai_video_create_url_from_api_root_base() {
        let mut transport = sample_transport("openai:video", "bearer");
        transport.endpoint.base_url = "https://api.openai.example/v1".to_string();
        let url = build_video_create_upstream_url(
            &transport,
            "/v1/videos",
            Some("trace=1"),
            "sora-upstream",
            ProviderVideoCreateFamily::OpenAi,
        )
        .expect("url should build");

        assert_eq!(url, "https://api.openai.example/v1/videos?trace=1");
    }

    #[test]
    fn builds_gemini_video_create_url_and_removes_client_key_query() {
        let transport = sample_transport("gemini:video", "api_key");
        let url = build_video_create_upstream_url(
            &transport,
            "/v1beta/models/client-model:predictLongRunning",
            Some("key=client-key&trace=1"),
            "veo-upstream",
            ProviderVideoCreateFamily::Gemini,
        )
        .expect("url should build");

        assert_eq!(
            url,
            "https://example.com/v1beta/models/veo-upstream:predictLongRunning?trace=1"
        );
    }

    #[test]
    fn builds_video_create_headers_with_auth_and_rules() {
        let provider_request_body = json!({"prompt": "make a clip"});
        let original_request_body = provider_request_body.clone();
        let headers = build_video_create_headers(ProviderVideoCreateHeadersInput {
            headers: &http::HeaderMap::new(),
            auth_header: "authorization",
            auth_value: "Bearer secret",
            header_rules: Some(&json!([
                {"action":"set","key":"x-provider-tag","value":"video"}
            ])),
            provider_request_body: &provider_request_body,
            original_request_body: &original_request_body,
        })
        .expect("headers should build");

        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );
        assert_eq!(
            headers.get("x-provider-tag").map(String::as_str),
            Some("video")
        );
    }

    #[test]
    fn resolves_doubao_video_transport_with_bearer_auth() {
        let mut provider_transport = sample_transport("doubao:video", "api_key");
        provider_transport.provider.proxy = Some(json!({
            "enabled": true,
            "url": "http://provider-proxy.example:8080"
        }));
        let transport = resolve_local_video_task_transport(
            &provider_transport,
            "doubao:video",
            Some("doubao-seedance-2-0-260128".to_string()),
        )
        .expect("transport");

        assert_eq!(
            transport.headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );
        assert_eq!(
            transport.model_name.as_deref(),
            Some("doubao-seedance-2-0-260128")
        );
        assert_eq!(
            transport
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.url.as_deref()),
            Some("http://provider-proxy.example:8080")
        );
    }

    #[test]
    fn rejects_doubao_video_transport_without_a_nonempty_bearer_token() {
        for secret in ["", "   ", "__placeholder__"] {
            let mut provider_transport = sample_transport("doubao:video", "api_key");
            provider_transport.key.decrypted_api_key = secret.to_string();
            assert!(resolve_local_video_task_transport(
                &provider_transport,
                "doubao:video",
                Some("doubao-seedance-2-0-260128".to_string()),
            )
            .is_none());
            assert!(resolve_video_create_auth(
                &provider_transport,
                ProviderVideoCreateFamily::Doubao,
            )
            .is_none());
        }
    }

    #[test]
    fn rejects_expired_doubao_video_key_for_create_and_follow_ups() {
        let mut provider_transport = sample_transport("doubao:video", "api_key");
        provider_transport.key.expires_at_unix_secs = Some(1);

        assert!(resolve_local_video_task_transport(
            &provider_transport,
            "doubao:video",
            Some("doubao-seedance-2-0-260128".to_string()),
        )
        .is_none());
        assert!(
            resolve_video_create_auth(&provider_transport, ProviderVideoCreateFamily::Doubao,)
                .is_none()
        );
    }

    #[test]
    fn zero_expiry_keeps_legacy_doubao_key_usable() {
        let mut provider_transport = sample_transport("doubao:video", "api_key");
        provider_transport.key.expires_at_unix_secs = Some(0);

        assert!(resolve_local_video_task_transport(
            &provider_transport,
            "doubao:video",
            Some("doubao-seedance-2-0-260128".to_string()),
        )
        .is_some());
        assert!(
            resolve_video_create_auth(&provider_transport, ProviderVideoCreateFamily::Doubao,)
                .is_some()
        );
    }

    #[test]
    fn builds_doubao_video_create_url_from_client_api_v3_path() {
        let mut transport = sample_transport("doubao:video", "bearer");
        transport.endpoint.base_url = "https://ark.cn-beijing.volces.com/api".to_string();
        let url = build_video_create_upstream_url(
            &transport,
            "/api/v3/contents/generations/tasks",
            Some("trace=1"),
            "doubao-seedance-2-0-260128",
            ProviderVideoCreateFamily::Doubao,
        )
        .expect("url should build");

        assert_eq!(
            url,
            "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks?trace=1"
        );
    }

    #[test]
    fn builds_doubao_video_create_body_with_mapped_model_and_passthrough_fields() {
        let body = build_video_create_request_body(
            &json!({
                "model": "client-model",
                "content": [
                    {"type": "text", "text": "a clip"},
                    {"type": "audio_url", "audio_url": {"url": "https://a.example/a.mp3"}, "role": "reference_audio"}
                ],
                "generate_audio": true,
                "ratio": "16:9",
                "duration": 11
            }),
            ProviderVideoCreateFamily::Doubao,
            "doubao-seedance-2-0-260128",
            None,
            None,
        )
        .expect("body should build");

        assert_eq!(
            body.get("model"),
            Some(&json!("doubao-seedance-2-0-260128"))
        );
        // Unmodeled fields must survive verbatim so new Ark parameters keep working.
        assert_eq!(body.get("generate_audio"), Some(&json!(true)));
        assert_eq!(body.get("duration"), Some(&json!(11)));
        assert_eq!(
            body.get("content")
                .and_then(|value| value.as_array())
                .map(Vec::len),
            Some(2)
        );
    }

    #[test]
    fn converts_openai_video_create_body_to_doubao_shape() {
        let body = build_video_create_request_body_for_client(
            &json!({
                "model": "sora-2",
                "prompt": "a cat yawning",
                "size": "1280x720",
                "seconds": "8",
                "input_reference": {"url": "https://example.com/frame.jpg"}
            }),
            ProviderVideoCreateFamily::OpenAi,
            ProviderVideoCreateFamily::Doubao,
            "doubao-seedance-2-0-260128",
            None,
            None,
        )
        .expect("conversion should build");

        assert_eq!(body["model"], "doubao-seedance-2-0-260128");
        assert_eq!(body["ratio"], "16:9");
        assert_eq!(body["resolution"], "720p");
        assert_eq!(body["duration"], 8);
        assert_eq!(body["content"][1]["role"], "first_frame");
    }

    #[test]
    fn rejects_lossy_openai_video_conversion() {
        assert!(build_video_create_request_body_for_client(
            &json!({"prompt": "clip", "remix_video_id": "video-1"}),
            ProviderVideoCreateFamily::OpenAi,
            ProviderVideoCreateFamily::Doubao,
            "doubao-seedance-2-0-260128",
            None,
            None,
        )
        .is_none());
    }

    #[test]
    fn rejects_mismatched_video_transport_format() {
        let transport = sample_transport("openai:chat", "bearer");
        assert!(resolve_local_video_task_transport(&transport, "openai:video", None).is_none());
    }

    #[tokio::test]
    async fn reconstructs_openai_video_snapshot_via_lookup_trait() {
        let mut transport = sample_transport("openai:video", "bearer");
        transport.endpoint.header_rules = Some(json!([
            {"action": "set", "key": "x-tenant-id", "value": "tenant-1"}
        ]));
        let lookup = TestLookup(Some(transport));
        let snapshot = reconstruct_local_video_task_snapshot(&lookup, &sample_stored_video_task())
            .await
            .expect("lookup should succeed")
            .expect("snapshot");

        match snapshot {
            LocalVideoTaskSnapshot::OpenAi(seed) => {
                assert_eq!(seed.transport.provider_id, "provider-1");
                assert_eq!(seed.transport.model_name.as_deref(), Some("sora"));
                assert_eq!(
                    seed.transport
                        .headers
                        .get("x-tenant-id")
                        .map(String::as_str),
                    Some("tenant-1")
                );
            }
            other => panic!("expected openai snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reconstructs_doubao_video_snapshot_via_lookup_trait() {
        let mut task = sample_stored_video_task();
        task.client_api_format = Some("doubao:video".to_string());
        task.provider_api_format = Some("doubao:video".to_string());
        task.model = Some("doubao-seedance-2-0-260128".to_string());
        task.aspect_ratio = Some("16:9".to_string());
        task.duration_seconds = Some(11);

        let lookup = TestLookup(Some(sample_transport("doubao:video", "bearer")));
        let snapshot = reconstruct_local_video_task_snapshot(&lookup, &task)
            .await
            .expect("lookup should succeed")
            .expect("snapshot");

        match snapshot {
            LocalVideoTaskSnapshot::Doubao(seed) => {
                assert_eq!(seed.local_task_id, "task-1");
                assert_eq!(seed.upstream_task_id, "upstream-task-1");
                assert_eq!(seed.ratio.as_deref(), Some("16:9"));
                assert_eq!(seed.duration_seconds, Some(11));
                assert_eq!(seed.transport.provider_id, "provider-1");
            }
            other => panic!("expected doubao snapshot, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn reconstruction_uses_runtime_system_or_tunnel_proxy_resolution() {
        let lookup = TestRuntimeProxyLookup(sample_transport("openai:video", "bearer"));
        let snapshot = reconstruct_local_video_task_snapshot(&lookup, &sample_stored_video_task())
            .await
            .expect("lookup should succeed")
            .expect("snapshot");

        let LocalVideoTaskSnapshot::OpenAi(seed) = snapshot else {
            panic!("expected OpenAI snapshot");
        };
        assert_eq!(
            seed.transport
                .proxy
                .as_ref()
                .and_then(|proxy| proxy.node_id.as_deref()),
            Some("runtime-tunnel-node")
        );
    }

    #[test]
    fn rejects_video_create_custom_path_without_a_follow_up_resource_template() {
        let mut transport = sample_transport("doubao:video", "bearer");
        transport.endpoint.custom_path = Some("/custom/video/create".to_string());

        assert_eq!(
            video_create_transport_unsupported_reason(
                &transport,
                ProviderVideoCreateFamily::Doubao,
                "doubao:video",
            ),
            Some("video_task_custom_path_follow_up_unsupported")
        );
    }
}
