use std::time::Instant;

use aether_contracts::{ExecutionPlan, ExecutionResult, RequestBody};
use aether_data_contracts::repository::provider_catalog::{
    StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
};
use aether_provider_transport::{
    build_volc_action_request, VolcActionRequestInput, VolcActionTransportError,
};
use axum::{
    body::{Body, Bytes},
    http::{self, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::handlers::admin::provider::shared::paths::admin_asset_library_test_key_id;
use crate::handlers::admin::request::{AdminAppState, AdminRequestContext};
use crate::material_assets::{ARK_ASSET_API_FORMAT, ARK_ASSET_REQUIRED_CAPABILITY};
use crate::GatewayError;

const TEST_ACTION: &str = "ListAssetGroups";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AssetLibraryTestRequest {
    endpoint_id: String,
}

#[derive(Debug, Serialize)]
struct AssetLibraryTestResult {
    success: bool,
    action: &'static str,
    provider_id: String,
    endpoint_id: String,
    key_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    status_code: Option<u16>,
    latency_ms: u64,
    request_id: Option<String>,
    total: Option<u64>,
    error_code: Option<String>,
    error_message: Option<String>,
}

pub(super) async fn maybe_handle(
    state: &AdminAppState<'_>,
    request_context: &AdminRequestContext<'_>,
    request_body: Option<&Bytes>,
) -> Result<Option<Response<Body>>, GatewayError> {
    let Some(decision) = request_context.decision() else {
        return Ok(None);
    };
    if decision.route_family.as_deref() != Some("endpoints_manage")
        || decision.route_kind.as_deref() != Some("test_asset_library_connection")
        || request_context.method() != http::Method::POST
        || !request_context
            .path()
            .starts_with("/api/admin/endpoints/keys/")
        || !request_context.path().ends_with("/asset-library/test")
    {
        return Ok(None);
    }

    let Some(key_id) = admin_asset_library_test_key_id(request_context.path()) else {
        return Ok(Some(not_found_response("Key 不存在")));
    };
    let Some(request_body) = request_body.filter(|body| !body.is_empty()) else {
        return Ok(Some(bad_request_response("endpoint_id 为必填字段")));
    };
    let payload = match serde_json::from_slice::<AssetLibraryTestRequest>(request_body) {
        Ok(payload) => payload,
        Err(_) => {
            return Ok(Some(bad_request_response(
                "请求体必须是合法的 JSON 对象，且 endpoint_id 为必填字符串字段",
            )));
        }
    };
    let requested_endpoint_id = payload.endpoint_id.trim();
    if requested_endpoint_id.is_empty() {
        return Ok(Some(bad_request_response("endpoint_id 不能为空")));
    }

    let Some(key) = state
        .read_provider_catalog_keys_by_ids(std::slice::from_ref(&key_id))
        .await?
        .into_iter()
        .next()
    else {
        return Ok(Some(not_found_response(format!("Key {key_id} 不存在"))));
    };
    let Some(provider) = state
        .read_provider_catalog_providers_by_ids(std::slice::from_ref(&key.provider_id))
        .await?
        .into_iter()
        .next()
    else {
        return Ok(Some(not_found_response(format!(
            "Provider {} 不存在",
            key.provider_id
        ))));
    };

    if !provider.is_active {
        return Ok(Some(bad_request_response("Provider 未启用")));
    }
    if !key.is_active {
        return Ok(Some(bad_request_response("Key 未启用")));
    }
    if key
        .expires_at_unix_secs
        .is_some_and(|expires_at| expires_at < crate::clock::current_unix_secs())
    {
        return Ok(Some(bad_request_response("Key 已过期")));
    }
    if !crate::handlers::shared::provider_catalog_key_supports_format(
        &key,
        &provider.provider_type,
        ARK_ASSET_API_FORMAT,
    ) {
        return Ok(Some(bad_request_response(
            "Key 未启用 doubao:asset_library API 格式",
        )));
    }
    if !key_supports_asset_library(key.capabilities.as_ref()) {
        return Ok(Some(bad_request_response(
            "Key 缺少 ark_asset_library 能力",
        )));
    }

    let endpoint = match resolve_endpoint(state, &provider, requested_endpoint_id).await? {
        Ok(endpoint) => endpoint,
        Err(response) => return Ok(Some(response)),
    };
    let Some(transport) = state
        .read_provider_transport_snapshot_uncached(&provider.id, &endpoint.id, &key.id)
        .await?
    else {
        return Ok(Some(
            Json(configuration_failure(
                &provider,
                &endpoint,
                &key,
                "TransportUnavailable",
                "无法读取已保存的 Provider 凭据",
            ))
            .into_response(),
        ));
    };

    if transport.provider.id != provider.id
        || transport.endpoint.id != endpoint.id
        || transport.key.id != key.id
        || !transport.provider.is_active
        || !transport.endpoint.is_active
        || !transport.key.is_active
        || !crate::ai_serving::api_format_alias_matches(
            &transport.endpoint.api_format,
            ARK_ASSET_API_FORMAT,
        )
    {
        return Ok(Some(
            Json(configuration_failure(
                &provider,
                &endpoint,
                &key,
                "TransportMismatch",
                "已保存的 Provider、Endpoint 与 Key 无法组成有效素材库连接",
            ))
            .into_response(),
        ));
    }

    let probe_body = json!({
        "PageNumber": 1,
        "PageSize": 1,
        "Filter": {"GroupType": "AIGC"},
    });
    let request = match build_volc_action_request(VolcActionRequestInput {
        transport: &transport,
        action: TEST_ACTION,
        body: &probe_body,
        request_headers: &HeaderMap::new(),
        request_time: None,
    }) {
        Ok(request) => request,
        Err(error) => {
            let (code, message) = request_build_error(error);
            return Ok(Some(
                Json(configuration_failure(
                    &provider, &endpoint, &key, code, message,
                ))
                .into_response(),
            ));
        }
    };
    let plan = ExecutionPlan {
        request_id: request_context.trace_id().to_string(),
        candidate_id: Some(format!("asset-library-test:{}", key.id)),
        provider_name: Some(provider.name.clone()),
        provider_id: provider.id.clone(),
        endpoint_id: endpoint.id.clone(),
        key_id: key.id.clone(),
        method: "POST".to_string(),
        url: request.url,
        headers: request.headers,
        content_type: Some("application/json".to_string()),
        content_encoding: None,
        body: RequestBody {
            json_body: None,
            body_bytes_b64: Some(
                base64::engine::general_purpose::STANDARD.encode(request.body.as_slice()),
            ),
            body_ref: None,
        },
        stream: false,
        client_api_format: ARK_ASSET_API_FORMAT.to_string(),
        provider_api_format: ARK_ASSET_API_FORMAT.to_string(),
        model_name: None,
        proxy: state
            .resolve_transport_proxy_snapshot_with_tunnel_affinity(&transport)
            .await,
        transport_profile: state.resolve_transport_profile(&transport),
        timeouts: state.resolve_transport_execution_timeouts(&transport),
    };

    let started_at = Instant::now();
    let result = match state
        .execute_execution_runtime_sync_plan(Some(request_context.trace_id()), &plan)
        .await
    {
        Ok(result) => result,
        Err(_) => {
            return Ok(Some(
                Json(AssetLibraryTestResult {
                    success: false,
                    action: TEST_ACTION,
                    provider_id: provider.id,
                    endpoint_id: endpoint.id,
                    key_id: key.id,
                    status_code: None,
                    latency_ms: elapsed_millis(started_at),
                    request_id: None,
                    total: None,
                    error_code: Some("ExecutionRuntimeUnavailable".to_string()),
                    error_message: Some("素材库连接测试暂时不可用".to_string()),
                })
                .into_response(),
            ));
        }
    };
    let latency_ms = result
        .telemetry
        .as_ref()
        .and_then(|telemetry| telemetry.elapsed_ms)
        .unwrap_or_else(|| elapsed_millis(started_at));
    Ok(Some(
        Json(execution_result_payload(
            &provider, &endpoint, &key, &result, latency_ms,
        ))
        .into_response(),
    ))
}

async fn resolve_endpoint(
    state: &AdminAppState<'_>,
    provider: &StoredProviderCatalogProvider,
    endpoint_id: &str,
) -> Result<Result<StoredProviderCatalogEndpoint, Response<Body>>, GatewayError> {
    let Some(endpoint) = state
        .read_provider_catalog_endpoints_by_ids(&[endpoint_id.to_string()])
        .await?
        .into_iter()
        .next()
    else {
        return Ok(Err(not_found_response(format!(
            "Endpoint {endpoint_id} 不存在"
        ))));
    };

    if endpoint.provider_id != provider.id {
        return Ok(Err(bad_request_response(
            "Endpoint 与 Key 不属于同一个 Provider",
        )));
    }
    if !endpoint.is_active {
        return Ok(Err(bad_request_response("Endpoint 未启用")));
    }
    if !crate::ai_serving::api_format_alias_matches(&endpoint.api_format, ARK_ASSET_API_FORMAT) {
        return Ok(Err(bad_request_response(
            "Endpoint 不是 doubao:asset_library 格式",
        )));
    }
    Ok(Ok(endpoint))
}

fn execution_result_payload(
    provider: &StoredProviderCatalogProvider,
    endpoint: &StoredProviderCatalogEndpoint,
    key: &StoredProviderCatalogKey,
    result: &ExecutionResult,
    latency_ms: u64,
) -> AssetLibraryTestResult {
    let body = execution_result_json(result);
    let request_id = body
        .as_ref()
        .and_then(response_request_id)
        .or_else(|| response_header_request_id(result));
    if result.status_code == StatusCode::TEMPORARY_REDIRECT.as_u16() {
        return failure_result(
            provider,
            endpoint,
            key,
            Some(result.status_code),
            latency_ms,
            request_id,
            "UpstreamRedirect",
            "上游返回 307 重定向，Base URL 可能缺少末尾 `/`",
        );
    }

    let provider_error = body.as_ref().and_then(provider_error_value);
    if !(200..300).contains(&result.status_code) || provider_error.is_some() {
        let code = provider_error
            .and_then(provider_error_code)
            .unwrap_or_else(|| status_error_code(result.status_code).to_string());
        let message = safe_error_message(result.status_code, &code);
        return failure_result(
            provider,
            endpoint,
            key,
            Some(result.status_code),
            latency_ms,
            request_id,
            &code,
            message,
        );
    }

    let Some(body) = body else {
        return failure_result(
            provider,
            endpoint,
            key,
            Some(result.status_code),
            latency_ms,
            request_id,
            "InvalidUpstreamResponse",
            "素材库上游未返回有效 JSON",
        );
    };
    if !list_asset_groups_response_is_valid(&body) {
        return failure_result(
            provider,
            endpoint,
            key,
            Some(result.status_code),
            latency_ms,
            request_id,
            "InvalidUpstreamResponse",
            "素材库上游返回的 ListAssetGroups 响应结构无效",
        );
    }
    AssetLibraryTestResult {
        success: true,
        action: TEST_ACTION,
        provider_id: provider.id.clone(),
        endpoint_id: endpoint.id.clone(),
        key_id: key.id.clone(),
        status_code: Some(result.status_code),
        latency_ms,
        request_id,
        total: response_total(&body),
        error_code: None,
        error_message: None,
    }
}

fn configuration_failure(
    provider: &StoredProviderCatalogProvider,
    endpoint: &StoredProviderCatalogEndpoint,
    key: &StoredProviderCatalogKey,
    code: &str,
    message: &str,
) -> AssetLibraryTestResult {
    failure_result(provider, endpoint, key, None, 0, None, code, message)
}

#[allow(clippy::too_many_arguments)]
fn failure_result(
    provider: &StoredProviderCatalogProvider,
    endpoint: &StoredProviderCatalogEndpoint,
    key: &StoredProviderCatalogKey,
    status_code: Option<u16>,
    latency_ms: u64,
    request_id: Option<String>,
    code: &str,
    message: &str,
) -> AssetLibraryTestResult {
    AssetLibraryTestResult {
        success: false,
        action: TEST_ACTION,
        provider_id: provider.id.clone(),
        endpoint_id: endpoint.id.clone(),
        key_id: key.id.clone(),
        status_code,
        latency_ms,
        request_id,
        total: None,
        error_code: Some(code.to_string()),
        error_message: Some(message.to_string()),
    }
}

fn request_build_error(error: VolcActionTransportError) -> (&'static str, &'static str) {
    match error {
        VolcActionTransportError::InvalidBaseUrl | VolcActionTransportError::InvalidCustomPath => (
            "InvalidEndpointConfiguration",
            "素材库 Endpoint 地址配置无效",
        ),
        VolcActionTransportError::UnsupportedAuthType
        | VolcActionTransportError::InvalidCredential
        | VolcActionTransportError::InvalidAkSkConfig
        | VolcActionTransportError::InvalidSigningUrl => {
            ("InvalidProviderCredential", "素材库 Provider 凭据配置无效")
        }
        VolcActionTransportError::BodyRulesApplyFailed
        | VolcActionTransportError::HeaderRulesApplyFailed => {
            ("InvalidEndpointRules", "素材库 Endpoint 请求规则无法应用")
        }
        VolcActionTransportError::InvalidAction | VolcActionTransportError::BodyEncodeFailed => {
            ("RequestBuildFailed", "素材库连接测试请求构建失败")
        }
    }
}

fn key_supports_asset_library(capabilities: Option<&Value>) -> bool {
    match capabilities {
        Some(Value::Array(values)) => values.iter().any(|value| {
            value
                .as_str()
                .is_some_and(|value| value.eq_ignore_ascii_case(ARK_ASSET_REQUIRED_CAPABILITY))
        }),
        Some(Value::Object(values)) => values.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case(ARK_ASSET_REQUIRED_CAPABILITY)
                && match value {
                    Value::Bool(value) => *value,
                    Value::String(value) => value.eq_ignore_ascii_case("true"),
                    Value::Number(value) => value.as_i64().is_some_and(|value| value > 0),
                    _ => false,
                }
        }),
        _ => false,
    }
}

fn execution_result_json(result: &ExecutionResult) -> Option<Value> {
    let body = result.body.as_ref()?;
    if let Some(body) = body.json_body.as_ref() {
        return Some(body.clone());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(body.body_bytes_b64.as_deref()?)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn provider_error_value(body: &Value) -> Option<&Value> {
    body.get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| metadata.get("Error").or_else(|| metadata.get("error")))
        .filter(|error| !error.is_null())
        .or_else(|| body.get("error").filter(|error| !error.is_null()))
        .or_else(|| string_field(body, &["Code", "code"]).map(|_| body))
}

fn provider_error_code(error: &Value) -> Option<String> {
    string_field(error, &["Code", "code", "Type", "type"])
        .and_then(|code| sanitize_identifier(&code))
}

fn response_request_id(body: &Value) -> Option<String> {
    body.get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| string_field(metadata, &["RequestId", "request_id"]))
        .or_else(|| string_field(body, &["RequestId", "request_id"]))
        .and_then(|request_id| sanitize_identifier(&request_id))
}

fn response_header_request_id(result: &ExecutionResult) -> Option<String> {
    ["x-request-id", "x-tt-logid", "x-amzn-requestid"]
        .into_iter()
        .find_map(|name| {
            result
                .headers
                .iter()
                .find(|(header, _)| header.eq_ignore_ascii_case(name))
                .map(|(_, value)| value)
        })
        .and_then(|request_id| sanitize_identifier(request_id))
}

fn response_total(body: &Value) -> Option<u64> {
    crate::material_assets::protocol_api::extract_result(body)
        .unwrap_or(body)
        .as_object()
        .and_then(|result| {
            ["TotalCount", "total_count", "Total", "total"]
                .into_iter()
                .find_map(|name| result.get(name))
        })
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
        })
}

fn list_asset_groups_response_is_valid(body: &Value) -> bool {
    let Some(result) =
        crate::material_assets::protocol_api::extract_result(body).and_then(Value::as_object)
    else {
        return false;
    };
    ["Items", "items", "Groups", "groups"]
        .into_iter()
        .filter_map(|name| result.get(name))
        .any(Value::is_array)
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    let object = value.as_object()?;
    names
        .iter()
        .find_map(|name| object.get(*name))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn sanitize_identifier(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()
        && value.len() <= 128
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.:".contains(character)))
    .then(|| value.to_string())
}

fn status_error_code(status_code: u16) -> &'static str {
    match status_code {
        401 => "UpstreamAuthenticationError",
        403 => "UpstreamAccessDenied",
        404 => "UpstreamNotFound",
        429 => "UpstreamRateLimited",
        500..=599 => "UpstreamServerError",
        _ => "UpstreamError",
    }
}

fn safe_error_message(status_code: u16, code: &str) -> &'static str {
    match code.trim().to_ascii_lowercase().as_str() {
        "subscriptionrequired" => "火山素材库服务未开通，请检查套餐状态",
        "signaturedoesnotmatch" | "invalidsignature" | "requestsignatureinvalid" => {
            "素材库上游签名校验失败，请检查 AK/SK、Region、Service 和系统时间"
        }
        "invalidaccesskeyid"
        | "invalidcredential"
        | "invalidsecuritytoken"
        | "requestexpired"
        | "requesttimetoolarge"
        | "requesttimetooskewed" => "素材库上游凭据无效",
        "accessdenied" | "forbidden" | "unauthorizedoperation" | "permissiondenied" => {
            "素材库上游拒绝访问，请检查账号权限与服务开通状态"
        }
        "throttling" | "ratelimitexceeded" | "requestlimitexceeded" => "素材库上游请求频率受限",
        _ => match status_code {
            401 => "素材库上游凭据无效",
            403 => "素材库上游拒绝访问，请检查账号权限与服务开通状态",
            429 => "素材库上游请求频率受限",
            500..=599 => "素材库上游服务异常",
            _ => "素材库上游请求失败",
        },
    }
}

fn elapsed_millis(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn bad_request_response(detail: impl Into<String>) -> Response<Body> {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "detail": detail.into() })),
    )
        .into_response()
}

fn not_found_response(detail: impl Into<String>) -> Response<Body> {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "detail": detail.into() })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_level_relay_error_keeps_code_and_request_id() {
        let body = json!({
            "code": "SubscriptionRequired",
            "detail": "subscription required",
            "request_id": "relay-request-id",
        });

        let error = provider_error_value(&body).expect("top-level error");
        assert_eq!(
            provider_error_code(error).as_deref(),
            Some("SubscriptionRequired")
        );
        assert_eq!(
            response_request_id(&body).as_deref(),
            Some("relay-request-id")
        );
    }
}
