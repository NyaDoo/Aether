use std::collections::{BTreeMap, HashMap};

use aether_contracts::{ExecutionPlan, RequestBody};
use aether_data_contracts::repository::asset_library::{
    AssetGroupListQuery, AssetListQuery, StoredArkVisualValidationSession, StoredAsset,
    StoredAssetGroup, UpsertArkVisualValidationSessionRecord, UpsertAssetGroupRecord,
    UpsertAssetRecord,
};
use aether_provider_transport::{
    build_volc_action_request, resolve_transport_execution_timeouts, resolve_transport_profile,
    GatewayProviderTransportSnapshot, VolcActionAuth, VolcActionRequestInput,
};
use axum::body::{Body, Bytes};
use axum::http::{self, HeaderMap, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::protocol_api::{build_error_envelope, extract_result, sanitize_action_body};
use super::{action_from_request, ArkAssetAction, ARK_ASSET_API_FORMAT};
use crate::control::GatewayPublicRequestContext;
use crate::{AppState, GatewayError};

const ASSET_URL_TTL_SECS: u64 = 12 * 60 * 60;
const VALIDATION_SESSION_TTL_SECS: u64 = 30 * 60;
const DEFAULT_PAGE_SIZE: usize = 20;
const MAX_PAGE_SIZE: usize = 500;

#[derive(Debug, Clone)]
struct AssetCaller {
    user_id: String,
    api_key_id: Option<String>,
    unrestricted_provider_access: bool,
    allowed_providers: Option<Vec<String>>,
    allowed_api_formats: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default)]
struct CallerAccessPolicy {
    unrestricted: bool,
    allowed_providers: Option<Vec<String>>,
    allowed_api_formats: Option<Vec<String>>,
}

#[derive(Debug)]
struct AssetServiceError {
    status: StatusCode,
    code: &'static str,
    message: String,
    provider_body: Option<Value>,
}

impl AssetServiceError {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            provider_body: None,
        }
    }

    fn provider(status: StatusCode, body: Value) -> Self {
        let upstream_auth = matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN);
        if upstream_auth {
            return Self::new(
                StatusCode::BAD_GATEWAY,
                "UpstreamAuthError",
                "素材库上游凭据无效或无权访问该资源",
            );
        }
        Self {
            status,
            code: "UpstreamError",
            message: "素材库上游请求失败".to_string(),
            provider_body: Some(sanitize_provider_error_body(&body)),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "InvalidParameter", message)
    }

    fn not_found() -> Self {
        Self::new(StatusCode::NOT_FOUND, "ResourceNotFound", "素材不存在")
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "ProviderUnavailable",
            message,
        )
    }
}

struct AssetTransport {
    snapshot: GatewayProviderTransportSnapshot,
    account_binding: String,
    project: Option<String>,
}

struct ActionResponse {
    body: Value,
    request_body: Value,
}

pub(crate) async fn maybe_handle_native_asset_request(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    request_body: Option<&Bytes>,
) -> Option<Response<Body>> {
    let decision = request_context.control_decision.as_ref()?;
    if decision.route_class.as_deref() != Some("ai_public")
        || decision.route_family.as_deref() != Some("doubao")
        || decision.route_kind.as_deref() != Some("asset_library")
    {
        return None;
    }

    let response = match native_caller(request_context).and_then(|caller| {
        validate_public_credential_carriers(headers, request_context).map(|_| caller)
    }) {
        Ok(caller) => {
            let body = match parse_json_body(request_body) {
                Ok(body) => body,
                Err(error) => return Some(native_error_response(error)),
            };
            let uri = request_uri(request_context);
            match action_from_request(&uri, &body)
                .map_err(|error| AssetServiceError::bad_request(error.to_string()))
                .and_then(|action| {
                    sanitize_action_body(&body)
                        .map(|body| (action, body))
                        .map_err(|error| AssetServiceError::bad_request(error.to_string()))
                }) {
                Ok((action, body)) => {
                    match handle_native_action(
                        state,
                        request_context,
                        headers,
                        &caller,
                        action,
                        body,
                    )
                    .await
                    {
                        Ok(body) => json_response(StatusCode::OK, body),
                        Err(error) => native_error_response(error),
                    }
                }
                Err(error) => native_error_response(error),
            }
        }
        Err(error) => native_error_response(error),
    };
    Some(response)
}

pub(crate) async fn maybe_handle_user_asset_request(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    request_body: Option<&Bytes>,
) -> Option<Response<Body>> {
    let decision = request_context.control_decision.as_ref()?;
    if decision.route_class.as_deref() != Some("public_support")
        || decision.route_family.as_deref() != Some("material_assets")
    {
        return None;
    }
    let control_has_valid_api_key = request_context
        .control_decision
        .as_ref()
        .and_then(|decision| decision.auth_context.as_ref())
        .is_some_and(|auth| auth.access_allowed);
    let caller = if !control_has_valid_api_key
        && crate::handlers::public::bearer_is_aether_access_token(headers)
    {
        let auth = match crate::handlers::public::resolve_authenticated_local_user(
            state,
            request_context,
            headers,
        )
        .await
        {
            Ok(auth) => auth,
            Err(response) => return Some(response),
        };
        let policies = match state
            .data
            .resolve_user_effective_list_policies(&auth.user)
            .await
        {
            Ok(policies) => policies,
            Err(error) => {
                return Some(rest_error_response(AssetServiceError::unavailable(
                    format!("用户权限策略读取失败: {error}"),
                )))
            }
        };
        Ok(AssetCaller {
            user_id: auth.user.id,
            api_key_id: None,
            unrestricted_provider_access: false,
            allowed_providers: policies.allowed_providers,
            allowed_api_formats: policies.allowed_api_formats,
        })
    } else {
        native_caller(request_context)
    };
    let response = match caller.and_then(|caller| {
        validate_public_credential_carriers(headers, request_context).map(|_| caller)
    }) {
        Ok(caller) => {
            match handle_rest_request(state, request_context, headers, request_body, caller, false)
                .await
            {
                Ok(response) => response,
                Err(error) => rest_error_response(error),
            }
        }
        Err(error) => rest_error_response(error),
    };
    Some(response)
}

pub(crate) async fn maybe_handle_admin_asset_request(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    request_body: Option<&Bytes>,
) -> Option<Response<Body>> {
    let decision = request_context.control_decision.as_ref()?;
    if decision.route_class.as_deref() != Some("admin_proxy")
        || decision.route_family.as_deref() != Some("material_assets_manage")
    {
        return None;
    }
    let response = match admin_caller(request_context, request_body) {
        Ok(caller) => {
            match handle_rest_request(state, request_context, headers, request_body, caller, true)
                .await
            {
                Ok(response) => response,
                Err(error) => rest_error_response(error),
            }
        }
        Err(error) => rest_error_response(error),
    };
    Some(response)
}

fn native_caller(
    request_context: &GatewayPublicRequestContext,
) -> Result<AssetCaller, AssetServiceError> {
    let auth = request_context
        .control_decision
        .as_ref()
        .and_then(|decision| decision.auth_context.as_ref())
        .filter(|auth| auth.access_allowed)
        .ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Authentication required",
            )
        })?;
    Ok(AssetCaller {
        user_id: auth.user_id.clone(),
        api_key_id: Some(auth.api_key_id.clone()),
        unrestricted_provider_access: false,
        allowed_providers: None,
        allowed_api_formats: None,
    })
}

fn admin_caller(
    request_context: &GatewayPublicRequestContext,
    request_body: Option<&Bytes>,
) -> Result<AssetCaller, AssetServiceError> {
    let decision = request_context
        .control_decision
        .as_ref()
        .and_then(|decision| decision.admin_principal.as_ref())
        .ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Administrator authentication required",
            )
        })?;
    let body = parse_json_body(request_body)?;
    let query_user = query_value(request_context.request_query_string.as_deref(), "user_id");
    let body_user = string_field(&body, &["user_id", "UserId"]);
    let user_id = body_user
        .or(query_user)
        .unwrap_or_else(|| decision.user_id.clone());
    Ok(AssetCaller {
        user_id,
        api_key_id: None,
        unrestricted_provider_access: true,
        allowed_providers: None,
        allowed_api_formats: None,
    })
}

fn validate_public_credential_carriers(
    headers: &HeaderMap,
    request_context: &GatewayPublicRequestContext,
) -> Result<(), AssetServiceError> {
    if request_context
        .request_query_string
        .as_deref()
        .is_some_and(|query| {
            url::form_urlencoded::parse(query.as_bytes()).any(|(key, _)| {
                matches!(
                    key.trim().to_ascii_lowercase().as_str(),
                    "key" | "api_key" | "apikey" | "access_token"
                )
            })
        })
    {
        return Err(AssetServiceError::bad_request(
            "API key must be sent in Authorization, X-Api-Key, or Api-Key",
        ));
    }

    let mut credentials = Vec::new();
    if let Some(value) = header_text(headers, http::header::AUTHORIZATION.as_str()) {
        let Some((scheme, value)) = value.split_once(char::is_whitespace) else {
            return Err(AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Authorization must use Bearer authentication",
            ));
        };
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Authorization must use Bearer authentication",
            ));
        }
        if !value.trim().is_empty() {
            credentials.push(value.trim().to_string());
        }
    }
    for name in ["x-api-key", "api-key"] {
        if let Some(value) = header_text(headers, name).filter(|value| !value.trim().is_empty()) {
            credentials.push(value.trim().to_string());
        }
    }
    credentials.sort();
    credentials.dedup();
    if credentials.len() > 1 {
        return Err(AssetServiceError::bad_request(
            "conflicting API credentials were supplied",
        ));
    }
    Ok(())
}

async fn handle_native_action(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    action: ArkAssetAction,
    body: Value,
) -> Result<Value, AssetServiceError> {
    match action {
        ArkAssetAction::ListAssetGroups => {
            list_groups_native(state, request_context, caller, &body).await
        }
        ArkAssetAction::ListAssets => {
            list_assets_native(state, request_context, caller, &body).await
        }
        ArkAssetAction::CreateAssetGroup => {
            let group = create_group(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                json!({"Group": group_native_json(&group)}),
            ))
        }
        ArkAssetAction::GetAssetGroup => {
            let id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
            let group = refresh_group(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(
                request_context,
                json!({"Group": group_native_json(&group)}),
            ))
        }
        ArkAssetAction::UpdateAssetGroup => {
            let group = update_group(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                json!({"Group": group_native_json(&group)}),
            ))
        }
        ArkAssetAction::DeleteAssetGroup => {
            let id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
            delete_group(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, json!({"GroupId": id})))
        }
        ArkAssetAction::CreateAsset => {
            let asset = create_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                json!({"Asset": asset_native_json(&asset)}),
            ))
        }
        ArkAssetAction::GetAsset => {
            let id = required_string_field(&body, &["AssetId", "asset_id"], "AssetId")?;
            let asset = refresh_asset(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(
                request_context,
                json!({"Asset": asset_native_json(&asset)}),
            ))
        }
        ArkAssetAction::UpdateAsset => {
            let asset = update_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                json!({"Asset": asset_native_json(&asset)}),
            ))
        }
        ArkAssetAction::DeleteAsset => {
            let id = required_string_field(&body, &["AssetId", "asset_id"], "AssetId")?;
            delete_asset(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, json!({"AssetId": id})))
        }
        ArkAssetAction::CreateVisualValidateSession => {
            let (session, upstream) =
                create_validation_session(state, request_context, headers, caller, body).await?;
            let mut result = extract_result(&upstream).cloned().unwrap_or(upstream);
            if let Some(object) = result.as_object_mut() {
                object.insert("SessionId".to_string(), Value::String(session.id));
            }
            Ok(native_envelope(request_context, result))
        }
        ArkAssetAction::GetVisualValidateResult => {
            let upstream =
                get_validation_result_native(state, request_context, headers, caller, &body)
                    .await?;
            Ok(upstream)
        }
    }
}

async fn handle_rest_request(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    request_body: Option<&Bytes>,
    caller: AssetCaller,
    is_admin: bool,
) -> Result<Response<Body>, AssetServiceError> {
    let body = parse_json_body(request_body)?;
    let kind = request_context
        .control_decision
        .as_ref()
        .and_then(|decision| decision.route_kind.as_deref())
        .unwrap_or_default();
    match kind {
        "list_groups" => {
            let page = list_groups_rest(state, request_context, &caller, is_admin).await?;
            Ok(json_response(StatusCode::OK, page))
        }
        "create_group" => {
            if is_admin && body_user_id(&body).is_none() {
                return Err(AssetServiceError::bad_request(
                    "admin group creation requires user_id",
                ));
            }
            let upstream_body = json!({
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
                "Description": string_field(&body, &["description", "Description"]),
                "GroupType": string_field(&body, &["group_type", "GroupType"]).unwrap_or_else(|| "AIGC".to_string()),
            });
            let group =
                create_group(state, request_context, headers, &caller, upstream_body).await?;
            Ok(json_response(
                StatusCode::CREATED,
                group_rest_json(&group, 0),
            ))
        }
        "get_group" => {
            let id = path_resource_id(&request_context.request_path, "groups")?;
            let group = load_group(state, &caller, &id, is_admin).await?;
            let count = group_asset_count(state, &group).await?;
            Ok(json_response(
                StatusCode::OK,
                group_rest_json(&group, count),
            ))
        }
        "update_group" => {
            let id = path_resource_id(&request_context.request_path, "groups")?;
            let mut upstream_body = json!({
                "GroupId": id,
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
            });
            if object_has_field(&body, &["description", "Description"]) {
                upstream_body.as_object_mut().expect("object body").insert(
                    "Description".to_string(),
                    string_field(&body, &["description", "Description"])
                        .map(Value::String)
                        .unwrap_or(Value::Null),
                );
            }
            let group =
                update_group(state, request_context, headers, &caller, upstream_body).await?;
            let count = group_asset_count(state, &group).await?;
            Ok(json_response(
                StatusCode::OK,
                group_rest_json(&group, count),
            ))
        }
        "delete_group" => {
            let id = path_resource_id(&request_context.request_path, "groups")?;
            delete_group(state, request_context, headers, &caller, &id).await?;
            Ok(empty_response(StatusCode::NO_CONTENT))
        }
        "list_assets" => {
            let page = list_assets_rest(state, request_context, &caller, is_admin).await?;
            Ok(json_response(StatusCode::OK, page))
        }
        "create_asset_url" => {
            if is_admin && body_user_id(&body).is_none() {
                return Err(AssetServiceError::bad_request(
                    "admin asset creation requires user_id",
                ));
            }
            let source_url = required_string_field(&body, &["url", "URL"], "url")?;
            validate_source_url(&source_url)?;
            let group_id = required_string_field(&body, &["group_id", "GroupId"], "group_id")?;
            let asset_type = required_asset_type(&body)?;
            let upstream_body = json!({
                "GroupId": group_id,
                "URL": source_url,
                "AssetType": asset_type,
                "Name": string_field(&body, &["name", "Name"]),
            });
            let asset =
                create_asset(state, request_context, headers, &caller, upstream_body).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin)
                .await
                .ok();
            Ok(json_response(
                StatusCode::CREATED,
                asset_rest_json(&asset, group.as_ref(), is_admin),
            ))
        }
        "upload_asset" => Err(AssetServiceError::new(
            StatusCode::NOT_IMPLEMENTED,
            "UnsupportedOperation",
            "Ark 素材库仅支持 URL 创建；本地文件上传需要先接入对象存储",
        )),
        "get_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            let asset = refresh_asset(state, request_context, headers, &caller, &id).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin)
                .await
                .ok();
            Ok(json_response(
                StatusCode::OK,
                asset_rest_json(&asset, group.as_ref(), is_admin),
            ))
        }
        "update_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            let upstream_body = json!({
                "AssetId": id,
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
            });
            let asset =
                update_asset(state, request_context, headers, &caller, upstream_body).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin)
                .await
                .ok();
            Ok(json_response(
                StatusCode::OK,
                asset_rest_json(&asset, group.as_ref(), is_admin),
            ))
        }
        "delete_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            delete_asset(state, request_context, headers, &caller, &id).await?;
            Ok(empty_response(StatusCode::NO_CONTENT))
        }
        "preview_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            preview_asset(state, request_context, headers, &caller, &id).await
        }
        "create_verification_session" => {
            if is_admin && body_user_id(&body).is_none() {
                return Err(AssetServiceError::bad_request(
                    "admin verification creation requires user_id",
                ));
            }
            let mut upstream_body = Map::new();
            if let Some(return_url) = string_field(&body, &["return_url", "ReturnUrl"]) {
                upstream_body.insert("ReturnUrl".to_string(), Value::String(return_url));
            }
            let (session, upstream) = create_validation_session(
                state,
                request_context,
                headers,
                &caller,
                Value::Object(upstream_body),
            )
            .await?;
            Ok(json_response(
                StatusCode::CREATED,
                validation_session_rest_json(state, &session, Some(&upstream))?,
            ))
        }
        "get_verification_session" => {
            let id = path_resource_id(&request_context.request_path, "verification-sessions")?;
            let session =
                refresh_validation_session(state, request_context, headers, &caller, &id, is_admin)
                    .await?;
            Ok(json_response(
                StatusCode::OK,
                validation_session_rest_json(state, &session, None)?,
            ))
        }
        _ => Err(AssetServiceError::new(
            StatusCode::NOT_FOUND,
            "RouteNotFound",
            "material asset route not found",
        )),
    }
}

async fn select_transport(
    state: &AppState,
    caller: &AssetCaller,
) -> Result<AssetTransport, AssetServiceError> {
    let now = crate::clock::current_unix_secs();
    let access_policy = resolve_caller_access_policy(state, caller, now).await?;
    if !access_policy_allows_format(&access_policy, ARK_ASSET_API_FORMAT) {
        return Err(AssetServiceError::new(
            StatusCode::FORBIDDEN,
            "ApiFormatNotAllowed",
            "API key is not allowed to use the Ark asset library",
        ));
    }

    let mut providers = state
        .list_provider_catalog_providers(true)
        .await
        .map_err(gateway_error)?;
    providers.retain(|provider| {
        access_policy_allows_provider(
            &access_policy,
            &provider.id,
            &provider.name,
            &provider.provider_type,
        )
    });
    providers.sort_by(|left, right| {
        left.provider_priority
            .cmp(&right.provider_priority)
            .then(left.id.cmp(&right.id))
    });
    let provider_ids = providers
        .iter()
        .map(|provider| provider.id.clone())
        .collect::<Vec<_>>();
    if provider_ids.is_empty() {
        return Err(AssetServiceError::unavailable(
            "没有可用于素材库的 Provider",
        ));
    }
    let endpoints = state
        .list_provider_catalog_endpoints_by_provider_ids(&provider_ids)
        .await
        .map_err(gateway_error)?;
    let keys = state
        .list_provider_catalog_keys_by_provider_ids(&provider_ids)
        .await
        .map_err(gateway_error)?;
    let provider_by_id = providers
        .iter()
        .map(|provider| (provider.id.as_str(), provider))
        .collect::<HashMap<_, _>>();
    let mut candidates = Vec::new();
    for endpoint in endpoints.iter().filter(|endpoint| {
        endpoint.is_active
            && crate::ai_serving::api_format_alias_matches(
                &endpoint.api_format,
                ARK_ASSET_API_FORMAT,
            )
    }) {
        let Some(provider) = provider_by_id.get(endpoint.provider_id.as_str()) else {
            continue;
        };
        for key in keys.iter().filter(|key| {
            key.provider_id == endpoint.provider_id
                && key.is_active
                && key
                    .expires_at_unix_secs
                    .is_none_or(|expires_at| expires_at >= now)
                && crate::handlers::shared::provider_catalog_key_supports_format(
                    key,
                    &provider.provider_type,
                    ARK_ASSET_API_FORMAT,
                )
                && key_supports_asset_library(key.capabilities.as_ref())
        }) {
            candidates.push((
                provider.provider_priority,
                key.internal_priority,
                provider.id.clone(),
                endpoint.id.clone(),
                key.id.clone(),
            ));
        }
    }
    candidates.sort();
    for (_, _, provider_id, endpoint_id, key_id) in candidates {
        let snapshot = state
            .read_provider_transport_snapshot(&provider_id, &endpoint_id, &key_id)
            .await
            .map_err(gateway_error)?;
        let Some(snapshot) = snapshot else {
            continue;
        };
        if !snapshot.provider.is_active
            || !snapshot.endpoint.is_active
            || !snapshot.key.is_active
            || snapshot
                .key
                .expires_at_unix_secs
                .is_some_and(|expires_at| expires_at < now)
        {
            continue;
        }
        if aether_provider_transport::resolve_volc_action_auth(&snapshot).is_err() {
            continue;
        }
        return Ok(asset_transport(snapshot));
    }
    Err(AssetServiceError::unavailable(
        "没有配置可用的 Ark 素材库 AK/SK、Bearer 或 API Key",
    ))
}

async fn resolve_caller_access_policy(
    state: &AppState,
    caller: &AssetCaller,
    now: u64,
) -> Result<CallerAccessPolicy, AssetServiceError> {
    if caller.unrestricted_provider_access {
        return Ok(CallerAccessPolicy {
            unrestricted: true,
            ..CallerAccessPolicy::default()
        });
    }
    if let Some(api_key_id) = caller.api_key_id.as_deref() {
        let snapshot = state
            .read_cached_auth_api_key_snapshot(&caller.user_id, api_key_id, now)
            .await
            .map_err(gateway_error)?
            .filter(|snapshot| snapshot.currently_usable)
            .ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::UNAUTHORIZED,
                    "Unauthorized",
                    "API key is unavailable",
                )
            })?;
        return Ok(CallerAccessPolicy {
            unrestricted: false,
            allowed_providers: snapshot
                .effective_allowed_providers()
                .map(ToOwned::to_owned),
            allowed_api_formats: snapshot
                .effective_allowed_api_formats()
                .map(ToOwned::to_owned),
        });
    }
    Ok(CallerAccessPolicy {
        unrestricted: false,
        allowed_providers: caller.allowed_providers.clone(),
        allowed_api_formats: caller.allowed_api_formats.clone(),
    })
}

fn access_policy_allows_format(policy: &CallerAccessPolicy, api_format: &str) -> bool {
    policy.unrestricted
        || policy.allowed_api_formats.as_ref().is_none_or(|formats| {
            formats.iter().any(|value| {
                aether_scheduler_core::api_format_matches_allowed_value(value, api_format)
            })
        })
}

fn access_policy_allows_provider(
    policy: &CallerAccessPolicy,
    provider_id: &str,
    provider_name: &str,
    provider_type: &str,
) -> bool {
    policy.unrestricted
        || policy.allowed_providers.as_ref().is_none_or(|providers| {
            providers.iter().any(|value| {
                aether_scheduler_core::provider_matches_allowed_value(
                    value,
                    provider_id,
                    provider_name,
                    provider_type,
                )
            })
        })
}

async fn exact_transport(
    state: &AppState,
    provider_id: &str,
    endpoint_id: &str,
    key_id: &str,
    expected_account_binding: Option<&str>,
    expected_project: Option<&str>,
) -> Result<AssetTransport, AssetServiceError> {
    let snapshot = state
        .read_provider_transport_snapshot(provider_id, endpoint_id, key_id)
        .await
        .map_err(gateway_error)?
        .ok_or_else(|| AssetServiceError::unavailable("创建素材时使用的 Provider 凭据已不可用"))?;
    let now = crate::clock::current_unix_secs();
    let format_allowed = snapshot.key.api_formats.as_ref().is_none_or(|formats| {
        formats.is_empty()
            || formats.iter().any(|format| {
                crate::ai_serving::api_format_permission_covers(format, ARK_ASSET_API_FORMAT)
            })
    });
    if !snapshot.provider.is_active
        || !snapshot.endpoint.is_active
        || !snapshot.key.is_active
        || !crate::ai_serving::api_format_alias_matches(
            &snapshot.endpoint.api_format,
            ARK_ASSET_API_FORMAT,
        )
        || !format_allowed
        || !key_supports_asset_library(snapshot.key.capabilities.as_ref())
        || snapshot
            .key
            .expires_at_unix_secs
            .is_some_and(|expires_at| expires_at < now)
        || aether_provider_transport::resolve_volc_action_auth(&snapshot).is_err()
    {
        return Err(AssetServiceError::unavailable(
            "创建素材时使用的 Provider 凭据已被禁用、撤销或过期",
        ));
    }
    let mut transport = asset_transport(snapshot);
    if expected_account_binding != Some(transport.account_binding.as_str()) {
        return Err(AssetServiceError::unavailable(
            "创建素材时使用的上游账号或项目已发生变化",
        ));
    }
    match (expected_project, transport.project.as_deref()) {
        (Some(expected), Some(current)) if expected != current => {
            return Err(AssetServiceError::unavailable(
                "创建素材时使用的上游账号或项目已发生变化",
            ));
        }
        (None, Some(_)) => {
            return Err(AssetServiceError::unavailable(
                "创建素材时使用的上游账号或项目已发生变化",
            ));
        }
        (Some(expected), None) => transport.project = Some(expected.to_string()),
        _ => {}
    }
    Ok(transport)
}

async fn exact_transport_for_group(
    state: &AppState,
    caller: &AssetCaller,
    group: &StoredAssetGroup,
) -> Result<AssetTransport, AssetServiceError> {
    let transport = exact_transport(
        state,
        &group.provider_id,
        &group.endpoint_id,
        &group.key_id,
        group.account_binding.as_deref(),
        group.project.as_deref(),
    )
    .await?;
    ensure_caller_can_use_transport(state, caller, &transport.snapshot).await?;
    Ok(transport)
}

async fn exact_transport_for_session(
    state: &AppState,
    caller: &AssetCaller,
    session: &StoredArkVisualValidationSession,
) -> Result<AssetTransport, AssetServiceError> {
    let transport = exact_transport(
        state,
        &session.provider_id,
        &session.endpoint_id,
        &session.key_id,
        session.account_binding.as_deref(),
        session.project.as_deref(),
    )
    .await?;
    ensure_caller_can_use_transport(state, caller, &transport.snapshot).await?;
    Ok(transport)
}

async fn ensure_caller_can_use_transport(
    state: &AppState,
    caller: &AssetCaller,
    transport: &GatewayProviderTransportSnapshot,
) -> Result<(), AssetServiceError> {
    let policy =
        resolve_caller_access_policy(state, caller, crate::clock::current_unix_secs()).await?;
    if !access_policy_allows_format(&policy, ARK_ASSET_API_FORMAT) {
        return Err(AssetServiceError::new(
            StatusCode::FORBIDDEN,
            "ApiFormatNotAllowed",
            "当前凭据无权访问 Ark 素材库",
        ));
    }
    if !access_policy_allows_provider(
        &policy,
        &transport.provider.id,
        &transport.provider.name,
        &transport.provider.provider_type,
    ) {
        return Err(AssetServiceError::new(
            StatusCode::FORBIDDEN,
            "ProviderNotAllowed",
            "当前凭据无权访问该素材所属 Provider",
        ));
    }
    Ok(())
}

fn asset_transport(snapshot: GatewayProviderTransportSnapshot) -> AssetTransport {
    let account_binding = action_account_binding(&snapshot);
    let project = action_project(&snapshot);
    AssetTransport {
        snapshot,
        account_binding,
        project,
    }
}

async fn execute_action(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    transport: &AssetTransport,
    action: ArkAssetAction,
    body: &Value,
) -> Result<ActionResponse, AssetServiceError> {
    let mut bound_body = body.clone();
    if let Some(project) = transport.project.as_deref() {
        if let Some(requested) = string_field(&bound_body, &["ProjectName", "project_name"]) {
            if requested != project {
                return Err(AssetServiceError::bad_request(
                    "ProjectName 与素材库凭据绑定的项目不一致",
                ));
            }
        } else if let Some(object) = bound_body.as_object_mut() {
            object.insert(
                "ProjectName".to_string(),
                Value::String(project.to_string()),
            );
        }
    }
    let request = build_volc_action_request(VolcActionRequestInput {
        transport: &transport.snapshot,
        action: action.as_str(),
        body: &bound_body,
        request_headers: headers,
        request_time: None,
    })
    .map_err(|error| AssetServiceError::unavailable(format!("Ark 素材库请求构建失败: {error}")))?;
    if let Some(project) = transport.project.as_deref() {
        let final_project = string_field(&request.body_json, &["ProjectName", "project_name"]);
        if final_project.as_deref() != Some(project) {
            return Err(AssetServiceError::unavailable(
                "素材库 endpoint body rules 改写了已绑定的 ProjectName",
            ));
        }
    }
    let request_body = request.body_json.clone();
    let proxy = state
        .resolve_transport_proxy_snapshot_with_tunnel_affinity(&transport.snapshot)
        .await;
    let plan = ExecutionPlan {
        request_id: request_context.trace_id.clone(),
        candidate_id: None,
        provider_name: Some(transport.snapshot.provider.name.clone()),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
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
        proxy,
        transport_profile: resolve_transport_profile(&transport.snapshot),
        timeouts: resolve_transport_execution_timeouts(&transport.snapshot),
    };
    let result = crate::execution_runtime::execute_execution_runtime_sync_plan(
        state,
        Some(request_context.trace_id.as_str()),
        &plan,
    )
    .await
    .map_err(|_| AssetServiceError::unavailable("Ark 素材库上游请求暂时不可用"))?;
    let status = StatusCode::from_u16(result.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = execution_result_json(&result).unwrap_or(Value::Null);
    if !status.is_success() {
        return Err(AssetServiceError::provider(status, body));
    }
    if provider_error_value(&body).is_some() {
        let status = super::protocol_api::response_status_from_body(&body)
            .and_then(|status| StatusCode::from_u16(status).ok())
            .filter(|status| !status.is_success())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        return Err(AssetServiceError::provider(status, body));
    }
    Ok(ActionResponse { body, request_body })
}

async fn create_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: Value,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let name = required_string_field(&body, &["Name", "name"], "Name")?;
    let group_type = string_field(&body, &["GroupType", "Type", "group_type"])
        .unwrap_or_else(|| "AIGC".to_string());
    if !matches!(group_type.as_str(), "AIGC" | "LivenessFace") {
        return Err(AssetServiceError::bad_request(
            "GroupType must be AIGC or LivenessFace",
        ));
    }
    let transport = select_transport(state, caller).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::CreateAssetGroup,
        &body,
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let upstream_group_id =
        string_field(result, &["GroupId", "Id", "group_id", "id"]).ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark CreateAssetGroup response is missing GroupId",
            )
        })?;
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetGroupRecord {
        id: local_id("agrp"),
        upstream_group_id: Some(upstream_group_id),
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
        account_binding: Some(transport.account_binding),
        project: string_field(&response.request_body, &["ProjectName", "project_name"])
            .or(transport.project),
        group_type,
        name,
        description: string_field(&body, &["Description", "description"]),
        status: string_field(result, &["Status", "status"]).unwrap_or_else(|| "Active".to_string()),
        created_at_unix_secs: number_field(result, &["CreateTime", "CreatedAt"]).unwrap_or(now),
        updated_at_unix_secs: number_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
        deleted_at_unix_secs: None,
    };
    write_repo(state)?
        .upsert_group(record)
        .await
        .map_err(data_error)
}

async fn create_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAsset, AssetServiceError> {
    let group_id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
    let group = load_group(
        state,
        caller,
        &group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let upstream_group_id = group
        .upstream_group_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材组尚未与上游完成绑定"))?;
    let asset_type = required_asset_type(&body)?;
    if let Some(url) = string_field(&body, &["URL", "Url", "url"]) {
        validate_source_url(&url)?;
    }
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "GroupId".to_string(),
            Value::String(upstream_group_id.to_string()),
        );
        object.remove("group_id");
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::CreateAsset,
        &body,
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let upstream_asset_id =
        string_field(result, &["AssetId", "Id", "asset_id", "id"]).ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark CreateAsset response is missing AssetId",
            )
        })?;
    let source_url = string_field(&body, &["URL", "Url", "url"]);
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetRecord {
        id: local_id("asset"),
        upstream_asset_id: Some(upstream_asset_id),
        group_id: group.id,
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        asset_type,
        name: string_field(result, &["Name", "name"])
            .or_else(|| string_field(&body, &["Name", "name"]))
            .unwrap_or_else(|| "未命名素材".to_string()),
        status: string_field(result, &["Status", "status"])
            .unwrap_or_else(|| "Processing".to_string()),
        error_code: error_field(result, "Code"),
        error_message: error_field(result, "Message"),
        moderation: object_field(result, &["ModerationResult", "Moderation", "moderation"]),
        last_inference_at_unix_secs: number_field(
            result,
            &["LastInferenceTime", "LastInferenceAt"],
        ),
        source_url_fingerprint: source_url.as_deref().map(sha256_text),
        provider_url: None,
        provider_url_expires_at_unix_secs: None,
        sanitized_metadata: sanitize_asset_metadata(result),
        is_deleted: false,
        deleted_at_unix_secs: None,
        created_at_unix_secs: number_field(result, &["CreateTime", "CreatedAt"]).unwrap_or(now),
        updated_at_unix_secs: number_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
    };
    write_repo(state)?
        .upsert_asset(record)
        .await
        .map_err(data_error)
}

async fn refresh_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    group_id: &str,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let group = load_group(state, caller, group_id, caller.unrestricted_provider_access).await?;
    let upstream_id = group
        .upstream_group_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材组尚未与上游完成绑定"))?;
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::GetAssetGroup,
        &json!({"GroupId": upstream_id}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    persist_group_projection(state, group, result).await
}

async fn update_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let group_id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
    let group = load_group(
        state,
        caller,
        &group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let upstream_id = group
        .upstream_group_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材组尚未与上游完成绑定"))?;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "GroupId".to_string(),
            Value::String(upstream_id.to_string()),
        );
        object.remove("group_id");
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::UpdateAssetGroup,
        &body,
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let mut updated = group;
    if let Some(name) = string_field(&body, &["Name", "name"]) {
        updated.name = name;
    }
    if body.get("Description").is_some() || body.get("description").is_some() {
        updated.description = string_field(&body, &["Description", "description"]);
    }
    persist_group_projection(state, updated, result).await
}

async fn delete_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    group_id: &str,
) -> Result<(), AssetServiceError> {
    let group = load_group(state, caller, group_id, caller.unrestricted_provider_access).await?;
    let upstream_id = group
        .upstream_group_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材组尚未与上游完成绑定"))?;
    let transport = exact_transport_for_group(state, caller, &group).await?;
    execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::DeleteAssetGroup,
        &json!({"GroupId": upstream_id}),
    )
    .await?;
    let now = crate::clock::current_unix_secs();
    if !write_repo(state)?
        .soft_delete_group(&group.id, now)
        .await
        .map_err(data_error)?
    {
        return Err(AssetServiceError::not_found());
    }
    Ok(())
}

async fn refresh_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    asset_id: &str,
) -> Result<StoredAsset, AssetServiceError> {
    let asset = load_asset(state, caller, asset_id, caller.unrestricted_provider_access).await?;
    let group = load_group(
        state,
        caller,
        &asset.group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let upstream_id = asset
        .upstream_asset_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材尚未与上游完成绑定"))?;
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::GetAsset,
        &json!({"AssetId": upstream_id}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    persist_asset_projection(state, asset, result).await
}

async fn update_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAsset, AssetServiceError> {
    let asset_id = required_string_field(&body, &["AssetId", "asset_id"], "AssetId")?;
    let asset = load_asset(
        state,
        caller,
        &asset_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let group = load_group(
        state,
        caller,
        &asset.group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let upstream_id = asset
        .upstream_asset_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材尚未与上游完成绑定"))?;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "AssetId".to_string(),
            Value::String(upstream_id.to_string()),
        );
        object.remove("asset_id");
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::UpdateAsset,
        &body,
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let mut updated = asset;
    if let Some(name) = string_field(&body, &["Name", "name"]) {
        updated.name = name;
    }
    persist_asset_projection(state, updated, result).await
}

async fn delete_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    asset_id: &str,
) -> Result<(), AssetServiceError> {
    let asset = load_asset(state, caller, asset_id, caller.unrestricted_provider_access).await?;
    let group = load_group(
        state,
        caller,
        &asset.group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let upstream_id = asset
        .upstream_asset_id
        .as_deref()
        .ok_or_else(|| AssetServiceError::unavailable("素材尚未与上游完成绑定"))?;
    let transport = exact_transport_for_group(state, caller, &group).await?;
    execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::DeleteAsset,
        &json!({"AssetId": upstream_id}),
    )
    .await?;
    let now = crate::clock::current_unix_secs();
    if !write_repo(state)?
        .soft_delete_asset(&asset.id, now)
        .await
        .map_err(data_error)?
    {
        return Err(AssetServiceError::not_found());
    }
    Ok(())
}

async fn persist_group_projection(
    state: &AppState,
    group: StoredAssetGroup,
    result: &Value,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetGroupRecord {
        id: group.id,
        upstream_group_id: group.upstream_group_id,
        user_id: group.user_id,
        api_key_id: group.api_key_id,
        provider_id: group.provider_id,
        endpoint_id: group.endpoint_id,
        key_id: group.key_id,
        account_binding: group.account_binding,
        project: group.project,
        group_type: string_field(result, &["GroupType", "Type"]).unwrap_or(group.group_type),
        name: string_field(result, &["Name", "name"]).unwrap_or(group.name),
        description: string_field(result, &["Description", "description"]).or(group.description),
        status: string_field(result, &["Status", "status"]).unwrap_or(group.status),
        created_at_unix_secs: number_field(result, &["CreateTime", "CreatedAt"])
            .unwrap_or(group.created_at_unix_secs),
        updated_at_unix_secs: number_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
        deleted_at_unix_secs: group.deleted_at_unix_secs,
    };
    write_repo(state)?
        .upsert_group(record)
        .await
        .map_err(data_error)
}

async fn persist_asset_projection(
    state: &AppState,
    asset: StoredAsset,
    result: &Value,
) -> Result<StoredAsset, AssetServiceError> {
    let now = crate::clock::current_unix_secs();
    let next_url = provider_url(result);
    let record = UpsertAssetRecord {
        id: asset.id,
        upstream_asset_id: asset.upstream_asset_id,
        group_id: asset.group_id,
        user_id: asset.user_id,
        api_key_id: asset.api_key_id,
        asset_type: string_field(result, &["AssetType", "Type", "asset_type"])
            .unwrap_or(asset.asset_type),
        name: string_field(result, &["Name", "name"]).unwrap_or(asset.name),
        status: string_field(result, &["Status", "status"]).unwrap_or(asset.status),
        error_code: error_field(result, "Code").or(asset.error_code),
        error_message: error_field(result, "Message").or(asset.error_message),
        moderation: object_field(result, &["ModerationResult", "Moderation", "moderation"])
            .or(asset.moderation),
        last_inference_at_unix_secs: number_field(
            result,
            &["LastInferenceTime", "LastInferenceAt"],
        )
        .or(asset.last_inference_at_unix_secs),
        source_url_fingerprint: asset.source_url_fingerprint,
        provider_url: None,
        provider_url_expires_at_unix_secs: None,
        sanitized_metadata: sanitize_asset_metadata(result).or(asset.sanitized_metadata),
        is_deleted: asset.is_deleted,
        deleted_at_unix_secs: asset.deleted_at_unix_secs,
        created_at_unix_secs: number_field(result, &["CreateTime", "CreatedAt"])
            .unwrap_or(asset.created_at_unix_secs),
        updated_at_unix_secs: number_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
    };
    let mut persisted = write_repo(state)?
        .upsert_asset(record)
        .await
        .map_err(data_error)?;
    persisted.provider_url = next_url;
    persisted.provider_url_expires_at_unix_secs = persisted
        .provider_url
        .as_ref()
        .map(|_| now.saturating_add(ASSET_URL_TTL_SECS));
    Ok(persisted)
}

async fn list_groups_native(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let page = page_number(body);
    let page_size = page_size(body);
    let filter = native_list_filter(body);
    let query = AssetGroupListQuery {
        user_id: Some(caller.user_id.clone()),
        api_key_id: None,
        provider_id: None,
        group_type: string_field(filter, &["GroupType", "Type"]),
        status: string_field(filter, &["Status"]),
        search: string_field(filter, &["Name", "Search"]),
        include_deleted: false,
        offset: (page - 1).saturating_mul(page_size),
        limit: page_size,
    };
    let response = read_repo(state)?
        .list_groups(&query)
        .await
        .map_err(data_error)?;
    let items = response
        .items
        .iter()
        .map(group_native_json)
        .collect::<Vec<_>>();
    Ok(native_envelope(
        request_context,
        json!({"Total": response.total, "Items": items, "Groups": items}),
    ))
}

async fn list_assets_native(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let page = page_number(body);
    let page_size = page_size(body);
    let filter = native_list_filter(body);
    let group_id = string_field(filter, &["GroupId", "group_id"]);
    if let Some(group_id) = group_id.as_deref() {
        let _ = load_group(state, caller, group_id, caller.unrestricted_provider_access).await?;
    }
    let query = AssetListQuery {
        group_id,
        user_id: Some(caller.user_id.clone()),
        api_key_id: None,
        asset_type: string_field(filter, &["AssetType", "Type"]),
        status: string_field(filter, &["Status"]),
        search: string_field(filter, &["Name", "Search"]),
        include_deleted: false,
        offset: (page - 1).saturating_mul(page_size),
        limit: page_size,
    };
    let response = read_repo(state)?
        .list_assets(&query)
        .await
        .map_err(data_error)?;
    let items = response
        .items
        .iter()
        .map(asset_native_json)
        .collect::<Vec<_>>();
    Ok(native_envelope(
        request_context,
        json!({"Total": response.total, "Items": items, "Assets": items}),
    ))
}

async fn list_groups_rest(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    caller: &AssetCaller,
    _is_admin: bool,
) -> Result<Value, AssetServiceError> {
    let query = AssetGroupListQuery {
        user_id: Some(caller.user_id.clone()),
        api_key_id: None,
        provider_id: None,
        group_type: query_value(
            request_context.request_query_string.as_deref(),
            "group_type",
        ),
        status: query_value(request_context.request_query_string.as_deref(), "status"),
        search: query_value(request_context.request_query_string.as_deref(), "search"),
        include_deleted: false,
        offset: 0,
        limit: MAX_PAGE_SIZE,
    };
    let response = read_repo(state)?
        .list_groups(&query)
        .await
        .map_err(data_error)?;
    let mut counts = HashMap::<String, usize>::with_capacity(response.items.len());
    for group in &response.items {
        let total = read_repo(state)?
            .list_assets(&AssetListQuery {
                group_id: Some(group.id.clone()),
                user_id: Some(caller.user_id.clone()),
                include_deleted: false,
                offset: 0,
                limit: 1,
                ..AssetListQuery::default()
            })
            .await
            .map_err(data_error)?
            .total;
        counts.insert(group.id.clone(), total);
    }
    Ok(json!({
        "items": response.items.iter().map(|group| {
            group_rest_json(group, counts.get(&group.id).copied().unwrap_or_default())
        }).collect::<Vec<_>>(),
        "total": response.total,
    }))
}

async fn list_assets_rest(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    caller: &AssetCaller,
    is_admin: bool,
) -> Result<Value, AssetServiceError> {
    let page = query_value(request_context.request_query_string.as_deref(), "page")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);
    let page_size = query_value(request_context.request_query_string.as_deref(), "page_size")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PAGE_SIZE)
        .clamp(1, MAX_PAGE_SIZE);
    let group_id = query_value(request_context.request_query_string.as_deref(), "group_id");
    if let Some(group_id) = group_id.as_deref() {
        let _ = load_group(state, caller, group_id, is_admin).await?;
    }
    let response = read_repo(state)?
        .list_assets(&AssetListQuery {
            group_id,
            user_id: Some(caller.user_id.clone()),
            api_key_id: None,
            asset_type: query_value(request_context.request_query_string.as_deref(), "type")
                .map(normalize_asset_type_filter),
            status: query_value(request_context.request_query_string.as_deref(), "status"),
            search: query_value(request_context.request_query_string.as_deref(), "search"),
            include_deleted: false,
            offset: (page - 1).saturating_mul(page_size),
            limit: page_size,
        })
        .await
        .map_err(data_error)?;
    let groups = read_repo(state)?
        .list_groups(&AssetGroupListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            offset: 0,
            limit: MAX_PAGE_SIZE,
            ..AssetGroupListQuery::default()
        })
        .await
        .map_err(data_error)?;
    let groups = groups
        .items
        .into_iter()
        .map(|group| (group.id.clone(), group))
        .collect::<HashMap<_, _>>();
    Ok(json!({
        "items": response.items.iter().map(|asset| {
            asset_rest_json(asset, groups.get(&asset.group_id), is_admin)
        }).collect::<Vec<_>>(),
        "total": response.total,
        "page": page,
        "page_size": page_size,
        "pages": response.total.div_ceil(page_size),
    }))
}

async fn load_group(
    state: &AppState,
    caller: &AssetCaller,
    group_id: &str,
    is_admin: bool,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let group = if is_admin {
        read_repo(state)?.find_group_by_id(group_id).await
    } else {
        read_repo(state)?
            .find_group_for_user(group_id, &caller.user_id)
            .await
    }
    .map_err(data_error)?
    .filter(|group| !is_admin || group.user_id == caller.user_id)
    .filter(|group| group.deleted_at_unix_secs.is_none())
    .ok_or_else(AssetServiceError::not_found)?;
    Ok(group)
}

async fn load_asset(
    state: &AppState,
    caller: &AssetCaller,
    asset_id: &str,
    is_admin: bool,
) -> Result<StoredAsset, AssetServiceError> {
    let asset = if is_admin {
        read_repo(state)?.find_asset_by_id(asset_id).await
    } else {
        read_repo(state)?
            .find_asset_for_user(asset_id, &caller.user_id)
            .await
    }
    .map_err(data_error)?
    .filter(|asset| !is_admin || asset.user_id == caller.user_id)
    .filter(|asset| !asset.is_deleted)
    .ok_or_else(AssetServiceError::not_found)?;
    Ok(asset)
}

async fn group_asset_count(
    state: &AppState,
    group: &StoredAssetGroup,
) -> Result<usize, AssetServiceError> {
    Ok(read_repo(state)?
        .list_assets(&AssetListQuery {
            group_id: Some(group.id.clone()),
            user_id: Some(group.user_id.clone()),
            include_deleted: false,
            offset: 0,
            limit: 1,
            ..AssetListQuery::default()
        })
        .await
        .map_err(data_error)?
        .total)
}

async fn create_validation_session(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: Value,
) -> Result<(StoredArkVisualValidationSession, Value), AssetServiceError> {
    let transport = select_transport(state, caller).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::CreateVisualValidateSession,
        &body,
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let byted_token = string_field(result, &["BytedToken", "Token", "byted_token", "token"])
        .ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark CreateVisualValidateSession response is missing BytedToken",
            )
        })?;
    let session_id = string_field(result, &["SessionId", "session_id", "Id", "id"])
        .unwrap_or_else(|| sha256_text(&byted_token)[..32].to_string());
    let encryption_key = state
        .encryption_key()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AssetServiceError::unavailable("素材库真人验证需要配置数据加密密钥"))?;
    let encrypted_byted_token =
        aether_crypto::encrypt_python_fernet_plaintext(encryption_key, &byted_token).map_err(
            |error| AssetServiceError::unavailable(format!("真人验证 token 加密失败: {error}")),
        )?;
    let encrypted_verification_url = validation_verification_url(result)
        .map(|url| {
            aether_crypto::encrypt_python_fernet_plaintext(encryption_key, &url).map_err(|error| {
                AssetServiceError::unavailable(format!("真人验证链接加密失败: {error}"))
            })
        })
        .transpose()?;
    let callback_state = local_id("vstate");
    let now = crate::clock::current_unix_secs();
    let expires_at = number_field(result, &["ExpireAt", "ExpiresAt", "Expiration"])
        .unwrap_or_else(|| now.saturating_add(VALIDATION_SESSION_TTL_SECS));
    let record = UpsertArkVisualValidationSessionRecord {
        id: local_id("vsess"),
        session_id,
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
        account_binding: Some(transport.account_binding),
        project: string_field(&response.request_body, &["ProjectName", "project_name"])
            .or(transport.project),
        byted_token_hash: sha256_text(&byted_token),
        encrypted_byted_token,
        callback_state_hash: sha256_text(&callback_state),
        status: string_field(result, &["Status", "status"])
            .unwrap_or_else(|| "Pending".to_string()),
        expires_at_unix_secs: expires_at.max(now.saturating_add(1)),
        consumed_at_unix_secs: None,
        group_id: None,
        sanitized_result: merge_sanitized_validation_result(
            None,
            result,
            None,
            encrypted_verification_url,
        ),
        created_at_unix_secs: now,
        updated_at_unix_secs: now,
    };
    let session = write_repo(state)?
        .upsert_visual_validation_session(record)
        .await
        .map_err(data_error)?;
    Ok((session, response.body))
}

async fn get_validation_result_native(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let token = required_string_field(
        body,
        &["BytedToken", "Token", "byted_token", "token"],
        "BytedToken",
    )?;
    let session = read_repo(state)?
        .find_visual_validation_session_by_byted_token_hash(&sha256_text(&token))
        .await
        .map_err(data_error)?
        .filter(|session| session.user_id == caller.user_id)
        .ok_or_else(AssetServiceError::not_found)?;
    if validation_session_is_terminal(&session) {
        return Ok(native_envelope(
            request_context,
            public_validation_result(&session),
        ));
    }
    if session.expires_at_unix_secs <= crate::clock::current_unix_secs() {
        return Err(AssetServiceError::new(
            StatusCode::GONE,
            "ValidationSessionExpired",
            "真人验证会话已过期，请重新创建",
        ));
    }
    let response =
        poll_validation_session(state, request_context, headers, caller, session, token).await?;
    Ok(response)
}

async fn refresh_validation_session(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    id: &str,
    is_admin: bool,
) -> Result<StoredArkVisualValidationSession, AssetServiceError> {
    let session = if is_admin {
        read_repo(state)?
            .find_visual_validation_session_by_id(id)
            .await
    } else {
        read_repo(state)?
            .find_visual_validation_session_for_user(id, &caller.user_id)
            .await
    }
    .map_err(data_error)?
    .filter(|session| !is_admin || session.user_id == caller.user_id)
    .ok_or_else(AssetServiceError::not_found)?;
    if validation_session_is_terminal(&session) {
        return Ok(session);
    }
    if session.expires_at_unix_secs <= crate::clock::current_unix_secs() {
        return Err(AssetServiceError::new(
            StatusCode::GONE,
            "ValidationSessionExpired",
            "真人验证会话已过期，请重新创建",
        ));
    }
    let encryption_key = state
        .encryption_key()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AssetServiceError::unavailable("数据加密密钥不可用"))?;
    let token = aether_crypto::decrypt_python_fernet_ciphertext(
        encryption_key,
        &session.encrypted_byted_token,
    )
    .map_err(|_| AssetServiceError::unavailable("真人验证 token 无法解密"))?;
    let owner_caller;
    let polling_caller = if is_admin {
        owner_caller = AssetCaller {
            user_id: session.user_id.clone(),
            api_key_id: session.api_key_id.clone(),
            unrestricted_provider_access: true,
            allowed_providers: None,
            allowed_api_formats: None,
        };
        &owner_caller
    } else {
        caller
    };
    let _ = poll_validation_session(
        state,
        request_context,
        headers,
        polling_caller,
        session,
        token,
    )
    .await?;
    read_repo(state)?
        .find_visual_validation_session_by_id(id)
        .await
        .map_err(data_error)?
        .ok_or_else(AssetServiceError::not_found)
}

async fn poll_validation_session(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    session: StoredArkVisualValidationSession,
    token: String,
) -> Result<Value, AssetServiceError> {
    if session.user_id != caller.user_id {
        return Err(AssetServiceError::not_found());
    }
    let transport = exact_transport_for_session(state, caller, &session).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        &transport,
        ArkAssetAction::GetVisualValidateResult,
        &json!({"BytedToken": token}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let status = validation_result_status(result, &session.status);
    let upstream_group_id = string_field(result, &["GroupId", "group_id"]);
    let group_id = if let Some(upstream_group_id) = upstream_group_id.as_deref() {
        ensure_validation_group(state, caller, &transport, upstream_group_id, result)
            .await?
            .map(|group| group.id)
    } else {
        session.group_id.clone()
    };
    let projected_group_id = group_id.clone();
    let now = crate::clock::current_unix_secs();
    let record = UpsertArkVisualValidationSessionRecord {
        id: session.id,
        session_id: session.session_id,
        user_id: session.user_id,
        api_key_id: session.api_key_id,
        provider_id: session.provider_id,
        endpoint_id: session.endpoint_id,
        key_id: session.key_id,
        account_binding: session.account_binding,
        project: session.project,
        byted_token_hash: session.byted_token_hash,
        encrypted_byted_token: session.encrypted_byted_token,
        callback_state_hash: session.callback_state_hash,
        status: status.clone(),
        expires_at_unix_secs: session.expires_at_unix_secs,
        consumed_at_unix_secs: matches!(
            status.to_ascii_lowercase().as_str(),
            "succeeded" | "success" | "failed"
        )
        .then_some(now),
        group_id,
        sanitized_result: merge_sanitized_validation_result(
            session.sanitized_result,
            result,
            projected_group_id.clone(),
            None,
        ),
        created_at_unix_secs: session.created_at_unix_secs,
        updated_at_unix_secs: now,
    };
    write_repo(state)?
        .upsert_visual_validation_session(record)
        .await
        .map_err(data_error)?;
    let mut projected = response.body;
    if let Some(group_id) = projected_group_id {
        let mut replaced = false;
        if let Some(object) = projected.get_mut("Result").and_then(Value::as_object_mut) {
            object.insert("GroupId".to_string(), Value::String(group_id.clone()));
            object.remove("group_id");
            replaced = true;
        }
        if !replaced {
            if let Some(object) = projected.get_mut("result").and_then(Value::as_object_mut) {
                object.insert("GroupId".to_string(), Value::String(group_id.clone()));
                object.remove("group_id");
                replaced = true;
            }
        }
        if !replaced {
            if let Some(object) = projected.as_object_mut() {
                object.insert("GroupId".to_string(), Value::String(group_id));
                object.remove("group_id");
            }
        }
    }
    Ok(projected)
}

async fn ensure_validation_group(
    state: &AppState,
    caller: &AssetCaller,
    transport: &AssetTransport,
    upstream_group_id: &str,
    result: &Value,
) -> Result<Option<StoredAssetGroup>, AssetServiceError> {
    if let Some(group) = read_repo(state)?
        .find_group_by_canonical_upstream(
            &transport.snapshot.provider.id,
            &transport.account_binding,
            transport.project.as_deref(),
            upstream_group_id,
        )
        .await
        .map_err(data_error)?
    {
        if group.user_id != caller.user_id
            || group.deleted_at_unix_secs.is_some()
            || group.account_binding.as_deref() != Some(transport.account_binding.as_str())
            || group.project != transport.project
        {
            return Err(AssetServiceError::new(
                StatusCode::CONFLICT,
                "AssetGroupOwnershipConflict",
                "真人验证返回的素材组已绑定到其他用户、账号或项目",
            ));
        }
        return Ok(Some(group));
    }
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetGroupRecord {
        id: local_id("agrp"),
        upstream_group_id: Some(upstream_group_id.to_string()),
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
        account_binding: Some(transport.account_binding.clone()),
        project: transport.project.clone(),
        group_type: "LivenessFace".to_string(),
        name: string_field(result, &["GroupName", "Name"])
            .unwrap_or_else(|| "真人素材".to_string()),
        description: None,
        status: "Active".to_string(),
        created_at_unix_secs: now,
        updated_at_unix_secs: now,
        deleted_at_unix_secs: None,
    };
    write_repo(state)?
        .upsert_group(record)
        .await
        .map(Some)
        .map_err(data_error)
}

async fn preview_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    asset_id: &str,
) -> Result<Response<Body>, AssetServiceError> {
    let asset = refresh_asset(state, request_context, headers, caller, asset_id).await?;
    let url = asset.provider_url.as_deref().ok_or_else(|| {
        AssetServiceError::new(
            StatusCode::CONFLICT,
            "AssetContentUnavailable",
            "素材尚未生成可预览内容",
        )
    })?;
    let group = load_group(
        state,
        caller,
        &asset.group_id,
        caller.unrestricted_provider_access,
    )
    .await?;
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let proxy = state
        .resolve_transport_proxy_snapshot_with_tunnel_affinity(&transport.snapshot)
        .await;
    let transport_profile = resolve_transport_profile(&transport.snapshot);
    let mut request_headers = BTreeMap::from([(
        aether_contracts::EXECUTION_REQUEST_DIRECT_VIDEO_ASSET_HEADER.to_string(),
        "1".to_string(),
    )]);
    if let Some(range) = header_text(headers, http::header::RANGE.as_str()) {
        request_headers.insert(http::header::RANGE.as_str().to_string(), range);
    }
    let plan = ExecutionPlan {
        request_id: request_context.trace_id.clone(),
        candidate_id: None,
        provider_name: Some(transport.snapshot.provider.name.clone()),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
        method: "GET".to_string(),
        url: url.to_string(),
        headers: request_headers,
        content_type: None,
        content_encoding: None,
        body: RequestBody::from_json(Value::Null),
        stream: true,
        client_api_format: ARK_ASSET_API_FORMAT.to_string(),
        provider_api_format: ARK_ASSET_API_FORMAT.to_string(),
        model_name: None,
        proxy,
        transport_profile,
        timeouts: resolve_transport_execution_timeouts(&transport.snapshot),
    };
    crate::execution_runtime::execute_direct_asset_response(&plan)
        .await
        .map_err(|error| AssetServiceError::unavailable(error.to_string()))
}

pub(crate) async fn project_video_asset_references(
    state: &AppState,
    user_id: &str,
    transport: &GatewayProviderTransportSnapshot,
    body: &Value,
) -> Result<Value, String> {
    let mut projected = body.clone();
    let Some(content) = projected
        .as_object_mut()
        .and_then(|object| object.get_mut("content"))
    else {
        return Ok(projected);
    };
    let mut local_ids = Vec::new();
    collect_asset_reference_ids(content, &mut local_ids);
    local_ids.sort();
    local_ids.dedup();
    if local_ids.is_empty() {
        return Ok(projected);
    }
    let account_binding = action_account_binding(transport);
    let configured_project = action_project(transport);
    let request_project = string_field(body, &["ProjectName", "project_name"]);
    if configured_project.is_some()
        && request_project.is_some()
        && configured_project != request_project
    {
        return Err("视频请求项目与视频凭据配置的 Ark Project 不一致".to_string());
    }
    let project = configured_project.or(request_project);
    let reader = state
        .data
        .asset_library_read_repository()
        .ok_or_else(|| "素材库数据读取服务不可用".to_string())?;
    let mut replacements = HashMap::new();
    for local_id in local_ids {
        let asset = reader
            .find_asset_for_user(&local_id, user_id)
            .await
            .map_err(|error| format!("读取素材 {local_id} 失败: {error}"))?
            .filter(|asset| !asset.is_deleted)
            .ok_or_else(|| format!("素材 {local_id} 不存在或不属于当前用户"))?;
        if !asset.status.eq_ignore_ascii_case("Active") {
            return Err(format!(
                "素材 {local_id} 当前状态为 {}，必须为 Active",
                asset.status
            ));
        }
        let group = reader
            .find_group_for_user(&asset.group_id, user_id)
            .await
            .map_err(|error| format!("读取素材组失败: {error}"))?
            .filter(|group| group.deleted_at_unix_secs.is_none())
            .ok_or_else(|| format!("素材 {local_id} 所属素材组不存在"))?;
        if group.provider_id != transport.provider.id
            || group.account_binding.as_deref() != Some(account_binding.as_str())
            || group.project != project
        {
            return Err(format!(
                "素材 {local_id} 与本次视频生成的上游账号或项目不一致"
            ));
        }
        let upstream_id = asset
            .upstream_asset_id
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| format!("素材 {local_id} 尚未完成上游绑定"))?;
        replacements.insert(local_id, upstream_id);
    }
    replace_asset_reference_ids(content, &replacements);
    Ok(projected)
}

fn collect_asset_reference_ids(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(value) => {
            if let Some(id) = value
                .strip_prefix("asset://")
                .map(str::trim)
                .filter(|id| !id.is_empty())
            {
                output.push(id.to_string());
            }
        }
        Value::Array(values) => {
            for value in values {
                collect_asset_reference_ids(value, output);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_asset_reference_ids(value, output);
            }
        }
        _ => {}
    }
}

fn replace_asset_reference_ids(value: &mut Value, replacements: &HashMap<String, String>) {
    match value {
        Value::String(value) => {
            if let Some(id) = value.strip_prefix("asset://").map(str::trim) {
                if let Some(upstream_id) = replacements.get(id) {
                    *value = format!("asset://{upstream_id}");
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                replace_asset_reference_ids(value, replacements);
            }
        }
        Value::Object(values) => {
            for value in values.values_mut() {
                replace_asset_reference_ids(value, replacements);
            }
        }
        _ => {}
    }
}

fn group_native_json(group: &StoredAssetGroup) -> Value {
    json!({
        "GroupId": group.id,
        "GroupType": group.group_type,
        "Name": group.name,
        "Description": group.description,
        "Status": group.status,
        "ProjectName": group.project,
        "CreateTime": group.created_at_unix_secs,
        "UpdateTime": group.updated_at_unix_secs,
    })
}

fn asset_native_json(asset: &StoredAsset) -> Value {
    let metadata = asset.sanitized_metadata.as_ref();
    json!({
        "AssetId": asset.id,
        "GroupId": asset.group_id,
        "AssetType": asset.asset_type,
        "Name": asset.name,
        "Status": asset.status,
        "URL": format!("/api/material-assets/assets/{}/preview", asset.id),
        "Error": if asset.error_code.is_some() || asset.error_message.is_some() {
            Some(json!({"Code": asset.error_code, "Message": asset.error_message}))
        } else { None },
        "ModerationResult": asset.moderation,
        "Moderation": asset.moderation,
        "MimeType": metadata.and_then(|value| value_field(value, &["MimeType", "mime_type"])).cloned(),
        "Size": metadata.and_then(|value| value_field(value, &["Size", "Bytes", "size_bytes"])).cloned(),
        "Width": metadata.and_then(|value| value_field(value, &["Width", "width"])).cloned(),
        "Height": metadata.and_then(|value| value_field(value, &["Height", "height"])).cloned(),
        "Duration": metadata.and_then(|value| value_field(value, &["Duration", "duration"])).cloned(),
        "CreateTime": asset.created_at_unix_secs,
        "UpdateTime": asset.updated_at_unix_secs,
    })
}

fn group_rest_json(group: &StoredAssetGroup, asset_count: usize) -> Value {
    json!({
        "id": group.id,
        "name": group.name,
        "description": group.description,
        "group_type": group.group_type,
        "status": group.status,
        "asset_count": asset_count,
        "created_at": unix_secs_rfc3339(group.created_at_unix_secs),
        "updated_at": unix_secs_rfc3339(group.updated_at_unix_secs),
    })
}

fn asset_rest_json(asset: &StoredAsset, group: Option<&StoredAssetGroup>, is_admin: bool) -> Value {
    let metadata = asset.sanitized_metadata.as_ref();
    let mut body = json!({
        "id": asset.id,
        "uri": format!("asset://{}", asset.id),
        "name": asset.name,
        "status": asset.status,
        "asset_type": asset.asset_type,
        "media_type": asset.asset_type.to_ascii_lowercase(),
        "mime_type": metadata.and_then(|value| value_field(value, &["MimeType", "mime_type"])).cloned(),
        "group_id": asset.group_id,
        "group_name": group.map(|group| group.name.as_str()),
        "source_type": "url",
        "size_bytes": metadata.and_then(|value| value_field(value, &["Size", "Bytes", "size_bytes"])).cloned(),
        "width": metadata.and_then(|value| value_field(value, &["Width", "width"])).cloned(),
        "height": metadata.and_then(|value| value_field(value, &["Height", "height"])).cloned(),
        "duration_seconds": metadata.and_then(|value| value_field(value, &["Duration", "duration"])).cloned(),
        "error": if asset.error_code.is_some() || asset.error_message.is_some() {
            Some(json!({"code": asset.error_code, "message": asset.error_message}))
        } else { None },
        "error_code": asset.error_code,
        "error_message": asset.error_message,
        "created_at": unix_secs_rfc3339(asset.created_at_unix_secs),
        "updated_at": unix_secs_rfc3339(asset.updated_at_unix_secs),
    });
    if is_admin {
        if let Some(object) = body.as_object_mut() {
            object.insert("user_id".to_string(), Value::String(asset.user_id.clone()));
            if let Some(group) = group {
                object.insert(
                    "provider_id".to_string(),
                    Value::String(group.provider_id.clone()),
                );
            }
        }
    }
    body
}

fn validation_session_rest_json(
    state: &AppState,
    session: &StoredArkVisualValidationSession,
    upstream: Option<&Value>,
) -> Result<Value, AssetServiceError> {
    let result = upstream.and_then(extract_result).or(upstream);
    let verification_url =
        validation_verification_url(result.unwrap_or(&Value::Null)).or_else(|| {
            decrypt_persisted_validation_url(state, session)
                .ok()
                .flatten()
        });
    Ok(json!({
        "id": session.id,
        "status": session.status,
        "verification_url": verification_url,
        "h5_link": verification_url,
        "group_id": session.group_id,
        "error_message": session.sanitized_result.as_ref().and_then(|value| string_field(value, &["ErrorMessage", "Message"])),
        "expires_at": unix_secs_rfc3339(session.expires_at_unix_secs),
    }))
}

fn validation_session_is_terminal(session: &StoredArkVisualValidationSession) -> bool {
    matches!(
        session.status.trim().to_ascii_lowercase().as_str(),
        "succeeded" | "success" | "failed"
    ) || session.consumed_at_unix_secs.is_some()
}

fn validation_result_status(result: &Value, fallback: &str) -> String {
    if let Some(status) = string_field(result, &["Status", "status"]) {
        return status;
    }
    match number_field(result, &["ResultCode", "resultCode", "result_code"]) {
        Some(10000) => "Succeeded".to_string(),
        Some(_) => "Failed".to_string(),
        None => fallback.to_string(),
    }
}

fn public_validation_result(session: &StoredArkVisualValidationSession) -> Value {
    let mut result = session
        .sanitized_result
        .clone()
        .unwrap_or_else(|| json!({"Status": session.status}));
    if let Some(object) = result.as_object_mut() {
        object.remove("EncryptedVerificationUrl");
        object.insert("Status".to_string(), Value::String(session.status.clone()));
        if let Some(group_id) = session.group_id.as_ref() {
            object.insert("GroupId".to_string(), Value::String(group_id.clone()));
        }
    }
    result
}

fn validation_verification_url(value: &Value) -> Option<String> {
    string_field(value, &["H5Url", "H5Link", "VerificationUrl", "URL"])
        .filter(|url| safe_provider_url(url))
}

fn decrypt_persisted_validation_url(
    state: &AppState,
    session: &StoredArkVisualValidationSession,
) -> Result<Option<String>, AssetServiceError> {
    let Some(ciphertext) = session
        .sanitized_result
        .as_ref()
        .and_then(|value| string_field(value, &["EncryptedVerificationUrl"]))
    else {
        return Ok(None);
    };
    let encryption_key = state
        .encryption_key()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AssetServiceError::unavailable("数据加密密钥不可用"))?;
    let url = aether_crypto::decrypt_python_fernet_ciphertext(encryption_key, &ciphertext)
        .map_err(|_| AssetServiceError::unavailable("真人验证链接无法解密"))?;
    Ok(safe_provider_url(&url).then_some(url))
}

fn merge_sanitized_validation_result(
    existing: Option<Value>,
    value: &Value,
    group_id: Option<String>,
    encrypted_verification_url: Option<String>,
) -> Option<Value> {
    let mut safe = existing
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    if let Some(current) =
        sanitize_validation_result(value).and_then(|value| value.as_object().cloned())
    {
        safe.extend(current);
    }
    if let Some(group_id) = group_id {
        safe.insert("GroupId".to_string(), Value::String(group_id));
    }
    if let Some(ciphertext) = encrypted_verification_url {
        safe.insert(
            "EncryptedVerificationUrl".to_string(),
            Value::String(ciphertext),
        );
    }
    (!safe.is_empty()).then_some(Value::Object(safe))
}

fn native_envelope(request_context: &GatewayPublicRequestContext, result: Value) -> Value {
    json!({
        "ResponseMetadata": {"RequestId": request_context.trace_id},
        "Result": result,
    })
}

fn native_error_response(error: AssetServiceError) -> Response<Body> {
    let body = error
        .provider_body
        .unwrap_or_else(|| build_error_envelope(error.code, &error.message));
    json_response(error.status, body)
}

fn rest_error_response(error: AssetServiceError) -> Response<Body> {
    json_response(
        error.status,
        json!({
            "detail": error.message,
            "code": error.code,
        }),
    )
}

fn json_response(status: StatusCode, body: Value) -> Response<Body> {
    (status, Json(body)).into_response()
}

fn empty_response(status: StatusCode) -> Response<Body> {
    Response::builder()
        .status(status)
        .body(Body::empty())
        .unwrap_or_else(|_| Response::new(Body::empty()))
}

fn parse_json_body(request_body: Option<&Bytes>) -> Result<Value, AssetServiceError> {
    let Some(body) = request_body.filter(|body| !body.is_empty()) else {
        return Ok(json!({}));
    };
    serde_json::from_slice::<Value>(body.as_ref())
        .map_err(|_| AssetServiceError::bad_request("request body must be valid JSON"))
}

fn provider_error_value(body: &Value) -> Option<&Value> {
    body.get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| metadata.get("Error").or_else(|| metadata.get("error")))
        .filter(|error| !error.is_null())
        .or_else(|| body.get("error").filter(|error| !error.is_null()))
}

fn sanitize_provider_error_body(body: &Value) -> Value {
    let error = provider_error_value(body).unwrap_or(body);
    let code = string_field(error, &["Code", "code", "Type", "type"])
        .unwrap_or_else(|| "UpstreamError".to_string());
    let message = string_field(error, &["Message", "message"])
        .unwrap_or_else(|| "素材库上游请求失败".to_string());
    let request_id = body
        .get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| string_field(metadata, &["RequestId", "request_id"]));
    let mut metadata = Map::from_iter([(
        "Error".to_string(),
        json!({"Code": code, "Message": message}),
    )]);
    if let Some(request_id) = request_id {
        metadata.insert("RequestId".to_string(), Value::String(request_id));
    }
    json!({"ResponseMetadata": Value::Object(metadata)})
}

fn request_uri(request_context: &GatewayPublicRequestContext) -> Uri {
    request_context
        .request_path_and_query()
        .parse()
        .unwrap_or_else(|_| Uri::from_static("/"))
}

fn execution_result_json(result: &aether_contracts::ExecutionResult) -> Option<Value> {
    let body = result.body.as_ref()?;
    if let Some(body) = body.json_body.as_ref() {
        return Some(body.clone());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(body.body_bytes_b64.as_deref()?)
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_repo(
    state: &AppState,
) -> Result<
    std::sync::Arc<
        dyn aether_data_contracts::repository::asset_library::AssetLibraryReadRepository,
    >,
    AssetServiceError,
> {
    state
        .data
        .asset_library_read_repository()
        .ok_or_else(|| AssetServiceError::unavailable("素材库数据读取服务不可用"))
}

fn write_repo(
    state: &AppState,
) -> Result<
    std::sync::Arc<
        dyn aether_data_contracts::repository::asset_library::AssetLibraryWriteRepository,
    >,
    AssetServiceError,
> {
    state
        .data
        .asset_library_write_repository()
        .ok_or_else(|| AssetServiceError::unavailable("素材库数据写入服务不可用"))
}

fn gateway_error(error: GatewayError) -> AssetServiceError {
    AssetServiceError::unavailable(error.into_message())
}

fn data_error(error: aether_data_contracts::DataLayerError) -> AssetServiceError {
    AssetServiceError::unavailable(error.to_string())
}

fn required_string_field(
    value: &Value,
    names: &[&str],
    display_name: &str,
) -> Result<String, AssetServiceError> {
    string_field(value, names)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| AssetServiceError::bad_request(format!("{display_name} is required")))
}

fn required_asset_type(value: &Value) -> Result<String, AssetServiceError> {
    let asset_type =
        required_string_field(value, &["AssetType", "asset_type", "type"], "AssetType")?;
    let normalized = match asset_type.trim().to_ascii_lowercase().as_str() {
        "image" => "Image",
        "video" => "Video",
        "audio" => "Audio",
        _ => {
            return Err(AssetServiceError::bad_request(
                "AssetType must be Image, Video, or Audio",
            ));
        }
    };
    Ok(normalized.to_string())
}

fn normalize_asset_type_filter(value: String) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "image" => "Image".to_string(),
        "video" => "Video".to_string(),
        "audio" => "Audio".to_string(),
        _ => value,
    }
}

fn value_field<'a>(value: &'a Value, names: &[&str]) -> Option<&'a Value> {
    let object = value.as_object()?;
    for name in names {
        if let Some(value) = object.get(*name) {
            return Some(value);
        }
        if let Some((_, value)) = object
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
        {
            return Some(value);
        }
    }
    for wrapper in ["Group", "Asset", "Data", "Result"] {
        if let Some(nested) = object.get(wrapper) {
            if let Some(value) = value_field(nested, names) {
                return Some(value);
            }
        }
    }
    None
}

fn string_field(value: &Value, names: &[&str]) -> Option<String> {
    value_field(value, names)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn number_field(value: &Value, names: &[&str]) -> Option<u64> {
    value_field(value, names).and_then(|value| {
        value
            .as_u64()
            .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
            .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
    })
}

fn object_field(value: &Value, names: &[&str]) -> Option<Value> {
    value_field(value, names)
        .filter(|value| value.is_object())
        .cloned()
}

fn native_list_filter(value: &Value) -> &Value {
    value_field(value, &["Filter", "filter"])
        .filter(|value| value.is_object())
        .unwrap_or(value)
}

fn object_has_field(value: &Value, names: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    names.iter().any(|name| {
        object.contains_key(*name) || object.keys().any(|key| key.eq_ignore_ascii_case(name))
    })
}

fn error_field(value: &Value, name: &str) -> Option<String> {
    value_field(value, &["Error", "error"]).and_then(|error| string_field(error, &[name]))
}

fn provider_url(value: &Value) -> Option<String> {
    string_field(
        value,
        &[
            "URL",
            "Url",
            "url",
            "AssetUrl",
            "AssetURL",
            "ContentUrl",
            "DownloadUrl",
        ],
    )
    .filter(|url| safe_provider_url(url))
}

fn sanitize_asset_metadata(value: &Value) -> Option<Value> {
    let mut safe = Map::new();
    for name in [
        "MimeType",
        "FileFormat",
        "Size",
        "Bytes",
        "Width",
        "Height",
        "Duration",
        "FrameRate",
        "Resolution",
        "LastInferenceTime",
    ] {
        if let Some(value) = value_field(value, &[name]) {
            if value.is_string() || value.is_number() || value.is_boolean() || value.is_null() {
                safe.insert(name.to_string(), value.clone());
            }
        }
    }
    (!safe.is_empty()).then_some(Value::Object(safe))
}

fn sanitize_validation_result(value: &Value) -> Option<Value> {
    let mut safe = Map::new();
    for name in [
        "Status",
        "Message",
        "ErrorCode",
        "ErrorMessage",
        "ResultCode",
    ] {
        if let Some(value) = value_field(value, &[name]) {
            if value.is_string() || value.is_number() || value.is_boolean() || value.is_null() {
                safe.insert(name.to_string(), value.clone());
            }
        }
    }
    (!safe.is_empty()).then_some(Value::Object(safe))
}

fn validate_source_url(value: &str) -> Result<(), AssetServiceError> {
    let url = url::Url::parse(value.trim())
        .map_err(|_| AssetServiceError::bad_request("url must be a valid HTTPS URL"))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(AssetServiceError::bad_request(
            "url must be an HTTPS URL without user information",
        ));
    }
    if url
        .host_str()
        .and_then(|host| host.parse::<std::net::IpAddr>().ok())
        .is_some_and(|ip| !literal_ip_is_public(ip))
    {
        return Err(AssetServiceError::bad_request(
            "url must not target a private or reserved address",
        ));
    }
    Ok(())
}

fn safe_provider_url(value: &str) -> bool {
    url::Url::parse(value.trim()).is_ok_and(|url| {
        url.scheme() == "https"
            && url.host_str().is_some()
            && url.username().is_empty()
            && url.password().is_none()
            && url
                .host_str()
                .and_then(|host| host.parse::<std::net::IpAddr>().ok())
                .is_none_or(literal_ip_is_public)
    })
}

fn literal_ip_is_public(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ip) => {
            let octets = ip.octets();
            !(octets[0] == 0
                || octets[0] == 10
                || octets[0] == 127
                || (octets[0] == 100 && (64..=127).contains(&octets[1]))
                || (octets[0] == 169 && octets[1] == 254)
                || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 168)
                || octets[0] >= 224)
        }
        std::net::IpAddr::V6(ip) => {
            if let Some(ip) = ip.to_ipv4_mapped() {
                return literal_ip_is_public(std::net::IpAddr::V4(ip));
            }
            let first = ip.segments()[0];
            !(ip.is_unspecified()
                || ip.is_loopback()
                || first == 0
                || (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || (first & 0xff00) == 0xff00)
        }
    }
}

fn key_supports_asset_library(capabilities: Option<&Value>) -> bool {
    match capabilities {
        Some(Value::Array(values)) => values.iter().any(|value| {
            value.as_str().is_some_and(|value| {
                value.eq_ignore_ascii_case(super::ARK_ASSET_REQUIRED_CAPABILITY)
            })
        }),
        Some(Value::Object(values)) => values.iter().any(|(name, value)| {
            name.eq_ignore_ascii_case(super::ARK_ASSET_REQUIRED_CAPABILITY)
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

fn action_account_binding(transport: &GatewayProviderTransportSnapshot) -> String {
    let auth_config = transport
        .key
        .decrypted_auth_config
        .as_deref()
        .and_then(|value| serde_json::from_str::<Value>(value).ok());
    for value in [
        auth_config.as_ref(),
        transport.key.upstream_metadata.as_ref(),
        transport.key.fingerprint.as_ref(),
        transport.endpoint.config.as_ref(),
        transport.provider.config.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(account_id) = string_field(
            value,
            &[
                "account_id",
                "account",
                "tenant_id",
                "asset_account_binding",
                "account_binding",
            ],
        ) {
            return format!("account:{}", sha256_text(&account_id)[..24].to_string());
        }
    }
    if let Some(access_key_id) = auth_config
        .as_ref()
        .and_then(|config| string_field(config, &["access_key_id", "AccessKeyId"]))
    {
        return format!("ak:{}", sha256_text(&access_key_id)[..24].to_string());
    }
    if let Ok(auth) = aether_provider_transport::resolve_volc_action_auth(transport) {
        let identity = match auth {
            VolcActionAuth::AkSk(credentials) => credentials.access_key_id,
            VolcActionAuth::Bearer(secret) => secret,
            VolcActionAuth::ApiKey { secret, .. } => secret,
        };
        return format!("credential:{}", &sha256_text(&identity)[..24]);
    }
    format!("key:{}", transport.key.id)
}

fn action_project(transport: &GatewayProviderTransportSnapshot) -> Option<String> {
    for value in [
        transport
            .key
            .decrypted_auth_config
            .as_deref()
            .and_then(|value| serde_json::from_str::<Value>(value).ok()),
        transport.key.upstream_metadata.clone(),
        transport.endpoint.config.clone(),
        transport.provider.config.clone(),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(project) = string_field(&value, &["project", "project_name", "ProjectName"]) {
            return Some(project);
        }
    }
    None
}

fn page_number(value: &Value) -> usize {
    number_field(value, &["PageNumber", "PageNum", "page_num"])
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1)
        .max(1)
}

fn page_size(value: &Value) -> usize {
    number_field(value, &["PageSize", "page_size"])
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(DEFAULT_PAGE_SIZE)
        .clamp(1, MAX_PAGE_SIZE)
}

fn body_user_id(value: &Value) -> Option<String> {
    string_field(value, &["user_id", "UserId"])
}

fn query_value(query: Option<&str>, name: &str) -> Option<String> {
    url::form_urlencoded::parse(query.unwrap_or_default().as_bytes())
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.into_owned())
        .filter(|value| !value.trim().is_empty())
}

fn path_resource_id(path: &str, resource: &str) -> Result<String, AssetServiceError> {
    let marker = format!("/{resource}/");
    let id = path
        .split_once(&marker)
        .map(|(_, suffix)| suffix)
        .and_then(|suffix| suffix.split('/').next())
        .map(str::trim)
        .filter(|id| !id.is_empty())
        .ok_or_else(AssetServiceError::not_found)?;
    Ok(id.to_string())
}

fn header_text(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn local_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

fn sha256_text(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn unix_secs_rfc3339(value: u64) -> Option<String> {
    let value = i64::try_from(value).ok()?;
    chrono::DateTime::<chrono::Utc>::from_timestamp(value, 0)
        .map(|value| value.to_rfc3339_opts(chrono::SecondsFormat::Secs, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use aether_data::repository::asset_library::{
        AssetLibraryReadRepository, AssetLibraryWriteRepository, InMemoryAssetLibraryRepository,
    };

    use crate::data::GatewayDataState;

    fn stored_group(id: &str, user_id: &str) -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: id.to_string(),
            upstream_group_id: Some(format!("upstream-{id}")),
            user_id: user_id.to_string(),
            api_key_id: Some(format!("key-{user_id}")),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "provider-key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            group_type: "AIGC".to_string(),
            name: "group".to_string(),
            description: None,
            status: "Active".to_string(),
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
            deleted_at_unix_secs: None,
        }
    }

    fn stored_asset(id: &str, group_id: &str, user_id: &str) -> UpsertAssetRecord {
        UpsertAssetRecord {
            id: id.to_string(),
            upstream_asset_id: Some(format!("upstream-{id}")),
            group_id: group_id.to_string(),
            user_id: user_id.to_string(),
            api_key_id: Some(format!("key-{user_id}")),
            asset_type: "Image".to_string(),
            name: "asset".to_string(),
            status: "Processing".to_string(),
            error_code: None,
            error_message: None,
            moderation: None,
            last_inference_at_unix_secs: None,
            source_url_fingerprint: Some("source-fingerprint".to_string()),
            provider_url: None,
            provider_url_expires_at_unix_secs: None,
            sanitized_metadata: None,
            is_deleted: false,
            deleted_at_unix_secs: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
        }
    }

    fn caller(user_id: &str) -> AssetCaller {
        AssetCaller {
            user_id: user_id.to_string(),
            api_key_id: Some(format!("key-{user_id}")),
            unrestricted_provider_access: false,
            allowed_providers: None,
            allowed_api_formats: None,
        }
    }

    fn state_with_asset_repository(repository: Arc<InMemoryAssetLibraryRepository>) -> AppState {
        AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(GatewayDataState::with_asset_library_repository_for_tests(
                repository,
            ))
    }

    fn encrypted_state_with_asset_repository(
        repository: Arc<InMemoryAssetLibraryRepository>,
    ) -> AppState {
        let data = GatewayDataState::with_asset_library_repository_for_tests(repository)
            .with_encryption_key_for_tests(aether_crypto::DEVELOPMENT_ENCRYPTION_KEY);
        AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(data)
    }

    #[test]
    fn rejects_conflicting_public_credentials() {
        let mut headers = HeaderMap::new();
        headers.insert(http::header::AUTHORIZATION, "Bearer one".parse().unwrap());
        headers.insert("x-api-key", "two".parse().unwrap());
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace",
            &http::Method::POST,
            &"/?Action=ListAssets&Version=2024-01-01".parse().unwrap(),
            &headers,
            None,
        );
        let error = validate_public_credential_carriers(&headers, &context).unwrap_err();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn accepts_mixed_case_bearer_scheme() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            "bEaReR material-asset-key".parse().unwrap(),
        );
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace",
            &http::Method::POST,
            &"/?Action=ListAssets&Version=2024-01-01".parse().unwrap(),
            &headers,
            None,
        );

        validate_public_credential_carriers(&headers, &context)
            .expect("Bearer authentication scheme is case-insensitive");
    }

    #[test]
    fn malformed_json_is_a_bad_request() {
        let error = parse_json_body(Some(&Bytes::from_static(br#"{"Name":"unfinished"#)))
            .expect_err("malformed JSON must not be accepted");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "InvalidParameter");
        assert_eq!(
            native_error_response(error).status(),
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn provider_error_without_http_code_is_detected_and_sanitized() {
        let body = json!({
            "ResponseMetadata": {
                "RequestId": "upstream-request-id",
                "InternalDebug": "must-not-leak",
                "Error": {
                    "Code": "InvalidParameter",
                    "Message": "invalid asset",
                    "AccessKey": "must-not-leak",
                    "Details": {"BytedToken": "must-not-leak"}
                }
            }
        });

        assert_eq!(
            provider_error_value(&body),
            Some(&body["ResponseMetadata"]["Error"])
        );
        assert_eq!(
            sanitize_provider_error_body(&body),
            json!({
                "ResponseMetadata": {
                    "RequestId": "upstream-request-id",
                    "Error": {
                        "Code": "InvalidParameter",
                        "Message": "invalid asset"
                    }
                }
            })
        );
    }

    #[test]
    fn nested_native_filters_and_result_aliases_are_supported() {
        let body = json!({
            "PageNumber": 2,
            "PageSize": 50,
            "Filter": {
                "GroupId": "group-1",
                "AssetType": "Image",
                "Status": "Active"
            }
        });
        let filter = native_list_filter(&body);

        assert_eq!(page_number(&body), 2);
        assert_eq!(page_size(&body), 50);
        assert_eq!(string_field(filter, &["GroupId"]), Some("group-1".into()));
        assert_eq!(string_field(filter, &["AssetType"]), Some("Image".into()));
        assert_eq!(string_field(filter, &["Status"]), Some("Active".into()));
    }

    #[test]
    fn validation_result_code_drives_terminal_status() {
        assert_eq!(
            validation_result_status(&json!({"resultCode": 10000}), "Pending"),
            "Succeeded"
        );
        assert_eq!(
            validation_result_status(&json!({"ResultCode": 10001}), "Pending"),
            "Failed"
        );
        assert_eq!(
            validation_result_status(&json!({"Status": "Processing"}), "Pending"),
            "Processing"
        );
    }

    #[test]
    fn encrypted_validation_url_survives_refresh_projection() {
        let state = encrypted_state_with_asset_repository(Arc::new(
            InMemoryAssetLibraryRepository::default(),
        ));
        let verification_url = "https://verify.example.test/session";
        let encrypted_url = aether_crypto::encrypt_python_fernet_plaintext(
            aether_crypto::DEVELOPMENT_ENCRYPTION_KEY,
            verification_url,
        )
        .expect("encrypted validation URL");
        let mut session = StoredArkVisualValidationSession {
            id: "vsess-local".to_string(),
            session_id: "session-upstream".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: Some("key-user-1".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "provider-key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            byted_token_hash: "token-hash".to_string(),
            encrypted_byted_token: "encrypted-token".to_string(),
            callback_state_hash: "callback-hash".to_string(),
            status: "Pending".to_string(),
            expires_at_unix_secs: 1_800,
            consumed_at_unix_secs: None,
            group_id: None,
            sanitized_result: Some(json!({"EncryptedVerificationUrl": encrypted_url})),
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
        };

        let body = validation_session_rest_json(&state, &session, None)
            .expect("persisted validation URL projection");
        assert_eq!(body["verification_url"], verification_url);
        assert!(!serde_json::to_string(&body).unwrap().contains("gAAAA"));

        session.status = "Succeeded".to_string();
        let native = public_validation_result(&session);
        assert_eq!(native["Status"], "Succeeded");
        assert!(native.get("EncryptedVerificationUrl").is_none());
    }

    #[test]
    fn recursively_rewrites_only_asset_references() {
        let mut value = json!({
            "content": [
                {"image_url": {"url": "asset://local-1"}},
                {"text": "asset://local-2"},
                {"url": "https://example.test/asset://local-1"}
            ]
        });
        replace_asset_reference_ids(
            &mut value,
            &HashMap::from([
                ("local-1".to_string(), "upstream-1".to_string()),
                ("local-2".to_string(), "upstream-2".to_string()),
            ]),
        );
        assert_eq!(
            value["content"][0]["image_url"]["url"],
            "asset://upstream-1"
        );
        assert_eq!(value["content"][1]["text"], "asset://upstream-2");
        assert_eq!(
            value["content"][2]["url"],
            "https://example.test/asset://local-1"
        );
    }

    #[tokio::test]
    async fn owner_scoping_and_group_deletion_hide_assets() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(stored_group("group-1", "user-1"))
            .await
            .expect("group");
        repository
            .upsert_asset(stored_asset("asset-1", "group-1", "user-1"))
            .await
            .expect("asset");
        let state = state_with_asset_repository(Arc::clone(&repository));

        assert!(load_group(&state, &caller("user-1"), "group-1", false)
            .await
            .is_ok());
        assert_eq!(
            load_group(&state, &caller("user-2"), "group-1", false)
                .await
                .expect_err("another user must not see the group")
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            load_asset(&state, &caller("user-2"), "asset-1", false)
                .await
                .expect_err("another user must not see the asset")
                .status,
            StatusCode::NOT_FOUND
        );

        assert!(repository
            .soft_delete_group("group-1", 20)
            .await
            .expect("delete group"));
        assert_eq!(
            load_group(&state, &caller("user-1"), "group-1", false)
                .await
                .expect_err("deleted group must be hidden")
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            load_asset(&state, &caller("user-1"), "asset-1", false)
                .await
                .expect_err("group deletion must hide child assets")
                .status,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn admin_resource_lookup_rejects_mismatched_owner() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(stored_group("group-1", "user-1"))
            .await
            .expect("group");
        repository
            .upsert_asset(stored_asset("asset-1", "group-1", "user-1"))
            .await
            .expect("asset");
        let state = state_with_asset_repository(repository);
        let caller = AssetCaller {
            user_id: "user-2".to_string(),
            api_key_id: None,
            unrestricted_provider_access: true,
            allowed_providers: None,
            allowed_api_formats: None,
        };

        assert_eq!(
            load_group(&state, &caller, "group-1", true)
                .await
                .expect_err("admin lookup must honor the selected owner")
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            load_asset(&state, &caller, "asset-1", true)
                .await
                .expect_err("admin lookup must honor the selected owner")
                .status,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn expired_validation_session_stops_before_decryption_or_polling() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_visual_validation_session(UpsertArkVisualValidationSessionRecord {
                id: "vsess-expired".to_string(),
                session_id: "upstream-session".to_string(),
                user_id: "user-1".to_string(),
                api_key_id: Some("key-user-1".to_string()),
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "provider-key-1".to_string(),
                account_binding: Some("account-1".to_string()),
                project: Some("project-1".to_string()),
                byted_token_hash: "token-hash".to_string(),
                encrypted_byted_token: "deliberately-invalid-ciphertext".to_string(),
                callback_state_hash: "callback-hash".to_string(),
                status: "Pending".to_string(),
                expires_at_unix_secs: 2,
                consumed_at_unix_secs: None,
                group_id: None,
                sanitized_result: None,
                created_at_unix_secs: 1,
                updated_at_unix_secs: 1,
            })
            .await
            .expect("validation session");
        let state = state_with_asset_repository(repository);
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace",
            &http::Method::GET,
            &"/api/material-assets/visual-validation/sessions/vsess-expired"
                .parse()
                .unwrap(),
            &headers,
            None,
        );

        let error = refresh_validation_session(
            &state,
            &context,
            &headers,
            &caller("user-1"),
            "vsess-expired",
            false,
        )
        .await
        .expect_err("expired session must stop before decrypting or polling");
        assert_eq!(error.status, StatusCode::GONE);
        assert_eq!(error.code, "ValidationSessionExpired");
    }

    #[tokio::test]
    async fn provider_signed_url_is_ephemeral_and_never_persisted() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(stored_group("group-1", "user-1"))
            .await
            .expect("group");
        let asset = repository
            .upsert_asset(stored_asset("asset-1", "group-1", "user-1"))
            .await
            .expect("asset");
        let state = state_with_asset_repository(Arc::clone(&repository));
        let signed_url = "https://media.example.test/a.png?X-Signature=secret";

        let response_asset = persist_asset_projection(
            &state,
            asset,
            &json!({
                "Status": "Active",
                "URL": signed_url,
                "MimeType": "image/png"
            }),
        )
        .await
        .expect("projection");
        assert_eq!(response_asset.provider_url.as_deref(), Some(signed_url));
        assert!(response_asset.provider_url_expires_at_unix_secs.is_some());

        let persisted = repository
            .find_asset_by_id("asset-1")
            .await
            .expect("read asset")
            .expect("stored asset");
        assert_eq!(persisted.status, "Active");
        assert_eq!(persisted.provider_url, None);
        assert_eq!(persisted.provider_url_expires_at_unix_secs, None);
        assert_eq!(
            persisted.sanitized_metadata,
            Some(json!({"MimeType": "image/png"}))
        );
        assert!(!serde_json::to_string(&persisted.sanitized_metadata)
            .unwrap()
            .contains("X-Signature"));
    }

    #[test]
    fn native_asset_projection_exposes_only_local_ids_and_proxy_url() {
        let mut asset = stored_asset("asset-local", "group-local", "user-1").into_stored();
        asset.upstream_asset_id = Some("asset-upstream-secret".to_string());
        asset.provider_url =
            Some("https://media.example.test/a.png?X-Signature=provider-secret".to_string());
        asset.provider_url_expires_at_unix_secs = Some(99);

        let projection = asset_native_json(&asset);
        assert_eq!(projection["AssetId"], "asset-local");
        assert_eq!(projection["GroupId"], "group-local");
        assert_eq!(
            projection["URL"],
            "/api/material-assets/assets/asset-local/preview"
        );
        let serialized = serde_json::to_string(&projection).unwrap();
        assert!(!serialized.contains("asset-upstream-secret"));
        assert!(!serialized.contains("provider-secret"));
    }

    #[test]
    fn rest_validation_projection_does_not_expose_byted_token() {
        let state = AppState::new().expect("gateway state");
        let session = StoredArkVisualValidationSession {
            id: "vsess-local".to_string(),
            session_id: "session-upstream".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: Some("key-user-1".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "provider-key-1".to_string(),
            account_binding: Some("account-1".to_string()),
            project: Some("project-1".to_string()),
            byted_token_hash: "token-hash".to_string(),
            encrypted_byted_token: "encrypted-token".to_string(),
            callback_state_hash: "callback-hash".to_string(),
            status: "Pending".to_string(),
            expires_at_unix_secs: 1_800,
            consumed_at_unix_secs: None,
            group_id: None,
            sanitized_result: None,
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
        };
        let body = validation_session_rest_json(
            &state,
            &session,
            Some(&json!({
                "Result": {
                    "BytedToken": "raw-secret-token",
                    "H5Link": "https://verify.example.test/session"
                }
            })),
        )
        .expect("validation projection");

        assert_eq!(
            body["verification_url"],
            "https://verify.example.test/session"
        );
        let serialized = serde_json::to_string(&body).unwrap();
        assert!(!serialized.contains("raw-secret-token"));
        assert!(!serialized.contains("encrypted-token"));
        assert!(!serialized.contains("token-hash"));
    }
}
