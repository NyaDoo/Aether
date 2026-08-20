use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use aether_contracts::{ExecutionPlan, RequestBody};
use aether_data_contracts::repository::asset_library::{
    AssetGroupListQuery, AssetListQuery, StoredArkVisualValidationSession, StoredAsset,
    StoredAssetGroup, UpsertArkVisualValidationSessionRecord, UpsertAssetGroupRecord,
    UpsertAssetRecord,
};
use aether_data_contracts::repository::candidates::RequestCandidateStatus;
use aether_data_contracts::repository::provider_catalog::{
    StoredProviderCatalogEndpoint, StoredProviderCatalogKey, StoredProviderCatalogProvider,
};
use aether_provider_transport::{
    build_volc_action_request, resolve_transport_execution_timeouts, resolve_transport_profile,
    GatewayProviderTransportSnapshot, VolcActionRequestInput,
};
use aether_scheduler_core::{
    extract_global_priority_for_format, SchedulerMinimalCandidateSelectionCandidate,
    SchedulerRequestCandidateStatusUpdate,
};
use axum::body::{Body, Bytes};
use axum::http::{self, HeaderMap, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::Json;
use base64::Engine as _;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::protocol_api::{
    build_error_envelope, canonicalize_provider_body, extract_result, sanitize_action_body,
};
use super::{action_from_request, ArkAssetAction, ARK_ASSET_API_FORMAT};
use crate::control::GatewayPublicRequestContext;
use crate::{AppState, GatewayError};

const ASSET_URL_TTL_SECS: u64 = 12 * 60 * 60;
const VALIDATION_SESSION_TTL_SECS: u64 = 30 * 60;
const DEFAULT_PAGE_SIZE: usize = 20;
const MAX_PAGE_SIZE: usize = 500;
const ARK_DEFAULT_PAGE_SIZE: usize = 10;
const ARK_MAX_PAGE_SIZE: usize = 100;
const ASSET_ROUTING_MODEL: &str = "__ark_asset_library__";

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

struct ResolvedCallerAccess {
    policy: CallerAccessPolicy,
    auth_snapshot: Option<crate::data::auth::GatewayAuthApiKeySnapshot>,
}

#[derive(Debug)]
struct AssetServiceError {
    status: StatusCode,
    code: String,
    message: String,
    provider_body: Option<Value>,
}

impl AssetServiceError {
    fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
            provider_body: None,
        }
    }

    fn provider(status: StatusCode, body: Value) -> Self {
        let upstream_code = provider_error_code(&body);
        if let Some((code, field)) = upstream_code
            .as_deref()
            .and_then(missing_parameter_code_and_field)
        {
            let message = format!("素材库上游缺少必填参数：{field}");
            let provider_body = sanitize_provider_error_body_with_override(&body, code, &message);
            return Self {
                status: StatusCode::BAD_REQUEST,
                code: code.to_string(),
                message,
                provider_body: Some(provider_body),
            };
        }
        let normalized_code = upstream_code.as_deref().map(str::to_ascii_lowercase);
        let (mapped_status, mapped_code, mapped_message) = match normalized_code.as_deref() {
            Some("subscriptionrequired") => (
                StatusCode::FORBIDDEN,
                "SubscriptionRequired".to_string(),
                "火山素材库服务未开通，请在火山控制台开通对应套餐",
            ),
            Some("accessdenied" | "forbidden" | "unauthorizedoperation" | "permissiondenied") => (
                StatusCode::FORBIDDEN,
                "AccessDenied".to_string(),
                "火山素材库账号没有执行此操作的权限",
            ),
            Some("signaturedoesnotmatch" | "invalidsignature" | "requestsignatureinvalid") => (
                StatusCode::BAD_GATEWAY,
                "SignatureDoesNotMatch".to_string(),
                "火山素材库请求签名校验失败，请检查 AK/SK、Region、Service 和系统时间",
            ),
            Some(
                "invalidaccesskeyid"
                | "invalidcredential"
                | "invalidsecuritytoken"
                | "requestexpired"
                | "requesttimetoolarge"
                | "requesttimetooskewed",
            ) => (
                StatusCode::BAD_GATEWAY,
                "InvalidCredentials".to_string(),
                "火山素材库凭据无效，请检查 AK/SK 或安全令牌",
            ),
            Some("throttling" | "ratelimitexceeded" | "requestlimitexceeded") => (
                StatusCode::TOO_MANY_REQUESTS,
                "RateLimitExceeded".to_string(),
                "火山素材库请求频率受限，请稍后重试",
            ),
            Some("resourcenotfound" | "notfound") => (
                StatusCode::NOT_FOUND,
                "ResourceNotFound".to_string(),
                "火山素材库资源不存在",
            ),
            _ if status == StatusCode::UNAUTHORIZED => (
                StatusCode::BAD_GATEWAY,
                "UpstreamAuthenticationError".to_string(),
                "素材库上游凭据无效",
            ),
            _ if status == StatusCode::FORBIDDEN => (
                StatusCode::FORBIDDEN,
                "UpstreamAccessDenied".to_string(),
                "火山素材库账号没有执行此操作的权限",
            ),
            _ => (
                status,
                upstream_code.unwrap_or_else(|| "UpstreamError".to_string()),
                "素材库上游请求失败",
            ),
        };
        let provider_body =
            sanitize_provider_error_body_with_override(&body, &mapped_code, mapped_message);
        Self {
            status: mapped_status,
            code: mapped_code,
            message: mapped_message.to_string(),
            provider_body: Some(provider_body),
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
}

struct ActionResponse {
    body: Value,
}

struct AssetCandidateTerminalGuard {
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    started_at_unix_ms: u64,
    started_at: Instant,
    armed: bool,
}

impl AssetCandidateTerminalGuard {
    fn new(
        state: &AppState,
        plan: &ExecutionPlan,
        report_context: Option<Value>,
        started_at_unix_ms: u64,
        started_at: Instant,
    ) -> Self {
        Self {
            state: state.clone(),
            plan: plan.clone(),
            report_context,
            started_at_unix_ms,
            started_at,
            armed: true,
        }
    }

    async fn finish(
        &mut self,
        status: RequestCandidateStatus,
        status_code: Option<u16>,
        error_type: Option<&str>,
        error_message: Option<String>,
    ) {
        if !self.armed {
            return;
        }
        record_asset_candidate_terminal(
            &self.state,
            &self.plan,
            self.report_context.as_ref(),
            self.started_at_unix_ms,
            self.started_at,
            status,
            status_code,
            error_type,
            error_message,
        )
        .await;
        self.armed = false;
    }
}

impl Drop for AssetCandidateTerminalGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        self.armed = false;
        let state = self.state.clone();
        let plan = self.plan.clone();
        let report_context = self.report_context.clone();
        let started_at_unix_ms = self.started_at_unix_ms;
        let started_at = self.started_at;
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                record_asset_candidate_terminal(
                    &state,
                    &plan,
                    report_context.as_ref(),
                    started_at_unix_ms,
                    started_at,
                    RequestCandidateStatus::Cancelled,
                    Some(499),
                    Some("asset_request_cancelled"),
                    Some(
                        "Ark asset request was cancelled before terminal finalization".to_string(),
                    ),
                )
                .await;
            });
        }
    }
}

fn begin_asset_candidate_attempt(
    state: &AppState,
    plan: &mut ExecutionPlan,
    report_context: &mut Option<Value>,
    started_at_unix_ms: u64,
    started_at: Instant,
) -> AssetCandidateTerminalGuard {
    crate::request_candidate_runtime::assign_execution_request_candidate_slot(plan, report_context);
    AssetCandidateTerminalGuard::new(
        state,
        plan,
        report_context.clone(),
        started_at_unix_ms,
        started_at,
    )
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
    let mut uses_hmac = false;
    if let Some(value) = header_text(headers, http::header::AUTHORIZATION.as_str()) {
        let Some((scheme, value)) = value.split_once(char::is_whitespace) else {
            return Err(AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Authorization must use Bearer authentication",
            ));
        };
        if scheme.eq_ignore_ascii_case("HMAC-SHA256") {
            uses_hmac = true;
        } else if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(AssetServiceError::new(
                StatusCode::UNAUTHORIZED,
                "Unauthorized",
                "Authorization must use Bearer authentication",
            ));
        }
        if !uses_hmac && !value.trim().is_empty() {
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
    if uses_hmac && !credentials.is_empty() {
        return Err(AssetServiceError::bad_request(
            "conflicting API credentials were supplied",
        ));
    }
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
    validate_native_project(&body)?;
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
                action,
                json!({"Id": group.id}),
            ))
        }
        ArkAssetAction::GetAssetGroup => {
            let id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
            let group = refresh_group(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(
                request_context,
                action,
                group_native_json(&group),
            ))
        }
        ArkAssetAction::UpdateAssetGroup => {
            let group = update_group(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": group.id}),
            ))
        }
        ArkAssetAction::DeleteAssetGroup => {
            let id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
            delete_group(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, action, json!({})))
        }
        ArkAssetAction::CreateAsset => {
            let asset = create_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": asset.id}),
            ))
        }
        ArkAssetAction::GetAsset => {
            let id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
            let asset = refresh_asset(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(
                request_context,
                action,
                asset_native_json(&asset),
            ))
        }
        ArkAssetAction::UpdateAsset => {
            let asset = update_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": asset.id}),
            ))
        }
        ArkAssetAction::DeleteAsset => {
            let id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
            delete_asset(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, action, json!({})))
        }
        ArkAssetAction::CreateVisualValidateSession => {
            if string_field(
                &body,
                &["CallbackURL", "callback_url", "ReturnUrl", "return_url"],
            )
            .is_none()
            {
                return Err(AssetServiceError::new(
                    StatusCode::BAD_REQUEST,
                    "MissingParameter.CallbackURL",
                    "CallbackURL is required",
                ));
            }
            let (_session, upstream) =
                create_validation_session(state, request_context, headers, caller, body).await?;
            Ok(native_create_validation_payload(upstream))
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
            let mut upstream_body = json!({
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
                "Description": string_field(&body, &["description", "Description"]),
            });
            if let Some(group_type) = value_field(&body, &["group_type", "GroupType"]) {
                upstream_body
                    .as_object_mut()
                    .expect("group request body is an object")
                    .insert("GroupType".to_string(), group_type.clone());
            }
            if is_admin {
                validate_admin_group_owner(state, &caller.user_id).await?;
            }
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
                "Id": id,
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
                "Id": id,
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
            let callback_url = required_string_field(
                &body,
                &["callback_url", "CallbackURL", "return_url", "ReturnUrl"],
                "callback_url",
            )?;
            let upstream_body = json!({"CallbackURL": callback_url});
            let (session, upstream) =
                create_validation_session(state, request_context, headers, &caller, upstream_body)
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

async fn validate_admin_group_owner(
    state: &AppState,
    user_id: &str,
) -> Result<(), AssetServiceError> {
    let user = state.find_user_auth_by_id(user_id).await.map_err(|_| {
        AssetServiceError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "UserLookupUnavailable",
            "目标用户信息暂时无法读取",
        )
    })?;
    let Some(user) = user.filter(|user| !user.is_deleted) else {
        return Err(AssetServiceError::new(
            StatusCode::NOT_FOUND,
            "UserNotFound",
            "目标用户不存在或已删除",
        ));
    };
    if !user.is_active {
        return Err(AssetServiceError::new(
            StatusCode::BAD_REQUEST,
            "UserInactive",
            "目标用户已停用",
        ));
    }
    Ok(())
}

async fn select_transport(
    state: &AppState,
    caller: &AssetCaller,
) -> Result<AssetTransport, AssetServiceError> {
    let now = crate::clock::current_unix_secs();
    let caller_access = resolve_caller_access(state, caller, now).await?;
    if !access_policy_allows_format(&caller_access.policy, ARK_ASSET_API_FORMAT) {
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
            &caller_access.policy,
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
            candidates.push(asset_scheduler_candidate(provider, endpoint, key)?);
        }
    }
    let (candidates, skipped) =
        crate::scheduler::candidate::list_selectable_enumerated_candidates_with_skip_reasons(
            state,
            ARK_ASSET_API_FORMAT,
            ASSET_ROUTING_MODEL,
            candidates,
            None,
            caller_access.auth_snapshot.as_ref(),
            None,
            now,
        )
        .await
        .map_err(gateway_error)?;
    if crate::scheduler::candidate::is_exact_all_skipped_by_auth_limit(&candidates, &skipped) {
        return Err(auth_concurrency_limit_error());
    }
    for candidate in candidates {
        let snapshot = state
            .read_provider_transport_snapshot(
                &candidate.provider_id,
                &candidate.endpoint_id,
                &candidate.key_id,
            )
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

fn asset_scheduler_candidate(
    provider: &StoredProviderCatalogProvider,
    endpoint: &StoredProviderCatalogEndpoint,
    key: &StoredProviderCatalogKey,
) -> Result<SchedulerMinimalCandidateSelectionCandidate, AssetServiceError> {
    let key_global_priority_for_format = extract_global_priority_for_format(
        key.global_priority_by_format.as_ref(),
        ARK_ASSET_API_FORMAT,
    )
    .map_err(data_error)?;
    Ok(SchedulerMinimalCandidateSelectionCandidate {
        provider_id: provider.id.clone(),
        provider_name: provider.name.clone(),
        provider_type: provider.provider_type.clone(),
        provider_priority: provider.provider_priority,
        endpoint_id: endpoint.id.clone(),
        endpoint_api_format: endpoint.api_format.clone(),
        key_id: key.id.clone(),
        key_name: key.name.clone(),
        key_auth_type: key.auth_type.clone(),
        key_internal_priority: key.internal_priority,
        key_global_priority_for_format,
        key_capabilities: key.capabilities.clone(),
        model_id: ASSET_ROUTING_MODEL.to_string(),
        global_model_id: ASSET_ROUTING_MODEL.to_string(),
        global_model_name: ASSET_ROUTING_MODEL.to_string(),
        selected_provider_model_name: ASSET_ROUTING_MODEL.to_string(),
        supports_streaming: false,
        mapping_matched_model: None,
    })
}

fn asset_scheduler_candidate_from_transport(
    transport: &GatewayProviderTransportSnapshot,
) -> Result<SchedulerMinimalCandidateSelectionCandidate, AssetServiceError> {
    let key_global_priority_for_format = extract_global_priority_for_format(
        transport.key.global_priority_by_format.as_ref(),
        ARK_ASSET_API_FORMAT,
    )
    .map_err(data_error)?;
    Ok(SchedulerMinimalCandidateSelectionCandidate {
        provider_id: transport.provider.id.clone(),
        provider_name: transport.provider.name.clone(),
        provider_type: transport.provider.provider_type.clone(),
        provider_priority: 0,
        endpoint_id: transport.endpoint.id.clone(),
        endpoint_api_format: transport.endpoint.api_format.clone(),
        key_id: transport.key.id.clone(),
        key_name: transport.key.name.clone(),
        key_auth_type: transport.key.auth_type.clone(),
        key_internal_priority: 0,
        key_global_priority_for_format,
        key_capabilities: transport.key.capabilities.clone(),
        model_id: ASSET_ROUTING_MODEL.to_string(),
        global_model_id: ASSET_ROUTING_MODEL.to_string(),
        global_model_name: ASSET_ROUTING_MODEL.to_string(),
        selected_provider_model_name: ASSET_ROUTING_MODEL.to_string(),
        supports_streaming: false,
        mapping_matched_model: None,
    })
}

async fn resolve_caller_access(
    state: &AppState,
    caller: &AssetCaller,
    now: u64,
) -> Result<ResolvedCallerAccess, AssetServiceError> {
    if caller.unrestricted_provider_access {
        return Ok(ResolvedCallerAccess {
            policy: CallerAccessPolicy {
                unrestricted: true,
                ..CallerAccessPolicy::default()
            },
            auth_snapshot: None,
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
        let policy = CallerAccessPolicy {
            unrestricted: false,
            allowed_providers: snapshot
                .effective_allowed_providers()
                .map(ToOwned::to_owned),
            allowed_api_formats: snapshot
                .effective_allowed_api_formats()
                .map(ToOwned::to_owned),
        };
        return Ok(ResolvedCallerAccess {
            policy,
            auth_snapshot: Some(snapshot),
        });
    }
    Ok(ResolvedCallerAccess {
        policy: CallerAccessPolicy {
            unrestricted: false,
            allowed_providers: caller.allowed_providers.clone(),
            allowed_api_formats: caller.allowed_api_formats.clone(),
        },
        auth_snapshot: None,
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
    Ok(asset_transport(snapshot))
}

async fn exact_transport_for_group(
    state: &AppState,
    caller: &AssetCaller,
    group: &StoredAssetGroup,
) -> Result<AssetTransport, AssetServiceError> {
    let transport =
        exact_transport(state, &group.provider_id, &group.endpoint_id, &group.key_id).await?;
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
    let now = crate::clock::current_unix_secs();
    let caller_access = resolve_caller_access(state, caller, now).await?;
    if !access_policy_allows_format(&caller_access.policy, ARK_ASSET_API_FORMAT) {
        return Err(AssetServiceError::new(
            StatusCode::FORBIDDEN,
            "ApiFormatNotAllowed",
            "当前凭据无权访问 Ark 素材库",
        ));
    }
    if !access_policy_allows_provider(
        &caller_access.policy,
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
    let candidate = asset_scheduler_candidate_from_transport(transport)?;
    let (selected, skipped) =
        crate::scheduler::candidate::list_selectable_enumerated_candidates_with_skip_reasons(
            state,
            ARK_ASSET_API_FORMAT,
            ASSET_ROUTING_MODEL,
            vec![candidate],
            None,
            caller_access.auth_snapshot.as_ref(),
            None,
            now,
        )
        .await
        .map_err(gateway_error)?;
    if crate::scheduler::candidate::is_exact_all_skipped_by_auth_limit(&selected, &skipped) {
        return Err(auth_concurrency_limit_error());
    }
    if selected.is_empty() {
        return Err(AssetServiceError::unavailable(
            "创建素材时使用的 Provider 凭据当前受并发、配额或健康状态限制",
        ));
    }
    Ok(())
}

fn auth_concurrency_limit_error() -> AssetServiceError {
    AssetServiceError::new(
        StatusCode::TOO_MANY_REQUESTS,
        "ConcurrencyLimitExceeded",
        "当前 API Key 的并发请求数已达到上限",
    )
}

fn asset_transport(snapshot: GatewayProviderTransportSnapshot) -> AssetTransport {
    AssetTransport { snapshot }
}

async fn execute_action(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    transport: &AssetTransport,
    action: ArkAssetAction,
    body: &Value,
) -> Result<ActionResponse, AssetServiceError> {
    let provider_body = canonicalize_provider_body(action, body)
        .map_err(|error| AssetServiceError::bad_request(error.to_string()))?;
    let request = build_volc_action_request(VolcActionRequestInput {
        transport: &transport.snapshot,
        action: action.as_str(),
        body: &provider_body,
        request_headers: headers,
        request_time: None,
    })
    .map_err(|error| AssetServiceError::unavailable(format!("Ark 素材库请求构建失败: {error}")))?;
    let proxy = state
        .resolve_transport_proxy_snapshot_with_tunnel_affinity(&transport.snapshot)
        .await;
    let mut plan = ExecutionPlan {
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
    let started_at = Instant::now();
    let started_at_unix_ms = crate::clock::current_unix_ms();
    let mut report_context = Some(json!({
        "request_id": request_context.trace_id,
        "user_id": caller.user_id,
        "api_key_id": caller.api_key_id,
        "client_api_format": ARK_ASSET_API_FORMAT,
        "provider_api_format": ARK_ASSET_API_FORMAT,
        "request_path": request_context.request_path,
        "request_query_string": request_context.request_query_string,
        "asset_action": action.as_str(),
    }));
    let mut terminal_guard = begin_asset_candidate_attempt(
        state,
        &mut plan,
        &mut report_context,
        started_at_unix_ms,
        started_at,
    );
    crate::request_candidate_runtime::record_local_request_candidate_status(
        state,
        &plan,
        report_context.as_ref(),
        SchedulerRequestCandidateStatusUpdate {
            status: RequestCandidateStatus::Pending,
            status_code: None,
            error_type: None,
            error_message: None,
            latency_ms: None,
            started_at_unix_ms: Some(started_at_unix_ms),
            finished_at_unix_ms: None,
        },
    )
    .await;
    let _upstream_execution_permit =
        match crate::execution_runtime::acquire_upstream_execution_gate(
            state,
            request_context.trace_id.as_str(),
        )
        .await
        {
            Ok(permit) => permit,
            Err(error) => {
                terminal_guard
                    .finish(
                        RequestCandidateStatus::Failed,
                        Some(StatusCode::TOO_MANY_REQUESTS.as_u16()),
                        Some("gateway_admission_failed"),
                        Some(error.into_message()),
                    )
                    .await;
                return Err(AssetServiceError::new(
                    StatusCode::TOO_MANY_REQUESTS,
                    "AdmissionTimeout",
                    "素材库上游并发队列暂时不可用",
                ));
            }
        };
    let result =
        match crate::execution_runtime::execute_execution_runtime_sync_plan_with_report_context(
            state,
            Some(request_context.trace_id.as_str()),
            &plan,
            report_context.as_ref(),
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                terminal_guard
                    .finish(
                        RequestCandidateStatus::Failed,
                        None,
                        Some("execution_runtime_unavailable"),
                        Some(error.into_message()),
                    )
                    .await;
                return Err(AssetServiceError::unavailable(
                    "Ark 素材库上游请求暂时不可用",
                ));
            }
        };
    let mut status = StatusCode::from_u16(result.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut body = execution_result_json(&result).unwrap_or(Value::Null);
    if visual_validation_result_is_pending(action, &body) {
        status = StatusCode::OK;
        body = json!({"Status": "Pending"});
    }
    if !status.is_success() {
        let (error_type, error_message) = result
            .error
            .as_ref()
            .map(|error| (format!("{:?}", error.kind), error.message.clone()))
            .unwrap_or_else(|| {
                (
                    "upstream_http_error".to_string(),
                    "Ark 素材库上游返回错误".to_string(),
                )
            });
        terminal_guard
            .finish(
                RequestCandidateStatus::Failed,
                Some(status.as_u16()),
                Some(&error_type),
                Some(error_message),
            )
            .await;
        return Err(AssetServiceError::provider(status, body));
    }
    if provider_error_value(&body).is_some() {
        let status = super::protocol_api::response_status_from_body(&body)
            .and_then(|status| StatusCode::from_u16(status).ok())
            .filter(|status| !status.is_success())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        terminal_guard
            .finish(
                RequestCandidateStatus::Failed,
                Some(status.as_u16()),
                Some("upstream_protocol_error"),
                Some("Ark 素材库上游返回协议错误".to_string()),
            )
            .await;
        return Err(AssetServiceError::provider(status, body));
    }
    terminal_guard
        .finish(
            RequestCandidateStatus::Success,
            Some(status.as_u16()),
            None,
            None,
        )
        .await;
    Ok(ActionResponse { body })
}

#[allow(clippy::too_many_arguments)]
async fn record_asset_candidate_terminal(
    state: &AppState,
    plan: &ExecutionPlan,
    report_context: Option<&Value>,
    started_at_unix_ms: u64,
    started_at: Instant,
    status: RequestCandidateStatus,
    status_code: Option<u16>,
    error_type: Option<&str>,
    error_message: Option<String>,
) {
    crate::request_candidate_runtime::record_local_request_candidate_status(
        state,
        plan,
        report_context,
        SchedulerRequestCandidateStatusUpdate {
            status,
            status_code,
            error_type: error_type.map(ToOwned::to_owned),
            error_message,
            latency_ms: Some(started_at.elapsed().as_millis() as u64),
            started_at_unix_ms: Some(started_at_unix_ms),
            finished_at_unix_ms: Some(crate::clock::current_unix_ms()),
        },
    )
    .await;
}

async fn create_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: Value,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let name = required_string_field(&body, &["Name", "name"], "Name")?;
    validate_group_text_lengths(&body)?;
    let group_type = create_group_type(&body)?;
    let transport = select_transport(state, caller).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        caller,
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
        group_type,
        name,
        description: string_field(&body, &["Description", "description"]),
        status: string_field(result, &["Status", "status"]).unwrap_or_else(|| "Active".to_string()),
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"]).unwrap_or(now),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
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
    let url = required_string_field(&body, &["URL", "Url", "url"], "URL")?;
    validate_source_url(&url)?;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "GroupId".to_string(),
            Value::String(upstream_group_id.to_string()),
        );
        object.remove("group_id");
        object.insert("URL".to_string(), Value::String(url));
        object.remove("Url");
        object.remove("url");
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        caller,
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
        last_inference_at_unix_secs: timestamp_field(
            result,
            &["LastInferenceTime", "LastInferenceAt"],
        ),
        source_url_fingerprint: source_url.as_deref().map(sha256_text),
        provider_url: None,
        provider_url_expires_at_unix_secs: None,
        sanitized_metadata: sanitize_asset_metadata(result),
        is_deleted: false,
        deleted_at_unix_secs: None,
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"]).unwrap_or(now),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
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
        caller,
        &transport,
        ArkAssetAction::GetAssetGroup,
        &json!({"Id": upstream_id}),
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
    let group_id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
    validate_group_text_lengths(&body)?;
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
        object.insert("Id".to_string(), Value::String(upstream_id.to_string()));
        object.remove("GroupId");
        object.remove("group_id");
        object.remove("id");
        if object_has_field(
            &Value::Object(object.clone()),
            &["Description", "description"],
        ) && string_field(
            &Value::Object(object.clone()),
            &["Description", "description"],
        )
        .is_none()
        {
            object.insert("Description".to_string(), Value::String(String::new()));
            object.remove("description");
        }
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        caller,
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
        caller,
        &transport,
        ArkAssetAction::DeleteAssetGroup,
        &json!({"Id": upstream_id}),
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
        caller,
        &transport,
        ArkAssetAction::GetAsset,
        &json!({"Id": upstream_id}),
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
    let asset_id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
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
        object.insert("Id".to_string(), Value::String(upstream_id.to_string()));
        object.remove("AssetId");
        object.remove("asset_id");
        object.remove("id");
    }
    let transport = exact_transport_for_group(state, caller, &group).await?;
    let response = execute_action(
        state,
        request_context,
        headers,
        caller,
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
        caller,
        &transport,
        ArkAssetAction::DeleteAsset,
        &json!({"Id": upstream_id}),
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
        group_type: string_field(result, &["GroupType", "Type"]).unwrap_or(group.group_type),
        name: string_field(result, &["Name", "name"]).unwrap_or(group.name),
        description: string_field(result, &["Description", "description"]).or(group.description),
        status: string_field(result, &["Status", "status"]).unwrap_or(group.status),
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"])
            .unwrap_or(group.created_at_unix_secs),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
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
        last_inference_at_unix_secs: timestamp_field(
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
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"])
            .unwrap_or(asset.created_at_unix_secs),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
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
    let page_size = page_size(body)?;
    let filter = value_field(body, &["Filter", "filter"])
        .filter(|value| value.is_object())
        .ok_or_else(|| AssetServiceError::bad_request("Filter.GroupType is required"))?;
    let group_type = required_string_field(
        filter,
        &["GroupType", "group_type", "Type"],
        "Filter.GroupType",
    )?;
    if !matches!(group_type.as_str(), "AIGC" | "LivenessFace") {
        return Err(AssetServiceError::bad_request(
            "Filter.GroupType must be AIGC or LivenessFace",
        ));
    }
    let group_ids = string_list_field(filter, &["GroupIds", "group_ids", "GroupId", "group_id"]);
    let statuses = string_list_field(filter, &["Statuses", "statuses", "Status", "status"]);
    let query = AssetGroupListQuery {
        user_id: Some(caller.user_id.clone()),
        api_key_id: None,
        provider_id: None,
        group_type: Some(group_type),
        search: string_field(filter, &["Name", "Search"]),
        include_deleted: false,
        ..AssetGroupListQuery::default()
    };
    let groups = list_all_groups(state, query).await?;
    let filtered = groups.into_iter().filter(|group| {
        (group_ids.is_empty()
            || group_ids.iter().any(|id| {
                id == &group.id || group.upstream_group_id.as_deref() == Some(id.as_str())
            }))
            && (statuses.is_empty()
                || statuses
                    .iter()
                    .any(|status| status.eq_ignore_ascii_case(&group.status)))
    });
    let mut filtered = filtered.collect::<Vec<_>>();
    sort_native_groups(&mut filtered, body)?;
    let total = filtered.len();
    let offset = (page - 1).saturating_mul(page_size);
    let items = filtered
        .iter()
        .skip(offset)
        .take(page_size)
        .map(group_native_json)
        .collect::<Vec<_>>();
    Ok(native_envelope(
        request_context,
        ArkAssetAction::ListAssetGroups,
        json!({
            "TotalCount": total,
            "PageNumber": page,
            "PageSize": page_size,
            "Items": items,
        }),
    ))
}

async fn list_assets_native(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let page = page_number(body);
    let page_size = page_size(body)?;
    let filter = native_list_filter(body);
    let group_ids = string_list_field(filter, &["GroupIds", "group_ids", "GroupId", "group_id"]);
    let statuses = string_list_field(filter, &["Statuses", "statuses", "Status", "status"]);
    let group_type = string_field(filter, &["GroupType", "group_type"]);
    let query = AssetListQuery {
        user_id: Some(caller.user_id.clone()),
        api_key_id: None,
        asset_type: string_field(filter, &["AssetType", "Type"]),
        search: string_field(filter, &["Name", "Search"]),
        include_deleted: false,
        ..AssetListQuery::default()
    };
    let assets = list_all_assets(state, query).await?;
    let groups = list_all_groups(
        state,
        AssetGroupListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            ..AssetGroupListQuery::default()
        },
    )
    .await?
    .into_iter()
    .map(|group| (group.id.clone(), group))
    .collect::<HashMap<_, _>>();
    let filtered = assets.into_iter().filter(|asset| {
        let group = groups.get(&asset.group_id);
        (group_ids.is_empty()
            || group_ids.iter().any(|id| {
                id == &asset.group_id
                    || group.and_then(|group| group.upstream_group_id.as_deref())
                        == Some(id.as_str())
            }))
            && (statuses.is_empty()
                || statuses
                    .iter()
                    .any(|status| status.eq_ignore_ascii_case(&asset.status)))
            && group_type.as_ref().is_none_or(|expected| {
                group.is_some_and(|group| group.group_type.eq_ignore_ascii_case(expected))
            })
    });
    let mut filtered = filtered.collect::<Vec<_>>();
    sort_native_assets(&mut filtered, body)?;
    let total = filtered.len();
    let offset = (page - 1).saturating_mul(page_size);
    let items = filtered
        .iter()
        .skip(offset)
        .take(page_size)
        .map(asset_native_json)
        .collect::<Vec<_>>();
    Ok(native_envelope(
        request_context,
        ArkAssetAction::ListAssets,
        json!({
            "TotalCount": total,
            "PageNumber": page,
            "PageSize": page_size,
            "Items": items,
        }),
    ))
}

async fn list_all_groups(
    state: &AppState,
    mut query: AssetGroupListQuery,
) -> Result<Vec<StoredAssetGroup>, AssetServiceError> {
    query.offset = 0;
    query.limit = MAX_PAGE_SIZE;
    let mut items = Vec::new();
    loop {
        query.offset = items.len();
        let page = read_repo(state)?
            .list_groups(&query)
            .await
            .map_err(data_error)?;
        let page_len = page.items.len();
        items.extend(page.items);
        if page_len == 0 || items.len() >= page.total {
            break;
        }
    }
    Ok(items)
}

async fn list_all_assets(
    state: &AppState,
    mut query: AssetListQuery,
) -> Result<Vec<StoredAsset>, AssetServiceError> {
    query.offset = 0;
    query.limit = MAX_PAGE_SIZE;
    let mut items = Vec::new();
    loop {
        query.offset = items.len();
        let page = read_repo(state)?
            .list_assets(&query)
            .await
            .map_err(data_error)?;
        let page_len = page.items.len();
        items.extend(page.items);
        if page_len == 0 || items.len() >= page.total {
            break;
        }
    }
    Ok(items)
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
        caller,
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
    let expires_at = timestamp_field(result, &["ExpireAt", "ExpiresAt", "Expiration"])
        .unwrap_or_else(|| now.saturating_add(VALIDATION_SESSION_TTL_SECS));
    let record = UpsertArkVisualValidationSessionRecord {
        id: local_id("vsess"),
        session_id,
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
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
        return Ok(public_validation_result(&session));
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
    Ok(native_validation_payload(response))
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
        caller,
        &transport,
        ArkAssetAction::GetVisualValidateResult,
        &json!({"BytedToken": token}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let status = validation_result_status(result, &session.status);
    let upstream_group_id = string_field(result, &["GroupId", "group_id"]);
    let group_id = if let Some(upstream_group_id) = upstream_group_id.as_deref() {
        let group =
            ensure_validation_group(state, caller, &transport, upstream_group_id, result).await?;
        if let Some(group) = group {
            sync_validation_group_assets(
                state,
                request_context,
                headers,
                caller,
                &transport,
                &group,
                upstream_group_id,
            )
            .await?;
            Some(group.id)
        } else {
            None
        }
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
        .find_group_by_canonical_upstream(&transport.snapshot.provider.id, upstream_group_id)
        .await
        .map_err(data_error)?
    {
        if group.user_id != caller.user_id || group.deleted_at_unix_secs.is_some() {
            return Err(AssetServiceError::new(
                StatusCode::CONFLICT,
                "AssetGroupOwnershipConflict",
                "真人验证返回的素材组已绑定到其他用户",
            ));
        }
        return Ok(Some(group));
    }
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetGroupRecord {
        id: deterministic_validation_group_id(&transport.snapshot.provider.id, upstream_group_id),
        upstream_group_id: Some(upstream_group_id.to_string()),
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
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

async fn sync_validation_group_assets(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    transport: &AssetTransport,
    group: &StoredAssetGroup,
    upstream_group_id: &str,
) -> Result<(), AssetServiceError> {
    let mut synced = 0usize;
    for page_number in 1..=100usize {
        let response = execute_action(
            state,
            request_context,
            headers,
            caller,
            transport,
            ArkAssetAction::ListAssets,
            &json!({
                "Filter": {"GroupIds": [upstream_group_id]},
                "PageNumber": page_number,
                "PageSize": ARK_MAX_PAGE_SIZE,
            }),
        )
        .await?;
        let result = extract_result(&response.body).unwrap_or(&response.body);
        let total = number_field(result, &["TotalCount", "Total"]).ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark ListAssets response is missing TotalCount",
            )
        })?;
        let items = value_field(result, &["Items", "Assets"])
            .and_then(Value::as_array)
            .ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    "Ark ListAssets response is missing Items",
                )
            })?;
        for item in items {
            upsert_validation_asset(state, group, upstream_group_id, item).await?;
        }
        synced = synced.saturating_add(items.len());
        let total = usize::try_from(total).unwrap_or(usize::MAX);
        if validation_asset_page_complete(items.len(), synced, total) {
            return Ok(());
        }
    }
    Err(AssetServiceError::new(
        StatusCode::BAD_GATEWAY,
        "InvalidUpstreamResponse",
        "Ark ListAssets response exceeded 100 pages",
    ))
}

fn validation_asset_page_complete(_page_len: usize, synced: usize, total: usize) -> bool {
    synced >= total
}

async fn upsert_validation_asset(
    state: &AppState,
    group: &StoredAssetGroup,
    upstream_group_id: &str,
    item: &Value,
) -> Result<StoredAsset, AssetServiceError> {
    let upstream_asset_id = upstream_required_string(item, &["Id", "AssetId"], "Id")?;
    if let Some(item_group_id) = string_field(item, &["GroupId", "group_id"]) {
        if item_group_id != upstream_group_id {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark ListAssets returned an asset from another group",
            ));
        }
    }
    let asset_type = upstream_required_string(item, &["AssetType", "asset_type"], "AssetType")?;
    if !asset_type.eq_ignore_ascii_case("Image") {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark ListAssets returned a non-Image asset",
        ));
    }
    let existing = read_repo(state)?
        .find_asset_by_upstream(&group.id, &upstream_asset_id)
        .await
        .map_err(data_error)?;
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetRecord {
        id: existing
            .as_ref()
            .map(|asset| asset.id.clone())
            .unwrap_or_else(|| {
                deterministic_validation_asset_id(&group.provider_id, &group.id, &upstream_asset_id)
            }),
        upstream_asset_id: Some(upstream_asset_id),
        group_id: group.id.clone(),
        user_id: group.user_id.clone(),
        api_key_id: group.api_key_id.clone(),
        asset_type: "Image".to_string(),
        name: string_field(item, &["Name", "name"]).unwrap_or_else(|| "真人素材".to_string()),
        status: string_field(item, &["Status", "status"]).unwrap_or_else(|| "Active".to_string()),
        error_code: error_field(item, "Code"),
        error_message: error_field(item, "Message"),
        moderation: object_field(item, &["ModerationResult", "Moderation", "moderation"]),
        last_inference_at_unix_secs: timestamp_field(
            item,
            &["LastInferenceTime", "LastInferenceAt"],
        ),
        source_url_fingerprint: None,
        provider_url: None,
        provider_url_expires_at_unix_secs: None,
        sanitized_metadata: sanitize_asset_metadata(item),
        is_deleted: existing.as_ref().is_some_and(|asset| asset.is_deleted),
        deleted_at_unix_secs: existing
            .as_ref()
            .and_then(|asset| asset.deleted_at_unix_secs),
        created_at_unix_secs: timestamp_field(item, &["CreateTime", "CreatedAt"])
            .or_else(|| existing.as_ref().map(|asset| asset.created_at_unix_secs))
            .unwrap_or(now),
        updated_at_unix_secs: timestamp_field(item, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
    };
    write_repo(state)?
        .upsert_asset(record)
        .await
        .map_err(data_error)
}

fn upstream_required_string(
    value: &Value,
    names: &[&str],
    display_name: &str,
) -> Result<String, AssetServiceError> {
    string_field(value, names).ok_or_else(|| {
        AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark ListAssets item is missing {display_name}"),
        )
    })
}

fn deterministic_validation_asset_id(
    provider_id: &str,
    group_id: &str,
    upstream_asset_id: &str,
) -> String {
    let digest = sha256_text(&format!("{provider_id}\0{group_id}\0{upstream_asset_id}"));
    format!("asset-{}", &digest[..32])
}

fn deterministic_validation_group_id(provider_id: &str, upstream_group_id: &str) -> String {
    let digest = sha256_text(&format!("{provider_id}\0{upstream_group_id}"));
    format!("agrp-{}", &digest[..32])
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
        if group.provider_id != transport.provider.id {
            return Err(format!("素材 {local_id} 与本次视频生成的 Provider 不一致"));
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
        "Id": group.id,
        "GroupType": group.group_type,
        "Name": group.name,
        "Description": group.description,
        "Status": group.status,
        "CreateTime": unix_secs_rfc3339(group.created_at_unix_secs),
        "UpdateTime": unix_secs_rfc3339(group.updated_at_unix_secs),
    })
}

fn asset_native_json(asset: &StoredAsset) -> Value {
    let metadata = asset.sanitized_metadata.as_ref();
    json!({
        "Id": asset.id,
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
        "CreateTime": unix_secs_rfc3339(asset.created_at_unix_secs),
        "UpdateTime": unix_secs_rfc3339(asset.updated_at_unix_secs),
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
    if string_field(result, &["GroupId", "group_id"]).is_some() {
        return "Succeeded".to_string();
    }
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
    if matches!(
        session.status.trim().to_ascii_lowercase().as_str(),
        "succeeded" | "success"
    ) {
        if let Some(group_id) = session.group_id.as_ref() {
            return json!({"GroupId": group_id});
        }
    }
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

fn native_envelope(
    request_context: &GatewayPublicRequestContext,
    action: ArkAssetAction,
    result: Value,
) -> Value {
    json!({
        "ResponseMetadata": {
            "RequestId": request_context.trace_id,
            "Action": action.as_str(),
            "Version": super::ARK_ASSET_VERSION,
            "Service": "ark",
            "Region": "cn-beijing",
        },
        "Result": result,
    })
}

fn native_validation_payload(body: Value) -> Value {
    extract_result(&body).cloned().unwrap_or(body)
}

fn native_create_validation_payload(body: Value) -> Value {
    let result = extract_result(&body).unwrap_or(&body);
    let mut projected = Map::new();
    for (name, aliases) in [
        (
            "BytedToken",
            &["BytedToken", "Token", "byted_token", "token"][..],
        ),
        ("H5Link", &["H5Link", "H5Url", "VerificationUrl", "URL"][..]),
        (
            "CallbackURL",
            &["CallbackURL", "callback_url", "ReturnUrl", "return_url"][..],
        ),
    ] {
        if let Some(value) = string_field(result, aliases) {
            projected.insert(name.to_string(), Value::String(value));
        }
    }
    Value::Object(projected)
}

fn native_error_response(error: AssetServiceError) -> Response<Body> {
    let body = error
        .provider_body
        .unwrap_or_else(|| build_error_envelope(&error.code, &error.message));
    json_response(error.status, body)
}

fn rest_error_response(error: AssetServiceError) -> Response<Body> {
    let request_id = error
        .provider_body
        .as_ref()
        .and_then(|body| body.pointer("/ResponseMetadata/RequestId"))
        .and_then(Value::as_str);
    json_response(
        error.status,
        json!({
            "detail": error.message,
            "code": error.code,
            "request_id": request_id,
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
        .or_else(|| string_field(body, &["Code", "code"]).map(|_| body))
}

fn provider_error_code(body: &Value) -> Option<String> {
    provider_error_value(body)
        .and_then(|error| string_field(error, &["Code", "code", "Type", "type"]))
}

fn visual_validation_result_is_pending(action: ArkAssetAction, body: &Value) -> bool {
    if action != ArkAssetAction::GetVisualValidateResult {
        return false;
    }
    let Some(code) = provider_error_code(body) else {
        return false;
    };
    let code = code.trim().to_ascii_lowercase();
    code == "notfound"
        || code.starts_with("notfound.")
        || code == "resourcenotfound"
        || code.starts_with("resourcenotfound.")
}

fn missing_parameter_code_and_field(code: &str) -> Option<(&str, &str)> {
    let (prefix, field) = code.split_once('.')?;
    (prefix.eq_ignore_ascii_case("MissingParameter")
        && !field.is_empty()
        && field.len() <= 64
        && field
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_'))
    .then_some((code, field))
}

fn sanitize_provider_error_body(body: &Value) -> Value {
    let error = provider_error_value(body).unwrap_or(body);
    let code = string_field(error, &["Code", "code", "Type", "type"])
        .unwrap_or_else(|| "UpstreamError".to_string());
    let message = string_field(error, &["Message", "message"])
        .or_else(|| string_field(error, &["Detail", "detail"]))
        .unwrap_or_else(|| "素材库上游请求失败".to_string());
    let request_id = body
        .get("ResponseMetadata")
        .or_else(|| body.get("response_metadata"))
        .and_then(|metadata| string_field(metadata, &["RequestId", "request_id"]))
        .or_else(|| string_field(body, &["RequestId", "request_id"]));
    let mut metadata = Map::from_iter([(
        "Error".to_string(),
        json!({"Code": code, "Message": message}),
    )]);
    if let Some(request_id) = request_id {
        metadata.insert("RequestId".to_string(), Value::String(request_id));
    }
    json!({"ResponseMetadata": Value::Object(metadata)})
}

fn sanitize_provider_error_body_with_override(body: &Value, code: &str, message: &str) -> Value {
    let mut sanitized = sanitize_provider_error_body(body);
    let Some(error) = sanitized
        .pointer_mut("/ResponseMetadata/Error")
        .and_then(Value::as_object_mut)
    else {
        return sanitized;
    };
    error.insert("Code".to_string(), Value::String(code.to_string()));
    error.insert("Message".to_string(), Value::String(message.to_string()));
    sanitized
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

fn validate_group_text_lengths(value: &Value) -> Result<(), AssetServiceError> {
    if string_field(value, &["Name", "name"]).is_some_and(|name| name.chars().count() > 64) {
        return Err(AssetServiceError::bad_request(
            "Name must be at most 64 characters",
        ));
    }
    if string_field(value, &["Description", "description"])
        .is_some_and(|description| description.chars().count() > 300)
    {
        return Err(AssetServiceError::bad_request(
            "Description must be at most 300 characters",
        ));
    }
    Ok(())
}

fn required_asset_type(value: &Value) -> Result<String, AssetServiceError> {
    let asset_type =
        required_string_field(value, &["AssetType", "asset_type", "type"], "AssetType")?;
    if !asset_type.trim().eq_ignore_ascii_case("image") {
        return Err(AssetServiceError::bad_request("AssetType must be Image"));
    }
    Ok("Image".to_string())
}

fn create_group_type(value: &Value) -> Result<String, AssetServiceError> {
    match value_field(value, &["GroupType", "Type", "group_type"]) {
        None | Some(Value::Null) => Ok("AIGC".to_string()),
        Some(Value::String(value)) if value.trim().is_empty() => Ok("AIGC".to_string()),
        Some(Value::String(value)) if value.trim() == "AIGC" => Ok("AIGC".to_string()),
        Some(Value::String(_)) => Err(AssetServiceError::bad_request("GroupType must be AIGC")),
        Some(_) => Err(AssetServiceError::bad_request("GroupType must be a string")),
    }
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

fn timestamp_field(value: &Value, names: &[&str]) -> Option<u64> {
    let value = value_field(value, names)?;
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
        .or_else(|| value.as_str()?.trim().parse::<u64>().ok())
        .or_else(|| {
            let timestamp = chrono::DateTime::parse_from_rfc3339(value.as_str()?.trim())
                .ok()?
                .timestamp();
            u64::try_from(timestamp).ok()
        })
}

fn string_list_field(value: &Value, names: &[&str]) -> Vec<String> {
    match value_field(value, names) {
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
        Some(Value::String(value)) => {
            let value = value.trim();
            (!value.is_empty())
                .then(|| vec![value.to_string()])
                .unwrap_or_default()
        }
        _ => Vec::new(),
    }
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

fn page_number(value: &Value) -> usize {
    number_field(value, &["PageNumber", "PageNum", "page_num"])
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1)
        .max(1)
}

fn page_size(value: &Value) -> Result<usize, AssetServiceError> {
    let page_size = number_field(value, &["PageSize", "page_size"])
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(ARK_DEFAULT_PAGE_SIZE);
    if !(1..=ARK_MAX_PAGE_SIZE).contains(&page_size) {
        return Err(AssetServiceError::bad_request(
            "PageSize must be between 1 and 100",
        ));
    }
    Ok(page_size)
}

#[derive(Clone, Copy)]
enum NativeSortField {
    CreateTime,
    UpdateTime,
}

fn native_sort_options(value: &Value) -> Result<(NativeSortField, bool), AssetServiceError> {
    let field = match string_field(value, &["SortBy", "sort_by"])
        .unwrap_or_else(|| "CreateTime".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "createtime" => NativeSortField::CreateTime,
        "updatetime" => NativeSortField::UpdateTime,
        _ => {
            return Err(AssetServiceError::bad_request(
                "SortBy must be CreateTime or UpdateTime",
            ));
        }
    };
    let descending = match string_field(value, &["SortOrder", "sort_order"])
        .unwrap_or_else(|| "Desc".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "desc" => true,
        "asc" => false,
        _ => {
            return Err(AssetServiceError::bad_request(
                "SortOrder must be Asc or Desc",
            ));
        }
    };
    Ok((field, descending))
}

fn sort_native_groups(
    groups: &mut [StoredAssetGroup],
    body: &Value,
) -> Result<(), AssetServiceError> {
    let (field, descending) = native_sort_options(body)?;
    groups.sort_by(|left, right| {
        let order = match field {
            NativeSortField::CreateTime => {
                left.created_at_unix_secs.cmp(&right.created_at_unix_secs)
            }
            NativeSortField::UpdateTime => {
                left.updated_at_unix_secs.cmp(&right.updated_at_unix_secs)
            }
        };
        let order = if descending { order.reverse() } else { order };
        order.then_with(|| left.id.cmp(&right.id))
    });
    Ok(())
}

fn sort_native_assets(assets: &mut [StoredAsset], body: &Value) -> Result<(), AssetServiceError> {
    let (field, descending) = native_sort_options(body)?;
    assets.sort_by(|left, right| {
        let order = match field {
            NativeSortField::CreateTime => {
                left.created_at_unix_secs.cmp(&right.created_at_unix_secs)
            }
            NativeSortField::UpdateTime => {
                left.updated_at_unix_secs.cmp(&right.updated_at_unix_secs)
            }
        };
        let order = if descending { order.reverse() } else { order };
        order.then_with(|| left.id.cmp(&right.id))
    });
    Ok(())
}

fn validate_native_project(body: &Value) -> Result<(), AssetServiceError> {
    let Some(project) = value_field(body, &["ProjectName", "project_name"]) else {
        return Ok(());
    };
    if project.is_null()
        || project
            .as_str()
            .map(str::trim)
            .is_some_and(|project| project.eq_ignore_ascii_case("default"))
    {
        return Ok(());
    }
    Err(AssetServiceError::bad_request(
        "ProjectName must be omitted or default",
    ))
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use aether_data::repository::asset_library::{
        AssetLibraryReadRepository, AssetLibraryWriteRepository, InMemoryAssetLibraryRepository,
    };
    use aether_data::repository::candidates::InMemoryRequestCandidateRepository;
    use aether_data::repository::users::{InMemoryUserReadRepository, StoredUserAuthRecord};
    use aether_data_contracts::repository::candidates::{
        PublicHealthStatusCount, PublicHealthTimelineBucket, RequestCandidateReadRepository,
        RequestCandidateWriteRepository, StoredRequestCandidate, UpsertRequestCandidateRecord,
    };
    use aether_data_contracts::DataLayerError;
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use tokio::sync::Notify;

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

    fn admin_caller_for(user_id: &str) -> AssetCaller {
        AssetCaller {
            user_id: user_id.to_string(),
            api_key_id: None,
            unrestricted_provider_access: true,
            allowed_providers: None,
            allowed_api_formats: None,
        }
    }

    fn stored_user(user_id: &str, is_active: bool, is_deleted: bool) -> StoredUserAuthRecord {
        StoredUserAuthRecord::new(
            user_id.to_string(),
            Some(format!("{user_id}@example.test")),
            true,
            user_id.to_string(),
            Some("password-hash".to_string()),
            "user".to_string(),
            "local".to_string(),
            None,
            None,
            None,
            is_active,
            is_deleted,
            None,
            None,
        )
        .expect("test user")
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

    fn candidate_plan(request_id: &str, candidate_id: &str) -> ExecutionPlan {
        ExecutionPlan {
            request_id: request_id.to_string(),
            candidate_id: Some(candidate_id.to_string()),
            provider_name: Some("Ark".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "provider-key-1".to_string(),
            method: "POST".to_string(),
            url: "https://ark.example/?Action=ListAssets&Version=2024-01-01".to_string(),
            headers: BTreeMap::new(),
            content_type: Some("application/json".to_string()),
            content_encoding: None,
            body: RequestBody::from_json(json!({})),
            stream: false,
            client_api_format: ARK_ASSET_API_FORMAT.to_string(),
            provider_api_format: ARK_ASSET_API_FORMAT.to_string(),
            model_name: None,
            proxy: None,
            transport_profile: None,
            timeouts: None,
        }
    }

    #[derive(Default)]
    struct BlockingPendingRequestCandidateRepository {
        inner: InMemoryRequestCandidateRepository,
        block_next_pending: AtomicBool,
        pending_persisted: Notify,
        release_pending: Notify,
    }

    impl BlockingPendingRequestCandidateRepository {
        fn blocking() -> Self {
            Self {
                block_next_pending: AtomicBool::new(true),
                ..Self::default()
            }
        }
    }

    #[async_trait]
    impl RequestCandidateReadRepository for BlockingPendingRequestCandidateRepository {
        async fn list_by_request_id(
            &self,
            request_id: &str,
        ) -> Result<Vec<StoredRequestCandidate>, DataLayerError> {
            self.inner.list_by_request_id(request_id).await
        }

        async fn list_recent(
            &self,
            limit: usize,
        ) -> Result<Vec<StoredRequestCandidate>, DataLayerError> {
            self.inner.list_recent(limit).await
        }

        async fn list_by_provider_id(
            &self,
            provider_id: &str,
            limit: usize,
        ) -> Result<Vec<StoredRequestCandidate>, DataLayerError> {
            self.inner.list_by_provider_id(provider_id, limit).await
        }

        async fn list_finalized_by_endpoint_ids_since(
            &self,
            endpoint_ids: &[String],
            since_unix_secs: u64,
            limit: usize,
        ) -> Result<Vec<StoredRequestCandidate>, DataLayerError> {
            self.inner
                .list_finalized_by_endpoint_ids_since(endpoint_ids, since_unix_secs, limit)
                .await
        }

        async fn count_finalized_statuses_by_endpoint_ids_since(
            &self,
            endpoint_ids: &[String],
            since_unix_secs: u64,
        ) -> Result<Vec<PublicHealthStatusCount>, DataLayerError> {
            self.inner
                .count_finalized_statuses_by_endpoint_ids_since(endpoint_ids, since_unix_secs)
                .await
        }

        async fn aggregate_finalized_timeline_by_endpoint_ids_since(
            &self,
            endpoint_ids: &[String],
            since_unix_secs: u64,
            until_unix_secs: u64,
            segments: u32,
        ) -> Result<Vec<PublicHealthTimelineBucket>, DataLayerError> {
            self.inner
                .aggregate_finalized_timeline_by_endpoint_ids_since(
                    endpoint_ids,
                    since_unix_secs,
                    until_unix_secs,
                    segments,
                )
                .await
        }
    }

    #[async_trait]
    impl RequestCandidateWriteRepository for BlockingPendingRequestCandidateRepository {
        async fn upsert(
            &self,
            candidate: UpsertRequestCandidateRecord,
        ) -> Result<StoredRequestCandidate, DataLayerError> {
            let should_block = candidate.status == RequestCandidateStatus::Pending
                && self.block_next_pending.swap(false, Ordering::SeqCst);
            let stored = self.inner.upsert(candidate).await?;
            if should_block {
                self.pending_persisted.notify_one();
                self.release_pending.notified().await;
            }
            Ok(stored)
        }

        async fn delete_created_before(
            &self,
            created_before_unix_secs: u64,
            limit: usize,
        ) -> Result<usize, DataLayerError> {
            self.inner
                .delete_created_before(created_before_unix_secs, limit)
                .await
        }
    }

    #[tokio::test]
    async fn dropped_asset_candidate_is_finalized_as_cancelled() {
        let repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_repository_for_tests(Arc::clone(
                    &repository,
                )),
            );
        let request_id = "asset-cancelled-request";
        let plan = candidate_plan(request_id, "asset-cancelled-candidate");
        let report_context = Some(json!({
            "request_id": request_id,
            "candidate_id": "asset-cancelled-candidate",
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-1",
            "api_key_id": "api-key-1",
        }));
        crate::request_candidate_runtime::record_local_request_candidate_status(
            &state,
            &plan,
            report_context.as_ref(),
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Pending,
                status_code: None,
                error_type: None,
                error_message: None,
                latency_ms: None,
                started_at_unix_ms: Some(1),
                finished_at_unix_ms: None,
            },
        )
        .await;

        {
            let _guard =
                AssetCandidateTerminalGuard::new(&state, &plan, report_context, 1, Instant::now());
        }

        let candidate = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let rows = repository
                    .list_by_request_id(request_id)
                    .await
                    .expect("request candidate read");
                if let Some(candidate) = rows
                    .into_iter()
                    .find(|candidate| candidate.status == RequestCandidateStatus::Cancelled)
                {
                    break candidate;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled candidate should be persisted");

        assert_eq!(candidate.status_code, Some(499));
        assert_eq!(
            candidate.error_type.as_deref(),
            Some("asset_request_cancelled")
        );
        assert!(candidate.finished_at_unix_ms.is_some());
    }

    #[tokio::test]
    async fn cancellation_during_pending_persist_is_finalized_as_cancelled() {
        let repository = Arc::new(BlockingPendingRequestCandidateRepository::blocking());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_repository_for_tests(Arc::clone(
                    &repository,
                )),
            );
        let request_id = "asset-cancelled-during-pending";
        let task = tokio::spawn({
            let state = state.clone();
            async move {
                let mut plan = candidate_plan(request_id, "");
                plan.candidate_id = None;
                let mut report_context = Some(json!({
                    "request_id": request_id,
                    "candidate_index": 0,
                    "retry_index": 0,
                    "user_id": "user-1",
                    "api_key_id": "api-key-1",
                }));
                let started_at_unix_ms = 1;
                let _guard = begin_asset_candidate_attempt(
                    &state,
                    &mut plan,
                    &mut report_context,
                    started_at_unix_ms,
                    Instant::now(),
                );
                crate::request_candidate_runtime::record_local_request_candidate_status(
                    &state,
                    &plan,
                    report_context.as_ref(),
                    SchedulerRequestCandidateStatusUpdate {
                        status: RequestCandidateStatus::Pending,
                        status_code: None,
                        error_type: None,
                        error_message: None,
                        latency_ms: None,
                        started_at_unix_ms: Some(started_at_unix_ms),
                        finished_at_unix_ms: None,
                    },
                )
                .await;
            }
        });

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            repository.pending_persisted.notified(),
        )
        .await
        .expect("pending candidate should be persisted before cancellation");
        task.abort();
        assert!(task
            .await
            .expect_err("request task should be cancelled")
            .is_cancelled());

        let candidate = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let rows = repository
                    .list_by_request_id(request_id)
                    .await
                    .expect("request candidate read");
                if let Some(candidate) = rows
                    .into_iter()
                    .find(|candidate| candidate.status == RequestCandidateStatus::Cancelled)
                {
                    break candidate;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled candidate should replace pending state");

        assert_eq!(candidate.status_code, Some(499));
        assert_eq!(
            candidate.error_type.as_deref(),
            Some("asset_request_cancelled")
        );
        assert!(candidate.finished_at_unix_ms.is_some());
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

    #[tokio::test]
    async fn rest_visual_validation_requires_callback_url_before_provider_selection() {
        let state = AppState::new().expect("gateway state");
        let headers = HeaderMap::new();
        let uri: Uri = "/api/material-assets/verification-sessions"
            .parse()
            .unwrap();
        let decision = crate::control::classify_control_route(&http::Method::POST, &uri, &headers)
            .expect("public material asset route");
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-create-validation-missing-callback",
            &http::Method::POST,
            &uri,
            &headers,
            Some(decision),
        );
        let body = Bytes::from_static(br#"{}"#);

        let error = handle_rest_request(
            &state,
            &context,
            &headers,
            Some(&body),
            caller("user-1"),
            false,
        )
        .await
        .expect_err("missing callback URL must be rejected before provider selection");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "InvalidParameter");
        assert_eq!(error.message, "callback_url is required");
    }

    #[tokio::test]
    async fn native_visual_validation_requires_callback_url_before_provider_selection() {
        let state = AppState::new().expect("gateway state");
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-native-validation-missing-callback",
            &http::Method::POST,
            &"/?Action=CreateVisualValidateSession&Version=2024-01-01"
                .parse()
                .unwrap(),
            &headers,
            None,
        );

        let error = handle_native_action(
            &state,
            &context,
            &headers,
            &caller("user-1"),
            ArkAssetAction::CreateVisualValidateSession,
            json!({}),
        )
        .await
        .expect_err("missing CallbackURL must be rejected before provider selection");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code, "MissingParameter.CallbackURL");
        assert_eq!(error.message, "CallbackURL is required");
    }

    #[test]
    fn create_asset_accepts_only_k23_image_type() {
        assert_eq!(
            required_asset_type(&json!({"AssetType": "image"})).unwrap(),
            "Image"
        );
        for asset_type in ["Video", "Audio"] {
            let error = required_asset_type(&json!({"AssetType": asset_type}))
                .expect_err("k23 only accepts Image assets");
            assert_eq!(error.status, StatusCode::BAD_REQUEST, "{asset_type}");
            assert_eq!(error.message, "AssetType must be Image", "{asset_type}");
        }
    }

    #[tokio::test]
    async fn create_group_rejects_liveness_face_before_provider_selection() {
        let state = AppState::new().expect("gateway state");
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-create-liveness-group",
            &http::Method::POST,
            &"/?Action=CreateAssetGroup&Version=2024-01-01"
                .parse()
                .unwrap(),
            &headers,
            None,
        );
        let error = create_group(
            &state,
            &context,
            &headers,
            &caller("user-1"),
            json!({"Name": "face", "GroupType": "LivenessFace"}),
        )
        .await
        .expect_err("LivenessFace groups are created only by visual validation");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "GroupType must be AIGC");
    }

    #[test]
    fn create_group_type_defaults_when_omitted_null_or_empty() {
        for body in [
            json!({"Name": "人物参考素材"}),
            json!({"Name": "人物参考素材", "GroupType": null}),
            json!({"Name": "人物参考素材", "GroupType": ""}),
            json!({"Name": "人物参考素材", "GroupType": "   "}),
        ] {
            assert_eq!(create_group_type(&body).unwrap(), "AIGC");
        }
    }

    #[test]
    fn create_group_type_accepts_aliases_and_rejects_invalid_values() {
        for body in [
            json!({"Type": "AIGC"}),
            json!({"group_type": "AIGC"}),
            json!({"GroupType": " AIGC "}),
        ] {
            assert_eq!(create_group_type(&body).unwrap(), "AIGC");
        }

        let error = create_group_type(&json!({"GroupType": "LivenessFace"}))
            .expect_err("unsupported explicit group types must be rejected");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "GroupType must be AIGC");

        let error = create_group_type(&json!({"GroupType": 1}))
            .expect_err("non-string group types must be rejected");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "GroupType must be a string");
    }

    #[test]
    fn group_text_limits_count_unicode_characters() {
        validate_group_text_lengths(&json!({
            "Name": "人".repeat(64),
            "Description": "描".repeat(300),
        }))
        .expect("documented Unicode character limits");

        let name_error = validate_group_text_lengths(&json!({"Name": "人".repeat(65)}))
            .expect_err("Name must not exceed 64 characters");
        assert_eq!(name_error.status, StatusCode::BAD_REQUEST);
        assert_eq!(name_error.message, "Name must be at most 64 characters");

        let description_error =
            validate_group_text_lengths(&json!({"Description": "描".repeat(301)}))
                .expect_err("Description must not exceed 300 characters");
        assert_eq!(description_error.status, StatusCode::BAD_REQUEST);
        assert_eq!(
            description_error.message,
            "Description must be at most 300 characters"
        );
    }

    #[tokio::test]
    async fn create_asset_requires_url_before_provider_selection() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(stored_group("group-local", "user-1"))
            .await
            .expect("group");
        let state = state_with_asset_repository(repository);
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-create-asset-missing-url",
            &http::Method::POST,
            &"/?Action=CreateAsset&Version=2024-01-01".parse().unwrap(),
            &headers,
            None,
        );

        let error = create_asset(
            &state,
            &context,
            &headers,
            &caller("user-1"),
            json!({"GroupId": "group-local", "AssetType": "Image"}),
        )
        .await
        .expect_err("URL is required by the k23 contract");

        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "URL is required");
    }

    #[tokio::test]
    async fn admin_create_group_rejects_unknown_owner_before_upstream_execution() {
        let upstream_calls = Arc::new(AtomicUsize::new(0));
        let calls_for_override = Arc::clone(&upstream_calls);
        let users = Arc::new(InMemoryUserReadRepository::seed_auth_users(Vec::new()));
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(GatewayDataState::with_user_reader_for_tests(users))
            .with_execution_runtime_sync_override_for_tests(move |_| {
                calls_for_override.fetch_add(1, Ordering::SeqCst);
                Err(GatewayError::Internal(
                    "unexpected upstream invocation".to_string(),
                ))
            });
        let headers = HeaderMap::new();
        let uri: Uri = "/api/admin/material-assets/groups".parse().unwrap();
        let decision = crate::control::classify_control_route(&http::Method::POST, &uri, &headers)
            .expect("admin material asset route");
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-admin-create-group-unknown-owner",
            &http::Method::POST,
            &uri,
            &headers,
            Some(decision),
        );
        let request_body = Bytes::from(
            serde_json::to_vec(&json!({
                "name": "人物参考素材",
                "group_type": "AIGC",
                "user_id": "missing-user",
            }))
            .unwrap(),
        );

        let error = handle_rest_request(
            &state,
            &context,
            &headers,
            Some(&request_body),
            admin_caller_for("missing-user"),
            true,
        )
        .await
        .expect_err("unknown owner must be rejected");

        assert_eq!(error.status, StatusCode::NOT_FOUND);
        assert_eq!(error.code, "UserNotFound");
        assert_eq!(error.message, "目标用户不存在或已删除");
        assert_eq!(upstream_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn admin_create_group_rejects_inactive_or_deleted_owner() {
        for (user, expected_status, expected_code, expected_message) in [
            (
                stored_user("inactive-user", false, false),
                StatusCode::BAD_REQUEST,
                "UserInactive",
                "目标用户已停用",
            ),
            (
                stored_user("deleted-user", true, true),
                StatusCode::NOT_FOUND,
                "UserNotFound",
                "目标用户不存在或已删除",
            ),
        ] {
            let user_id = user.id.clone();
            let state = AppState::new()
                .expect("gateway state")
                .with_data_state_for_tests(GatewayDataState::with_user_reader_for_tests(Arc::new(
                    InMemoryUserReadRepository::seed_auth_users(vec![user]),
                )));

            let error = validate_admin_group_owner(&state, &user_id)
                .await
                .expect_err("unavailable owner must be rejected");

            assert_eq!(error.status, expected_status, "{user_id}");
            assert_eq!(error.code, expected_code, "{user_id}");
            assert_eq!(error.message, expected_message, "{user_id}");
        }
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
    fn top_level_relay_error_is_detected_and_keeps_safe_request_id() {
        let body = json!({
            "code": "MissingParameter.URL",
            "detail": "missing URL",
            "request_id": "relay-request-id",
            "debug": {"authorization": "must-not-leak"}
        });

        assert_eq!(provider_error_value(&body), Some(&body));
        assert_eq!(
            provider_error_code(&body).as_deref(),
            Some("MissingParameter.URL")
        );
        assert_eq!(
            sanitize_provider_error_body(&body),
            json!({
                "ResponseMetadata": {
                    "RequestId": "relay-request-id",
                    "Error": {
                        "Code": "MissingParameter.URL",
                        "Message": "missing URL"
                    }
                }
            })
        );
    }

    #[test]
    fn visual_validation_not_found_is_pending_only_for_result_polling() {
        for code in [
            "NotFound",
            "NotFound.20260819022657D1490FE9F21E22D81D08",
            "ResourceNotFound",
            "ResourceNotFound.session",
        ] {
            let body = json!({
                "code": code,
                "detail": "validation is not complete",
                "request_id": "relay-request-id"
            });
            assert!(visual_validation_result_is_pending(
                ArkAssetAction::GetVisualValidateResult,
                &body
            ));
            assert!(!visual_validation_result_is_pending(
                ArkAssetAction::GetAsset,
                &body
            ));
        }

        assert!(!visual_validation_result_is_pending(
            ArkAssetAction::GetVisualValidateResult,
            &json!({"code": "AccessDenied"})
        ));
    }

    #[test]
    fn provider_errors_map_actionable_upstream_codes_without_exposing_credentials() {
        let cases = [
            (
                "SubscriptionRequired",
                StatusCode::FORBIDDEN,
                "SubscriptionRequired",
                "火山素材库服务未开通，请在火山控制台开通对应套餐",
            ),
            (
                "SignatureDoesNotMatch",
                StatusCode::BAD_GATEWAY,
                "SignatureDoesNotMatch",
                "火山素材库请求签名校验失败，请检查 AK/SK、Region、Service 和系统时间",
            ),
            (
                "AccessDenied",
                StatusCode::FORBIDDEN,
                "AccessDenied",
                "火山素材库账号没有执行此操作的权限",
            ),
            (
                "UnknownProviderCode",
                StatusCode::FORBIDDEN,
                "UpstreamAccessDenied",
                "火山素材库账号没有执行此操作的权限",
            ),
        ];

        for (upstream_code, expected_status, expected_code, expected_message) in cases {
            let error = AssetServiceError::provider(
                StatusCode::FORBIDDEN,
                json!({
                    "ResponseMetadata": {
                        "RequestId": "provider-request-id",
                        "Error": {
                            "Code": upstream_code,
                            "Message": "secret_access_key=must-not-leak"
                        }
                    }
                }),
            );

            assert_eq!(error.status, expected_status, "{upstream_code}");
            assert_eq!(error.code, expected_code, "{upstream_code}");
            assert_eq!(error.message, expected_message, "{upstream_code}");
            let provider_body = error.provider_body.expect("sanitized provider body");
            assert_eq!(
                provider_body["ResponseMetadata"]["RequestId"],
                "provider-request-id"
            );
            assert_eq!(
                provider_body["ResponseMetadata"]["Error"]["Code"],
                expected_code
            );
            assert!(!provider_body.to_string().contains("secret_access_key"));
        }
    }

    #[test]
    fn provider_auth_status_without_code_keeps_safe_auth_mapping() {
        let unauthorized = AssetServiceError::provider(
            StatusCode::UNAUTHORIZED,
            json!({"message": "upstream credential rejected"}),
        );
        assert_eq!(unauthorized.status, StatusCode::BAD_GATEWAY);
        assert_eq!(unauthorized.code, "UpstreamAuthenticationError");

        let forbidden = AssetServiceError::provider(
            StatusCode::FORBIDDEN,
            json!({"message": "permission rejected"}),
        );
        assert_eq!(forbidden.status, StatusCode::FORBIDDEN);
        assert_eq!(forbidden.code, "UpstreamAccessDenied");
    }

    #[test]
    fn missing_parameter_errors_keep_safe_code_and_name_the_field() {
        for field in ["Filter", "CallbackURL"] {
            let code = format!("MissingParameter.{field}");
            let error = AssetServiceError::provider(
                StatusCode::BAD_REQUEST,
                json!({
                    "ResponseMetadata": {
                        "RequestId": "provider-request-id",
                        "Error": {
                            "Code": code,
                            "Message": "internal provider details",
                            "SecretAccessKey": "must-not-leak"
                        }
                    }
                }),
            );

            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.code, code);
            assert_eq!(error.message, format!("素材库上游缺少必填参数：{field}"));
            let provider_body = error.provider_body.expect("sanitized provider body");
            assert_eq!(provider_body["ResponseMetadata"]["Error"]["Code"], code);
            assert_eq!(
                provider_body["ResponseMetadata"]["Error"]["Message"],
                format!("素材库上游缺少必填参数：{field}")
            );
            assert!(!provider_body.to_string().contains("SecretAccessKey"));
            assert!(!provider_body
                .to_string()
                .contains("internal provider details"));
        }
    }

    #[tokio::test]
    async fn provider_error_responses_expose_actionable_code_and_request_id_only() {
        let body = json!({
            "ResponseMetadata": {
                "RequestId": "provider-request-id",
                "Error": {
                    "Code": "SubscriptionRequired",
                    "Message": "secret_access_key=must-not-leak"
                }
            }
        });

        let native = native_error_response(AssetServiceError::provider(
            StatusCode::FORBIDDEN,
            body.clone(),
        ));
        assert_eq!(native.status(), StatusCode::FORBIDDEN);
        let native_body: Value = serde_json::from_slice(
            &to_bytes(native.into_body(), usize::MAX)
                .await
                .expect("native response body"),
        )
        .expect("native response JSON");
        assert_eq!(
            native_body["ResponseMetadata"]["Error"]["Code"],
            "SubscriptionRequired"
        );
        assert_eq!(
            native_body["ResponseMetadata"]["RequestId"],
            "provider-request-id"
        );
        assert!(!native_body.to_string().contains("secret_access_key"));

        let rest = rest_error_response(AssetServiceError::provider(StatusCode::FORBIDDEN, body));
        assert_eq!(rest.status(), StatusCode::FORBIDDEN);
        let rest_body: Value = serde_json::from_slice(
            &to_bytes(rest.into_body(), usize::MAX)
                .await
                .expect("REST response body"),
        )
        .expect("REST response JSON");
        assert_eq!(rest_body["code"], "SubscriptionRequired");
        assert_eq!(rest_body["request_id"], "provider-request-id");
        assert_eq!(
            rest_body["detail"],
            "火山素材库服务未开通，请在火山控制台开通对应套餐"
        );
        assert!(!rest_body.to_string().contains("secret_access_key"));
    }

    #[test]
    fn nested_native_filters_and_result_aliases_are_supported() {
        let body = json!({
            "PageNumber": 2,
            "PageSize": 50,
            "Filter": {
                "GroupIds": ["group-1", "group-2"],
                "AssetType": "Image",
                "Statuses": ["Active", "Processing"]
            }
        });
        let filter = native_list_filter(&body);

        assert_eq!(page_number(&body), 2);
        assert_eq!(page_size(&body).unwrap(), 50);
        assert_eq!(
            page_size(&json!({"PageSize": 101}))
                .expect_err("k23 PageSize has a hard upper bound")
                .message,
            "PageSize must be between 1 and 100"
        );
        assert_eq!(
            string_list_field(filter, &["GroupIds"]),
            vec!["group-1", "group-2"]
        );
        assert_eq!(string_field(filter, &["AssetType"]), Some("Image".into()));
        assert_eq!(
            string_list_field(filter, &["Statuses"]),
            vec!["Active", "Processing"]
        );
        assert_eq!(
            timestamp_field(
                &json!({"CreateTime": "2026-03-28T12:34:56Z"}),
                &["CreateTime"]
            ),
            Some(1_774_701_296)
        );

        let mut groups = vec![
            stored_group("newer", "user-1").into_stored(),
            stored_group("older", "user-1").into_stored(),
        ];
        groups[0].updated_at_unix_secs = 20;
        groups[1].updated_at_unix_secs = 10;
        sort_native_groups(
            &mut groups,
            &json!({"SortBy": "UpdateTime", "SortOrder": "Asc"}),
        )
        .unwrap();
        assert_eq!(groups[0].id, "older");

        assert!(validate_native_project(&json!({"ProjectName": "default"})).is_ok());
        assert_eq!(
            validate_native_project(&json!({"ProjectName": "another-project"}))
                .expect_err("Aether does not persist Ark project bindings")
                .message,
            "ProjectName must be omitted or default"
        );
    }

    #[test]
    fn k23_documented_response_fixtures_are_consumable() {
        let created_group = json!({
            "ResponseMetadata": {"RequestId": "request-create-group"},
            "Result": {"Id": "group-upstream"}
        });
        assert_eq!(
            string_field(
                extract_result(&created_group).unwrap(),
                &["GroupId", "Id", "group_id", "id"]
            ),
            Some("group-upstream".to_string())
        );

        let group = json!({
            "ResponseMetadata": {"RequestId": "request-get-group"},
            "Result": {
                "Id": "group-upstream",
                "Name": "products",
                "GroupType": "AIGC",
                "CreateTime": "2026-03-28T12:34:56Z",
                "UpdateTime": "2026-03-28T12:34:56Z"
            }
        });
        let group = extract_result(&group).unwrap();
        assert_eq!(string_field(group, &["Id"]), Some("group-upstream".into()));
        assert_eq!(timestamp_field(group, &["CreateTime"]), Some(1_774_701_296));

        let assets = json!({
            "ResponseMetadata": {"RequestId": "request-list-assets"},
            "Result": {
                "TotalCount": 1,
                "Items": [{
                    "Id": "asset-upstream",
                    "GroupId": "group-upstream",
                    "AssetType": "Image",
                    "Status": "Active"
                }],
                "PageNumber": 1,
                "PageSize": 10
            }
        });
        let assets = extract_result(&assets).unwrap();
        assert_eq!(number_field(assets, &["TotalCount", "Total"]), Some(1));
        assert_eq!(number_field(assets, &["PageNumber"]), Some(1));
        assert_eq!(number_field(assets, &["PageSize"]), Some(10));

        let validation = json!({
            "BytedToken": "token-upstream",
            "H5Link": "https://verify.example.com/session",
            "CallbackURL": "https://example.com/callback",
            "SessionId": "must-not-be-exposed",
            "Status": "Pending"
        });
        assert_eq!(
            string_field(&validation, &["BytedToken"]),
            Some("token-upstream".into())
        );
        assert_eq!(
            validation_verification_url(&validation),
            Some("https://verify.example.com/session".into())
        );
        assert_eq!(
            native_create_validation_payload(json!({"Result": validation})),
            json!({
                "BytedToken": "token-upstream",
                "H5Link": "https://verify.example.com/session",
                "CallbackURL": "https://example.com/callback"
            })
        );

        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "request-native-shape",
            &http::Method::POST,
            &"/?Action=CreateAsset&Version=2024-01-01".parse().unwrap(),
            &headers,
            None,
        );
        assert_eq!(
            native_envelope(
                &context,
                ArkAssetAction::CreateAsset,
                json!({"Id": "asset-local"})
            )["Result"],
            json!({"Id": "asset-local"})
        );
        assert_eq!(
            native_envelope(&context, ArkAssetAction::DeleteAsset, json!({}))["Result"],
            json!({})
        );
        let envelope = native_envelope(
            &context,
            ArkAssetAction::CreateAsset,
            json!({"Id": "asset-local"}),
        );
        let metadata = &envelope["ResponseMetadata"];
        assert_eq!(metadata["Action"], "CreateAsset");
        assert_eq!(metadata["Version"], "2024-01-01");
        assert_eq!(metadata["Service"], "ark");
        assert_eq!(metadata["Region"], "cn-beijing");
        let group_resource =
            group_native_json(&stored_group("group-local", "user-1").into_stored());
        assert_eq!(group_resource["Id"], "group-local");
        assert!(group_resource.get("Group").is_none());
        assert!(group_resource.get("GroupId").is_none());
    }

    #[tokio::test]
    async fn validation_asset_fixture_is_idempotent_and_does_not_persist_urls() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        let group = repository
            .upsert_group(stored_group("group-local", "user-1"))
            .await
            .expect("group");
        let state = state_with_asset_repository(Arc::clone(&repository));
        let item = json!({
            "Id": "asset-upstream",
            "GroupId": "upstream-group-local",
            "Name": "face",
            "URL": "https://media.example.com/face.png?signature=secret",
            "AssetType": "Image",
            "Status": "Active",
            "Width": 1024,
            "Height": 1024,
            "CreateTime": "2026-03-28T12:34:56Z",
            "UpdateTime": "2026-03-28T12:34:56Z"
        });

        let first = upsert_validation_asset(&state, &group, "upstream-group-local", &item)
            .await
            .expect("first projection");
        let second = upsert_validation_asset(&state, &group, "upstream-group-local", &item)
            .await
            .expect("idempotent projection");
        assert_eq!(first.id, second.id);
        assert_eq!(
            first.id,
            deterministic_validation_asset_id("provider-1", "group-local", "asset-upstream")
        );
        assert_eq!(second.provider_url, None);
        assert_eq!(second.provider_url_expires_at_unix_secs, None);
        assert_eq!(second.source_url_fingerprint, None);
        assert_eq!(
            second.sanitized_metadata,
            Some(json!({"Width": 1024, "Height": 1024}))
        );
        let mismatch = upsert_validation_asset(&state, &group, "another-group", &item)
            .await
            .expect_err("cross-group item must be rejected");
        assert_eq!(mismatch.status, StatusCode::BAD_GATEWAY);

        assert_eq!(
            deterministic_validation_group_id("provider-1", "upstream-group-local"),
            deterministic_validation_group_id("provider-1", "upstream-group-local")
        );
        assert_ne!(
            deterministic_validation_group_id("provider-1", "upstream-group-local"),
            deterministic_validation_group_id("provider-1", "another-upstream-group")
        );

        assert!(!validation_asset_page_complete(100, 100, 101));
        assert!(!validation_asset_page_complete(1, 100, 101));
        assert!(!validation_asset_page_complete(0, 100, 101));
        assert!(validation_asset_page_complete(1, 101, 101));
    }

    #[tokio::test]
    async fn terminal_validation_result_keeps_k23_top_level_shape_on_repeated_queries() {
        let token = "token-upstream";
        let now = crate::clock::current_unix_secs();
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(stored_group("group-local", "user-1"))
            .await
            .expect("validation group");
        repository
            .upsert_visual_validation_session(UpsertArkVisualValidationSessionRecord {
                id: "vsess-local".to_string(),
                session_id: "session-upstream".to_string(),
                user_id: "user-1".to_string(),
                api_key_id: Some("key-user-1".to_string()),
                provider_id: "provider-1".to_string(),
                endpoint_id: "endpoint-1".to_string(),
                key_id: "provider-key-1".to_string(),
                byted_token_hash: sha256_text(token),
                encrypted_byted_token: "unused-terminal-ciphertext".to_string(),
                callback_state_hash: "callback-hash".to_string(),
                status: "Succeeded".to_string(),
                expires_at_unix_secs: now.saturating_add(60),
                consumed_at_unix_secs: Some(now),
                group_id: Some("group-local".to_string()),
                sanitized_result: Some(json!({"GroupId": "group-local"})),
                created_at_unix_secs: 1,
                updated_at_unix_secs: now,
            })
            .await
            .expect("validation session");
        let state = state_with_asset_repository(repository);
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-k23-validation-result",
            &http::Method::POST,
            &"/?Action=GetVisualValidateResult&Version=2024-01-01"
                .parse()
                .unwrap(),
            &headers,
            None,
        );
        let request = json!({"BytedToken": token});

        for _ in 0..2 {
            let result = get_validation_result_native(
                &state,
                &context,
                &headers,
                &caller("user-1"),
                &request,
            )
            .await
            .expect("terminal validation result");
            assert_eq!(result, json!({"GroupId": "group-local"}));
            assert!(result.get("Result").is_none());
            assert!(result.get("ResponseMetadata").is_none());
        }
    }

    #[tokio::test]
    async fn k23_native_lists_apply_plural_filters_before_pagination() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        let mut active_group = stored_group("group-active", "user-1");
        active_group.name = "产品人物".to_string();
        let mut disabled_group = stored_group("group-disabled", "user-1");
        disabled_group.status = "Disabled".to_string();
        let mut face_group = stored_group("group-face", "user-1");
        face_group.group_type = "LivenessFace".to_string();
        for group in [active_group, disabled_group, face_group] {
            repository.upsert_group(group).await.expect("group");
        }
        let mut active_asset = stored_asset("asset-active", "group-active", "user-1");
        active_asset.status = "Active".to_string();
        let mut disabled_asset = stored_asset("asset-disabled", "group-disabled", "user-1");
        disabled_asset.status = "Disabled".to_string();
        for asset in [active_asset, disabled_asset] {
            repository.upsert_asset(asset).await.expect("asset");
        }
        let state = state_with_asset_repository(repository);
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-k23-lists",
            &http::Method::POST,
            &"/?Action=ListAssetGroups&Version=2024-01-01"
                .parse()
                .unwrap(),
            &headers,
            None,
        );

        let groups = list_groups_native(
            &state,
            &context,
            &caller("user-1"),
            &json!({
                "Filter": {
                    "GroupType": "AIGC",
                    "GroupIds": ["group-active", "group-disabled"],
                    "Statuses": ["Active"]
                },
                "PageNumber": 1,
                "PageSize": 10
            }),
        )
        .await
        .expect("group list");
        assert_eq!(groups["Result"]["TotalCount"], 1);
        assert_eq!(groups["Result"]["PageNumber"], 1);
        assert_eq!(groups["Result"]["PageSize"], 10);
        assert_eq!(groups["Result"]["Items"][0]["Id"], "group-active");
        assert!(groups["Result"].get("Total").is_none());
        assert!(groups["Result"].get("Groups").is_none());
        assert!(groups["Result"]["Items"][0].get("GroupId").is_none());

        let assets = list_assets_native(
            &state,
            &context,
            &caller("user-1"),
            &json!({
                "Filter": {
                    "GroupType": "AIGC",
                    "GroupIds": ["group-active", "group-disabled"],
                    "Statuses": ["Active"]
                },
                "PageNumber": 1,
                "PageSize": 10
            }),
        )
        .await
        .expect("asset list");
        assert_eq!(assets["Result"]["TotalCount"], 1);
        assert_eq!(assets["Result"]["Items"][0]["Id"], "asset-active");
        assert!(assets["Result"].get("Total").is_none());
        assert!(assets["Result"].get("Assets").is_none());
        assert!(assets["Result"]["Items"][0].get("AssetId").is_none());
    }

    #[tokio::test]
    async fn k23_group_list_requires_valid_group_type_filter() {
        let state =
            state_with_asset_repository(Arc::new(InMemoryAssetLibraryRepository::default()));
        let headers = HeaderMap::new();
        let context = GatewayPublicRequestContext::from_request_parts(
            "trace-k23-group-filter",
            &http::Method::POST,
            &"/?Action=ListAssetGroups&Version=2024-01-01"
                .parse()
                .unwrap(),
            &headers,
            None,
        );
        for (body, message) in [
            (json!({"Filter": {}}), "Filter.GroupType is required"),
            (json!({"GroupType": "AIGC"}), "Filter.GroupType is required"),
            (
                json!({"Filter": {"GroupType": "Unknown"}}),
                "Filter.GroupType must be AIGC or LivenessFace",
            ),
        ] {
            let error = list_groups_native(&state, &context, &caller("user-1"), &body)
                .await
                .expect_err("invalid GroupType filter must fail");
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.message, message);
        }
    }

    #[test]
    fn validation_result_code_drives_terminal_status() {
        assert_eq!(
            validation_result_status(&json!({"GroupId": "group-upstream"}), "Pending"),
            "Succeeded"
        );
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
        assert_eq!(projection["Id"], "asset-local");
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
