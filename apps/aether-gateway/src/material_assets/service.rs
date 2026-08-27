use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};

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
use aether_usage_runtime::{
    build_sync_terminal_usage_outcome, build_terminal_usage_event_from_outcome,
};
use axum::body::{Body, Bytes};
use axum::http::{self, HeaderMap, Response, StatusCode, Uri};
use axum::response::IntoResponse;
use axum::Json;
use base64::Engine as _;
use futures_util::{stream, StreamExt};
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
const MAX_LIST_MERGE_WINDOW: usize = 10_000;
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

struct UpstreamListFetch {
    items: Vec<Value>,
    total: usize,
}

struct AssetCandidateTerminalGuard {
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    started_at_unix_ms: u64,
    started_at: Instant,
    armed: bool,
}

#[derive(Clone, Default)]
struct AssetResponseAuditCapture {
    provider_headers: std::collections::BTreeMap<String, String>,
    provider_body_json: Option<Value>,
    provider_body_base64: Option<String>,
    client_headers: std::collections::BTreeMap<String, String>,
    client_body: Option<Value>,
    telemetry: Option<aether_contracts::ExecutionTelemetry>,
}

impl AssetResponseAuditCapture {
    fn from_execution_result(result: &aether_contracts::ExecutionResult) -> Self {
        let provider_body_json = result
            .body
            .as_ref()
            .and_then(|body| body.json_body.clone())
            .or_else(|| execution_result_json(result));
        let provider_body_base64 = result
            .body
            .as_ref()
            .and_then(|body| body.body_bytes_b64.clone());
        let client_body = provider_body_json.clone().or_else(|| {
            provider_body_base64
                .as_ref()
                .map(|body| Value::String(body.clone()))
        });
        Self {
            provider_headers: result.headers.clone(),
            provider_body_json,
            provider_body_base64,
            client_headers: BTreeMap::from([(
                "content-type".to_string(),
                "application/json".to_string(),
            )]),
            client_body,
            telemetry: result.telemetry.clone(),
        }
    }

    fn with_client_body(mut self, body: Value) -> Self {
        self.client_body = Some(body);
        self
    }
}

#[derive(Clone)]
struct AssetCandidateTerminalOutcome {
    status: RequestCandidateStatus,
    status_code: u16,
    error_type: Option<String>,
    error_message: Option<String>,
    response: AssetResponseAuditCapture,
}

impl AssetCandidateTerminalOutcome {
    fn failed(
        status_code: u16,
        error_type: impl Into<String>,
        error_message: impl Into<String>,
        response: AssetResponseAuditCapture,
    ) -> Self {
        Self {
            status: RequestCandidateStatus::Failed,
            status_code,
            error_type: Some(error_type.into()),
            error_message: Some(error_message.into()),
            response,
        }
    }

    fn cancelled() -> Self {
        Self {
            status: RequestCandidateStatus::Cancelled,
            status_code: 499,
            error_type: Some("asset_request_cancelled".to_string()),
            error_message: Some(
                "Ark asset request was cancelled before terminal finalization".to_string(),
            ),
            response: AssetResponseAuditCapture::default(),
        }
    }

    fn success(status_code: u16, response: AssetResponseAuditCapture) -> Self {
        Self {
            status: RequestCandidateStatus::Success,
            status_code,
            error_type: None,
            error_message: None,
            response,
        }
    }

    fn report_kind(&self) -> &'static str {
        match self.status {
            RequestCandidateStatus::Success => "ark_asset_sync_success",
            RequestCandidateStatus::Failed => "ark_asset_sync_failed",
            RequestCandidateStatus::Cancelled => "ark_asset_sync_cancelled",
            _ => "ark_asset_sync_failed",
        }
    }
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

    async fn finish(&mut self, outcome: AssetCandidateTerminalOutcome) {
        if !self.armed {
            return;
        }
        // Transfer ownership before the first await. If the request is dropped
        // while usage or candidate persistence is in flight, the process-
        // lifetime continuation still completes the same candidate identity
        // instead of racing it with a synthetic 499.
        self.armed = false;
        let handoff = spawn_asset_candidate_terminal(
            self.state.clone(),
            self.plan.clone(),
            self.report_context.clone(),
            self.started_at_unix_ms,
            self.started_at,
            outcome,
        );
        let _ = tokio::time::timeout(Duration::from_secs(5), handoff).await;
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
        // Drop can run while the request runtime is being cancelled or shut
        // down. The usage runtime is process-lifetime and owns both halves of
        // the usage -> candidate handoff.
        spawn_asset_candidate_terminal(
            state,
            plan,
            report_context,
            started_at_unix_ms,
            started_at,
            AssetCandidateTerminalOutcome::cancelled(),
        );
    }
}

fn spawn_asset_candidate_terminal(
    state: AppState,
    plan: ExecutionPlan,
    report_context: Option<Value>,
    started_at_unix_ms: u64,
    started_at: Instant,
    outcome: AssetCandidateTerminalOutcome,
) -> tokio::task::JoinHandle<()> {
    aether_usage_runtime::spawn_on_usage_background_runtime(async move {
        record_asset_candidate_terminal(
            &state,
            &plan,
            report_context.as_ref(),
            started_at_unix_ms,
            started_at,
            outcome,
        )
        .await;
    })
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

fn asset_action_lifecycle_request_id(parent_request_id: &str, action: ArkAssetAction) -> String {
    let parent_fragment = parent_request_id
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || *character == '-')
        .take(24)
        .collect::<String>();
    format!(
        "asset-{}-{}-{}",
        if parent_fragment.is_empty() {
            "request"
        } else {
            parent_fragment.as_str()
        },
        action.as_str().to_ascii_lowercase(),
        uuid::Uuid::new_v4().simple()
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
                Err(error) => {
                    return Some(native_error_response_with_context(
                        error,
                        request_context,
                        None,
                    ))
                }
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
                        Err(error) => {
                            native_error_response_with_context(error, request_context, Some(action))
                        }
                    }
                }
                Err(error) => native_error_response_with_context(error, request_context, None),
            }
        }
        Err(error) => native_error_response_with_context(error, request_context, None),
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
    let _ = native_project_name(&body)?;
    match action {
        ArkAssetAction::ListAssetGroups => {
            list_groups_native(state, request_context, headers, caller, &body).await
        }
        ArkAssetAction::ListAssets => {
            list_assets_native(state, request_context, headers, caller, &body).await
        }
        ArkAssetAction::CreateAssetGroup => {
            let group = create_group(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": public_group_id(&group)}),
            ))
        }
        ArkAssetAction::GetAssetGroup => {
            let id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
            let owned = load_group_by_upstream_id(state, caller, &id).await?;
            ensure_project_matches(&body, &owned.project_name)?;
            let (_group, upstream_result) =
                refresh_group_with_result(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, action, upstream_result))
        }
        ArkAssetAction::UpdateAssetGroup => {
            let id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
            let owned = load_group_by_upstream_id(state, caller, &id).await?;
            ensure_project_matches(&body, &owned.project_name)?;
            let group = update_group(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": public_group_id(&group)}),
            ))
        }
        ArkAssetAction::DeleteAssetGroup => {
            let id = required_string_field(&body, &["Id", "GroupId", "id", "group_id"], "Id")?;
            let owned = load_group_by_upstream_id(state, caller, &id).await?;
            ensure_project_matches(&body, &owned.project_name)?;
            delete_group(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, action, json!({})))
        }
        ArkAssetAction::CreateAsset => {
            let group_id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
            let owned = load_group_by_upstream_id(state, caller, &group_id).await?;
            ensure_project_matches(&body, &owned.project_name)?;
            let asset = create_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": public_asset_id(&asset)}),
            ))
        }
        ArkAssetAction::GetAsset => {
            let id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
            let owned = load_asset_by_upstream_id(state, caller, &id).await?;
            let owned_group = load_group(state, caller, &owned.group_id, false).await?;
            ensure_project_matches(&body, &owned_group.project_name)?;
            let (_asset, upstream_result) =
                refresh_asset_with_result(state, request_context, headers, caller, &id).await?;
            Ok(native_envelope(request_context, action, upstream_result))
        }
        ArkAssetAction::UpdateAsset => {
            let id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
            let owned = load_asset_by_upstream_id(state, caller, &id).await?;
            let owned_group = load_group(state, caller, &owned.group_id, false).await?;
            ensure_project_matches(&body, &owned_group.project_name)?;
            let asset = update_asset(state, request_context, headers, caller, body).await?;
            Ok(native_envelope(
                request_context,
                action,
                json!({"Id": public_asset_id(&asset)}),
            ))
        }
        ArkAssetAction::DeleteAsset => {
            let id = required_string_field(&body, &["Id", "AssetId", "id", "asset_id"], "Id")?;
            let owned = load_asset_by_upstream_id(state, caller, &id).await?;
            let owned_group = load_group(state, caller, &owned.group_id, false).await?;
            ensure_project_matches(&body, &owned_group.project_name)?;
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
            canonical_visual_success(request_context, action, upstream)
        }
        ArkAssetAction::GetVisualValidateResult => {
            let upstream =
                get_validation_result_native(state, request_context, headers, caller, &body)
                    .await?;
            canonical_visual_success(request_context, action, upstream)
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
            let page = list_groups_rest(state, request_context, headers, &caller, is_admin).await?;
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
                "ProjectName": native_project_name(&body)?,
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
            validate_official_request_id(&id, "group-", "Id")?;
            let group = refresh_group(state, request_context, headers, &caller, &id).await?;
            let count = group_asset_count(state, &group).await?;
            Ok(json_response(
                StatusCode::OK,
                group_rest_json(&group, count),
            ))
        }
        "update_group" => {
            let id = path_resource_id(&request_context.request_path, "groups")?;
            validate_official_request_id(&id, "group-", "Id")?;
            let current = load_group(state, &caller, &id, is_admin).await?;
            let mut upstream_body = json!({
                "Id": id,
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
                "ProjectName": current.project_name,
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
            validate_official_request_id(&id, "group-", "Id")?;
            delete_group(state, request_context, headers, &caller, &id).await?;
            Ok(empty_response(StatusCode::NO_CONTENT))
        }
        "list_assets" => {
            let page = list_assets_rest(state, request_context, headers, &caller, is_admin).await?;
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
            let group_id = required_string_field(&body, &["group_id", "GroupId"], "GroupId")?;
            validate_official_request_id(&group_id, "group-", "GroupId")?;
            let group = load_group(state, &caller, &group_id, is_admin).await?;
            let asset_type = required_asset_type(&body)?;
            let upstream_body = json!({
                "GroupId": group_id,
                "URL": source_url,
                "AssetType": asset_type,
                "Name": string_field(&body, &["name", "Name"]),
                "ProjectName": group.project_name,
            });
            let asset =
                create_asset(state, request_context, headers, &caller, upstream_body).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin).await?;
            Ok(json_response(
                StatusCode::CREATED,
                asset_rest_json(&asset, &group, is_admin),
            ))
        }
        "upload_asset" => Err(AssetServiceError::new(
            StatusCode::NOT_IMPLEMENTED,
            "UnsupportedOperation",
            "Ark 素材库仅支持 URL 创建；本地文件上传需要先接入对象存储",
        )),
        "get_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            validate_official_request_id(&id, "asset-", "Id")?;
            let asset = refresh_asset(state, request_context, headers, &caller, &id).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin).await?;
            Ok(json_response(
                StatusCode::OK,
                asset_rest_json(&asset, &group, is_admin),
            ))
        }
        "update_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            validate_official_request_id(&id, "asset-", "Id")?;
            let current = load_asset(state, &caller, &id, is_admin).await?;
            let current_group = load_group(state, &caller, &current.group_id, is_admin).await?;
            let upstream_body = json!({
                "Id": id,
                "Name": required_string_field(&body, &["name", "Name"], "name")?,
                "ProjectName": current_group.project_name,
            });
            let asset =
                update_asset(state, request_context, headers, &caller, upstream_body).await?;
            let group = load_group(state, &caller, &asset.group_id, is_admin).await?;
            Ok(json_response(
                StatusCode::OK,
                asset_rest_json(&asset, &group, is_admin),
            ))
        }
        "delete_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            validate_official_request_id(&id, "asset-", "Id")?;
            delete_asset(state, request_context, headers, &caller, &id).await?;
            Ok(empty_response(StatusCode::NO_CONTENT))
        }
        "preview_asset" => {
            let id = path_resource_id(&request_context.request_path, "assets")?;
            validate_official_request_id(&id, "asset-", "Id")?;
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
                "CallbackURL",
            )?;
            let upstream_body = json!({
                "CallbackURL": callback_url,
                "ProjectName": native_project_name(&body)?,
            });
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
    let lifecycle_request_id =
        asset_action_lifecycle_request_id(request_context.trace_id.as_str(), action);
    let mut plan = ExecutionPlan {
        request_id: lifecycle_request_id.clone(),
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
        model_name: Some(ASSET_ROUTING_MODEL.to_string()),
        proxy,
        transport_profile: resolve_transport_profile(&transport.snapshot),
        timeouts: resolve_transport_execution_timeouts(&transport.snapshot),
    };
    let started_at = Instant::now();
    let started_at_unix_ms = crate::clock::current_unix_ms();
    let mut report_context = Some(json!({
        "request_id": lifecycle_request_id,
        "parent_request_id": request_context.trace_id,
        "client_request_id": request_context.trace_id,
        "client_trace_id": request_context.trace_id,
        "user_id": caller.user_id,
        "api_key_id": caller.api_key_id,
        "client_api_format": ARK_ASSET_API_FORMAT,
        "provider_api_format": ARK_ASSET_API_FORMAT,
        "model": ASSET_ROUTING_MODEL,
        "mapped_model": ASSET_ROUTING_MODEL,
        "request_path": request_context.request_path,
        "request_query_string": request_context.request_query_string,
        "asset_action": action.as_str(),
        "usage_scope": "asset_upstream_action",
        "client_capture_scope": "asset_action_projection",
        "original_headers": crate::headers::collect_control_headers(headers),
        "original_request_body": body,
        "provider_request_headers": &plan.headers,
        "provider_request_body": provider_body,
    }));
    let mut terminal_guard = begin_asset_candidate_attempt(
        state,
        &mut plan,
        &mut report_context,
        started_at_unix_ms,
        started_at,
    );
    // Seed the audit lifecycle immediately after the guard owns the exact
    // child request/candidate identity. Ordered terminal handoff for that same
    // child id cannot overtake this pending write, even if the HTTP task drops.
    state.usage_runtime.record_pending(
        state.usage_lifecycle_data_state().as_ref(),
        aether_usage_runtime::build_lifecycle_usage_seed(&plan, report_context.as_ref()),
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
                    .finish(AssetCandidateTerminalOutcome::failed(
                        StatusCode::TOO_MANY_REQUESTS.as_u16(),
                        "gateway_admission_failed",
                        error.into_message(),
                        AssetResponseAuditCapture::default(),
                    ))
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
            Some(plan.request_id.as_str()),
            &plan,
            report_context.as_ref(),
        )
        .await
        {
            Ok(result) => result,
            Err(error) => {
                terminal_guard
                    .finish(AssetCandidateTerminalOutcome::failed(
                        StatusCode::BAD_GATEWAY.as_u16(),
                        "execution_runtime_unavailable",
                        error.into_message(),
                        AssetResponseAuditCapture::default(),
                    ))
                    .await;
                return Err(AssetServiceError::unavailable(
                    "Ark 素材库上游请求暂时不可用",
                ));
            }
        };
    let response_audit = AssetResponseAuditCapture::from_execution_result(&result);
    let mut status = StatusCode::from_u16(result.status_code).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut body = execution_result_json(&result).unwrap_or(Value::Null);
    if visual_validation_result_is_pending(action, &body) {
        status = StatusCode::OK;
        body = native_envelope(request_context, action, json!({"Status": "Pending"}));
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
            .finish(AssetCandidateTerminalOutcome::failed(
                status.as_u16(),
                error_type,
                error_message,
                response_audit.clone().with_client_body(body.clone()),
            ))
            .await;
        return Err(AssetServiceError::provider(status, body));
    }
    if provider_error_value(&body).is_some() {
        let status = super::protocol_api::response_status_from_body(&body)
            .and_then(|status| StatusCode::from_u16(status).ok())
            .filter(|status| !status.is_success())
            .unwrap_or(StatusCode::BAD_GATEWAY);
        terminal_guard
            .finish(AssetCandidateTerminalOutcome::failed(
                status.as_u16(),
                "upstream_protocol_error",
                "Ark 素材库上游返回协议错误",
                response_audit.clone().with_client_body(body.clone()),
            ))
            .await;
        return Err(AssetServiceError::provider(status, body));
    }
    terminal_guard
        .finish(AssetCandidateTerminalOutcome::success(
            status.as_u16(),
            response_audit.with_client_body(body.clone()),
        ))
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
    outcome: AssetCandidateTerminalOutcome,
) {
    let latency_ms = started_at.elapsed().as_millis() as u64;
    let mut telemetry =
        outcome
            .response
            .telemetry
            .clone()
            .unwrap_or(aether_contracts::ExecutionTelemetry {
                ttfb_ms: None,
                elapsed_ms: None,
                upstream_bytes: None,
            });
    telemetry.elapsed_ms = telemetry.elapsed_ms.or(Some(latency_ms));
    let mut terminal_report_context = report_context.cloned().unwrap_or_else(|| json!({}));
    if !terminal_report_context.is_object() {
        terminal_report_context = json!({"report_context": terminal_report_context});
    }
    if let Some(context) = terminal_report_context.as_object_mut() {
        context.insert(
            "provider_response_headers".to_string(),
            serde_json::to_value(&outcome.response.provider_headers).unwrap_or_else(|_| json!({})),
        );
        // Always insert this field, including an empty object. That explicitly
        // represents no client response observation and prevents the sync
        // builder from falling back to (and mislabelling) provider headers.
        context.insert(
            "client_response_headers".to_string(),
            serde_json::to_value(&outcome.response.client_headers).unwrap_or_else(|_| json!({})),
        );
        context.insert(
            "provider_response_capture_state".to_string(),
            Value::String(
                if outcome.response.provider_body_json.is_some()
                    || outcome.response.provider_body_base64.is_some()
                {
                    "inline"
                } else {
                    "none"
                }
                .to_string(),
            ),
        );
        context.insert(
            "client_response_capture_state".to_string(),
            Value::String(
                if outcome.response.client_body.is_some() {
                    "inline"
                } else {
                    "none"
                }
                .to_string(),
            ),
        );
    }
    let payload = crate::usage::GatewaySyncReportRequest {
        trace_id: plan.request_id.clone(),
        report_kind: outcome.report_kind().to_string(),
        report_context: Some(terminal_report_context.clone()),
        status_code: outcome.status_code,
        headers: outcome.response.provider_headers.clone(),
        body_json: outcome.response.provider_body_json.clone(),
        client_body_json: outcome.response.client_body.clone(),
        body_base64: outcome.response.provider_body_base64.clone(),
        telemetry: Some(telemetry),
    };

    let usage_persisted = if !state.usage_runtime.is_enabled() {
        true
    } else {
        let mut usage_outcome =
            build_sync_terminal_usage_outcome(plan, Some(&terminal_report_context), &payload);
        // Empty objects in the report context are intentional sentinels: they
        // stop the generic sync builder from falling back to provider headers
        // for the client-facing capture.  They do not mean that either side's
        // response headers were actually observed (notably on Drop/499), so
        // restore the storage representation to `None` after that fallback has
        // been suppressed.
        if outcome.response.provider_headers.is_empty() {
            usage_outcome.provider_response_headers = None;
        }
        if outcome.response.client_headers.is_empty() {
            usage_outcome.client_response_headers = None;
        }
        usage_outcome.request_type = "asset_library".to_string();
        usage_outcome.terminal_error_message = outcome.error_message.clone();
        usage_outcome.terminal_failure_category = outcome.error_type.clone();
        usage_outcome.billing_treat_as_completed = false;
        match build_terminal_usage_event_from_outcome(usage_outcome) {
            Ok(mut event) => {
                // Asset-library actions are audit lifecycle children, not AI
                // inference units. Even a verified upstream 2xx is explicitly
                // zero-cost/void and can never debit the parent request once
                // per page or per internal action.
                event.data.billing_treat_as_void = Some(true);
                state
                    .usage_runtime
                    .record_terminal_event_direct_with_handoff(
                        state.usage_lifecycle_data_state().as_ref(),
                        event,
                    )
                    .await
            }
            Err(error) => {
                tracing::warn!(
                    event_name = "asset_usage_terminal_event_build_failed",
                    log_type = "event",
                    request_id = %plan.request_id,
                    candidate_id = ?plan.candidate_id,
                    error = %error,
                    "gateway could not build the Ark asset terminal usage event"
                );
                false
            }
        }
    };

    let terminal_unix_ms = crate::clock::current_unix_ms();
    let desired_update = SchedulerRequestCandidateStatusUpdate {
        status: outcome.status,
        status_code: Some(outcome.status_code),
        error_type: outcome.error_type.clone(),
        error_message: outcome.error_message.clone(),
        latency_ms: Some(latency_ms),
        started_at_unix_ms: Some(started_at_unix_ms),
        finished_at_unix_ms: Some(terminal_unix_ms),
    };

    if !usage_persisted {
        // Never publish a terminal candidate ahead of its audit/billing row.
        // The durable usage runtime retains failed handoffs for retry; once the
        // row is visible, reconciliation publishes this exact terminal state.
        crate::request_candidate_runtime::record_local_request_candidate_status(
            state,
            plan,
            report_context,
            SchedulerRequestCandidateStatusUpdate {
                status: RequestCandidateStatus::Streaming,
                status_code: Some(outcome.status_code),
                error_type: Some("usage_terminal_handoff_unconfirmed".to_string()),
                error_message: Some(
                    "Ark asset terminal usage persistence was not confirmed".to_string(),
                ),
                latency_ms: Some(latency_ms),
                started_at_unix_ms: Some(started_at_unix_ms),
                finished_at_unix_ms: None,
            },
        )
        .await;
        crate::request_candidate_runtime::spawn_terminal_candidate_reconciliation(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            desired_update,
        );
        return;
    }

    let candidate_persisted =
        crate::request_candidate_runtime::record_local_request_candidate_status(
            state,
            plan,
            report_context,
            desired_update.clone(),
        )
        .await;
    if !candidate_persisted {
        crate::request_candidate_runtime::spawn_candidate_persistence_retry_after_usage_handoff(
            state.clone(),
            plan.clone(),
            report_context.cloned(),
            desired_update,
            None,
        );
    }
}

async fn create_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let name = required_string_field(&body, &["Name", "name"], "Name")?;
    validate_group_text_lengths(&body)?;
    let group_type = create_group_type(&body)?;
    let project_name = native_project_name(&body)?;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "ProjectName".to_string(),
            Value::String(project_name.clone()),
        );
        object.remove("project_name");
    }
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
    let upstream_group_id = upstream_required_official_id(result, &["Id"], "Id", "group-")?;
    let now = crate::clock::current_unix_secs();
    let record = UpsertAssetGroupRecord {
        id: local_id("agrp"),
        upstream_group_id: Some(upstream_group_id),
        user_id: caller.user_id.clone(),
        api_key_id: caller.api_key_id.clone(),
        provider_id: transport.snapshot.provider.id.clone(),
        endpoint_id: transport.snapshot.endpoint.id.clone(),
        key_id: transport.snapshot.key.id.clone(),
        project_name,
        group_type,
        name,
        description: string_field(&body, &["Description", "description"]),
        status: string_field(result, &["Status", "status"]).unwrap_or_else(|| "Active".to_string()),
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"]).unwrap_or(now),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
        deleted_at_unix_secs: None,
    };
    let mut last_error = None;
    for _ in 0..3 {
        match write_repo(state)?.upsert_group(record.clone()).await {
            Ok(group) => return Ok(group),
            Err(error) => last_error = Some(error),
        }
    }
    let projection_error =
        data_error(last_error.expect("three failed group upserts record an error"));
    let existing = read_repo(state)?
        .find_group_by_canonical_upstream(
            &record.provider_id,
            record
                .upstream_group_id
                .as_deref()
                .expect("validated upstream group ID"),
        )
        .await;
    if let Ok(Some(existing)) = existing {
        if existing.user_id == record.user_id
            && existing.endpoint_id == record.endpoint_id
            && existing.key_id == record.key_id
            && existing.project_name == record.project_name
            && existing.deleted_at_unix_secs.is_none()
        {
            return Ok(existing);
        }
        return Err(AssetServiceError::new(
            StatusCode::CONFLICT,
            "AssetGroupOwnershipConflict",
            "Ark 素材组 ID 已绑定到另一用户、端点、密钥或项目",
        ));
    }
    tracing::error!(
        upstream_group_id = %record.upstream_group_id.as_deref().unwrap_or("<missing>"),
        provider_id = %record.provider_id,
        reconciliation_required = true,
        binding_lookup_failed = existing.is_err(),
        "asset group projection failed after upstream create; destructive compensation was skipped"
    );
    Err(projection_error)
}

async fn create_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAsset, AssetServiceError> {
    let group_id = required_string_field(&body, &["GroupId", "group_id"], "GroupId")?;
    validate_asset_text_lengths(&body)?;
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
    ensure_project_matches(&body, &group.project_name)?;
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
        let asset_type_aliases = object
            .keys()
            .filter(|name| {
                name.as_str() != "AssetType"
                    && (name.eq_ignore_ascii_case("AssetType") || name.eq_ignore_ascii_case("type"))
            })
            .cloned()
            .collect::<Vec<_>>();
        for alias in asset_type_aliases {
            object.remove(&alias);
        }
        object.insert("AssetType".to_string(), Value::String(asset_type.clone()));
        object.insert(
            "ProjectName".to_string(),
            Value::String(group.project_name.clone()),
        );
        object.remove("project_name");
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
    let upstream_asset_id = upstream_required_official_id(result, &["Id"], "Id", "asset-")?;
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
    let mut last_error = None;
    for _ in 0..3 {
        match write_repo(state)?.upsert_asset(record.clone()).await {
            Ok(asset) => return Ok(asset),
            Err(error) => last_error = Some(error),
        }
    }
    let projection_error =
        data_error(last_error.expect("three failed asset upserts record an error"));
    let existing = read_repo(state)?
        .find_asset_by_upstream(
            &record.group_id,
            record
                .upstream_asset_id
                .as_deref()
                .expect("validated upstream asset ID"),
        )
        .await;
    if let Ok(Some(existing)) = existing {
        if existing.user_id == record.user_id && !existing.is_deleted {
            return Ok(existing);
        }
        return Err(AssetServiceError::new(
            StatusCode::CONFLICT,
            "AssetOwnershipConflict",
            "Ark 素材 ID 已绑定到另一资源或已删除资源",
        ));
    }
    tracing::error!(
        upstream_asset_id = %record.upstream_asset_id.as_deref().unwrap_or("<missing>"),
        group_id = %record.group_id,
        reconciliation_required = true,
        binding_lookup_failed = existing.is_err(),
        "asset projection failed after upstream create; destructive compensation was skipped"
    );
    Err(projection_error)
}

async fn refresh_group(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    group_id: &str,
) -> Result<StoredAssetGroup, AssetServiceError> {
    refresh_group_with_result(state, request_context, headers, caller, group_id)
        .await
        .map(|(group, _)| group)
}

async fn refresh_group_with_result(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    group_id: &str,
) -> Result<(StoredAssetGroup, Value), AssetServiceError> {
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
        &json!({"Id": upstream_id, "ProjectName": group.project_name}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let mut upstream_result = result.clone();
    normalize_upstream_project_name(&mut upstream_result, &group.project_name)?;
    validate_upstream_identity(&upstream_result, "Id", upstream_id)?;
    validate_upstream_group_resource(&upstream_result)?;
    let group = persist_group_projection(state, group, &upstream_result).await?;
    Ok((group, upstream_result))
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
    ensure_project_matches(&body, &group.project_name)?;
    if let Some(object) = body.as_object_mut() {
        object.insert("Id".to_string(), Value::String(upstream_id.to_string()));
        object.remove("GroupId");
        object.remove("group_id");
        object.remove("id");
        object.insert(
            "ProjectName".to_string(),
            Value::String(group.project_name.clone()),
        );
        object.remove("project_name");
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
    validate_upstream_identity(result, "Id", upstream_id)?;
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
    let upstream_delete = execute_action(
        state,
        request_context,
        headers,
        caller,
        &transport,
        ArkAssetAction::DeleteAssetGroup,
        &json!({"Id": upstream_id, "ProjectName": group.project_name}),
    )
    .await;
    if let Err(error) = upstream_delete {
        if !is_upstream_resource_not_found(&error) {
            return Err(error);
        }
    }
    soft_delete_group_with_retry(state, &group.id, crate::clock::current_unix_secs()).await
}

async fn refresh_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    asset_id: &str,
) -> Result<StoredAsset, AssetServiceError> {
    refresh_asset_with_result(state, request_context, headers, caller, asset_id)
        .await
        .map(|(asset, _)| asset)
}

async fn refresh_asset_with_result(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    asset_id: &str,
) -> Result<(StoredAsset, Value), AssetServiceError> {
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
        &json!({"Id": upstream_id, "ProjectName": group.project_name}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let mut upstream_result = result.clone();
    normalize_upstream_project_name(&mut upstream_result, &group.project_name)?;
    validate_upstream_identity(&upstream_result, "Id", upstream_id)?;
    validate_upstream_identity(&upstream_result, "GroupId", public_group_id(&group))?;
    validate_upstream_asset_resource(&upstream_result)?;
    let asset = persist_asset_projection(state, asset, &upstream_result).await?;
    Ok((asset, upstream_result))
}

async fn update_asset(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    mut body: Value,
) -> Result<StoredAsset, AssetServiceError> {
    validate_asset_text_lengths(&body)?;
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
    ensure_project_matches(&body, &group.project_name)?;
    if let Some(object) = body.as_object_mut() {
        object.insert("Id".to_string(), Value::String(upstream_id.to_string()));
        object.remove("AssetId");
        object.remove("asset_id");
        object.remove("id");
        object.insert(
            "ProjectName".to_string(),
            Value::String(group.project_name.clone()),
        );
        object.remove("project_name");
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
    validate_upstream_identity(result, "Id", upstream_id)?;
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
    let upstream_delete = execute_action(
        state,
        request_context,
        headers,
        caller,
        &transport,
        ArkAssetAction::DeleteAsset,
        &json!({"Id": upstream_id, "ProjectName": group.project_name}),
    )
    .await;
    if let Err(error) = upstream_delete {
        if !is_upstream_resource_not_found(&error) {
            return Err(error);
        }
    }
    soft_delete_asset_with_retry(state, &asset.id, crate::clock::current_unix_secs()).await
}

fn is_upstream_resource_not_found(error: &AssetServiceError) -> bool {
    let code = error.code.trim().to_ascii_lowercase();
    code == "resourcenotfound"
        || code.starts_with("resourcenotfound.")
        || code == "notfound"
        || code.starts_with("notfound.")
        || code == "assetnotfound"
        || code == "assetgroupnotfound"
}

async fn soft_delete_group_with_retry(
    state: &AppState,
    group_id: &str,
    deleted_at_unix_secs: u64,
) -> Result<(), AssetServiceError> {
    let mut last_error = None;
    for _ in 0..3 {
        match write_repo(state)?
            .soft_delete_group(group_id, deleted_at_unix_secs)
            .await
        {
            Ok(true) => return Ok(()),
            Ok(false) => {
                let already_deleted = read_repo(state)?
                    .find_group_by_id(group_id)
                    .await
                    .map_err(data_error)?
                    .is_some_and(|group| group.deleted_at_unix_secs.is_some());
                if already_deleted {
                    return Ok(());
                }
            }
            Err(error) => last_error = Some(error),
        }
    }
    if let Some(error) = last_error {
        return Err(data_error(error));
    }
    Err(AssetServiceError::not_found())
}

async fn soft_delete_asset_with_retry(
    state: &AppState,
    asset_id: &str,
    deleted_at_unix_secs: u64,
) -> Result<(), AssetServiceError> {
    let mut last_error = None;
    for _ in 0..3 {
        match write_repo(state)?
            .soft_delete_asset(asset_id, deleted_at_unix_secs)
            .await
        {
            Ok(true) => return Ok(()),
            Ok(false) => {
                let already_deleted = read_repo(state)?
                    .find_asset_by_id(asset_id)
                    .await
                    .map_err(data_error)?
                    .is_some_and(|asset| asset.is_deleted);
                if already_deleted {
                    return Ok(());
                }
            }
            Err(error) => last_error = Some(error),
        }
    }
    if let Some(error) = last_error {
        return Err(data_error(error));
    }
    Err(AssetServiceError::not_found())
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
        project_name: group.project_name,
        group_type: string_field(result, &["GroupType", "Type"]).unwrap_or(group.group_type),
        name: string_field(result, &["Name", "name"]).unwrap_or(group.name),
        description: string_field(result, &["Description", "description"]).or(group.description),
        status: string_field(result, &["Status", "status"]).unwrap_or(group.status),
        created_at_unix_secs: timestamp_field(result, &["CreateTime", "CreatedAt"])
            .unwrap_or(group.created_at_unix_secs),
        updated_at_unix_secs: timestamp_field(result, &["UpdateTime", "UpdatedAt"]).unwrap_or(now),
        deleted_at_unix_secs: group.deleted_at_unix_secs,
    };
    let projected = record.clone().into_stored();
    match write_repo(state)?.upsert_group(record).await {
        Ok(stored) => Ok(stored),
        Err(error) => {
            tracing::error!(
                group_id = %projected.id,
                upstream_group_id = ?projected.upstream_group_id,
                error = %error,
                "official Ark group response succeeded but cache projection needs retry"
            );
            Ok(projected)
        }
    }
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
    let projected = record.clone().into_stored();
    let mut persisted = match write_repo(state)?.upsert_asset(record).await {
        Ok(stored) => stored,
        Err(error) => {
            tracing::error!(
                asset_id = %projected.id,
                upstream_asset_id = ?projected.upstream_asset_id,
                error = %error,
                "official Ark asset response succeeded but cache projection needs retry"
            );
            projected
        }
    };
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
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let page = required_native_page_number(body)?;
    let page_size = required_native_page_size(body)?;
    let filter = required_object_field(body, &["Filter", "filter"], "Filter")?;
    reject_unknown_fields(
        body,
        &[
            "Filter",
            "PageNumber",
            "PageSize",
            "SortBy",
            "SortOrder",
            "ProjectName",
        ],
        "ListAssetGroups",
    )?;
    reject_unknown_fields(filter, &["GroupIds", "GroupType", "Name"], "Filter")?;
    sort_upstream_items(&mut [], body, false)?;
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
    let requested_ids =
        optional_string_array_field(filter, &["GroupIds", "group_ids"], "Filter.GroupIds")?;
    validate_requested_group_ids(&requested_ids)?;
    let name = optional_native_string(filter, &["Name", "name"], "Filter.Name")?;
    if name.as_ref().is_some_and(|name| name.chars().count() > 64) {
        return Err(AssetServiceError::bad_request(
            "Filter.Name must be at most 64 characters",
        ));
    }
    let project_name = native_project_name(body)?;
    let mut groups = policy_visible_bound_groups(state, caller).await?;
    groups.retain(|group| {
        group.project_name == project_name
            && group.group_type.eq_ignore_ascii_case(&group_type)
            && (requested_ids.is_empty()
                || requested_ids
                    .iter()
                    .any(|id| group.upstream_group_id.as_deref() == Some(id.as_str())))
    });
    let mut partitions = BTreeMap::<(String, String, String), Vec<StoredAssetGroup>>::new();
    for group in groups {
        partitions
            .entry((
                group.provider_id.clone(),
                group.endpoint_id.clone(),
                group.key_id.clone(),
            ))
            .or_default()
            .push(group);
    }
    let source_count = partitions
        .values()
        .map(|groups| groups.len().div_ceil(ARK_MAX_PAGE_SIZE))
        .sum::<usize>();
    let direct_page = source_count <= 1;
    let merge_window = page
        .checked_mul(page_size)
        .ok_or_else(|| AssetServiceError::bad_request("PageNumber is too large"))?;
    if !direct_page && merge_window > MAX_LIST_MERGE_WINDOW {
        return Err(AssetServiceError::bad_request(format!(
            "multi-provider pagination window must not exceed {MAX_LIST_MERGE_WINDOW} items"
        )));
    }
    let fetch_start_page = if direct_page { page } else { 1 };
    let fetch_limit = if direct_page { page_size } else { merge_window };
    let (sort_by, sort_order) = upstream_sort_options(body, false)?;
    let mut reported_total = 0usize;
    let mut items_by_id = BTreeMap::<String, Value>::new();
    for groups in partitions.into_values() {
        let transport = exact_transport_for_group(state, caller, &groups[0]).await?;
        let owned = groups
            .iter()
            .filter_map(|group| {
                group
                    .upstream_group_id
                    .as_ref()
                    .map(|id| (id.clone(), group))
            })
            .collect::<HashMap<_, _>>();
        let ids = owned.keys().cloned().collect::<Vec<_>>();
        for chunk in ids.chunks(ARK_MAX_PAGE_SIZE) {
            let mut filter = json!({
                "GroupIds": chunk,
                "GroupType": group_type,
            });
            if let Some(name) = name.as_ref() {
                filter
                    .as_object_mut()
                    .expect("list group filter is an object")
                    .insert("Name".to_string(), Value::String(name.clone()));
            }
            let fetched = fetch_upstream_list_items(
                state,
                request_context,
                headers,
                caller,
                &transport,
                ArkAssetAction::ListAssetGroups,
                filter,
                &project_name,
                fetch_start_page,
                fetch_limit,
                &sort_by,
                &sort_order,
            )
            .await?;
            reported_total = reported_total.checked_add(fetched.total).ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    "Ark ListAssetGroups TotalCount overflowed",
                )
            })?;
            for item in fetched.items {
                let id = upstream_required_official_id(&item, &["Id"], "Items.Id", "group-")?;
                let Some(local) = owned.get(&id) else {
                    return Err(AssetServiceError::new(
                        StatusCode::BAD_GATEWAY,
                        "InvalidUpstreamResponse",
                        "Ark ListAssetGroups returned an unowned GroupId",
                    ));
                };
                validate_upstream_group_resource(&item)?;
                let item_group_type =
                    upstream_required_string(&item, &["GroupType"], "Items.GroupType")?;
                if !item_group_type.eq_ignore_ascii_case(&group_type) {
                    return Err(AssetServiceError::new(
                        StatusCode::BAD_GATEWAY,
                        "InvalidUpstreamResponse",
                        "Ark ListAssetGroups returned an unexpected GroupType",
                    ));
                }
                let _ = persist_group_projection(state, (*local).clone(), &item).await?;
                if items_by_id.insert(id, item).is_some() {
                    return Err(AssetServiceError::new(
                        StatusCode::CONFLICT,
                        "UpstreamIdentityConflict",
                        "多个素材库来源返回了相同的官方素材组 ID",
                    ));
                }
            }
        }
    }
    let mut items = items_by_id.into_values().collect::<Vec<_>>();
    sort_upstream_items(&mut items, body, false)?;
    let total = reported_total;
    let offset = if direct_page {
        0
    } else {
        (page - 1).saturating_mul(page_size)
    };
    let items = items
        .into_iter()
        .skip(offset)
        .take(page_size)
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
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: &Value,
) -> Result<Value, AssetServiceError> {
    let page = optional_native_page_number(body)?;
    let page_size = optional_native_page_size(body)?;
    reject_unknown_fields(
        body,
        &[
            "Filter",
            "PageNumber",
            "PageSize",
            "SortBy",
            "SortOrder",
            "ProjectName",
        ],
        "ListAssets",
    )?;
    let filter =
        optional_object_field(body, &["Filter", "filter"], "Filter")?.unwrap_or(&Value::Null);
    if !filter.is_null() {
        reject_unknown_fields(
            filter,
            &["GroupIds", "GroupType", "Name", "Statuses"],
            "Filter",
        )?;
    }
    sort_upstream_items(&mut [], body, true)?;
    if value_field(filter, &["AssetType", "asset_type", "Type"]).is_some() {
        return Err(AssetServiceError::bad_request(
            "Filter.AssetType is not part of the official ListAssets request",
        ));
    }
    let requested_ids =
        optional_string_array_field(filter, &["GroupIds", "group_ids"], "Filter.GroupIds")?;
    validate_requested_group_ids(&requested_ids)?;
    let statuses =
        optional_string_array_field(filter, &["Statuses", "statuses"], "Filter.Statuses")?;
    if statuses
        .iter()
        .any(|status| !matches!(status.as_str(), "Active" | "Processing" | "Failed"))
    {
        return Err(AssetServiceError::bad_request(
            "Filter.Statuses values must be Active, Processing, or Failed",
        ));
    }
    let group_type =
        optional_native_string(filter, &["GroupType", "group_type"], "Filter.GroupType")?;
    if group_type
        .as_ref()
        .is_some_and(|value| !matches!(value.as_str(), "AIGC" | "LivenessFace"))
    {
        return Err(AssetServiceError::bad_request(
            "Filter.GroupType must be AIGC or LivenessFace",
        ));
    }
    let name = optional_native_string(filter, &["Name", "name"], "Filter.Name")?;
    if name.as_ref().is_some_and(|name| name.chars().count() > 64) {
        return Err(AssetServiceError::bad_request(
            "Filter.Name must be at most 64 characters",
        ));
    }
    let project_name = native_project_name(body)?;
    let mut groups = policy_visible_bound_groups(state, caller).await?;
    groups.retain(|group| {
        group.project_name == project_name
            && group_type
                .as_ref()
                .is_none_or(|expected| group.group_type.eq_ignore_ascii_case(expected))
            && (requested_ids.is_empty()
                || requested_ids
                    .iter()
                    .any(|id| group.upstream_group_id.as_deref() == Some(id.as_str())))
    });
    let mut partitions = BTreeMap::<(String, String, String, String), Vec<StoredAssetGroup>>::new();
    for group in groups {
        partitions
            .entry((
                group.provider_id.clone(),
                group.endpoint_id.clone(),
                group.key_id.clone(),
                group.group_type.clone(),
            ))
            .or_default()
            .push(group);
    }
    let source_count = partitions
        .values()
        .map(|groups| groups.len().div_ceil(ARK_MAX_PAGE_SIZE))
        .sum::<usize>();
    let direct_page = source_count <= 1;
    let merge_window = page
        .checked_mul(page_size)
        .ok_or_else(|| AssetServiceError::bad_request("PageNumber is too large"))?;
    if !direct_page && merge_window > MAX_LIST_MERGE_WINDOW {
        return Err(AssetServiceError::bad_request(format!(
            "multi-provider pagination window must not exceed {MAX_LIST_MERGE_WINDOW} items"
        )));
    }
    let fetch_start_page = if direct_page { page } else { 1 };
    let fetch_limit = if direct_page { page_size } else { merge_window };
    let (sort_by, sort_order) = upstream_sort_options(body, true)?;
    let mut reported_total = 0usize;
    let mut items_by_id = BTreeMap::<String, Value>::new();
    for groups in partitions.into_values() {
        let transport = exact_transport_for_group(state, caller, &groups[0]).await?;
        let owned = groups
            .iter()
            .filter_map(|group| {
                group
                    .upstream_group_id
                    .as_ref()
                    .map(|id| (id.clone(), group))
            })
            .collect::<HashMap<_, _>>();
        let ids = owned.keys().cloned().collect::<Vec<_>>();
        for chunk in ids.chunks(ARK_MAX_PAGE_SIZE) {
            let mut upstream_filter = json!({
                "GroupIds": chunk,
                "GroupType": groups[0].group_type,
            });
            if let Some(name) = name.as_ref() {
                upstream_filter
                    .as_object_mut()
                    .expect("list asset filter is an object")
                    .insert("Name".to_string(), Value::String(name.clone()));
            }
            if !statuses.is_empty() {
                upstream_filter
                    .as_object_mut()
                    .expect("list asset filter is an object")
                    .insert("Statuses".to_string(), json!(statuses));
            }
            let fetched = fetch_upstream_list_items(
                state,
                request_context,
                headers,
                caller,
                &transport,
                ArkAssetAction::ListAssets,
                upstream_filter,
                &project_name,
                fetch_start_page,
                fetch_limit,
                &sort_by,
                &sort_order,
            )
            .await?;
            reported_total = reported_total.checked_add(fetched.total).ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    "Ark ListAssets TotalCount overflowed",
                )
            })?;
            for item in fetched.items {
                let id = upstream_required_official_id(&item, &["Id"], "Items.Id", "asset-")?;
                let group_id =
                    upstream_required_official_id(&item, &["GroupId"], "Items.GroupId", "group-")?;
                let Some(group) = owned.get(&group_id) else {
                    return Err(AssetServiceError::new(
                        StatusCode::BAD_GATEWAY,
                        "InvalidUpstreamResponse",
                        "Ark ListAssets returned an asset from an unowned group",
                    ));
                };
                validate_upstream_asset_resource(&item)?;
                let _ = upsert_asset_projection_from_list(state, group, &item).await?;
                if items_by_id.insert(id, item).is_some() {
                    return Err(AssetServiceError::new(
                        StatusCode::CONFLICT,
                        "UpstreamIdentityConflict",
                        "多个素材库来源返回了相同的官方素材 ID",
                    ));
                }
            }
        }
    }
    let mut items = items_by_id.into_values().collect::<Vec<_>>();
    sort_upstream_items(&mut items, body, true)?;
    let total = reported_total;
    let offset = if direct_page {
        0
    } else {
        (page - 1).saturating_mul(page_size)
    };
    let items = items
        .into_iter()
        .skip(offset)
        .take(page_size)
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

#[allow(clippy::too_many_arguments)]
async fn fetch_upstream_list_items(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    transport: &AssetTransport,
    action: ArkAssetAction,
    filter: Value,
    project_name: &str,
    start_page: usize,
    max_items: usize,
    sort_by: &str,
    sort_order: &str,
) -> Result<UpstreamListFetch, AssetServiceError> {
    if start_page == 0 || max_items == 0 {
        return Err(AssetServiceError::bad_request(
            "upstream list pagination must be positive",
        ));
    }
    let request_page_size = max_items.min(ARK_MAX_PAGE_SIZE);
    let pages_to_fetch = max_items.div_ceil(request_page_size);
    let mut items = Vec::new();
    let mut reported_total = None;
    for page_offset in 0..pages_to_fetch {
        let page_number = start_page
            .checked_add(page_offset)
            .ok_or_else(|| AssetServiceError::bad_request("PageNumber is too large"))?;
        let response = execute_action(
            state,
            request_context,
            headers,
            caller,
            transport,
            action,
            &json!({
                "Filter": filter,
                "PageNumber": page_number,
                "PageSize": request_page_size,
                "SortBy": sort_by,
                "SortOrder": sort_order,
                "ProjectName": project_name,
            }),
        )
        .await?;
        let result = extract_result(&response.body).ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!("Ark {} response is missing Result", action.as_str()),
            )
        })?;
        let total = upstream_required_usize(result, "TotalCount")?;
        if reported_total
            .replace(total)
            .is_some_and(|known| known != total)
        {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!(
                    "Ark {} changed TotalCount during pagination",
                    action.as_str()
                ),
            ));
        }
        let upstream_page = upstream_required_usize(result, "PageNumber")?;
        let upstream_page_size = upstream_required_usize(result, "PageSize")?;
        if upstream_page != page_number || upstream_page_size != request_page_size {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!(
                    "Ark {} returned invalid pagination metadata",
                    action.as_str()
                ),
            ));
        }
        let page_items = result
            .get("Items")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    format!("Ark {} response is missing Items", action.as_str()),
                )
            })?;
        if page_items.len() > request_page_size {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!("Ark {} returned too many Items", action.as_str()),
            ));
        }
        let upstream_offset = page_number
            .saturating_sub(1)
            .checked_mul(request_page_size)
            .ok_or_else(|| AssetServiceError::bad_request("PageNumber is too large"))?;
        if page_items.is_empty() {
            if upstream_offset >= total {
                return Ok(UpstreamListFetch { items, total });
            }
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!("Ark {} pagination ended before TotalCount", action.as_str()),
            ));
        }
        if upstream_offset >= total
            || upstream_offset.saturating_add(page_items.len()) > total
            || (page_items.len() < request_page_size
                && upstream_offset.saturating_add(page_items.len()) < total)
        {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!("Ark {} returned inconsistent pagination", action.as_str()),
            ));
        }
        let remaining = max_items.saturating_sub(items.len());
        for item in page_items.iter().take(remaining) {
            let mut item = item.clone();
            normalize_upstream_project_name(&mut item, project_name)?;
            items.push(item);
        }
        if upstream_offset.saturating_add(page_items.len()) >= total || items.len() >= max_items {
            return Ok(UpstreamListFetch { items, total });
        }
    }
    Ok(UpstreamListFetch {
        items,
        total: reported_total.unwrap_or_default(),
    })
}

async fn policy_visible_bound_groups(
    state: &AppState,
    caller: &AssetCaller,
) -> Result<Vec<StoredAssetGroup>, AssetServiceError> {
    let access = resolve_caller_access(state, caller, crate::clock::current_unix_secs()).await?;
    if !access_policy_allows_format(&access.policy, ARK_ASSET_API_FORMAT) {
        return Err(AssetServiceError::new(
            StatusCode::FORBIDDEN,
            "ApiFormatNotAllowed",
            "当前凭据无权访问 Ark 素材库",
        ));
    }
    let mut groups = list_all_groups(
        state,
        AssetGroupListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            ..AssetGroupListQuery::default()
        },
    )
    .await?;
    groups.retain(|group| {
        group
            .upstream_group_id
            .as_ref()
            .is_some_and(|id| id.starts_with("group-") && id.len() > "group-".len())
    });
    let provider_ids = groups
        .iter()
        .map(|group| group.provider_id.clone())
        .collect::<Vec<_>>();
    let providers = state
        .read_provider_catalog_providers_by_ids(&provider_ids)
        .await
        .map_err(gateway_error)?
        .into_iter()
        .map(|provider| (provider.id.clone(), provider))
        .collect::<HashMap<_, _>>();
    groups.retain(|group| {
        providers.get(&group.provider_id).is_some_and(|provider| {
            access_policy_allows_provider(
                &access.policy,
                &provider.id,
                &provider.name,
                &provider.provider_type,
            )
        })
    });
    Ok(groups)
}

async fn upsert_asset_projection_from_list(
    state: &AppState,
    group: &StoredAssetGroup,
    item: &Value,
) -> Result<StoredAsset, AssetServiceError> {
    let upstream_asset_id = upstream_required_official_id(item, &["Id"], "Items.Id", "asset-")?;
    let upstream_group_id =
        upstream_required_official_id(item, &["GroupId"], "Items.GroupId", "group-")?;
    if upstream_group_id != public_group_id(group) {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark ListAssets returned an unexpected GroupId",
        ));
    }
    let asset_type = normalize_upstream_asset_type(&upstream_required_string(
        item,
        &["AssetType"],
        "Items.AssetType",
    )?)?;
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
        asset_type,
        name: string_field(item, &["Name"])
            .or_else(|| existing.as_ref().map(|asset| asset.name.clone()))
            .unwrap_or_else(|| "未命名素材".to_string()),
        status: upstream_required_string(item, &["Status"], "Items.Status")?,
        error_code: error_field(item, "Code"),
        error_message: error_field(item, "Message"),
        moderation: object_field(item, &["Moderation"]),
        last_inference_at_unix_secs: timestamp_field(item, &["LastInferenceTime"]),
        source_url_fingerprint: existing
            .as_ref()
            .and_then(|asset| asset.source_url_fingerprint.clone()),
        provider_url: None,
        provider_url_expires_at_unix_secs: None,
        sanitized_metadata: sanitize_asset_metadata(item),
        is_deleted: false,
        deleted_at_unix_secs: None,
        created_at_unix_secs: timestamp_field(item, &["CreateTime"])
            .or_else(|| existing.as_ref().map(|asset| asset.created_at_unix_secs))
            .unwrap_or(now),
        updated_at_unix_secs: timestamp_field(item, &["UpdateTime"]).unwrap_or(now),
    };
    let projected = record.clone().into_stored();
    let mut persisted = match write_repo(state)?.upsert_asset(record).await {
        Ok(stored) => stored,
        Err(error) => {
            tracing::error!(
                asset_id = %projected.id,
                upstream_asset_id = ?projected.upstream_asset_id,
                error = %error,
                "official Ark list response succeeded but asset cache projection needs retry"
            );
            projected
        }
    };
    persisted.provider_url = provider_url(item);
    persisted.provider_url_expires_at_unix_secs = persisted
        .provider_url
        .as_ref()
        .map(|_| now.saturating_add(ASSET_URL_TTL_SECS));
    Ok(persisted)
}

fn upstream_required_usize(value: &Value, field: &str) -> Result<usize, AssetServiceError> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                format!("Ark response {field} must be a non-negative integer"),
            )
        })
}

fn sort_upstream_items(
    items: &mut [Value],
    body: &Value,
    allow_group_id: bool,
) -> Result<(), AssetServiceError> {
    let (sort_by, sort_order) = upstream_sort_options(body, allow_group_id)?;
    let descending = sort_order == "Desc";
    items.sort_by(|left, right| {
        let order = if sort_by == "GroupId" {
            string_field(left, &["GroupId"]).cmp(&string_field(right, &["GroupId"]))
        } else {
            timestamp_field(left, &[sort_by.as_str()])
                .cmp(&timestamp_field(right, &[sort_by.as_str()]))
        };
        let order = if descending { order.reverse() } else { order };
        order.then_with(|| string_field(left, &["Id"]).cmp(&string_field(right, &["Id"])))
    });
    Ok(())
}

fn upstream_sort_options(
    body: &Value,
    allow_group_id: bool,
) -> Result<(String, String), AssetServiceError> {
    let sort_by = optional_native_string(body, &["SortBy", "sort_by"], "SortBy")?
        .unwrap_or_else(|| "CreateTime".to_string());
    if !matches!(sort_by.as_str(), "CreateTime" | "UpdateTime")
        && !(allow_group_id && sort_by == "GroupId")
    {
        return Err(AssetServiceError::bad_request(if allow_group_id {
            "SortBy must be CreateTime, UpdateTime, or GroupId"
        } else {
            "SortBy must be CreateTime or UpdateTime"
        }));
    }
    let sort_order = match optional_native_string(body, &["SortOrder", "sort_order"], "SortOrder")?
        .unwrap_or_else(|| "Desc".to_string())
        .as_str()
    {
        "Desc" => "Desc".to_string(),
        "Asc" => "Asc".to_string(),
        _ => {
            return Err(AssetServiceError::bad_request(
                "SortOrder must be Asc or Desc",
            ));
        }
    };
    Ok((sort_by, sort_order))
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
    headers: &HeaderMap,
    caller: &AssetCaller,
    _is_admin: bool,
) -> Result<Value, AssetServiceError> {
    let group_type = query_value(
        request_context.request_query_string.as_deref(),
        "group_type",
    );
    let status = query_value(request_context.request_query_string.as_deref(), "status");
    let search_query = query_value(request_context.request_query_string.as_deref(), "search");
    let search = search_query
        .as_ref()
        .map(|value| value.to_ascii_lowercase());
    let visible_groups = policy_visible_bound_groups(state, caller).await?;
    let mut refresh_partitions = BTreeMap::<(String, String), Vec<String>>::new();
    for group in &visible_groups {
        refresh_partitions
            .entry((group.project_name.clone(), group.group_type.clone()))
            .or_default()
            .push(public_group_id(group).to_string());
    }
    let mut authoritative_ids = HashSet::new();
    for ((project_name, partition_group_type), group_ids) in refresh_partitions {
        let mut filter = json!({
            "GroupIds": group_ids,
            "GroupType": partition_group_type,
        });
        if let Some(search) = search_query.as_ref() {
            filter
                .as_object_mut()
                .expect("REST ListAssetGroups filter is an object")
                .insert("Name".to_string(), Value::String(search.clone()));
        }
        for page_number in 1..=MAX_LIST_MERGE_WINDOW.div_ceil(ARK_MAX_PAGE_SIZE) {
            let response = list_groups_native(
                state,
                request_context,
                headers,
                caller,
                &json!({
                    "Filter": filter,
                    "PageNumber": page_number,
                    "PageSize": ARK_MAX_PAGE_SIZE,
                    "ProjectName": project_name,
                }),
            )
            .await?;
            let result = extract_result(&response).ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalError",
                    "Aether ListAssetGroups projection is missing Result",
                )
            })?;
            let total = upstream_required_usize(result, "TotalCount")?;
            if total > MAX_LIST_MERGE_WINDOW {
                return Err(AssetServiceError::bad_request(format!(
                    "REST material group list supports at most {MAX_LIST_MERGE_WINDOW} items"
                )));
            }
            let items = result
                .get("Items")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    AssetServiceError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "InternalError",
                        "Aether ListAssetGroups projection is missing Items",
                    )
                })?;
            for item in items {
                authoritative_ids.insert(upstream_required_official_id(
                    item,
                    &["Id"],
                    "Items.Id",
                    "group-",
                )?);
            }
            if page_number.saturating_mul(ARK_MAX_PAGE_SIZE) >= total {
                break;
            }
        }
    }
    let mut groups = policy_visible_bound_groups(state, caller).await?;
    groups.retain(|group| {
        authoritative_ids.contains(public_group_id(group))
            && group_type
                .as_ref()
                .is_none_or(|expected| group.group_type.eq_ignore_ascii_case(expected))
            && status
                .as_ref()
                .is_none_or(|expected| group.status.eq_ignore_ascii_case(expected))
            && search.as_ref().is_none_or(|search| {
                group.name.to_ascii_lowercase().contains(search)
                    || group
                        .description
                        .as_ref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(search))
            })
    });
    groups.sort_by(|left, right| {
        right
            .created_at_unix_secs
            .cmp(&left.created_at_unix_secs)
            .then_with(|| public_group_id(left).cmp(public_group_id(right)))
    });
    let mut counts = HashMap::<String, usize>::with_capacity(groups.len());
    for group in &groups {
        let total = group_asset_count(state, group).await?;
        counts.insert(group.id.clone(), total);
    }
    Ok(json!({
        "items": groups.iter().map(|group| {
            group_rest_json(group, counts.get(&group.id).copied().unwrap_or_default())
        }).collect::<Vec<_>>(),
        "total": groups.len(),
    }))
}

async fn list_assets_rest(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
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
    let requested_group_id =
        query_value(request_context.request_query_string.as_deref(), "group_id");
    let requested_group = if let Some(group_id) = requested_group_id.as_deref() {
        validate_official_request_id(group_id, "group-", "GroupId")?;
        Some(load_group(state, caller, group_id, is_admin).await?)
    } else {
        None
    };
    let mut visible_groups = policy_visible_bound_groups(state, caller).await?;
    if let Some(requested_group) = requested_group.as_ref() {
        visible_groups.retain(|group| group.id == requested_group.id);
    }
    let mut project_groups = BTreeMap::<String, Vec<String>>::new();
    for group in &visible_groups {
        project_groups
            .entry(group.project_name.clone())
            .or_default()
            .push(public_group_id(group).to_string());
    }
    let status = query_value(request_context.request_query_string.as_deref(), "status");
    let search = query_value(request_context.request_query_string.as_deref(), "search");
    let mut authoritative_ids = HashSet::new();
    for (project_name, group_ids) in project_groups {
        let mut filter = json!({"GroupIds": group_ids});
        if let Some(status) = status.as_ref() {
            filter
                .as_object_mut()
                .expect("REST ListAssets filter is an object")
                .insert("Statuses".to_string(), json!([status]));
        }
        if let Some(search) = search.as_ref() {
            filter
                .as_object_mut()
                .expect("REST ListAssets filter is an object")
                .insert("Name".to_string(), Value::String(search.clone()));
        }
        for page_number in 1..=MAX_LIST_MERGE_WINDOW.div_ceil(ARK_MAX_PAGE_SIZE) {
            let response = list_assets_native(
                state,
                request_context,
                headers,
                caller,
                &json!({
                    "Filter": filter,
                    "PageNumber": page_number,
                    "PageSize": ARK_MAX_PAGE_SIZE,
                    "ProjectName": project_name,
                }),
            )
            .await?;
            let result = extract_result(&response).ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalError",
                    "Aether ListAssets projection is missing Result",
                )
            })?;
            let total = upstream_required_usize(result, "TotalCount")?;
            if total > MAX_LIST_MERGE_WINDOW {
                return Err(AssetServiceError::bad_request(format!(
                    "REST material asset list supports at most {MAX_LIST_MERGE_WINDOW} items per project"
                )));
            }
            let items = result
                .get("Items")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    AssetServiceError::new(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "InternalError",
                        "Aether ListAssets projection is missing Items",
                    )
                })?;
            for item in items {
                authoritative_ids.insert(upstream_required_official_id(
                    item,
                    &["Id"],
                    "Items.Id",
                    "asset-",
                )?);
            }
            if page_number.saturating_mul(ARK_MAX_PAGE_SIZE) >= total {
                break;
            }
        }
    }
    let groups = visible_groups
        .into_iter()
        .map(|group| (group.id.clone(), group))
        .collect::<HashMap<_, _>>();
    let requested_type = query_value(request_context.request_query_string.as_deref(), "type")
        .map(normalize_asset_type_filter);
    let search_lower = search.as_ref().map(|value| value.to_ascii_lowercase());
    let mut assets = list_all_assets(
        state,
        AssetListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            ..AssetListQuery::default()
        },
    )
    .await?;
    assets.retain(|asset| {
        groups.contains_key(&asset.group_id)
            && authoritative_ids.contains(public_asset_id(asset))
            && asset
                .upstream_asset_id
                .as_ref()
                .is_some_and(|id| id.starts_with("asset-") && id.len() > "asset-".len())
            && requested_group
                .as_ref()
                .is_none_or(|group| asset.group_id == group.id)
            && requested_type
                .as_ref()
                .is_none_or(|expected| asset.asset_type.eq_ignore_ascii_case(expected))
            && status
                .as_ref()
                .is_none_or(|expected| asset.status.eq_ignore_ascii_case(expected))
            && search_lower
                .as_ref()
                .is_none_or(|search| asset.name.to_ascii_lowercase().contains(search))
    });
    assets.sort_by(|left, right| {
        right
            .created_at_unix_secs
            .cmp(&left.created_at_unix_secs)
            .then_with(|| public_asset_id(left).cmp(public_asset_id(right)))
    });
    let total = assets.len();
    let offset = (page - 1).saturating_mul(page_size);
    let page_assets = assets
        .into_iter()
        .skip(offset)
        .take(page_size)
        .collect::<Vec<_>>();
    let refreshed = stream::iter(page_assets.into_iter().map(|asset| async move {
        refresh_asset(
            state,
            request_context,
            headers,
            caller,
            public_asset_id(&asset),
        )
        .await
    }))
    .buffered(8)
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "items": refreshed.iter().map(|asset| {
            asset_rest_json(asset, &groups[&asset.group_id], is_admin)
        }).collect::<Vec<_>>(),
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": total.div_ceil(page_size),
    }))
}

async fn load_group(
    state: &AppState,
    caller: &AssetCaller,
    group_id: &str,
    is_admin: bool,
) -> Result<StoredAssetGroup, AssetServiceError> {
    let direct = if is_admin {
        read_repo(state)?.find_group_by_id(group_id).await
    } else {
        read_repo(state)?
            .find_group_for_user(group_id, &caller.user_id)
            .await
    }
    .map_err(data_error)?;
    let group = if direct.is_some() {
        direct
    } else {
        let groups = list_all_groups(
            state,
            AssetGroupListQuery {
                user_id: Some(caller.user_id.clone()),
                include_deleted: false,
                ..AssetGroupListQuery::default()
            },
        )
        .await?;
        groups.into_iter().find(|group| {
            group.upstream_group_id.as_deref() == Some(group_id)
                && (!is_admin || group.user_id == caller.user_id)
        })
    };
    let group = group
        .filter(|group| !is_admin || group.user_id == caller.user_id)
        .filter(|group| group.deleted_at_unix_secs.is_none())
        .ok_or_else(AssetServiceError::not_found)?;
    Ok(group)
}

async fn load_group_by_upstream_id(
    state: &AppState,
    caller: &AssetCaller,
    upstream_group_id: &str,
) -> Result<StoredAssetGroup, AssetServiceError> {
    if !upstream_group_id.starts_with("group-") || upstream_group_id.len() == "group-".len() {
        return Err(AssetServiceError::bad_request(
            "Id must be an official Ark group-* ID",
        ));
    }
    list_all_groups(
        state,
        AssetGroupListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            ..AssetGroupListQuery::default()
        },
    )
    .await?
    .into_iter()
    .find(|group| group.upstream_group_id.as_deref() == Some(upstream_group_id))
    .ok_or_else(AssetServiceError::not_found)
}

async fn load_asset(
    state: &AppState,
    caller: &AssetCaller,
    asset_id: &str,
    _is_admin: bool,
) -> Result<StoredAsset, AssetServiceError> {
    load_asset_by_upstream_id(state, caller, asset_id).await
}

async fn load_asset_by_upstream_id(
    state: &AppState,
    caller: &AssetCaller,
    upstream_asset_id: &str,
) -> Result<StoredAsset, AssetServiceError> {
    if !upstream_asset_id.starts_with("asset-") || upstream_asset_id.len() == "asset-".len() {
        return Err(AssetServiceError::bad_request(
            "Id must be an official Ark asset-* ID",
        ));
    }
    list_all_assets(
        state,
        AssetListQuery {
            user_id: Some(caller.user_id.clone()),
            include_deleted: false,
            ..AssetListQuery::default()
        },
    )
    .await?
    .into_iter()
    .find(|asset| asset.upstream_asset_id.as_deref() == Some(upstream_asset_id))
    .ok_or_else(AssetServiceError::not_found)
}

async fn group_asset_count(
    state: &AppState,
    group: &StoredAssetGroup,
) -> Result<usize, AssetServiceError> {
    Ok(list_all_assets(
        state,
        AssetListQuery {
            group_id: Some(group.id.clone()),
            user_id: Some(group.user_id.clone()),
            include_deleted: false,
            ..AssetListQuery::default()
        },
    )
    .await?
    .into_iter()
    .filter(|asset| {
        asset
            .upstream_asset_id
            .as_ref()
            .is_some_and(|id| id.starts_with("asset-") && id.len() > "asset-".len())
    })
    .count())
}

async fn create_validation_session(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    headers: &HeaderMap,
    caller: &AssetCaller,
    body: Value,
) -> Result<(StoredArkVisualValidationSession, Value), AssetServiceError> {
    let project_name = native_project_name(&body)?;
    let callback_url = required_string_field(
        &body,
        &["CallbackURL", "callback_url", "ReturnUrl", "return_url"],
        "CallbackURL",
    )?;
    let encryption_key = state
        .encryption_key()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| AssetServiceError::unavailable("素材库真人验证需要配置数据加密密钥"))?;
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
    let h5_link = validation_verification_url(result).ok_or_else(|| {
        AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark CreateVisualValidateSession response is missing a safe H5Link",
        )
    })?;
    let returned_callback = upstream_required_string(result, &["CallbackURL"], "CallbackURL")?;
    if returned_callback != callback_url {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark CreateVisualValidateSession returned a different CallbackURL",
        ));
    }
    let session_id = string_field(result, &["SessionId", "session_id", "Id", "id"])
        .unwrap_or_else(|| sha256_text(&byted_token)[..32].to_string());
    let encrypted_byted_token =
        aether_crypto::encrypt_python_fernet_plaintext(encryption_key, &byted_token).map_err(
            |error| AssetServiceError::unavailable(format!("真人验证 token 加密失败: {error}")),
        )?;
    let encrypted_verification_url = Some(
        aether_crypto::encrypt_python_fernet_plaintext(encryption_key, &h5_link).map_err(
            |error| AssetServiceError::unavailable(format!("真人验证链接加密失败: {error}")),
        )?,
    );
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
        project_name,
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
    let mut last_error = None;
    for _ in 0..3 {
        match write_repo(state)?
            .upsert_visual_validation_session(record.clone())
            .await
        {
            Ok(session) => return Ok((session, response.body)),
            Err(error) => last_error = Some(error),
        }
    }
    Err(data_error(last_error.expect(
        "three failed validation-session upserts record an error",
    )))
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
    ensure_project_matches(body, &session.project_name)?;
    if validation_session_is_terminal(&session) {
        return Ok(native_envelope(
            request_context,
            ArkAssetAction::GetVisualValidateResult,
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
        caller,
        &transport,
        ArkAssetAction::GetVisualValidateResult,
        &json!({"BytedToken": token, "ProjectName": session.project_name}),
    )
    .await?;
    let result = extract_result(&response.body).unwrap_or(&response.body);
    let status = validation_result_status(result, &session.status)?;
    let upstream_group_id = if value_field(result, &["GroupId", "group_id"]).is_some() {
        Some(upstream_required_official_id(
            result,
            &["GroupId", "group_id"],
            "GroupId",
            "group-",
        )?)
    } else {
        None
    };
    let group_id = if let Some(upstream_group_id) = upstream_group_id.as_deref() {
        let projection = async {
            let group = ensure_validation_group(
                state,
                caller,
                &transport,
                upstream_group_id,
                &session.project_name,
                result,
            )
            .await?;
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
                Ok::<_, AssetServiceError>(Some(group.id))
            } else {
                Ok(None)
            }
        }
        .await;
        match projection {
            Ok(group_id) => group_id,
            Err(error) if error.status == StatusCode::CONFLICT => return Err(error),
            Err(error) => {
                tracing::error!(
                    session_id = %session.id,
                    upstream_group_id,
                    error_code = %error.code,
                    error = %error.message,
                    "visual validation succeeded upstream but local asset projection needs retry"
                );
                None
            }
        }
    } else {
        session.group_id.clone()
    };
    let projection_pending = upstream_group_id.is_some() && group_id.is_none();
    let persisted_status = if projection_pending {
        "ProjectionPending".to_string()
    } else {
        status.clone()
    };
    let projected_group_id = upstream_group_id.clone();
    let now = crate::clock::current_unix_secs();
    let record = UpsertArkVisualValidationSessionRecord {
        id: session.id,
        session_id: session.session_id,
        user_id: session.user_id,
        api_key_id: session.api_key_id,
        provider_id: session.provider_id,
        endpoint_id: session.endpoint_id,
        key_id: session.key_id,
        project_name: session.project_name,
        byted_token_hash: session.byted_token_hash,
        encrypted_byted_token: session.encrypted_byted_token,
        callback_state_hash: session.callback_state_hash,
        status: persisted_status.clone(),
        expires_at_unix_secs: session.expires_at_unix_secs,
        consumed_at_unix_secs: matches!(
            persisted_status.to_ascii_lowercase().as_str(),
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
    if let Err(error) = write_repo(state)?
        .upsert_visual_validation_session(record)
        .await
    {
        tracing::error!(
            upstream_group_id = ?projected_group_id,
            error = %error,
            "visual validation succeeded upstream but session projection needs retry"
        );
    }
    Ok(response.body)
}

async fn ensure_validation_group(
    state: &AppState,
    caller: &AssetCaller,
    transport: &AssetTransport,
    upstream_group_id: &str,
    project_name: &str,
    result: &Value,
) -> Result<Option<StoredAssetGroup>, AssetServiceError> {
    if let Some(group) = read_repo(state)?
        .find_group_by_canonical_upstream(&transport.snapshot.provider.id, upstream_group_id)
        .await
        .map_err(data_error)?
    {
        if group.user_id != caller.user_id
            || group.project_name != project_name
            || group.endpoint_id != transport.snapshot.endpoint.id
            || group.key_id != transport.snapshot.key.id
            || !group.group_type.eq_ignore_ascii_case("LivenessFace")
            || group.deleted_at_unix_secs.is_some()
        {
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
        project_name: project_name.to_string(),
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
    let fetched = fetch_upstream_list_items(
        state,
        request_context,
        headers,
        caller,
        transport,
        ArkAssetAction::ListAssets,
        json!({"GroupIds": [upstream_group_id]}),
        &group.project_name,
        1,
        MAX_LIST_MERGE_WINDOW,
        "CreateTime",
        "Desc",
    )
    .await?;
    if fetched.items.len() != fetched.total {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark validation group contains more than {MAX_LIST_MERGE_WINDOW} assets"),
        ));
    }
    for item in &fetched.items {
        upsert_validation_asset(state, group, upstream_group_id, item).await?;
    }
    Ok(())
}

async fn upsert_validation_asset(
    state: &AppState,
    group: &StoredAssetGroup,
    upstream_group_id: &str,
    item: &Value,
) -> Result<StoredAsset, AssetServiceError> {
    let upstream_asset_id =
        upstream_required_official_id(item, &["Id", "AssetId"], "Id", "asset-")?;
    let item_group_id = upstream_required_official_id(item, &["GroupId"], "GroupId", "group-")?;
    if item_group_id != upstream_group_id {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark ListAssets returned an asset from another group",
        ));
    }
    validate_upstream_asset_resource(item)?;
    let asset_type = upstream_required_string(item, &["AssetType", "asset_type"], "AssetType")?;
    let asset_type = normalize_upstream_asset_type(&asset_type)?;
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
        asset_type,
        name: upstream_optional_asset_name(item)?
            .or_else(|| existing.as_ref().map(|asset| asset.name.clone()))
            .unwrap_or_else(|| "未命名素材".to_string()),
        status: upstream_required_string(item, &["Status"], "Status")?,
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
            format!("Ark response is missing {display_name}"),
        )
    })
}

fn upstream_optional_asset_name(value: &Value) -> Result<Option<String>, AssetServiceError> {
    match value.get("Name") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) if name.trim().is_empty() => Ok(None),
        Some(Value::String(name)) if name.chars().count() <= 64 => Ok(Some(name.clone())),
        Some(Value::String(_)) => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response Name must be at most 64 characters",
        )),
        Some(_) => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response Name must be a string",
        )),
    }
}

fn upstream_required_official_id(
    value: &Value,
    names: &[&str],
    display_name: &str,
    prefix: &str,
) -> Result<String, AssetServiceError> {
    let id = upstream_required_string(value, names, display_name)?;
    if !id.starts_with(prefix) || id.len() == prefix.len() {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark response {display_name} must use the official {prefix} ID format"),
        ));
    }
    Ok(id)
}

fn validate_upstream_identity(
    value: &Value,
    field: &str,
    expected: &str,
) -> Result<(), AssetServiceError> {
    let actual = upstream_required_string(value, &[field], field)?;
    if actual != expected {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark response returned an unexpected {field}"),
        ));
    }
    Ok(())
}

fn normalize_upstream_project_name(
    value: &mut Value,
    project_name: &str,
) -> Result<(), AssetServiceError> {
    let object = value.as_object_mut().ok_or_else(|| {
        AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response resource must be an object",
        )
    })?;
    let aliases = object
        .keys()
        .filter(|name| {
            name.eq_ignore_ascii_case("ProjectName") || name.eq_ignore_ascii_case("project_name")
        })
        .cloned()
        .collect::<Vec<_>>();
    for alias in aliases {
        object.remove(&alias);
    }
    object.insert(
        "ProjectName".to_string(),
        Value::String(project_name.to_string()),
    );
    Ok(())
}

fn validate_upstream_group_resource(value: &Value) -> Result<(), AssetServiceError> {
    let _ = upstream_required_official_id(value, &["Id"], "Id", "group-")?;
    let group_type = upstream_required_string(value, &["GroupType"], "GroupType")?;
    if !matches!(group_type.as_str(), "AIGC" | "LivenessFace") {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response GroupType must be AIGC or LivenessFace",
        ));
    }
    if let Some(name) = value.get("Name") {
        let name = name.as_str().ok_or_else(|| {
            AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark response Name must be a string",
            )
        })?;
        if name.chars().count() > 64 {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark response Name must be at most 64 characters",
            ));
        }
    }
    validate_upstream_timestamp(value, "CreateTime")?;
    validate_upstream_timestamp(value, "UpdateTime")?;
    Ok(())
}

fn validate_upstream_asset_resource(value: &Value) -> Result<(), AssetServiceError> {
    let _ = upstream_required_official_id(value, &["Id"], "Id", "asset-")?;
    let _ = upstream_required_official_id(value, &["GroupId"], "GroupId", "group-")?;
    let _ = upstream_optional_asset_name(value)?;
    let _ = normalize_upstream_asset_type(&upstream_required_string(
        value,
        &["AssetType"],
        "AssetType",
    )?)?;
    let status = upstream_required_string(value, &["Status"], "Status")?;
    if !matches!(status.as_str(), "Active" | "Processing" | "Failed") {
        return Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response Status must be Active, Processing, or Failed",
        ));
    }
    validate_upstream_timestamp(value, "CreateTime")?;
    validate_upstream_timestamp(value, "UpdateTime")?;
    match value.get("URL") {
        Some(Value::String(url)) if !url.trim().is_empty() => {
            if !safe_provider_url(url.trim()) {
                return Err(AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    "Ark response URL is not a safe HTTPS address",
                ));
            }
        }
        None | Some(Value::Null) | Some(Value::String(_)) if status != "Active" => {}
        None | Some(Value::Null) | Some(Value::String(_)) => {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark Active asset response is missing URL",
            ));
        }
        Some(_) => {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark response URL must be a string",
            ));
        }
    }
    Ok(())
}

fn validate_upstream_timestamp(value: &Value, field: &str) -> Result<(), AssetServiceError> {
    match value.get(field) {
        Some(Value::String(timestamp)) if !timestamp.trim().is_empty() => Ok(()),
        Some(Value::String(_)) | None => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark response is missing {field}"),
        )),
        Some(_) => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            format!("Ark response {field} must be a string"),
        )),
    }
}

fn normalize_upstream_asset_type(value: &str) -> Result<String, AssetServiceError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "image" => Ok("Image".to_string()),
        "video" => Ok("Video".to_string()),
        "audio" => Ok("Audio".to_string()),
        _ => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark response AssetType must be Image, Video, or Audio",
        )),
    }
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
    let mut references = Vec::new();
    collect_asset_references(content, &mut references);
    references.sort_by(|left, right| left.0.cmp(&right.0));
    let mut required_types = HashMap::<String, AssetReferenceType>::new();
    for (asset_id, expected_type) in references {
        if required_types
            .insert(asset_id.clone(), expected_type)
            .is_some_and(|existing| existing != expected_type)
        {
            return Err(format!("素材 {asset_id} 不能同时作为不同媒体类型引用"));
        }
    }
    if required_types.is_empty() {
        return Ok(projected);
    }
    let reader = state
        .data
        .asset_library_read_repository()
        .ok_or_else(|| "素材库数据读取服务不可用".to_string())?;
    let mut replacements = HashMap::new();
    for (requested_id, expected_type) in required_types {
        if !requested_id.starts_with("asset-") || requested_id.len() == "asset-".len() {
            return Err(format!(
                "素材引用 {requested_id} 必须使用方舟返回的官方 asset-* ID"
            ));
        }
        let mut asset = None;
        let mut offset = 0usize;
        loop {
            let page = reader
                .list_assets(&AssetListQuery {
                    user_id: Some(user_id.to_string()),
                    include_deleted: false,
                    offset,
                    limit: MAX_PAGE_SIZE,
                    ..AssetListQuery::default()
                })
                .await
                .map_err(|error| format!("读取素材 {requested_id} 失败: {error}"))?;
            let page_len = page.items.len();
            asset = page
                .items
                .into_iter()
                .find(|asset| asset.upstream_asset_id.as_deref() == Some(requested_id.as_str()));
            if asset.is_some() || page_len == 0 || offset + page_len >= page.total {
                break;
            }
            offset = offset.saturating_add(page_len);
        }
        let asset = asset
            .filter(|asset| !asset.is_deleted)
            .ok_or_else(|| format!("素材 {requested_id} 不存在或不属于当前用户"))?;
        if !asset.status.eq_ignore_ascii_case("Active") {
            return Err(format!(
                "素材 {requested_id} 当前状态为 {}，必须为 Active",
                asset.status
            ));
        }
        if !asset
            .asset_type
            .eq_ignore_ascii_case(expected_type.official_name())
        {
            return Err(format!(
                "素材 {requested_id} 的类型为 {}，不能用于 {} 字段",
                asset.asset_type,
                expected_type.field_name()
            ));
        }
        let group = reader
            .find_group_for_user(&asset.group_id, user_id)
            .await
            .map_err(|error| format!("读取素材组失败: {error}"))?
            .filter(|group| group.deleted_at_unix_secs.is_none())
            .ok_or_else(|| format!("素材 {requested_id} 所属素材组不存在"))?;
        if group.provider_id != transport.provider.id {
            return Err(format!(
                "素材 {requested_id} 与本次视频生成的 Provider 不一致"
            ));
        }
        let upstream_id = asset
            .upstream_asset_id
            .filter(|value| value.starts_with("asset-") && value.len() > "asset-".len())
            .ok_or_else(|| format!("素材 {requested_id} 尚未完成上游绑定"))?;
        replacements.insert(requested_id, upstream_id);
    }
    replace_asset_references(content, &replacements);
    Ok(projected)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssetReferenceType {
    Image,
    Video,
    Audio,
}

impl AssetReferenceType {
    fn from_content_type(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "image_url" | "input_image" => Some(Self::Image),
            "video_url" | "input_video" => Some(Self::Video),
            "audio_url" | "input_audio" => Some(Self::Audio),
            _ => None,
        }
    }

    fn field_name(self) -> &'static str {
        match self {
            Self::Image => "image_url",
            Self::Video => "video_url",
            Self::Audio => "audio_url",
        }
    }

    fn official_name(self) -> &'static str {
        match self {
            Self::Image => "Image",
            Self::Video => "Video",
            Self::Audio => "Audio",
        }
    }
}

fn media_reference_url<'a>(value: &'a Value, field: &str) -> Option<&'a str> {
    let media = value.as_object()?.get(field)?;
    media
        .as_str()
        .or_else(|| media.as_object()?.get("url")?.as_str())
}

fn media_reference_url_mut<'a>(
    object: &'a mut Map<String, Value>,
    field: &str,
) -> Option<&'a mut String> {
    let media = object.get_mut(field)?;
    match media {
        Value::String(url) => Some(url),
        Value::Object(object) => match object.get_mut("url") {
            Some(Value::String(url)) => Some(url),
            _ => None,
        },
        _ => None,
    }
}

fn collect_asset_references(value: &Value, output: &mut Vec<(String, AssetReferenceType)>) {
    match value {
        Value::Array(values) => {
            for value in values {
                collect_asset_references(value, output);
            }
        }
        Value::Object(object) => {
            if let Some(reference_type) = object
                .get("type")
                .and_then(Value::as_str)
                .and_then(AssetReferenceType::from_content_type)
            {
                if let Some(id) = media_reference_url(value, reference_type.field_name())
                    .and_then(|url| url.strip_prefix("asset://"))
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                {
                    output.push((id.to_string(), reference_type));
                }
            }
            if let Some(nested) = object.get("content") {
                collect_asset_references(nested, output);
            }
        }
        _ => {}
    }
}

fn replace_asset_references(value: &mut Value, replacements: &HashMap<String, String>) {
    match value {
        Value::Array(values) => {
            for value in values {
                replace_asset_references(value, replacements);
            }
        }
        Value::Object(object) => {
            let reference_type = object
                .get("type")
                .and_then(Value::as_str)
                .and_then(AssetReferenceType::from_content_type);
            if let Some(reference_type) = reference_type {
                if let Some(url) = media_reference_url_mut(object, reference_type.field_name()) {
                    if let Some(id) = url.strip_prefix("asset://").map(str::trim) {
                        if let Some(upstream_id) = replacements.get(id) {
                            *url = format!("asset://{upstream_id}");
                        }
                    }
                }
            }
            if let Some(nested) = object.get_mut("content") {
                replace_asset_references(nested, replacements);
            }
        }
        _ => {}
    }
}

fn public_group_id(group: &StoredAssetGroup) -> &str {
    group
        .upstream_group_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(group.id.as_str())
}

fn public_asset_id(asset: &StoredAsset) -> &str {
    asset
        .upstream_asset_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(asset.id.as_str())
}

fn group_rest_json(group: &StoredAssetGroup, asset_count: usize) -> Value {
    json!({
        "id": public_group_id(group),
        "name": group.name,
        "description": group.description,
        "group_type": group.group_type,
        "project_name": group.project_name,
        "status": group.status,
        "asset_count": asset_count,
        "created_at": unix_secs_rfc3339(group.created_at_unix_secs),
        "updated_at": unix_secs_rfc3339(group.updated_at_unix_secs),
    })
}

fn asset_rest_json(asset: &StoredAsset, group: &StoredAssetGroup, is_admin: bool) -> Value {
    let metadata = asset.sanitized_metadata.as_ref();
    let public_id = public_asset_id(asset);
    let public_group_id = public_group_id(group);
    let preview_prefix = if is_admin {
        "/api/admin/material-assets/assets"
    } else {
        "/api/material-assets/assets"
    };
    let mut body = json!({
        "id": public_id,
        "uri": format!("asset://{public_id}"),
        "name": asset.name,
        "status": asset.status,
        "asset_type": asset.asset_type,
        "media_type": asset.asset_type.to_ascii_lowercase(),
        "mime_type": metadata.and_then(|value| value_field(value, &["MimeType", "mime_type"])).cloned(),
        "group_id": public_group_id,
        "group_name": group.name,
        "project_name": group.project_name,
        "source_type": "url",
        "url": asset.provider_url,
        "preview_url": format!("{preview_prefix}/{public_id}/preview"),
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
            object.insert(
                "provider_id".to_string(),
                Value::String(group.provider_id.clone()),
            );
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
        "group_id": public_validation_group_id(session),
        "error_message": session.sanitized_result.as_ref().and_then(|value| string_field(value, &["ErrorMessage", "Message"])),
        "expires_at": unix_secs_rfc3339(session.expires_at_unix_secs),
    }))
}

fn validation_session_is_terminal(session: &StoredArkVisualValidationSession) -> bool {
    let normalized = session.status.trim().to_ascii_lowercase();
    normalized == "failed"
        || ((normalized == "succeeded" || normalized == "success")
            && public_validation_group_id(session).is_some())
}

fn validation_result_status(result: &Value, fallback: &str) -> Result<String, AssetServiceError> {
    if string_field(result, &["GroupId", "group_id"]).is_some() {
        return Ok("Succeeded".to_string());
    }
    if let Some(status) = string_field(result, &["Status", "status"]) {
        if matches!(
            status.to_ascii_lowercase().as_str(),
            "succeeded" | "success"
        ) {
            return Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark GetVisualValidateResult reported success without GroupId",
            ));
        }
        return Ok(status);
    }
    match number_field(result, &["ResultCode", "resultCode", "result_code"]) {
        Some(10000) => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark GetVisualValidateResult returned resultCode=10000 without GroupId",
        )),
        Some(_) => Ok("Failed".to_string()),
        None if matches!(
            fallback.trim().to_ascii_lowercase().as_str(),
            "succeeded" | "success"
        ) =>
        {
            Err(AssetServiceError::new(
                StatusCode::BAD_GATEWAY,
                "InvalidUpstreamResponse",
                "Ark GetVisualValidateResult is missing GroupId",
            ))
        }
        None => Err(AssetServiceError::new(
            StatusCode::BAD_GATEWAY,
            "InvalidUpstreamResponse",
            "Ark GetVisualValidateResult response is missing GroupId, Status, and ResultCode",
        )),
    }
}

fn public_validation_group_id(session: &StoredArkVisualValidationSession) -> Option<String> {
    session
        .sanitized_result
        .as_ref()
        .and_then(|value| string_field(value, &["GroupId"]))
}

fn public_validation_result(session: &StoredArkVisualValidationSession) -> Value {
    if matches!(
        session.status.trim().to_ascii_lowercase().as_str(),
        "succeeded" | "success"
    ) {
        if let Some(group_id) = public_validation_group_id(session) {
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
        if let Some(group_id) = public_validation_group_id(session) {
            object.insert("GroupId".to_string(), Value::String(group_id));
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

fn canonical_visual_success(
    request_context: &GatewayPublicRequestContext,
    action: ArkAssetAction,
    body: Value,
) -> Result<Value, AssetServiceError> {
    let upstream = extract_result(&body).unwrap_or(&body);
    let result = match action {
        ArkAssetAction::CreateVisualValidateSession => {
            let byted_token = upstream_required_string(
                upstream,
                &["BytedToken", "Token", "byted_token", "token"],
                "BytedToken",
            )?;
            let callback_url = upstream_required_string(
                upstream,
                &["CallbackURL", "callback_url"],
                "CallbackURL",
            )?;
            let h5_link = validation_verification_url(upstream).ok_or_else(|| {
                AssetServiceError::new(
                    StatusCode::BAD_GATEWAY,
                    "InvalidUpstreamResponse",
                    "Ark CreateVisualValidateSession response is missing a safe H5Link",
                )
            })?;
            json!({
                "BytedToken": byted_token,
                "CallbackURL": callback_url,
                "H5Link": h5_link,
            })
        }
        ArkAssetAction::GetVisualValidateResult => {
            if value_field(upstream, &["GroupId", "group_id"]).is_some() {
                let group_id = upstream_required_official_id(
                    upstream,
                    &["GroupId", "group_id"],
                    "GroupId",
                    "group-",
                )?;
                json!({"GroupId": group_id})
            } else {
                let status = upstream_required_string(upstream, &["Status", "status"], "Status")?;
                if !matches!(
                    status.to_ascii_lowercase().as_str(),
                    "pending" | "processing" | "failed"
                ) {
                    return Err(AssetServiceError::new(
                        StatusCode::BAD_GATEWAY,
                        "InvalidUpstreamResponse",
                        "Ark GetVisualValidateResult returned success without GroupId",
                    ));
                }
                let mut result = json!({"Status": status});
                if let Some(result_code) =
                    number_field(upstream, &["ResultCode", "resultCode", "result_code"])
                {
                    result
                        .as_object_mut()
                        .expect("canonical visual result is an object")
                        .insert("ResultCode".to_string(), json!(result_code));
                }
                result
            }
        }
        _ => {
            return Err(AssetServiceError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                "canonical visual response used for a non-visual action",
            ));
        }
    };
    Ok(native_envelope(request_context, action, result))
}

fn native_error_response(error: AssetServiceError) -> Response<Body> {
    let body = error
        .provider_body
        .unwrap_or_else(|| build_error_envelope(&error.code, &error.message));
    json_response(error.status, body)
}

fn native_error_response_with_context(
    error: AssetServiceError,
    request_context: &GatewayPublicRequestContext,
    action: Option<ArkAssetAction>,
) -> Response<Body> {
    let status = error.status;
    let body = if let Some(mut provider_body) = error.provider_body {
        if let Some(metadata) = provider_body
            .get_mut("ResponseMetadata")
            .and_then(Value::as_object_mut)
        {
            metadata
                .entry("Version".to_string())
                .or_insert_with(|| Value::String(super::ARK_ASSET_VERSION.to_string()));
            metadata
                .entry("Service".to_string())
                .or_insert_with(|| Value::String("ark".to_string()));
            metadata
                .entry("Region".to_string())
                .or_insert_with(|| Value::String("cn-beijing".to_string()));
            if let Some(action) = action {
                metadata
                    .entry("Action".to_string())
                    .or_insert_with(|| Value::String(action.as_str().to_string()));
            }
        }
        provider_body
    } else {
        let mut metadata = Map::from_iter([
            (
                "RequestId".to_string(),
                Value::String(request_context.trace_id.clone()),
            ),
            (
                "Version".to_string(),
                Value::String(super::ARK_ASSET_VERSION.to_string()),
            ),
            ("Service".to_string(), Value::String("ark".to_string())),
            (
                "Region".to_string(),
                Value::String("cn-beijing".to_string()),
            ),
            (
                "Error".to_string(),
                json!({"Code": error.code, "Message": error.message}),
            ),
        ]);
        if let Some(action) = action {
            metadata.insert(
                "Action".to_string(),
                Value::String(action.as_str().to_string()),
            );
        }
        json!({"ResponseMetadata": metadata})
    };
    json_response(status, body)
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

fn missing_parameter(display_name: &str) -> AssetServiceError {
    AssetServiceError::new(
        StatusCode::BAD_REQUEST,
        format!("MissingParameter.{display_name}"),
        format!("{display_name} is required"),
    )
}

fn required_string_field(
    value: &Value,
    names: &[&str],
    display_name: &str,
) -> Result<String, AssetServiceError> {
    match value_field(value, names) {
        None | Some(Value::Null) => Err(missing_parameter(display_name)),
        Some(Value::String(value)) if value.trim().is_empty() => {
            Err(missing_parameter(display_name))
        }
        Some(Value::String(value)) => Ok(value.trim().to_string()),
        Some(_) => Err(AssetServiceError::bad_request(format!(
            "{display_name} must be a string"
        ))),
    }
}

fn optional_native_string(
    value: &Value,
    names: &[&str],
    display_name: &str,
) -> Result<Option<String>, AssetServiceError> {
    match value_field(value, names) {
        None => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Ok(Some(value.trim().to_string()))
        }
        Some(Value::String(_)) => Err(AssetServiceError::bad_request(format!(
            "{display_name} must not be empty"
        ))),
        Some(_) => Err(AssetServiceError::bad_request(format!(
            "{display_name} must be a string"
        ))),
    }
}

fn required_object_field<'a>(
    value: &'a Value,
    names: &[&str],
    display_name: &str,
) -> Result<&'a Value, AssetServiceError> {
    optional_object_field(value, names, display_name)?
        .ok_or_else(|| missing_parameter(display_name))
}

fn optional_object_field<'a>(
    value: &'a Value,
    names: &[&str],
    display_name: &str,
) -> Result<Option<&'a Value>, AssetServiceError> {
    match value_field(value, names) {
        None => Ok(None),
        Some(value) if value.is_object() => Ok(Some(value)),
        Some(_) => Err(AssetServiceError::bad_request(format!(
            "{display_name} must be an object"
        ))),
    }
}

fn optional_string_array_field(
    value: &Value,
    names: &[&str],
    display_name: &str,
) -> Result<Vec<String>, AssetServiceError> {
    let Some(value) = value_field(value, names) else {
        return Ok(Vec::new());
    };
    let values = value.as_array().ok_or_else(|| {
        AssetServiceError::bad_request(format!("{display_name} must be an array of strings"))
    })?;
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let value = value
            .as_str()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                AssetServiceError::bad_request(format!(
                    "{display_name} must contain non-empty strings"
                ))
            })?;
        if !output.iter().any(|existing| existing == value) {
            output.push(value.to_string());
        }
    }
    Ok(output)
}

fn validate_requested_group_ids(ids: &[String]) -> Result<(), AssetServiceError> {
    if ids
        .iter()
        .any(|id| !id.starts_with("group-") || id.len() == "group-".len())
    {
        return Err(AssetServiceError::bad_request(
            "Filter.GroupIds must contain official Ark group-* IDs",
        ));
    }
    Ok(())
}

fn validate_official_request_id(
    id: &str,
    prefix: &str,
    field: &str,
) -> Result<(), AssetServiceError> {
    if id.starts_with(prefix) && id.len() > prefix.len() {
        return Ok(());
    }
    Err(AssetServiceError::bad_request(format!(
        "{field} must use the official Ark {prefix}* ID format"
    )))
}

fn reject_unknown_fields(
    value: &Value,
    allowed: &[&str],
    display_name: &str,
) -> Result<(), AssetServiceError> {
    let Some(object) = value.as_object() else {
        return Err(AssetServiceError::bad_request(format!(
            "{display_name} must be an object"
        )));
    };
    if let Some(name) = object.keys().find(|name| {
        !allowed
            .iter()
            .any(|allowed| name.eq_ignore_ascii_case(allowed))
    }) {
        return Err(AssetServiceError::bad_request(format!(
            "{display_name}.{name} is not part of the official request"
        )));
    }
    Ok(())
}

fn native_positive_integer(
    value: &Value,
    names: &[&str],
    display_name: &str,
    required: bool,
    default: usize,
) -> Result<usize, AssetServiceError> {
    let Some(value) = value_field(value, names) else {
        return required
            .then(|| missing_parameter(display_name))
            .map_or(Ok(default), Err);
    };
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            AssetServiceError::bad_request(format!("{display_name} must be a positive integer"))
        })
}

fn required_native_page_number(value: &Value) -> Result<usize, AssetServiceError> {
    native_positive_integer(value, &["PageNumber", "page_number"], "PageNumber", true, 1)
}

fn required_native_page_size(value: &Value) -> Result<usize, AssetServiceError> {
    let page_size = native_positive_integer(
        value,
        &["PageSize", "page_size"],
        "PageSize",
        true,
        ARK_DEFAULT_PAGE_SIZE,
    )?;
    if page_size > ARK_MAX_PAGE_SIZE {
        return Err(AssetServiceError::bad_request(
            "PageSize must be between 1 and 100",
        ));
    }
    Ok(page_size)
}

fn optional_native_page_number(value: &Value) -> Result<usize, AssetServiceError> {
    native_positive_integer(
        value,
        &["PageNumber", "page_number"],
        "PageNumber",
        false,
        1,
    )
}

fn optional_native_page_size(value: &Value) -> Result<usize, AssetServiceError> {
    let page_size = native_positive_integer(
        value,
        &["PageSize", "page_size"],
        "PageSize",
        false,
        ARK_DEFAULT_PAGE_SIZE,
    )?;
    if page_size > ARK_MAX_PAGE_SIZE {
        return Err(AssetServiceError::bad_request(
            "PageSize must be between 1 and 100",
        ));
    }
    Ok(page_size)
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

fn validate_asset_text_lengths(value: &Value) -> Result<(), AssetServiceError> {
    if string_field(value, &["Name", "name"]).is_some_and(|name| name.chars().count() > 64) {
        return Err(AssetServiceError::bad_request(
            "Name must be at most 64 characters",
        ));
    }
    Ok(())
}

fn required_asset_type(value: &Value) -> Result<String, AssetServiceError> {
    let asset_type =
        required_string_field(value, &["AssetType", "asset_type", "type"], "AssetType")?;
    match asset_type.trim().to_ascii_lowercase().as_str() {
        "image" => Ok("Image".to_string()),
        "video" => Ok("Video".to_string()),
        "audio" => Ok("Audio".to_string()),
        _ => Err(AssetServiceError::bad_request(
            "AssetType must be Image, Video, or Audio",
        )),
    }
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

fn native_project_name(body: &Value) -> Result<String, AssetServiceError> {
    match value_field(body, &["ProjectName", "project_name"]) {
        None | Some(Value::Null) => Ok("default".to_string()),
        Some(Value::String(value)) if !value.trim().is_empty() => {
            let value = value.trim();
            if value.chars().count() > 128 {
                return Err(AssetServiceError::bad_request(
                    "ProjectName must be at most 128 characters",
                ));
            }
            Ok(value.to_string())
        }
        Some(Value::String(_)) => Err(AssetServiceError::bad_request(
            "ProjectName must not be empty",
        )),
        Some(_) => Err(AssetServiceError::bad_request(
            "ProjectName must be a string",
        )),
    }
}

fn ensure_project_matches(body: &Value, expected: &str) -> Result<String, AssetServiceError> {
    let project_name = native_project_name(body)?;
    if project_name != expected {
        return Err(AssetServiceError::bad_request(format!(
            "ProjectName must be {expected} for this resource"
        )));
    }
    Ok(project_name)
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
    use aether_data::repository::usage::InMemoryUsageReadRepository;
    use aether_data::repository::users::{InMemoryUserReadRepository, StoredUserAuthRecord};
    use aether_data_contracts::repository::candidates::{
        PublicHealthStatusCount, PublicHealthTimelineBucket, RequestCandidateReadRepository,
        RequestCandidateWriteRepository, StoredRequestCandidate, UpsertRequestCandidateRecord,
    };
    use aether_data_contracts::repository::usage::UsageReadRepository;
    use aether_data_contracts::DataLayerError;
    use aether_usage_runtime::UsageRuntimeConfig;
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use tokio::sync::Notify;

    use crate::data::GatewayDataState;

    fn stored_group(id: &str, user_id: &str) -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: id.to_string(),
            upstream_group_id: Some(format!("group-upstream-{id}")),
            user_id: user_id.to_string(),
            api_key_id: Some(format!("key-{user_id}")),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "provider-key-1".to_string(),
            project_name: "default".to_string(),
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
            upstream_asset_id: Some(format!("asset-upstream-{id}")),
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

    fn enabled_usage_runtime_config() -> UsageRuntimeConfig {
        UsageRuntimeConfig {
            enabled: true,
            enqueue_retry_initial_backoff_ms: 1,
            enqueue_retry_max_backoff_ms: 1,
            ..UsageRuntimeConfig::default()
        }
    }

    async fn record_pending_candidate_for_test(
        state: &AppState,
        plan: &ExecutionPlan,
        report_context: Option<&Value>,
    ) {
        crate::request_candidate_runtime::record_local_request_candidate_status(
            state,
            plan,
            report_context,
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
    }

    #[derive(Default)]
    struct BlockingPendingRequestCandidateRepository {
        inner: InMemoryRequestCandidateRepository,
        block_next_pending: AtomicBool,
        fail_next_terminal: AtomicBool,
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

        fn fail_first_terminal() -> Self {
            Self {
                fail_next_terminal: AtomicBool::new(true),
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
            if matches!(
                candidate.status,
                RequestCandidateStatus::Success
                    | RequestCandidateStatus::Failed
                    | RequestCandidateStatus::Cancelled
            ) && self.fail_next_terminal.swap(false, Ordering::SeqCst)
            {
                return Err(DataLayerError::sql(
                    "injected first terminal candidate write failure",
                ));
            }
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
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
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
        let usage = usage_repository
            .find_by_request_id(request_id)
            .await
            .expect("usage read")
            .expect("cancelled asset usage should exist before candidate terminal");
        assert_eq!(usage.status, "cancelled");
        assert_eq!(usage.billing_status, "void");
        assert_eq!(usage.status_code, Some(499));
        assert!(usage.response_headers.is_none());
        assert!(usage.response_body.is_none());
        assert!(usage.client_response_headers.is_none());
        assert!(usage.client_response_body.is_none());
    }

    #[tokio::test]
    async fn cancellation_during_pending_persist_is_finalized_as_cancelled() {
        let repository = Arc::new(BlockingPendingRequestCandidateRepository::blocking());
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
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
        let usage = usage_repository
            .find_by_request_id(request_id)
            .await
            .expect("usage read")
            .expect("cancelled usage should survive request task cancellation");
        assert_eq!(usage.status, "cancelled");
        assert_eq!(usage.billing_status, "void");
        assert_eq!(usage.status_code, Some(499));
    }

    #[tokio::test]
    async fn asset_success_persists_full_usage_audit_before_candidate_terminal() {
        let candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
        let request_id = "asset-success-audit-request";
        let plan = candidate_plan(request_id, "asset-success-audit-candidate");
        let report_context = json!({
            "candidate_id": "asset-success-audit-candidate",
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-1",
            "api_key_id": "api-key-1",
            "client_api_format": ARK_ASSET_API_FORMAT,
            "provider_api_format": ARK_ASSET_API_FORMAT,
            "model": ASSET_ROUTING_MODEL,
            "mapped_model": ASSET_ROUTING_MODEL,
            "original_headers": {
                "authorization": "Bearer secret-client-token",
                "content-type": "application/json"
            },
            "original_request_body": {"GroupId": "group-1"},
            "provider_request_headers": {
                "authorization": "HMAC secret-provider-signature",
                "content-type": "application/json"
            },
            "provider_request_body": {"GroupId": "group-1", "Action": "ListAssets"},
            "asset_action": "ListAssets"
        });
        record_pending_candidate_for_test(&state, &plan, Some(&report_context)).await;

        record_asset_candidate_terminal(
            &state,
            &plan,
            Some(&report_context),
            1,
            Instant::now(),
            AssetCandidateTerminalOutcome::success(
                200,
                AssetResponseAuditCapture {
                    provider_headers: BTreeMap::from([
                        ("content-type".to_string(), "application/json".to_string()),
                        (
                            "set-cookie".to_string(),
                            "provider-secret-cookie".to_string(),
                        ),
                    ]),
                    provider_body_json: Some(json!({"Result": {"Items": [1]}})),
                    provider_body_base64: None,
                    client_headers: BTreeMap::from([(
                        "content-type".to_string(),
                        "application/json".to_string(),
                    )]),
                    client_body: Some(json!({"Result": {"Items": [1]}})),
                    telemetry: Some(aether_contracts::ExecutionTelemetry {
                        ttfb_ms: Some(7),
                        elapsed_ms: Some(11),
                        upstream_bytes: Some(31),
                    }),
                },
            ),
        )
        .await;

        let usage = usage_repository
            .find_by_request_id(request_id)
            .await
            .expect("usage read")
            .expect("completed asset usage should be durable");
        assert_eq!(usage.status, "completed");
        assert_eq!(usage.status_code, Some(200));
        assert_eq!(usage.request_type.as_deref(), Some("asset_library"));
        assert_eq!(usage.billing_status, "void");
        assert_eq!(usage.total_cost_usd, 0.0);
        assert_eq!(usage.actual_total_cost_usd, 0.0);
        assert_eq!(usage.model, ASSET_ROUTING_MODEL);
        assert_eq!(usage.response_time_ms, Some(11));
        assert_eq!(usage.first_byte_time_ms, Some(7));
        assert_eq!(usage.request_body, Some(json!({"GroupId": "group-1"})));
        assert_eq!(
            usage.provider_request_body,
            Some(json!({"GroupId": "group-1", "Action": "ListAssets"}))
        );
        assert_eq!(usage.response_body, Some(json!({"Result": {"Items": [1]}})));
        assert_eq!(
            usage.client_response_body,
            Some(json!({"Result": {"Items": [1]}}))
        );
        assert_ne!(
            usage
                .request_headers
                .as_ref()
                .and_then(|headers| headers.get("authorization"))
                .and_then(Value::as_str),
            Some("Bearer secret-client-token")
        );
        assert_ne!(
            usage
                .response_headers
                .as_ref()
                .and_then(|headers| headers.get("set-cookie"))
                .and_then(Value::as_str),
            Some("provider-secret-cookie")
        );
        let candidates = candidate_repository
            .list_by_request_id(request_id)
            .await
            .expect("candidate read");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].status, RequestCandidateStatus::Success);
    }

    #[tokio::test]
    async fn asset_terminal_candidate_stays_non_terminal_when_usage_handoff_is_unconfirmed() {
        let candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_repository_for_tests(Arc::clone(
                    &candidate_repository,
                )),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
        let request_id = "asset-usage-unconfirmed-request";
        let plan = candidate_plan(request_id, "asset-usage-unconfirmed-candidate");
        let report_context = json!({
            "candidate_id": "asset-usage-unconfirmed-candidate",
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-1",
            "api_key_id": "api-key-1",
            "client_api_format": ARK_ASSET_API_FORMAT,
            "provider_api_format": ARK_ASSET_API_FORMAT,
            "model": ASSET_ROUTING_MODEL,
        });
        record_pending_candidate_for_test(&state, &plan, Some(&report_context)).await;

        record_asset_candidate_terminal(
            &state,
            &plan,
            Some(&report_context),
            1,
            Instant::now(),
            AssetCandidateTerminalOutcome::success(200, AssetResponseAuditCapture::default()),
        )
        .await;

        let candidates = candidate_repository
            .list_by_request_id(request_id)
            .await
            .expect("candidate read");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].status, RequestCandidateStatus::Streaming);
        assert!(candidates[0].finished_at_unix_ms.is_none());
        assert_eq!(
            candidates[0].error_type.as_deref(),
            Some("usage_terminal_handoff_unconfirmed")
        );
    }

    #[tokio::test]
    async fn asset_candidate_write_retries_after_durable_usage_handoff() {
        let candidate_repository =
            Arc::new(BlockingPendingRequestCandidateRepository::fail_first_terminal());
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
        let request_id = "asset-candidate-retry-request";
        let plan = candidate_plan(request_id, "asset-candidate-retry-candidate");
        let report_context = json!({
            "candidate_id": "asset-candidate-retry-candidate",
            "candidate_index": 0,
            "retry_index": 0,
            "user_id": "user-1",
            "api_key_id": "api-key-1",
            "client_api_format": ARK_ASSET_API_FORMAT,
            "provider_api_format": ARK_ASSET_API_FORMAT,
            "model": ASSET_ROUTING_MODEL,
        });
        record_pending_candidate_for_test(&state, &plan, Some(&report_context)).await;

        record_asset_candidate_terminal(
            &state,
            &plan,
            Some(&report_context),
            1,
            Instant::now(),
            AssetCandidateTerminalOutcome::success(200, AssetResponseAuditCapture::default()),
        )
        .await;

        let candidate = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let candidates = candidate_repository
                    .list_by_request_id(request_id)
                    .await
                    .expect("candidate read");
                if let Some(candidate) = candidates
                    .into_iter()
                    .find(|candidate| candidate.status == RequestCandidateStatus::Success)
                {
                    break candidate;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("candidate terminal retry should finish");
        assert_eq!(candidate.status_code, Some(200));
        let usage = usage_repository
            .find_by_request_id(request_id)
            .await
            .expect("usage read")
            .expect("usage must be durable before candidate retry");
        assert_eq!(usage.status, "completed");
    }

    #[tokio::test]
    async fn repeated_asset_actions_under_one_client_trace_use_independent_lifecycles() {
        let candidate_repository = Arc::new(InMemoryRequestCandidateRepository::default());
        let usage_repository = Arc::new(InMemoryUsageReadRepository::default());
        let state = AppState::new()
            .expect("gateway state")
            .with_data_state_for_tests(
                GatewayDataState::with_request_candidate_and_usage_repository_for_tests(
                    Arc::clone(&candidate_repository),
                    Arc::clone(&usage_repository),
                ),
            )
            .with_usage_runtime_for_tests(enabled_usage_runtime_config());
        let parent_request_id = "asset-client-trace-paginated";
        let child_request_ids = [
            asset_action_lifecycle_request_id(parent_request_id, ArkAssetAction::ListAssets),
            asset_action_lifecycle_request_id(parent_request_id, ArkAssetAction::ListAssets),
        ];
        assert_ne!(child_request_ids[0], child_request_ids[1]);

        for (index, child_request_id) in child_request_ids.iter().enumerate() {
            let candidate_id = format!("asset-page-candidate-{index}");
            let plan = candidate_plan(child_request_id, &candidate_id);
            let report_context = json!({
                "request_id": child_request_id,
                "parent_request_id": parent_request_id,
                "client_request_id": parent_request_id,
                "client_trace_id": parent_request_id,
                "candidate_id": candidate_id,
                "candidate_index": index,
                "retry_index": 0,
                "user_id": "user-1",
                "api_key_id": "api-key-1",
                "client_api_format": ARK_ASSET_API_FORMAT,
                "provider_api_format": ARK_ASSET_API_FORMAT,
                "model": ASSET_ROUTING_MODEL,
                "mapped_model": ASSET_ROUTING_MODEL,
                "asset_action": "ListAssets",
                "usage_scope": "asset_upstream_action",
                "client_capture_scope": "asset_action_projection",
            });
            state.usage_runtime.record_pending(
                state.usage_lifecycle_data_state().as_ref(),
                aether_usage_runtime::build_lifecycle_usage_seed(&plan, Some(&report_context)),
            );
            record_pending_candidate_for_test(&state, &plan, Some(&report_context)).await;
            record_asset_candidate_terminal(
                &state,
                &plan,
                Some(&report_context),
                1,
                Instant::now(),
                AssetCandidateTerminalOutcome::success(
                    200,
                    AssetResponseAuditCapture {
                        provider_headers: BTreeMap::from([(
                            "content-type".to_string(),
                            "application/json".to_string(),
                        )]),
                        provider_body_json: Some(json!({"Result": {"Page": index + 1}})),
                        provider_body_base64: None,
                        client_headers: BTreeMap::from([(
                            "content-type".to_string(),
                            "application/json".to_string(),
                        )]),
                        client_body: Some(json!({"Result": {"Page": index + 1}})),
                        telemetry: Some(aether_contracts::ExecutionTelemetry {
                            ttfb_ms: Some(1),
                            elapsed_ms: Some(2),
                            upstream_bytes: Some(16),
                        }),
                    },
                ),
            )
            .await;
        }

        for child_request_id in &child_request_ids {
            let usage = usage_repository
                .find_by_request_id(child_request_id)
                .await
                .expect("usage read")
                .expect("each asset page should have its own terminal usage");
            assert_eq!(usage.status, "completed");
            assert_eq!(usage.billing_status, "void");
            assert_eq!(usage.total_cost_usd, 0.0);
            assert_eq!(usage.actual_total_cost_usd, 0.0);
            let metadata = usage
                .request_metadata
                .as_ref()
                .expect("parent request linkage should be auditable");
            assert_eq!(
                metadata.get("parent_request_id").and_then(Value::as_str),
                Some(parent_request_id)
            );
            assert_eq!(
                metadata.get("client_request_id").and_then(Value::as_str),
                Some(parent_request_id)
            );
            let candidates = candidate_repository
                .list_by_request_id(child_request_id)
                .await
                .expect("candidate read");
            assert_eq!(candidates.len(), 1);
            assert_eq!(candidates[0].request_id.as_str(), child_request_id.as_str());
            assert_eq!(candidates[0].status, RequestCandidateStatus::Success);
        }
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
        assert_eq!(error.code, "MissingParameter.CallbackURL");
        assert_eq!(error.message, "CallbackURL is required");
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
    fn create_asset_accepts_all_official_asset_types() {
        for (input, expected) in [("image", "Image"), ("Video", "Video"), ("AUDIO", "Audio")] {
            assert_eq!(
                required_asset_type(&json!({"AssetType": input})).unwrap(),
                expected
            );
        }
        let error = required_asset_type(&json!({"AssetType": "Document"}))
            .expect_err("non-official asset types must be rejected");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.message, "AssetType must be Image, Video, or Audio");
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

        assert_eq!(
            native_project_name(&json!({"ProjectName": "another-project"})).unwrap(),
            "another-project"
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
            json!({"ResponseMetadata": {}, "Result": validation})["Result"]["BytedToken"],
            "token-upstream"
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
    }

    #[tokio::test]
    async fn upstream_project_name_is_mapped_to_the_bound_aether_project() {
        let mut group = json!({
            "Id": "group-20260825120000-a1b2c",
            "Name": "products",
            "GroupType": "AIGC",
            "ProjectName": "provider-internal-project",
            "CreateTime": "2026-08-25 12:00:00",
            "UpdateTime": "2026-08-25 12:00:00"
        });
        normalize_upstream_project_name(&mut group, "default")
            .expect("provider project name should be mapped");
        validate_upstream_group_resource(&group).expect("mapped group should remain valid");
        assert_eq!(group["ProjectName"], "default");

        for mut value in [
            json!({}),
            json!({"ProjectName": ""}),
            json!({"project_name": null}),
        ] {
            normalize_upstream_project_name(&mut value, "default")
                .expect("missing or empty provider project name should be mapped");
            assert_eq!(value, json!({"ProjectName": "default"}));
        }

        let mut asset = json!({
            "Id": "asset-20260825120000-d3e4f",
            "GroupId": "group-20260825120000-a1b2c",
            "Name": "portrait",
            "AssetType": "Image",
            "Status": "Processing",
            "project_name": 42,
            "CreateTime": "2026-08-25 12:00:00",
            "UpdateTime": "2026-08-25 12:00:00"
        });
        normalize_upstream_project_name(&mut asset, "default")
            .expect("missing or non-canonical provider project name should be mapped");
        validate_upstream_asset_resource(&asset).expect("mapped asset should remain valid");
        assert_eq!(asset["ProjectName"], "default");
        assert!(asset.get("project_name").is_none());

        let mut active_without_url = asset.clone();
        active_without_url["Status"] = Value::String("Active".to_string());
        assert!(validate_upstream_asset_resource(&active_without_url).is_err());

        let mut failed_without_url = asset.clone();
        failed_without_url["Status"] = Value::String("Failed".to_string());
        validate_upstream_asset_resource(&failed_without_url).expect("failed assets may omit URL");

        let mut unsafe_url = asset.clone();
        unsafe_url["URL"] = Value::String("http://127.0.0.1/private.png".to_string());
        assert!(validate_upstream_asset_resource(&unsafe_url).is_err());

        assert!(
            validate_upstream_identity(&asset, "GroupId", "group-20260825120000-other").is_err()
        );

        let request_mismatch = ensure_project_matches(
            &json!({"ProjectName": "another-logical-project"}),
            "default",
        )
        .expect_err("request project must still match the bound resource");
        assert_eq!(request_mismatch.status, StatusCode::BAD_REQUEST);

        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        let state = state_with_asset_repository(Arc::clone(&repository));
        let persisted = persist_group_projection(
            &state,
            stored_group("group-local", "user-1").into_stored(),
            &group,
        )
        .await
        .expect("mapped group should be persisted");
        assert_eq!(persisted.project_name, "default");
        assert_eq!(
            repository
                .find_group_by_id("group-local")
                .await
                .expect("group lookup")
                .expect("persisted group")
                .project_name,
            "default"
        );
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
            "GroupId": "group-upstream-group-local",
            "Name": "face",
            "URL": "https://media.example.com/face.png?signature=secret",
            "AssetType": "Image",
            "Status": "Active",
            "ProjectName": "default",
            "Width": 1024,
            "Height": 1024,
            "CreateTime": "2026-03-28T12:34:56Z",
            "UpdateTime": "2026-03-28T12:34:56Z"
        });

        let first = upsert_validation_asset(&state, &group, "group-upstream-group-local", &item)
            .await
            .expect("first projection");
        let second = upsert_validation_asset(&state, &group, "group-upstream-group-local", &item)
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
            deterministic_validation_group_id("provider-1", "group-upstream-group-local"),
            deterministic_validation_group_id("provider-1", "group-upstream-group-local")
        );
        assert_ne!(
            deterministic_validation_group_id("provider-1", "group-upstream-group-local"),
            deterministic_validation_group_id("provider-1", "another-upstream-group")
        );
    }

    #[tokio::test]
    async fn terminal_validation_result_keeps_official_envelope_and_upstream_group_id() {
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
                project_name: "default".to_string(),
                byted_token_hash: sha256_text(token),
                encrypted_byted_token: "unused-terminal-ciphertext".to_string(),
                callback_state_hash: "callback-hash".to_string(),
                status: "Succeeded".to_string(),
                expires_at_unix_secs: now.saturating_add(60),
                consumed_at_unix_secs: Some(now),
                group_id: Some("group-local".to_string()),
                sanitized_result: Some(json!({"GroupId": "group-20260825120000-a1b2c"})),
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
            assert_eq!(
                result["Result"],
                json!({"GroupId": "group-20260825120000-a1b2c"})
            );
            assert_eq!(
                result["ResponseMetadata"]["Action"],
                "GetVisualValidateResult"
            );
        }
    }

    #[test]
    fn official_list_filters_are_strict() {
        assert_eq!(
            optional_string_array_field(
                &json!({"GroupIds": ["group-1", "group-2"]}),
                &["GroupIds"],
                "Filter.GroupIds",
            )
            .unwrap(),
            vec!["group-1", "group-2"]
        );
        assert!(optional_string_array_field(
            &json!({"GroupIds": "group-1"}),
            &["GroupIds"],
            "Filter.GroupIds",
        )
        .is_err());
        assert!(reject_unknown_fields(
            &json!({"AssetType": "Image"}),
            &["GroupIds", "GroupType", "Name", "Statuses"],
            "Filter",
        )
        .is_err());
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
            (
                json!({"Filter": {}, "PageNumber": 1, "PageSize": 10}),
                "Filter.GroupType is required",
            ),
            (
                json!({"PageNumber": 1, "PageSize": 10}),
                "Filter is required",
            ),
            (
                json!({"Filter": {"GroupType": "Unknown"}, "PageNumber": 1, "PageSize": 10}),
                "Filter.GroupType must be AIGC or LivenessFace",
            ),
        ] {
            let error = list_groups_native(&state, &context, &headers, &caller("user-1"), &body)
                .await
                .expect_err("invalid GroupType filter must fail");
            assert_eq!(error.status, StatusCode::BAD_REQUEST);
            assert_eq!(error.message, message);
        }
    }

    #[test]
    fn validation_result_code_drives_terminal_status() {
        assert_eq!(
            validation_result_status(&json!({"GroupId": "group-upstream"}), "Pending").unwrap(),
            "Succeeded"
        );
        assert!(validation_result_status(&json!({"resultCode": 10000}), "Pending").is_err());
        assert_eq!(
            validation_result_status(&json!({"ResultCode": 10001}), "Pending").unwrap(),
            "Failed"
        );
        assert_eq!(
            validation_result_status(&json!({"Status": "Processing"}), "Pending").unwrap(),
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
            project_name: "default".to_string(),
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
    fn rewrites_only_typed_media_asset_references() {
        let mut value = json!({
            "content": [
                {"type": "image_url", "image_url": {"url": "asset://local-1"}},
                {"type": "text", "text": "asset://local-2"},
                {"url": "https://example.test/asset://local-1"}
            ]
        });
        replace_asset_references(
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
        assert_eq!(value["content"][1]["text"], "asset://local-2");
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
            load_asset(&state, &caller("user-2"), "asset-upstream-asset-1", false,)
                .await
                .expect_err("another user must not see the asset")
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            load_asset(&state, &caller("user-1"), "asset-1", false)
                .await
                .expect_err("internal asset IDs must not be accepted at the public boundary")
                .status,
            StatusCode::NOT_FOUND
        );
        assert!(
            load_asset(&state, &caller("user-1"), "asset-upstream-asset-1", false,)
                .await
                .is_ok()
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
            load_asset(&state, &caller("user-1"), "asset-upstream-asset-1", false,)
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
            load_asset(&state, &caller, "asset-upstream-asset-1", true)
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
                project_name: "default".to_string(),
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
            project_name: "default".to_string(),
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
