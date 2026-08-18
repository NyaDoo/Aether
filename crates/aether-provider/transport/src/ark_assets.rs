use std::collections::BTreeMap;
use std::fmt;

use chrono::{DateTime, NaiveDateTime, Utc};
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Digest, Sha256};
use url::{form_urlencoded, Host, Url};

use crate::auth::build_passthrough_headers;
use crate::headers::is_aether_internal_header;
use crate::rules::{
    apply_local_body_rules_with_request_headers, apply_local_header_rules_with_request_headers,
};
use crate::snapshot::GatewayProviderTransportSnapshot;

pub const ARK_ASSET_API_FORMAT: &str = "doubao:asset_library";
pub const ARK_ASSET_REQUIRED_CAPABILITY: &str = "ark_asset_library";
pub const VOLC_ACTION_VERSION: &str = "2024-01-01";
pub const VOLC_ACTION_DEFAULT_REGION: &str = "cn-beijing";
pub const VOLC_ACTION_DEFAULT_SERVICE: &str = "ark";
pub const VOLC_ACTION_DEFAULT_BASE_URL: &str = "https://ark.cn-beijing.volcengineapi.com";

const VOLC_AUTH_ALGORITHM: &str = "HMAC-SHA256";
const VOLC_AUTH_TERMINATOR: &str = "request";
const PLACEHOLDER_API_KEY: &str = "__placeholder__";
const BLOCKED_ACTION_QUERY_KEYS: &[&str] = &[
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "key",
    "token",
    "Action",
    "Version",
];
const PROTECTED_ACTION_HEADERS: &[&str] = &[
    "authorization",
    "api-key",
    "x-api-key",
    "content-type",
    "host",
    "x-content-sha256",
    "x-date",
    "x-security-token",
];

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VolcActionTransportError {
    #[error("Ark Action is empty or invalid")]
    InvalidAction,
    #[error("Ark Action endpoint base URL is invalid")]
    InvalidBaseUrl,
    #[error("Ark Action endpoint custom path is invalid")]
    InvalidCustomPath,
    #[error("Ark Action request body rules could not be applied")]
    BodyRulesApplyFailed,
    #[error("Ark Action request header rules could not be applied")]
    HeaderRulesApplyFailed,
    #[error("Ark Action request body could not be encoded as JSON")]
    BodyEncodeFailed,
    #[error("Ark Action provider authentication type is unsupported")]
    UnsupportedAuthType,
    #[error("Ark Action provider credential is missing or invalid")]
    InvalidCredential,
    #[error("Ark Action provider AK/SK configuration is missing or invalid")]
    InvalidAkSkConfig,
    #[error("Ark Action request URL cannot be signed")]
    InvalidSigningUrl,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VolcActionVerificationError {
    #[error("Ark Action authorization header is missing or unsupported")]
    MissingAuthorization,
    #[error("Ark Action authorization header is malformed")]
    MalformedAuthorization,
    #[error("Ark Action credential scope is invalid")]
    InvalidCredentialScope,
    #[error("Ark Action signed headers are invalid")]
    InvalidSignedHeaders,
    #[error("Ark Action signed header is missing or invalid: {0}")]
    InvalidSignedHeader(String),
    #[error("Ark Action payload hash does not match the request body")]
    PayloadHashMismatch,
    #[error("Ark Action request timestamp is invalid or outside the allowed window")]
    InvalidRequestTime,
    #[error("Ark Action signature is invalid")]
    InvalidSignature,
    #[error("Ark Action request URL cannot be verified")]
    InvalidRequestUrl,
}

#[derive(Clone, Copy)]
pub struct VolcActionVerificationInput<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a http::HeaderMap,
    pub body: &'a [u8],
    pub secret_access_key: &'a str,
    pub expected_access_key_id: &'a str,
    pub expected_region: &'a str,
    pub expected_service: &'a str,
    pub now: DateTime<Utc>,
    pub max_clock_skew_secs: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedVolcActionSignature {
    pub access_key_id: String,
    pub region: String,
    pub service: String,
    pub signed_at: DateTime<Utc>,
}

#[derive(Debug)]
struct ParsedVolcAuthorization {
    access_key_id: String,
    short_date: String,
    region: String,
    service: String,
    signed_headers: Vec<String>,
    signature: [u8; 32],
}

#[derive(Clone, PartialEq, Eq)]
pub struct VolcAkSkCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub security_token: Option<String>,
    pub region: String,
    pub service: String,
}

impl fmt::Debug for VolcAkSkCredentials {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VolcAkSkCredentials")
            .field("access_key_id", &"[REDACTED]")
            .field("secret_access_key", &"[REDACTED]")
            .field(
                "security_token",
                &self.security_token.as_ref().map(|_| "[REDACTED]"),
            )
            .field("region", &self.region)
            .field("service", &self.service)
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum VolcActionAuth {
    AkSk(VolcAkSkCredentials),
    Bearer(String),
    ApiKey { header_name: String, secret: String },
}

impl fmt::Debug for VolcActionAuth {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AkSk(credentials) => formatter.debug_tuple("AkSk").field(credentials).finish(),
            Self::Bearer(_) => formatter
                .debug_tuple("Bearer")
                .field(&"[REDACTED]")
                .finish(),
            Self::ApiKey { header_name, .. } => formatter
                .debug_struct("ApiKey")
                .field("header_name", header_name)
                .field("secret", &"[REDACTED]")
                .finish(),
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct VolcActionRequest {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
    pub body_json: Value,
}

impl VolcActionRequest {
    pub fn into_parts(self) -> (String, BTreeMap<String, String>, Vec<u8>) {
        (self.url, self.headers, self.body)
    }
}

impl fmt::Debug for VolcActionRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names = self.headers.keys().cloned().collect::<Vec<_>>();
        formatter
            .debug_struct("VolcActionRequest")
            .field("url", &self.url)
            .field("header_names", &header_names)
            .field("body_len", &self.body.len())
            .field("body_json", &self.body_json)
            .finish()
    }
}

#[derive(Clone, Copy)]
pub struct VolcActionRequestInput<'a> {
    pub transport: &'a GatewayProviderTransportSnapshot,
    pub action: &'a str,
    pub body: &'a Value,
    pub request_headers: &'a http::HeaderMap,
    pub request_time: Option<DateTime<Utc>>,
}

#[derive(Clone, PartialEq, Eq)]
pub struct VolcActionSignature {
    pub authorization: String,
    pub payload_hash: String,
    pub signed_headers: String,
    pub canonical_request: String,
    pub string_to_sign: String,
}

impl fmt::Debug for VolcActionSignature {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VolcActionSignature")
            .field("authorization", &"[REDACTED]")
            .field("payload_hash", &self.payload_hash)
            .field("signed_headers", &self.signed_headers)
            .field(
                "canonical_request_sha256",
                &sha256_hex(self.canonical_request.as_bytes()),
            )
            .field("string_to_sign", &"[REDACTED]")
            .finish()
    }
}

pub fn build_volc_action_url(
    upstream_base_url: &str,
    custom_path: Option<&str>,
    action: &str,
) -> Result<String, VolcActionTransportError> {
    let action = normalize_action(action)?;
    let mut url = Url::parse(upstream_base_url.trim())
        .map_err(|_| VolcActionTransportError::InvalidBaseUrl)?;
    if !volc_action_url_transport_is_allowed(&url) {
        return Err(VolcActionTransportError::InvalidBaseUrl);
    }

    let custom_query =
        if let Some(custom_path) = custom_path.map(str::trim).filter(|value| !value.is_empty()) {
            if !custom_path.starts_with('/')
                || custom_path.contains("://")
                || custom_path.contains('#')
                || custom_path.chars().any(char::is_control)
            {
                return Err(VolcActionTransportError::InvalidCustomPath);
            }
            let (path, query) = custom_path
                .split_once('?')
                .map(|(path, query)| (path, Some(query)))
                .unwrap_or((custom_path, None));
            if path.is_empty() {
                return Err(VolcActionTransportError::InvalidCustomPath);
            }
            url.set_path(path);
            query
        } else {
            None
        };

    let mut retained_query = BTreeMap::new();
    for query in [url.query(), custom_query].into_iter().flatten() {
        for (name, value) in form_urlencoded::parse(query.as_bytes()) {
            if BLOCKED_ACTION_QUERY_KEYS
                .iter()
                .any(|blocked| name.eq_ignore_ascii_case(blocked))
            {
                continue;
            }
            retained_query.insert(name.into_owned(), value.into_owned());
        }
    }
    retained_query.insert("Action".to_string(), action.to_string());
    retained_query.insert("Version".to_string(), VOLC_ACTION_VERSION.to_string());

    let mut query = form_urlencoded::Serializer::new(String::new());
    for (name, value) in retained_query {
        query.append_pair(&name, &value);
    }
    url.set_query(Some(&query.finish()));
    url.set_fragment(None);
    Ok(url.to_string())
}

fn volc_action_url_transport_is_allowed(url: &Url) -> bool {
    let Some(host) = url.host() else {
        return false;
    };
    match url.scheme() {
        "https" => true,
        "http" => match host {
            Host::Domain(host) => host.eq_ignore_ascii_case("localhost"),
            Host::Ipv4(address) => address.is_loopback(),
            Host::Ipv6(address) => {
                address.is_loopback()
                    || address
                        .to_ipv4_mapped()
                        .is_some_and(|address| address.is_loopback())
            }
        },
        _ => false,
    }
}

pub fn resolve_volc_action_auth(
    transport: &GatewayProviderTransportSnapshot,
) -> Result<VolcActionAuth, VolcActionTransportError> {
    let auth_type = normalize_volc_action_auth_type(&resolved_action_auth_type(transport))
        .ok_or(VolcActionTransportError::UnsupportedAuthType)?;
    match auth_type.as_str() {
        "volc_aksk" => parse_volc_aksk_credentials(transport)
            .map(VolcActionAuth::AkSk)
            .ok_or(VolcActionTransportError::InvalidAkSkConfig),
        "bearer" => resolve_raw_secret(transport)
            .map(|secret| VolcActionAuth::Bearer(secret.to_string()))
            .ok_or(VolcActionTransportError::InvalidCredential),
        "api_key" => resolve_raw_secret(transport)
            .map(|secret| VolcActionAuth::ApiKey {
                header_name: resolve_relay_api_key_header(transport),
                secret: secret.to_string(),
            })
            .ok_or(VolcActionTransportError::InvalidCredential),
        _ => Err(VolcActionTransportError::UnsupportedAuthType),
    }
}

pub fn normalize_volc_action_auth_type(value: &str) -> Option<String> {
    match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
        "volc_aksk" | "volcengine_aksk" | "aksk" | "hmac" | "hmac_sha256" => {
            Some("volc_aksk".to_string())
        }
        "api_key" | "apikey" => Some("api_key".to_string()),
        "bearer" | "bearer_token" | "authorization" => Some("bearer".to_string()),
        _ => None,
    }
}

pub fn build_volc_action_request(
    input: VolcActionRequestInput<'_>,
) -> Result<VolcActionRequest, VolcActionTransportError> {
    let url = build_volc_action_url(
        &input.transport.endpoint.base_url,
        input.transport.endpoint.custom_path.as_deref(),
        input.action,
    )?;
    let auth = resolve_volc_action_auth(input.transport)?;
    let mut body_json = input.body.clone();
    if !apply_local_body_rules_with_request_headers(
        &mut body_json,
        input.transport.endpoint.body_rules.as_ref(),
        Some(input.body),
        Some(input.request_headers),
    ) {
        return Err(VolcActionTransportError::BodyRulesApplyFailed);
    }
    let body =
        serde_json::to_vec(&body_json).map_err(|_| VolcActionTransportError::BodyEncodeFailed)?;

    let mut headers = build_passthrough_headers(
        input.request_headers,
        &BTreeMap::new(),
        Some("application/json"),
    );
    headers.insert("content-type".to_string(), "application/json".to_string());
    if !apply_local_header_rules_with_request_headers(
        &mut headers,
        input.transport.endpoint.header_rules.as_ref(),
        PROTECTED_ACTION_HEADERS,
        &body_json,
        Some(input.body),
        Some(input.request_headers),
    ) {
        return Err(VolcActionTransportError::HeaderRulesApplyFailed);
    }
    strip_action_runtime_internal_headers(&mut headers);

    match auth {
        VolcActionAuth::AkSk(credentials) => {
            apply_volc_action_signature(
                &mut headers,
                "POST",
                &url,
                &body,
                &credentials,
                input.request_time.unwrap_or_else(Utc::now),
            )?;
        }
        VolcActionAuth::Bearer(secret) => {
            strip_action_credential_headers(&mut headers);
            headers.insert("authorization".to_string(), format!("Bearer {secret}"));
        }
        VolcActionAuth::ApiKey {
            header_name,
            secret,
        } => {
            strip_action_credential_headers(&mut headers);
            headers.insert(header_name, secret);
        }
    }

    Ok(VolcActionRequest {
        url,
        headers,
        body,
        body_json,
    })
}

pub fn apply_volc_action_signature(
    headers: &mut BTreeMap<String, String>,
    method: &str,
    url: &str,
    body: &[u8],
    credentials: &VolcAkSkCredentials,
    request_time: DateTime<Utc>,
) -> Result<VolcActionSignature, VolcActionTransportError> {
    if credentials.access_key_id.trim().is_empty()
        || credentials.secret_access_key.trim().is_empty()
        || credentials.region.trim().is_empty()
        || credentials.service.trim().is_empty()
    {
        return Err(VolcActionTransportError::InvalidCredential);
    }
    let url = Url::parse(url).map_err(|_| VolcActionTransportError::InvalidSigningUrl)?;
    let host = canonical_host(&url).ok_or(VolcActionTransportError::InvalidSigningUrl)?;
    let payload_hash = sha256_hex(body);
    let x_date = request_time.format("%Y%m%dT%H%M%SZ").to_string();
    let short_date = request_time.format("%Y%m%d").to_string();

    normalize_action_header_names(headers);
    strip_action_signing_headers(headers);
    headers
        .entry("content-type".to_string())
        .or_insert_with(|| "application/json".to_string());
    headers.insert("host".to_string(), host);
    headers.insert("x-date".to_string(), x_date.clone());
    headers.insert("x-content-sha256".to_string(), payload_hash.clone());
    if let Some(token) = credentials
        .security_token
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        headers.insert("x-security-token".to_string(), token.to_string());
    }

    let (canonical_headers, signed_headers) = canonical_signing_headers(headers);
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method.trim().to_ascii_uppercase(),
        canonical_uri(&url),
        canonical_query(&url),
        canonical_headers,
        signed_headers,
        payload_hash,
    );
    let credential_scope = format!(
        "{}/{}/{}/{}",
        short_date,
        credentials.region.trim(),
        credentials.service.trim(),
        VOLC_AUTH_TERMINATOR
    );
    let string_to_sign = format!(
        "{}\n{}\n{}\n{}",
        VOLC_AUTH_ALGORITHM,
        x_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );
    let k_date = hmac_sha256(
        credentials.secret_access_key.trim().as_bytes(),
        short_date.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, credentials.region.trim().as_bytes());
    let k_service = hmac_sha256(&k_region, credentials.service.trim().as_bytes());
    let k_signing = hmac_sha256(&k_service, VOLC_AUTH_TERMINATOR.as_bytes());
    let signature = hex_lower(&hmac_sha256(&k_signing, string_to_sign.as_bytes()));
    let authorization = format!(
        "{} Credential={}/{}, SignedHeaders={}, Signature={}",
        VOLC_AUTH_ALGORITHM,
        credentials.access_key_id.trim(),
        credential_scope,
        signed_headers,
        signature,
    );
    headers.insert("authorization".to_string(), authorization.clone());

    Ok(VolcActionSignature {
        authorization,
        payload_hash,
        signed_headers,
        canonical_request,
        string_to_sign,
    })
}

pub fn volc_action_authorization_access_key_id(
    headers: &http::HeaderMap,
) -> Result<Option<String>, VolcActionVerificationError> {
    let Some(value) = headers.get(http::header::AUTHORIZATION) else {
        return Ok(None);
    };
    let value = value
        .to_str()
        .map_err(|_| VolcActionVerificationError::MalformedAuthorization)?
        .trim();
    if !value.starts_with(VOLC_AUTH_ALGORITHM) {
        return Ok(None);
    }
    Ok(Some(parse_volc_authorization(value)?.access_key_id))
}

pub fn verify_volc_action_signature(
    input: VolcActionVerificationInput<'_>,
) -> Result<VerifiedVolcActionSignature, VolcActionVerificationError> {
    let authorization = input
        .headers
        .get(http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or(VolcActionVerificationError::MissingAuthorization)?;
    let parsed = parse_volc_authorization(authorization.trim())?;
    if parsed.access_key_id != input.expected_access_key_id.trim()
        || parsed.region != input.expected_region.trim()
        || parsed.service != input.expected_service.trim()
    {
        return Err(VolcActionVerificationError::InvalidCredentialScope);
    }

    let x_date = signed_header_value(input.headers, "x-date")?;
    let signed_at = NaiveDateTime::parse_from_str(&x_date, "%Y%m%dT%H%M%SZ")
        .map(|value| value.and_utc())
        .map_err(|_| VolcActionVerificationError::InvalidRequestTime)?;
    if parsed.short_date != signed_at.format("%Y%m%d").to_string()
        || input.max_clock_skew_secs < 0
        || input
            .now
            .signed_duration_since(signed_at)
            .num_seconds()
            .unsigned_abs()
            > input.max_clock_skew_secs as u64
    {
        return Err(VolcActionVerificationError::InvalidRequestTime);
    }

    let actual_payload_hash = Sha256::digest(input.body);
    let supplied_payload_hash =
        decode_hex_32(&signed_header_value(input.headers, "x-content-sha256")?).ok_or(
            VolcActionVerificationError::InvalidSignedHeader("x-content-sha256".to_string()),
        )?;
    if !constant_time_eq_32(actual_payload_hash.as_ref(), &supplied_payload_hash) {
        return Err(VolcActionVerificationError::PayloadHashMismatch);
    }

    let url = Url::parse(input.url).map_err(|_| VolcActionVerificationError::InvalidRequestUrl)?;
    let canonical_headers = parsed
        .signed_headers
        .iter()
        .map(|name| {
            signed_header_value(input.headers, name)
                .map(|value| format!("{name}:{}\n", normalize_header_value(&value)))
        })
        .collect::<Result<String, _>>()?;
    let signed_headers = parsed.signed_headers.join(";");
    let payload_hash = hex_lower(actual_payload_hash.as_ref());
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        input.method.trim().to_ascii_uppercase(),
        canonical_uri(&url),
        canonical_query(&url),
        canonical_headers,
        signed_headers,
        payload_hash,
    );
    let credential_scope = format!(
        "{}/{}/{}/{}",
        parsed.short_date, parsed.region, parsed.service, VOLC_AUTH_TERMINATOR
    );
    let string_to_sign = format!(
        "{}\n{}\n{}\n{}",
        VOLC_AUTH_ALGORITHM,
        x_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );
    let k_date = hmac_sha256(
        input.secret_access_key.trim().as_bytes(),
        parsed.short_date.as_bytes(),
    );
    let k_region = hmac_sha256(&k_date, parsed.region.as_bytes());
    let k_service = hmac_sha256(&k_region, parsed.service.as_bytes());
    let k_signing = hmac_sha256(&k_service, VOLC_AUTH_TERMINATOR.as_bytes());
    let mut mac = HmacSha256::new_from_slice(&k_signing).expect("HMAC accepts any key length");
    mac.update(string_to_sign.as_bytes());
    mac.verify_slice(&parsed.signature)
        .map_err(|_| VolcActionVerificationError::InvalidSignature)?;

    Ok(VerifiedVolcActionSignature {
        access_key_id: parsed.access_key_id,
        region: parsed.region,
        service: parsed.service,
        signed_at,
    })
}

fn parse_volc_authorization(
    value: &str,
) -> Result<ParsedVolcAuthorization, VolcActionVerificationError> {
    let fields = value
        .strip_prefix(VOLC_AUTH_ALGORITHM)
        .filter(|rest| rest.starts_with(char::is_whitespace))
        .ok_or(VolcActionVerificationError::MalformedAuthorization)?
        .trim()
        .split(',')
        .map(str::trim)
        .map(|field| {
            field
                .split_once('=')
                .ok_or(VolcActionVerificationError::MalformedAuthorization)
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if fields.len() != 3 {
        return Err(VolcActionVerificationError::MalformedAuthorization);
    }
    let credential = fields
        .get("Credential")
        .ok_or(VolcActionVerificationError::MalformedAuthorization)?;
    let credential = credential.split('/').collect::<Vec<_>>();
    if credential.len() != 5
        || credential.iter().any(|value| value.trim().is_empty())
        || credential[4] != VOLC_AUTH_TERMINATOR
        || credential[1].len() != 8
        || !credential[1].bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(VolcActionVerificationError::InvalidCredentialScope);
    }
    let signed_headers = fields
        .get("SignedHeaders")
        .ok_or(VolcActionVerificationError::MalformedAuthorization)?
        .split(';')
        .map(str::trim)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut sorted_headers = signed_headers.clone();
    sorted_headers.sort();
    sorted_headers.dedup();
    if signed_headers != sorted_headers
        || signed_headers.iter().any(|name| {
            name.is_empty()
                || name != &name.to_ascii_lowercase()
                || http::HeaderName::from_bytes(name.as_bytes()).is_err()
        })
        || ["content-type", "host", "x-content-sha256", "x-date"]
            .iter()
            .any(|required| !signed_headers.iter().any(|name| name == required))
    {
        return Err(VolcActionVerificationError::InvalidSignedHeaders);
    }
    let signature = decode_hex_32(
        fields
            .get("Signature")
            .ok_or(VolcActionVerificationError::MalformedAuthorization)?,
    )
    .ok_or(VolcActionVerificationError::MalformedAuthorization)?;
    Ok(ParsedVolcAuthorization {
        access_key_id: credential[0].to_string(),
        short_date: credential[1].to_string(),
        region: credential[2].to_string(),
        service: credential[3].to_string(),
        signed_headers,
        signature,
    })
}

fn signed_header_value(
    headers: &http::HeaderMap,
    name: &str,
) -> Result<String, VolcActionVerificationError> {
    let values = headers
        .get_all(name)
        .iter()
        .map(|value| {
            value
                .to_str()
                .map(str::trim)
                .ok()
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .ok_or_else(|| VolcActionVerificationError::InvalidSignedHeader(name.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(VolcActionVerificationError::InvalidSignedHeader(
            name.to_string(),
        ));
    }
    Ok(values.join(","))
}

fn decode_hex_32(value: &str) -> Option<[u8; 32]> {
    let value = value.trim().as_bytes();
    if value.len() != 64 {
        return None;
    }
    let mut decoded = [0_u8; 32];
    for (index, pair) in value.chunks_exact(2).enumerate() {
        decoded[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Some(decoded)
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn constant_time_eq_32(left: &[u8], right: &[u8; 32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .fold(0_u8, |diff, (left, right)| diff | (left ^ right))
        == 0
}

fn normalize_action(action: &str) -> Result<&str, VolcActionTransportError> {
    let action = action.trim();
    let valid = !action.is_empty()
        && action.len() <= 128
        && action
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    valid
        .then_some(action)
        .ok_or(VolcActionTransportError::InvalidAction)
}

fn resolved_action_auth_type(transport: &GatewayProviderTransportSnapshot) -> String {
    let default = transport.key.auth_type.trim().to_ascii_lowercase();
    transport
        .key
        .auth_type_by_format
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|overrides| {
            overrides
                .get(ARK_ASSET_API_FORMAT)
                .or_else(|| overrides.get(transport.endpoint.api_format.trim()))
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or(default)
}

fn resolve_raw_secret(transport: &GatewayProviderTransportSnapshot) -> Option<&str> {
    let secret = transport.key.decrypted_api_key.trim();
    (!secret.is_empty() && secret != PLACEHOLDER_API_KEY).then_some(secret)
}

fn parse_volc_aksk_credentials(
    transport: &GatewayProviderTransportSnapshot,
) -> Option<VolcAkSkCredentials> {
    let raw = transport.key.decrypted_auth_config.as_deref()?.trim();
    let parsed: Value = serde_json::from_str(raw).ok()?;
    let root = parsed.as_object()?;
    let object = ["volc_aksk", "credentials", "credential"]
        .iter()
        .find_map(|key| root.get(*key).and_then(Value::as_object))
        .unwrap_or(root);
    let access_key_id = json_string_alias(
        object,
        &[
            "access_key_id",
            "access_key",
            "accessKeyId",
            "accessKey",
            "ak",
        ],
    )?;
    let secret_access_key = json_string_alias(
        object,
        &[
            "secret_access_key",
            "secret_key",
            "secretAccessKey",
            "secretKey",
            "sk",
        ],
    )?;
    let security_token = json_string_alias(
        object,
        &[
            "security_token",
            "session_token",
            "securityToken",
            "sessionToken",
        ],
    );
    let region = json_string_alias(object, &["region"])
        .unwrap_or_else(|| VOLC_ACTION_DEFAULT_REGION.to_string());
    let service = json_string_alias(object, &["service"])
        .unwrap_or_else(|| VOLC_ACTION_DEFAULT_SERVICE.to_string());
    Some(VolcAkSkCredentials {
        access_key_id,
        secret_access_key,
        security_token,
        region,
        service,
    })
}

fn json_string_alias(object: &serde_json::Map<String, Value>, aliases: &[&str]) -> Option<String> {
    aliases
        .iter()
        .find_map(|key| object.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn resolve_relay_api_key_header(transport: &GatewayProviderTransportSnapshot) -> String {
    for source in [
        transport.key.upstream_metadata.as_ref(),
        transport.endpoint.config.as_ref(),
        transport.provider.config.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        let Some(object) = source.as_object() else {
            continue;
        };
        let configured = ["api_key_header", "apiKeyHeader"]
            .iter()
            .find_map(|key| object.get(*key))
            .and_then(Value::as_str)
            .map(str::trim)
            .map(str::to_ascii_lowercase);
        if configured
            .as_deref()
            .is_some_and(|value| matches!(value, "api-key" | "x-api-key"))
        {
            return configured.unwrap_or_default();
        }
    }
    "x-api-key".to_string()
}

fn strip_action_credential_headers(headers: &mut BTreeMap<String, String>) {
    headers.retain(|name, _| {
        !PROTECTED_ACTION_HEADERS
            .iter()
            .any(|blocked| name.eq_ignore_ascii_case(blocked))
    });
    headers.insert("content-type".to_string(), "application/json".to_string());
}

fn strip_action_runtime_internal_headers(headers: &mut BTreeMap<String, String>) {
    headers.retain(|name, _| !is_aether_internal_header(name));
}

fn strip_action_signing_headers(headers: &mut BTreeMap<String, String>) {
    headers.retain(|name, _| {
        ![
            "authorization",
            "api-key",
            "x-api-key",
            "host",
            "x-content-sha256",
            "x-date",
            "x-security-token",
        ]
        .iter()
        .any(|blocked| name.eq_ignore_ascii_case(blocked))
    });
}

fn normalize_action_header_names(headers: &mut BTreeMap<String, String>) {
    let original = std::mem::take(headers);
    for (name, value) in original {
        let name = name.trim().to_ascii_lowercase();
        let value = value.trim();
        if name.is_empty() || value.is_empty() {
            continue;
        }
        headers.insert(name, value.to_string());
    }
}

fn canonical_host(url: &Url) -> Option<String> {
    let host = url.host_str()?;
    match url.port() {
        Some(80) if url.scheme() == "http" => Some(host.to_string()),
        Some(443) if url.scheme() == "https" => Some(host.to_string()),
        Some(port) => Some(format!("{host}:{port}")),
        _ => Some(host.to_string()),
    }
}

fn canonical_uri(url: &Url) -> String {
    let path = url.path().as_bytes();
    if path.is_empty() {
        "/".to_string()
    } else {
        rfc3986_encode(path, true)
    }
}

fn canonical_query(url: &Url) -> String {
    let mut pairs = url
        .query_pairs()
        .map(|(key, value)| {
            (
                rfc3986_encode(key.as_bytes(), false),
                rfc3986_encode(value.as_bytes(), false),
            )
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn canonical_signing_headers(headers: &BTreeMap<String, String>) -> (String, String) {
    let mut selected = BTreeMap::new();
    for (name, value) in headers {
        let name = name.trim().to_ascii_lowercase();
        if !matches!(name.as_str(), "content-type" | "content-md5" | "host")
            && !name.starts_with("x-")
        {
            continue;
        }
        selected.insert(name, normalize_header_value(value));
    }
    let signed_headers = selected.keys().cloned().collect::<Vec<_>>().join(";");
    let canonical_headers = selected
        .into_iter()
        .map(|(name, value)| format!("{name}:{value}\n"))
        .collect::<String>();
    (canonical_headers, signed_headers)
}

fn normalize_header_value(value: &str) -> String {
    value.split_ascii_whitespace().collect::<Vec<_>>().join(" ")
}

fn rfc3986_encode(value: &[u8], preserve_slash: bool) -> String {
    let mut encoded = String::with_capacity(value.len());
    for byte in value {
        if byte.is_ascii_alphanumeric()
            || matches!(byte, b'-' | b'_' | b'.' | b'~')
            || (preserve_slash && *byte == b'/')
        {
            encoded.push(*byte as char);
        } else {
            encoded.push('%');
            encoded.push(HEX[(byte >> 4) as usize] as char);
            encoded.push(HEX[(byte & 0x0f) as usize] as char);
        }
    }
    encoded
}

const HEX: &[u8; 16] = b"0123456789ABCDEF";

fn sha256_hex(value: &[u8]) -> String {
    hex_lower(&Sha256::digest(value))
}

fn hmac_sha256(key: &[u8], value: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(value);
    mac.finalize().into_bytes().to_vec()
}

fn hex_lower(value: &[u8]) -> String {
    let mut encoded = String::with_capacity(value.len() * 2);
    for byte in value {
        encoded.push((HEX[(byte >> 4) as usize] as char).to_ascii_lowercase());
        encoded.push((HEX[(byte & 0x0f) as usize] as char).to_ascii_lowercase());
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{
        GatewayProviderTransportEndpoint, GatewayProviderTransportKey,
        GatewayProviderTransportProvider,
    };
    use chrono::TimeZone;
    use serde_json::json;

    fn sample_transport(auth_type: &str) -> GatewayProviderTransportSnapshot {
        GatewayProviderTransportSnapshot {
            provider: GatewayProviderTransportProvider {
                id: "provider-1".to_string(),
                name: "Ark".to_string(),
                provider_type: "volcengine".to_string(),
                website: None,
                is_active: true,
                keep_priority_on_conversion: false,
                enable_format_conversion: false,
                concurrent_limit: None,
                max_retries: None,
                proxy: None,
                request_timeout_secs: None,
                stream_first_byte_timeout_secs: None,
                config: None,
            },
            endpoint: GatewayProviderTransportEndpoint {
                id: "endpoint-1".to_string(),
                provider_id: "provider-1".to_string(),
                api_format: ARK_ASSET_API_FORMAT.to_string(),
                api_family: Some("doubao".to_string()),
                endpoint_kind: Some("asset_library".to_string()),
                is_active: true,
                base_url: VOLC_ACTION_DEFAULT_BASE_URL.to_string(),
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
                api_formats: Some(vec![ARK_ASSET_API_FORMAT.to_string()]),
                auth_type_by_format: None,
                allow_auth_channel_mismatch_formats: None,
                allowed_models: None,
                capabilities: Some(json!([ARK_ASSET_REQUIRED_CAPABILITY])),
                rate_multipliers: None,
                global_priority_by_format: None,
                expires_at_unix_secs: None,
                proxy: None,
                fingerprint: None,
                upstream_metadata: None,
                decrypted_api_key: String::new(),
                decrypted_auth_config: None,
            },
        }
    }

    #[test]
    fn matches_reproducible_volc_signv4_golden_vector() {
        // Fixed from Volcengine's public SignV4 construction and independently
        // checked with Python's hashlib/hmac implementation.
        let credentials = VolcAkSkCredentials {
            access_key_id: "AKLTEXAMPLE".to_string(),
            secret_access_key: "secretEXAMPLE".to_string(),
            security_token: None,
            region: VOLC_ACTION_DEFAULT_REGION.to_string(),
            service: VOLC_ACTION_DEFAULT_SERVICE.to_string(),
        };
        let body = br#"{"AssetId":"asset-1","Name":"A B"}"#;
        let mut headers =
            BTreeMap::from([("content-type".to_string(), "application/json".to_string())]);
        let signed = apply_volc_action_signature(
            &mut headers,
            "POST",
            "https://ark.cn-beijing.volcengineapi.com/?Action=ListAssets&Version=2024-01-01&PageNumber=1",
            body,
            &credentials,
            Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap(),
        )
        .expect("request should sign");

        assert_eq!(
            signed.payload_hash,
            "e280d025e397a7b39c92c34ac0621014f9e24477a1d220e4121aa5486d850a73"
        );
        assert_eq!(
            signed.signed_headers,
            "content-type;host;x-content-sha256;x-date"
        );
        assert_eq!(
            sha256_hex(signed.canonical_request.as_bytes()),
            "48c234c5eeb387f0eef410fcd5674b69ed2755b91bacbe9332d001ba1a643275"
        );
        assert_eq!(
            signed.string_to_sign,
            concat!(
                "HMAC-SHA256\n",
                "20240102T030405Z\n",
                "20240102/cn-beijing/ark/request\n",
                "48c234c5eeb387f0eef410fcd5674b69ed2755b91bacbe9332d001ba1a643275"
            )
        );
        assert!(signed.authorization.ends_with(
            "Signature=43984fbc60e0ce6c51ba6c5fe5557ae621d9171bbdd96b28d4c5259f5babe731"
        ));
        assert_eq!(
            signed.canonical_request,
            concat!(
                "POST\n",
                "/\n",
                "Action=ListAssets&PageNumber=1&Version=2024-01-01\n",
                "content-type:application/json\n",
                "host:ark.cn-beijing.volcengineapi.com\n",
                "x-content-sha256:e280d025e397a7b39c92c34ac0621014f9e24477a1d220e4121aa5486d850a73\n",
                "x-date:20240102T030405Z\n",
                "\n",
                "content-type;host;x-content-sha256;x-date\n",
                "e280d025e397a7b39c92c34ac0621014f9e24477a1d220e4121aa5486d850a73"
            )
        );

        let url = "https://ark.cn-beijing.volcengineapi.com/?Action=ListAssets&Version=2024-01-01&PageNumber=1";
        let request_headers = headers
            .iter()
            .map(|(name, value)| {
                (
                    http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    http::HeaderValue::from_str(value).unwrap(),
                )
            })
            .collect::<http::HeaderMap>();
        let verified = verify_volc_action_signature(VolcActionVerificationInput {
            method: "POST",
            url,
            headers: &request_headers,
            body,
            secret_access_key: &credentials.secret_access_key,
            expected_access_key_id: &credentials.access_key_id,
            expected_region: VOLC_ACTION_DEFAULT_REGION,
            expected_service: VOLC_ACTION_DEFAULT_SERVICE,
            now: Utc.with_ymd_and_hms(2024, 1, 2, 3, 10, 0).unwrap(),
            max_clock_skew_secs: 15 * 60,
        })
        .expect("golden request should verify");
        assert_eq!(verified.access_key_id, "AKLTEXAMPLE");
    }

    #[test]
    fn verifier_recomputes_payload_hash_from_actual_body() {
        let credentials = VolcAkSkCredentials {
            access_key_id: "AKLTEXAMPLE".to_string(),
            secret_access_key: "secretEXAMPLE".to_string(),
            security_token: None,
            region: VOLC_ACTION_DEFAULT_REGION.to_string(),
            service: VOLC_ACTION_DEFAULT_SERVICE.to_string(),
        };
        let signed_body = br#"{"AssetId":"asset-1"}"#;
        let mut headers = BTreeMap::new();
        let signed_at = Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap();
        let url = "https://ark.cn-beijing.volcengineapi.com/?Action=GetAsset&Version=2024-01-01";
        apply_volc_action_signature(
            &mut headers,
            "POST",
            url,
            signed_body,
            &credentials,
            signed_at,
        )
        .unwrap();
        let request_headers = headers
            .iter()
            .map(|(name, value)| {
                (
                    http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    http::HeaderValue::from_str(value).unwrap(),
                )
            })
            .collect::<http::HeaderMap>();

        let error = verify_volc_action_signature(VolcActionVerificationInput {
            method: "POST",
            url,
            headers: &request_headers,
            body: br#"{"AssetId":"asset-2"}"#,
            secret_access_key: &credentials.secret_access_key,
            expected_access_key_id: &credentials.access_key_id,
            expected_region: VOLC_ACTION_DEFAULT_REGION,
            expected_service: VOLC_ACTION_DEFAULT_SERVICE,
            now: signed_at,
            max_clock_skew_secs: 15 * 60,
        })
        .unwrap_err();
        assert_eq!(error, VolcActionVerificationError::PayloadHashMismatch);
    }

    #[test]
    fn action_url_overrides_client_credentials_and_action_fields() {
        let url = build_volc_action_url(
            "https://relay.example.test/api?tenant=one&Action=Wrong&key=client",
            Some("/ark?tenant=two&Version=old"),
            "CreateAsset",
        )
        .expect("url should build");
        let parsed = Url::parse(&url).expect("url should parse");
        let query = parsed.query_pairs().collect::<BTreeMap<_, _>>();

        assert_eq!(parsed.path(), "/ark");
        assert_eq!(query.get("tenant").map(|value| value.as_ref()), Some("two"));
        assert_eq!(
            query.get("Action").map(|value| value.as_ref()),
            Some("CreateAsset")
        );
        assert_eq!(
            query.get("Version").map(|value| value.as_ref()),
            Some(VOLC_ACTION_VERSION)
        );
        assert!(!query.contains_key("key"));
    }

    #[test]
    fn action_url_accepts_action_names_with_leading_underscore() {
        let url = build_volc_action_url(VOLC_ACTION_DEFAULT_BASE_URL, None, "_RelayAction")
            .expect("URL should build");
        assert!(url.contains("Action=_RelayAction"));
    }

    #[test]
    fn action_url_rejects_remote_plain_http() {
        for base_url in [
            "http://relay.example.test",
            "http://192.168.1.10:8080",
            "http://localhost.example.test",
        ] {
            assert_eq!(
                build_volc_action_url(base_url, None, "ListAssets"),
                Err(VolcActionTransportError::InvalidBaseUrl),
                "plain HTTP must be rejected for {base_url}"
            );
        }
    }

    #[test]
    fn action_url_accepts_remote_https() {
        let url = build_volc_action_url("https://relay.example.test:8443/ark", None, "ListAssets")
            .expect("remote HTTPS should be accepted");

        assert!(url.starts_with("https://relay.example.test:8443/ark?"));
    }

    #[test]
    fn action_url_allows_plain_http_only_for_loopback_hosts() {
        for base_url in [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://127.42.0.9:8080",
            "http://[::1]:8080",
        ] {
            let url = build_volc_action_url(base_url, None, "ListAssets")
                .unwrap_or_else(|error| panic!("loopback HTTP should be accepted: {error}"));
            assert!(url.starts_with(base_url));
        }
    }

    #[test]
    fn builds_bearer_relay_request_after_rules_and_strips_client_auth() {
        let mut transport = sample_transport("bearer");
        transport.key.decrypted_api_key = "relay-token".to_string();
        transport.endpoint.body_rules = Some(json!([
            {"action":"set","path":"ProjectName","value":"project-1"}
        ]));
        transport.endpoint.header_rules = Some(json!([
            {"action":"set","key":"x-relay-tenant","value":"tenant-1"},
            {"action":"set","key":"authorization","value":"Bearer wrong"}
        ]));
        let mut incoming = http::HeaderMap::new();
        incoming.insert(
            "authorization",
            http::HeaderValue::from_static("Bearer client-token"),
        );
        incoming.insert("x-date", http::HeaderValue::from_static("client-date"));
        incoming.insert(
            "x-content-sha256",
            http::HeaderValue::from_static("client-payload-hash"),
        );

        let request = build_volc_action_request(VolcActionRequestInput {
            transport: &transport,
            action: "CreateAssetGroup",
            body: &json!({"Name":"group"}),
            request_headers: &incoming,
            request_time: None,
        })
        .expect("request should build");

        assert_eq!(request.body_json["ProjectName"], "project-1");
        assert_eq!(
            request.headers.get("authorization").map(String::as_str),
            Some("Bearer relay-token")
        );
        assert_eq!(
            request.headers.get("x-relay-tenant").map(String::as_str),
            Some("tenant-1")
        );
        assert!(!request.headers.contains_key("x-date"));
        assert!(!request.headers.contains_key("x-content-sha256"));
        assert!(!request.body.is_empty());
    }

    #[test]
    fn supports_configurable_api_key_relay_header() {
        let mut transport = sample_transport("api_key");
        transport.key.decrypted_api_key = "relay-key".to_string();
        transport.key.upstream_metadata = Some(json!({"api_key_header":"api-key"}));

        let request = build_volc_action_request(VolcActionRequestInput {
            transport: &transport,
            action: "ListAssets",
            body: &json!({}),
            request_headers: &http::HeaderMap::new(),
            request_time: None,
        })
        .expect("request should build");

        assert_eq!(
            request.headers.get("api-key").map(String::as_str),
            Some("relay-key")
        );
        assert!(!request.headers.contains_key("x-api-key"));
    }

    #[test]
    fn signs_the_rule_mutated_aksk_body_and_headers() {
        let mut transport = sample_transport("volc_aksk");
        transport.key.decrypted_auth_config = Some(
            json!({
                "access_key_id": "AKLTEXAMPLE",
                "secret_access_key": "secretEXAMPLE"
            })
            .to_string(),
        );
        transport.endpoint.body_rules = Some(json!([
            {"action":"set","path":"ProjectName","value":"project-1"}
        ]));
        transport.endpoint.header_rules = Some(json!([
            {"action":"set","key":"x-relay-tenant","value":"tenant-1"},
            {"action":"set","key":"X-Aether-Execution-Http1-Only","value":"true"}
        ]));

        let request = build_volc_action_request(VolcActionRequestInput {
            transport: &transport,
            action: "CreateAsset",
            body: &json!({"Name":"asset"}),
            request_headers: &http::HeaderMap::new(),
            request_time: Some(Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap()),
        })
        .expect("request should build");

        assert_eq!(request.body_json["ProjectName"], "project-1");
        let expected_payload_hash = sha256_hex(&request.body);
        assert_eq!(
            request.headers.get("x-content-sha256").map(String::as_str),
            Some(expected_payload_hash.as_str())
        );
        assert_eq!(
            request.headers.get("x-relay-tenant").map(String::as_str),
            Some("tenant-1")
        );
        let authorization = request
            .headers
            .get("authorization")
            .expect("signed request should include authorization");
        assert!(authorization
            .contains("SignedHeaders=content-type;host;x-content-sha256;x-date;x-relay-tenant"));
        assert!(!authorization.contains("x-aether-"));
        assert!(!request
            .headers
            .keys()
            .any(|name| name.starts_with("x-aether-")));
    }

    #[test]
    fn resolves_nested_aksk_config_and_redacts_debug_output() {
        let mut transport = sample_transport("volc_aksk");
        transport.key.decrypted_auth_config = Some(
            json!({
                "volc_aksk": {
                    "accessKeyId": "AKLT-NOT-FOR-LOGS",
                    "secretAccessKey": "secret-not-for-logs",
                    "sessionToken": "session-not-for-logs"
                }
            })
            .to_string(),
        );

        let auth = resolve_volc_action_auth(&transport).expect("AK/SK should resolve");
        let debug = format!("{auth:?}");
        assert!(!debug.contains("AKLT-NOT-FOR-LOGS"));
        assert!(!debug.contains("secret-not-for-logs"));
        assert!(!debug.contains("session-not-for-logs"));
        let VolcActionAuth::AkSk(credentials) = auth else {
            panic!("expected AK/SK");
        };
        assert_eq!(credentials.region, VOLC_ACTION_DEFAULT_REGION);
        assert_eq!(credentials.service, VOLC_ACTION_DEFAULT_SERVICE);
    }

    #[test]
    fn normalizes_supported_action_auth_aliases_only() {
        for (input, expected) in [
            ("VOLC-AKSK", "volc_aksk"),
            ("aksk", "volc_aksk"),
            ("hmac_sha256", "volc_aksk"),
            ("API-Key", "api_key"),
            ("bearer_token", "bearer"),
        ] {
            assert_eq!(
                normalize_volc_action_auth_type(input).as_deref(),
                Some(expected)
            );
        }
        assert_eq!(normalize_volc_action_auth_type("oauth"), None);
        assert_eq!(normalize_volc_action_auth_type("service_account"), None);
    }

    #[test]
    fn signed_request_includes_security_token_in_signed_headers() {
        let credentials = VolcAkSkCredentials {
            access_key_id: "ak".to_string(),
            secret_access_key: "sk".to_string(),
            security_token: Some("session".to_string()),
            region: VOLC_ACTION_DEFAULT_REGION.to_string(),
            service: VOLC_ACTION_DEFAULT_SERVICE.to_string(),
        };
        let mut headers = BTreeMap::new();
        let signature = apply_volc_action_signature(
            &mut headers,
            "POST",
            "https://ark.cn-beijing.volcengineapi.com/?Action=ListAssets&Version=2024-01-01",
            b"{}",
            &credentials,
            Utc.with_ymd_and_hms(2024, 1, 2, 3, 4, 5).unwrap(),
        )
        .expect("request should sign");

        assert!(signature.signed_headers.contains("x-security-token"));
        assert_eq!(
            headers.get("x-security-token").map(String::as_str),
            Some("session")
        );
    }
}
