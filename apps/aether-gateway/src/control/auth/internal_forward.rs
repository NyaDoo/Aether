use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Uri};
use base64::Engine as _;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use uuid::Uuid;

use crate::constants::{
    GATEWAY_HEADER, INTERNAL_FORWARD_AUTH_NONCE_HEADER, INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER,
    INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER, TRUSTED_ADMIN_MANAGEMENT_TOKEN_ID_HEADER,
    TRUSTED_ADMIN_SESSION_ID_HEADER, TRUSTED_ADMIN_USER_ID_HEADER, TRUSTED_ADMIN_USER_ROLE_HEADER,
    TRUSTED_AUTH_ACCESS_ALLOWED_HEADER, TRUSTED_AUTH_API_KEY_ID_HEADER,
    TRUSTED_AUTH_BALANCE_HEADER, TRUSTED_AUTH_USER_ID_HEADER, TUNNEL_AFFINITY_FORWARDED_BY_HEADER,
    TUNNEL_AFFINITY_OWNER_INSTANCE_HEADER,
};
use crate::headers::header_value_str;

const INTERNAL_FORWARD_MARKER: &str = "rust-phase3b-affinity";
const INTERNAL_FORWARD_AUTH_CONTEXT: &str = "aether-internal-auth-forward/v1";
const INTERNAL_FORWARD_ADMIN_MARKER: &str = "rust-phase3b-admin";
const INTERNAL_FORWARD_ADMIN_CONTEXT: &str = "aether-internal-admin-forward/v1";
const INTERNAL_FORWARD_AUTH_SECRET_ENV: &str = "AETHER_GATEWAY_INTERNAL_FORWARD_SECRET";
const JWT_SECRET_ENV: &str = "JWT_SECRET_KEY";
const MAX_PROOF_CLOCK_SKEW_SECS: u64 = 60;
const MAX_REPLAY_CACHE_ENTRIES: usize = 131_072;
const MIN_SECRET_BYTES: usize = 32;

#[cfg(test)]
const TEST_INTERNAL_FORWARD_SECRET: &str = "aether-internal-forward-test-secret-32-bytes-minimum";

type HmacSha256 = Hmac<Sha256>;

#[derive(Default)]
struct InternalForwardReplayCache {
    seen: HashMap<Uuid, u64>,
    expiry_order: VecDeque<(u64, Uuid)>,
}

static INTERNAL_FORWARD_REPLAY_CACHE: OnceLock<Mutex<InternalForwardReplayCache>> = OnceLock::new();

/// Adds a short-lived HMAC proof to the already-populated internal identity
/// headers. The proof is intentionally bound to the destination path and the
/// complete forwarded identity, so it cannot be moved to another user, key,
/// route, source gateway, or owner gateway.
pub(crate) fn sign_trusted_auth_forward_headers(
    headers: &mut HeaderMap,
    method: &Method,
    uri: &Uri,
) -> Result<(), &'static str> {
    sign_trusted_auth_forward_headers_at(headers, method, uri, current_unix_secs(), Uuid::new_v4())
}

fn sign_trusted_auth_forward_headers_at(
    headers: &mut HeaderMap,
    method: &Method,
    uri: &Uri,
    timestamp: u64,
    nonce: Uuid,
) -> Result<(), &'static str> {
    headers.remove(INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER);
    headers.remove(INTERNAL_FORWARD_AUTH_NONCE_HEADER);
    headers.remove(INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER);

    let timestamp = timestamp.to_string();
    let nonce = nonce.to_string();
    insert_header(headers, INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER, &timestamp)?;
    insert_header(headers, INTERNAL_FORWARD_AUTH_NONCE_HEADER, &nonce)?;

    let fields =
        signed_fields(headers, method, uri).ok_or("missing internal forwarding identity")?;
    let secret = internal_forward_secret().ok_or("internal forwarding secret is not configured")?;
    let signature = compute_signature(secret.as_bytes(), &fields)?;
    insert_header(headers, INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER, &signature)
}

/// Verifies the HMAC proof before any x-aether-auth-* header is allowed to
/// become a caller principal. Missing, malformed, expired, or tampered proofs
/// all fail closed.
pub(crate) fn verify_trusted_auth_forward_headers(
    headers: &HeaderMap,
    method: &Method,
    uri: &Uri,
) -> bool {
    if header_value_str(headers, GATEWAY_HEADER).as_deref() != Some(INTERNAL_FORWARD_MARKER) {
        return false;
    }
    verify_signed_fields(headers, signed_fields(headers, method, uri))
}

/// Signs an internally reconstructed administrator principal. Unlike the
/// historical `rust-phase3*` marker, this is cryptographic proof over the
/// destination and every privilege-bearing field.
pub(crate) fn sign_trusted_admin_forward_headers(
    headers: &mut HeaderMap,
    method: &Method,
    uri: &Uri,
) -> Result<(), &'static str> {
    headers.insert(
        HeaderName::from_static(GATEWAY_HEADER),
        HeaderValue::from_static(INTERNAL_FORWARD_ADMIN_MARKER),
    );
    sign_trusted_admin_forward_headers_at(headers, method, uri, current_unix_secs(), Uuid::new_v4())
}

fn sign_trusted_admin_forward_headers_at(
    headers: &mut HeaderMap,
    method: &Method,
    uri: &Uri,
    timestamp: u64,
    nonce: Uuid,
) -> Result<(), &'static str> {
    headers.remove(INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER);
    headers.remove(INTERNAL_FORWARD_AUTH_NONCE_HEADER);
    headers.remove(INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER);

    insert_header(
        headers,
        INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER,
        &timestamp.to_string(),
    )?;
    insert_header(
        headers,
        INTERNAL_FORWARD_AUTH_NONCE_HEADER,
        &nonce.to_string(),
    )?;

    let fields =
        signed_admin_fields(headers, method, uri).ok_or("missing internal admin identity")?;
    let secret = internal_forward_secret().ok_or("internal forwarding secret is not configured")?;
    let signature = compute_signature(secret.as_bytes(), &fields)?;
    insert_header(headers, INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER, &signature)
}

pub(crate) fn verify_trusted_admin_forward_headers(
    headers: &HeaderMap,
    method: &Method,
    uri: &Uri,
) -> bool {
    if header_value_str(headers, GATEWAY_HEADER).as_deref() != Some(INTERNAL_FORWARD_ADMIN_MARKER) {
        return false;
    }
    verify_signed_fields(headers, signed_admin_fields(headers, method, uri))
}

/// Admits a valid internal proof's nonce once per gateway process. Pure proof
/// verification remains repeatable because credential extraction intentionally
/// runs more than once inside one request; this admission hook is called once
/// at the `resolve_control_route` request boundary.
pub(crate) fn internal_forward_proof_is_replay(
    headers: &HeaderMap,
    method: &Method,
    uri: &Uri,
) -> bool {
    if !verify_trusted_auth_forward_headers(headers, method, uri)
        && !verify_trusted_admin_forward_headers(headers, method, uri)
    {
        return false;
    }

    let Some(nonce) = header_value_str(headers, INTERNAL_FORWARD_AUTH_NONCE_HEADER)
        .and_then(|value| Uuid::parse_str(&value).ok())
    else {
        return true;
    };
    let now = current_unix_secs();
    // A proof timestamp may be up to one skew window in the future and remain
    // valid for another skew window, so retain the nonce for two full windows.
    let valid_until = now.saturating_add(MAX_PROOF_CLOCK_SKEW_SECS.saturating_mul(2));
    let cache = INTERNAL_FORWARD_REPLAY_CACHE
        .get_or_init(|| Mutex::new(InternalForwardReplayCache::default()));
    let Ok(mut cache) = cache.lock() else {
        // A poisoned replay cache cannot safely establish freshness.
        return true;
    };

    while let Some((expires_at, expired_nonce)) = cache.expiry_order.front().copied() {
        if expires_at >= now {
            break;
        }
        cache.expiry_order.pop_front();
        if cache.seen.get(&expired_nonce).copied() == Some(expires_at) {
            cache.seen.remove(&expired_nonce);
        }
    }
    if cache.seen.contains_key(&nonce) {
        return true;
    }
    // Preserve replay safety under pressure: reject new trusted proofs rather
    // than evicting a still-valid nonce and making it replayable.
    if cache.seen.len() >= MAX_REPLAY_CACHE_ENTRIES {
        return true;
    }
    cache.seen.insert(nonce, valid_until);
    cache.expiry_order.push_back((valid_until, nonce));
    false
}

pub(crate) fn is_internal_forward_identity_header(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        GATEWAY_HEADER
            | TRUSTED_AUTH_USER_ID_HEADER
            | TRUSTED_AUTH_API_KEY_ID_HEADER
            | TRUSTED_AUTH_BALANCE_HEADER
            | TRUSTED_AUTH_ACCESS_ALLOWED_HEADER
            | TRUSTED_ADMIN_USER_ID_HEADER
            | TRUSTED_ADMIN_USER_ROLE_HEADER
            | TRUSTED_ADMIN_SESSION_ID_HEADER
            | TRUSTED_ADMIN_MANAGEMENT_TOKEN_ID_HEADER
            | TUNNEL_AFFINITY_FORWARDED_BY_HEADER
            | TUNNEL_AFFINITY_OWNER_INSTANCE_HEADER
            | INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER
            | INTERNAL_FORWARD_AUTH_NONCE_HEADER
            | INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER
    )
}

fn signed_admin_fields(headers: &HeaderMap, method: &Method, uri: &Uri) -> Option<Vec<String>> {
    let marker = header_value_str(headers, GATEWAY_HEADER)?;
    if marker != INTERNAL_FORWARD_ADMIN_MARKER {
        return None;
    }
    let user_id = header_value_str(headers, TRUSTED_ADMIN_USER_ID_HEADER)?;
    let user_role = header_value_str(headers, TRUSTED_ADMIN_USER_ROLE_HEADER)?;
    let session_id = header_value_str(headers, TRUSTED_ADMIN_SESSION_ID_HEADER).unwrap_or_default();
    let management_token_id =
        header_value_str(headers, TRUSTED_ADMIN_MANAGEMENT_TOKEN_ID_HEADER).unwrap_or_default();
    if user_id.trim().is_empty()
        || user_role.trim().is_empty()
        || (session_id.trim().is_empty() && management_token_id.trim().is_empty())
    {
        return None;
    }
    let timestamp = header_value_str(headers, INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER)?;
    let nonce = header_value_str(headers, INTERNAL_FORWARD_AUTH_NONCE_HEADER)?;
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");

    Some(vec![
        INTERNAL_FORWARD_ADMIN_CONTEXT.to_string(),
        marker,
        method.as_str().to_string(),
        path_and_query.to_string(),
        user_id,
        user_role,
        session_id,
        management_token_id,
        timestamp,
        nonce,
    ])
}

fn verify_signed_fields(headers: &HeaderMap, fields: Option<Vec<String>>) -> bool {
    let Some(timestamp_text) = header_value_str(headers, INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER)
    else {
        return false;
    };
    let Ok(timestamp) = timestamp_text.parse::<u64>() else {
        return false;
    };
    if timestamp.to_string() != timestamp_text
        || current_unix_secs().abs_diff(timestamp) > MAX_PROOF_CLOCK_SKEW_SECS
    {
        return false;
    }

    let Some(nonce) = header_value_str(headers, INTERNAL_FORWARD_AUTH_NONCE_HEADER) else {
        return false;
    };
    if Uuid::parse_str(&nonce).is_err() {
        return false;
    }

    let Some(signature) = header_value_str(headers, INTERNAL_FORWARD_AUTH_SIGNATURE_HEADER) else {
        return false;
    };
    let Ok(signature) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(signature) else {
        return false;
    };
    let Some(secret) = internal_forward_secret() else {
        return false;
    };
    let Some(fields) = fields else {
        return false;
    };
    let Ok(mut mac) = HmacSha256::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    update_mac(&mut mac, &fields);
    mac.verify_slice(&signature).is_ok()
}

fn signed_fields(headers: &HeaderMap, method: &Method, uri: &Uri) -> Option<Vec<String>> {
    let marker = header_value_str(headers, GATEWAY_HEADER)?;
    if marker != INTERNAL_FORWARD_MARKER {
        return None;
    }
    let user_id = header_value_str(headers, TRUSTED_AUTH_USER_ID_HEADER)?;
    let api_key_id = header_value_str(headers, TRUSTED_AUTH_API_KEY_ID_HEADER)?;
    let access_allowed = header_value_str(headers, TRUSTED_AUTH_ACCESS_ALLOWED_HEADER)?;
    if !matches!(access_allowed.as_str(), "true" | "false") {
        return None;
    }
    let forwarded_by = header_value_str(headers, TUNNEL_AFFINITY_FORWARDED_BY_HEADER)?;
    let owner_instance_id = header_value_str(headers, TUNNEL_AFFINITY_OWNER_INSTANCE_HEADER)?;
    if forwarded_by.trim().is_empty() || owner_instance_id.trim().is_empty() {
        return None;
    }
    let timestamp = header_value_str(headers, INTERNAL_FORWARD_AUTH_TIMESTAMP_HEADER)?;
    let nonce = header_value_str(headers, INTERNAL_FORWARD_AUTH_NONCE_HEADER)?;
    let path_and_query = uri
        .path_and_query()
        .map(|value| value.as_str())
        .unwrap_or("/");

    Some(vec![
        INTERNAL_FORWARD_AUTH_CONTEXT.to_string(),
        marker,
        method.as_str().to_string(),
        path_and_query.to_string(),
        user_id,
        api_key_id,
        header_value_str(headers, TRUSTED_AUTH_BALANCE_HEADER).unwrap_or_default(),
        access_allowed,
        forwarded_by,
        owner_instance_id,
        timestamp,
        nonce,
    ])
}

fn compute_signature(secret: &[u8], fields: &[String]) -> Result<String, &'static str> {
    let mut mac = HmacSha256::new_from_slice(secret).map_err(|_| "invalid internal secret")?;
    update_mac(&mut mac, fields);
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes()))
}

fn update_mac(mac: &mut HmacSha256, fields: &[String]) {
    for field in fields {
        let bytes = field.as_bytes();
        mac.update(&(bytes.len() as u64).to_be_bytes());
        mac.update(bytes);
    }
}

fn insert_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value: &str,
) -> Result<(), &'static str> {
    let value = HeaderValue::from_str(value).map_err(|_| "invalid internal forwarding header")?;
    headers.insert(HeaderName::from_static(name), value);
    Ok(())
}

fn internal_forward_secret() -> Option<String> {
    for name in [INTERNAL_FORWARD_AUTH_SECRET_ENV, JWT_SECRET_ENV] {
        if let Some(secret) = std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| secure_secret_candidate(value))
        {
            return Some(secret);
        }
    }

    #[cfg(test)]
    return Some(TEST_INTERNAL_FORWARD_SECRET.to_string());

    #[cfg(not(test))]
    None
}

fn secure_secret_candidate(value: &str) -> bool {
    value.as_bytes().len() >= MIN_SECRET_BYTES
        && !matches!(
            value,
            "aether-rust-dev-jwt-secret" | "change-this-to-a-secure-random-string"
        )
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signed_headers(method: &Method, uri: &Uri) -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert_header(&mut headers, GATEWAY_HEADER, INTERNAL_FORWARD_MARKER).unwrap();
        insert_header(&mut headers, TRUSTED_AUTH_USER_ID_HEADER, "user-1").unwrap();
        insert_header(&mut headers, TRUSTED_AUTH_API_KEY_ID_HEADER, "key-1").unwrap();
        insert_header(&mut headers, TRUSTED_AUTH_BALANCE_HEADER, "12.5").unwrap();
        insert_header(&mut headers, TRUSTED_AUTH_ACCESS_ALLOWED_HEADER, "true").unwrap();
        insert_header(
            &mut headers,
            TUNNEL_AFFINITY_FORWARDED_BY_HEADER,
            "gateway-a",
        )
        .unwrap();
        insert_header(
            &mut headers,
            TUNNEL_AFFINITY_OWNER_INSTANCE_HEADER,
            "gateway-b",
        )
        .unwrap();
        sign_trusted_auth_forward_headers(&mut headers, method, uri).unwrap();
        headers
    }

    fn signed_admin_headers(method: &Method, uri: &Uri) -> HeaderMap {
        let mut headers = HeaderMap::new();
        insert_header(&mut headers, TRUSTED_ADMIN_USER_ID_HEADER, "admin-user-1").unwrap();
        insert_header(&mut headers, TRUSTED_ADMIN_USER_ROLE_HEADER, "admin").unwrap();
        insert_header(&mut headers, TRUSTED_ADMIN_SESSION_ID_HEADER, "session-1").unwrap();
        insert_header(
            &mut headers,
            TRUSTED_ADMIN_MANAGEMENT_TOKEN_ID_HEADER,
            "management-token-1",
        )
        .unwrap();
        sign_trusted_admin_forward_headers(&mut headers, method, uri).unwrap();
        headers
    }

    #[test]
    fn signed_internal_forward_proof_round_trips() {
        let uri = "/v1/chat/completions?stream=false".parse().unwrap();
        let headers = signed_headers(&Method::POST, &uri);
        assert!(verify_trusted_auth_forward_headers(
            &headers,
            &Method::POST,
            &uri
        ));
    }

    #[test]
    fn proof_is_bound_to_path_and_identity() {
        let uri = "/v1/chat/completions?stream=false".parse().unwrap();
        let mut headers = signed_headers(&Method::POST, &uri);
        let other_uri = "/v1/chat/completions?stream=true".parse().unwrap();
        assert!(!verify_trusted_auth_forward_headers(
            &headers,
            &Method::POST,
            &other_uri
        ));

        insert_header(&mut headers, TRUSTED_AUTH_USER_ID_HEADER, "user-2").unwrap();
        assert!(!verify_trusted_auth_forward_headers(
            &headers,
            &Method::POST,
            &uri
        ));
    }

    #[test]
    fn signed_admin_proof_round_trips_and_binds_privileges() {
        let uri = "/api/admin/system/config".parse().unwrap();
        let mut headers = signed_admin_headers(&Method::GET, &uri);
        assert!(verify_trusted_admin_forward_headers(
            &headers,
            &Method::GET,
            &uri
        ));

        insert_header(&mut headers, TRUSTED_ADMIN_USER_ROLE_HEADER, "super_admin").unwrap();
        assert!(!verify_trusted_admin_forward_headers(
            &headers,
            &Method::GET,
            &uri
        ));
    }

    #[test]
    fn get_proof_cannot_be_replayed_as_delete() {
        let uri = "/api/v3/contents/generations/tasks/task-1".parse().unwrap();
        let headers = signed_headers(&Method::GET, &uri);

        assert!(!verify_trusted_auth_forward_headers(
            &headers,
            &Method::DELETE,
            &uri
        ));
    }

    #[test]
    fn affinity_proof_requires_nonempty_owner_and_forwarder() {
        let uri = "/v1/chat/completions".parse().unwrap();
        let mut headers = signed_headers(&Method::POST, &uri);
        insert_header(&mut headers, TUNNEL_AFFINITY_OWNER_INSTANCE_HEADER, " ").unwrap();
        assert!(sign_trusted_auth_forward_headers(&mut headers, &Method::POST, &uri).is_err());

        let mut headers = signed_headers(&Method::POST, &uri);
        insert_header(&mut headers, TUNNEL_AFFINITY_FORWARDED_BY_HEADER, " ").unwrap();
        assert!(sign_trusted_auth_forward_headers(&mut headers, &Method::POST, &uri).is_err());
    }

    #[test]
    fn valid_nonce_is_admitted_only_once_at_request_boundary() {
        let uri = "/api/v3/contents/generations/tasks/task-replay"
            .parse()
            .unwrap();
        let headers = signed_headers(&Method::GET, &uri);

        assert!(!internal_forward_proof_is_replay(
            &headers,
            &Method::GET,
            &uri
        ));
        assert!(internal_forward_proof_is_replay(
            &headers,
            &Method::GET,
            &uri
        ));
    }

    #[test]
    fn expired_proof_is_rejected() {
        let uri = "/v1/chat/completions".parse().unwrap();
        let mut headers = signed_headers(&Method::POST, &uri);
        sign_trusted_auth_forward_headers_at(
            &mut headers,
            &Method::POST,
            &uri,
            current_unix_secs().saturating_sub(MAX_PROOF_CLOCK_SKEW_SECS + 1),
            Uuid::new_v4(),
        )
        .unwrap();
        assert!(!verify_trusted_auth_forward_headers(
            &headers,
            &Method::POST,
            &uri
        ));
    }

    #[test]
    fn rejects_known_placeholder_secrets() {
        assert!(!secure_secret_candidate("aether-rust-dev-jwt-secret"));
        assert!(!secure_secret_candidate(
            "change-this-to-a-secure-random-string"
        ));
        assert!(!secure_secret_candidate("too-short"));
        assert!(secure_secret_candidate(
            "a-secure-internal-forward-secret-at-least-32-bytes"
        ));
    }
}
