use aether_provider_transport::{
    verify_volc_action_signature, volc_action_authorization_access_key_id,
    VolcActionVerificationInput, VOLC_ACTION_DEFAULT_REGION, VOLC_ACTION_DEFAULT_SERVICE,
};
use axum::body::Bytes;
use axum::http::{self, uri::Authority, HeaderMap, HeaderValue, Uri};

use crate::control::{GatewayControlAuthContext, GatewayPublicRequestContext};
use crate::handlers::shared::decrypt_catalog_secret_with_fallbacks;
use crate::{AppState, GatewayError};

const AETHER_AKSK_MAX_CLOCK_SKEW_SECS: i64 = 15 * 60;

pub(crate) fn authorization_uses_aether_aksk(headers: &HeaderMap) -> bool {
    matches!(
        volc_action_authorization_access_key_id(headers),
        Ok(Some(_))
    )
}

pub(crate) async fn resolve_aether_aksk_auth_context(
    state: &AppState,
    request_context: &GatewayPublicRequestContext,
    method: &http::Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: &Bytes,
) -> Result<Option<GatewayControlAuthContext>, GatewayError> {
    let Some(access_key_id) =
        volc_action_authorization_access_key_id(headers).map_err(invalid_signature)?
    else {
        return Ok(None);
    };
    if headers.contains_key("x-api-key") || headers.contains_key("api-key") {
        return Err(GatewayError::Client {
            status: http::StatusCode::BAD_REQUEST,
            message: "conflicting API credentials were supplied".to_string(),
        });
    }
    let decision = request_context.control_decision.as_ref().ok_or_else(|| {
        GatewayError::Internal("AK/SK request is missing its control decision".to_string())
    })?;
    if !route_accepts_aether_aksk(
        decision.route_class.as_deref(),
        decision.route_family.as_deref(),
        decision.route_kind.as_deref(),
    ) {
        return Err(GatewayError::Client {
            status: http::StatusCode::UNAUTHORIZED,
            message: "Aether AK/SK is only accepted by Ark material asset endpoints".to_string(),
        });
    }
    let signature = decision
        .auth_endpoint_signature
        .as_deref()
        .unwrap_or(super::ARK_ASSET_API_FORMAT);
    let snapshot = {
        let _permit = state.acquire_auth_snapshot_load_gate().await?;
        state
            .data
            .read_auth_api_key_snapshot_by_access_key_id_strong(
                &access_key_id,
                crate::clock::current_unix_secs(),
            )
            .await
            .map_err(|error| GatewayError::Internal(error.to_string()))?
    }
    .ok_or_else(|| GatewayError::Client {
        status: http::StatusCode::UNAUTHORIZED,
        message: "invalid Aether Access Key ID".to_string(),
    })?;
    let record = state
        .list_auth_api_key_export_records_by_ids(std::slice::from_ref(&snapshot.api_key_id))
        .await?
        .into_iter()
        .find(|record| {
            !record.is_standalone
                && record.api_key_id == snapshot.api_key_id
                && record.user_id == snapshot.user_id
                && record.credential_type == "volc_aksk"
                && record.access_key_id.as_deref() == Some(access_key_id.as_str())
        })
        .ok_or_else(|| GatewayError::Client {
            status: http::StatusCode::UNAUTHORIZED,
            message: "invalid Aether AK/SK credential".to_string(),
        })?;
    let ciphertext = record
        .key_encrypted
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| GatewayError::Internal("Aether AK/SK secret is unavailable".to_string()))?;
    let secret_access_key =
        decrypt_catalog_secret_with_fallbacks(state.encryption_key(), ciphertext).ok_or_else(
            || GatewayError::Internal("Aether AK/SK secret decrypt failed".to_string()),
        )?;
    let (verification_headers, host) = verification_headers_and_host(uri, headers)?;
    let url = format!(
        "https://{host}{}",
        uri.path_and_query()
            .map(|value| value.as_str())
            .unwrap_or("/")
    );
    verify_volc_action_signature(VolcActionVerificationInput {
        method: method.as_str(),
        url: &url,
        headers: &verification_headers,
        body,
        secret_access_key: &secret_access_key,
        expected_access_key_id: &access_key_id,
        expected_region: VOLC_ACTION_DEFAULT_REGION,
        expected_service: VOLC_ACTION_DEFAULT_SERVICE,
        now: chrono::Utc::now(),
        max_clock_skew_secs: AETHER_AKSK_MAX_CLOCK_SKEW_SECS,
    })
    .map_err(invalid_signature)?;

    crate::control::resolve_verified_api_key_snapshot_auth_context(state, snapshot, signature)
        .await
        .map(Some)
}

fn route_accepts_aether_aksk(
    route_class: Option<&str>,
    route_family: Option<&str>,
    route_kind: Option<&str>,
) -> bool {
    (route_class == Some("ai_public")
        && route_family == Some("doubao")
        && route_kind == Some("asset_library"))
        || (route_class == Some("public_support") && route_family == Some("material_assets"))
}

fn verification_headers_and_host(
    uri: &Uri,
    headers: &HeaderMap,
) -> Result<(HeaderMap, String), GatewayError> {
    let header_authority = host_header_authority(headers)?;
    let uri_authority = uri.authority();
    if let (Some(header_authority), Some(uri_authority)) =
        (header_authority.as_ref(), uri_authority)
    {
        if !authorities_match(header_authority, uri_authority) {
            return Err(invalid_host(
                "signed Host header does not match the HTTP request authority",
            ));
        }
    }

    let authority = header_authority
        .as_ref()
        .or(uri_authority)
        .ok_or_else(|| invalid_host("signed Host header or HTTP/2 authority is required"))?;
    let host = authority.as_str().to_string();
    let mut verification_headers = headers.clone();
    if header_authority.is_none() {
        let host_value = HeaderValue::from_str(&host)
            .map_err(|_| invalid_host("HTTP/2 authority cannot be represented as Host"))?;
        verification_headers.insert(http::header::HOST, host_value);
    }
    Ok((verification_headers, host))
}

fn host_header_authority(headers: &HeaderMap) -> Result<Option<Authority>, GatewayError> {
    let mut values = headers.get_all(http::header::HOST).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(invalid_host("multiple signed Host headers are not allowed"));
    }
    let value = value
        .to_str()
        .map(str::trim)
        .map_err(|_| invalid_host("signed Host header is invalid"))?;
    if value.is_empty() {
        return Err(invalid_host("signed Host header is empty"));
    }
    value
        .parse::<Authority>()
        .map(Some)
        .map_err(|_| invalid_host("signed Host header is invalid"))
}

fn authorities_match(left: &Authority, right: &Authority) -> bool {
    left.host().eq_ignore_ascii_case(right.host()) && left.port_u16() == right.port_u16()
}

fn invalid_host(message: &str) -> GatewayError {
    GatewayError::Client {
        status: http::StatusCode::UNAUTHORIZED,
        message: message.to_string(),
    }
}

fn invalid_signature(error: impl std::fmt::Display) -> GatewayError {
    GatewayError::Client {
        status: http::StatusCode::UNAUTHORIZED,
        message: format!("invalid Aether AK/SK signature: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{route_accepts_aether_aksk, verification_headers_and_host};
    use axum::http::{self, HeaderMap, HeaderValue, Uri};

    #[test]
    fn verification_uses_http2_authority_when_host_is_absent() {
        let uri: Uri = "https://assets.example.com:8443/?Action=ListMaterial"
            .parse()
            .expect("absolute HTTP/2 URI should parse");
        let original_headers = HeaderMap::new();

        let (verification_headers, host) = verification_headers_and_host(&uri, &original_headers)
            .expect("HTTP/2 authority should supply Host for verification");

        assert_eq!(host, "assets.example.com:8443");
        assert_eq!(
            verification_headers
                .get(http::header::HOST)
                .and_then(|value| value.to_str().ok()),
            Some("assets.example.com:8443")
        );
        assert!(original_headers.get(http::header::HOST).is_none());
    }

    #[test]
    fn verification_accepts_equivalent_host_and_authority() {
        let uri: Uri = "https://ASSETS.EXAMPLE.COM:8443/"
            .parse()
            .expect("absolute URI should parse");
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::HOST,
            HeaderValue::from_static("assets.example.com:8443"),
        );

        let (_, host) = verification_headers_and_host(&uri, &headers)
            .expect("equivalent authorities should be accepted");

        assert_eq!(host, "assets.example.com:8443");
    }

    #[test]
    fn verification_rejects_conflicting_host_and_authority() {
        let uri: Uri = "https://assets.example.com/"
            .parse()
            .expect("absolute URI should parse");
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::HOST,
            HeaderValue::from_static("attacker.example.com"),
        );

        let error = verification_headers_and_host(&uri, &headers)
            .expect_err("conflicting authorities must be rejected");

        assert!(matches!(
            &error,
            crate::GatewayError::Client {
                status: http::StatusCode::UNAUTHORIZED,
                ..
            }
        ));
        assert!(format!("{error:?}").contains("does not match"));
    }

    #[test]
    fn verification_rejects_request_without_host_or_authority() {
        let uri: Uri = "/?Action=ListMaterial"
            .parse()
            .expect("origin-form URI should parse");

        let error = verification_headers_and_host(&uri, &HeaderMap::new())
            .expect_err("a request without any authority must be rejected");

        assert!(format!("{error:?}").contains("authority is required"));
    }

    #[test]
    fn aksk_is_limited_to_native_and_user_material_asset_routes() {
        assert!(route_accepts_aether_aksk(
            Some("ai_public"),
            Some("doubao"),
            Some("asset_library")
        ));
        assert!(route_accepts_aether_aksk(
            Some("public_support"),
            Some("material_assets"),
            Some("preview_asset")
        ));
        assert!(!route_accepts_aether_aksk(
            Some("public_support"),
            Some("models"),
            Some("list")
        ));
        assert!(!route_accepts_aether_aksk(
            Some("admin"),
            Some("material_assets"),
            Some("preview_asset")
        ));
    }
}
