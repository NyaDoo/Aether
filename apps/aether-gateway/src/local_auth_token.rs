use base64::Engine as _;
use hmac::Mac as _;

pub(crate) fn jwt_secret() -> Result<String, String> {
    if let Ok(value) = std::env::var("JWT_SECRET_KEY") {
        let value = value.trim();
        if !value.is_empty() {
            return Ok(value.to_string());
        }
    }
    let environment = std::env::var("ENVIRONMENT")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "development".to_string());
    if environment.eq_ignore_ascii_case("production") {
        return Err("JWT_SECRET_KEY 未配置".to_string());
    }
    Ok("aether-rust-dev-jwt-secret".to_string())
}

pub(crate) fn is_aether_access_token(token: &str) -> bool {
    let mut parts = token.split('.');
    let (Some(header_segment), Some(payload_segment), Some(signature_segment)) =
        (parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    if parts.next().is_some() {
        return false;
    }
    let Ok(signature) = decode_base64url(signature_segment) else {
        return false;
    };
    let Ok(secret) = jwt_secret() else {
        return false;
    };
    let Ok(mut mac) = hmac::Hmac::<sha2::Sha256>::new_from_slice(secret.as_bytes()) else {
        return false;
    };
    mac.update(format!("{header_segment}.{payload_segment}").as_bytes());
    if mac.verify_slice(&signature).is_err() {
        return false;
    }
    decode_base64url(payload_segment)
        .ok()
        .and_then(|payload| serde_json::from_slice::<serde_json::Value>(&payload).ok())
        .and_then(|payload| payload.get("type")?.as_str().map(str::to_string))
        .is_some_and(|token_type| token_type == "access")
}

fn decode_base64url(value: &str) -> Result<Vec<u8>, base64::DecodeError> {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(value)
}

#[cfg(test)]
pub(crate) fn sign_for_tests(
    token_type: &str,
    mut payload: serde_json::Map<String, serde_json::Value>,
    expires_at: chrono::DateTime<chrono::Utc>,
) -> String {
    let header = serde_json::json!({"alg": "HS256", "typ": "JWT"});
    payload.insert("exp".to_string(), serde_json::json!(expires_at.timestamp()));
    payload.insert("type".to_string(), serde_json::json!(token_type));
    let encode = |bytes: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
    let header_segment = encode(&serde_json::to_vec(&header).expect("JWT header"));
    let payload_segment = encode(&serde_json::to_vec(&payload).expect("JWT payload"));
    let signing_input = format!("{header_segment}.{payload_segment}");
    let secret = jwt_secret().expect("test JWT secret");
    let mut mac =
        hmac::Hmac::<sha2::Sha256>::new_from_slice(secret.as_bytes()).expect("test JWT secret");
    mac.update(signing_input.as_bytes());
    format!(
        "{signing_input}.{}",
        encode(mac.finalize().into_bytes().as_slice())
    )
}
