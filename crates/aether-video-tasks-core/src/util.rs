use serde_json::Value;
use std::net::{Ipv4Addr, Ipv6Addr};

/// Maximum width of `video_tasks.error_code` in the logical schema.
pub(crate) const VIDEO_TASK_ERROR_CODE_MAX_CHARS: usize = 128;

/// Normalizes an untrusted provider error code for durable task storage.
///
/// PostgreSQL `varchar(n)` measures characters rather than UTF-8 bytes, so
/// truncate by Unicode scalar values to keep the result valid UTF-8. The raw
/// provider response is retained separately in sanitized task metadata.
pub(crate) fn normalize_video_task_error_code(value: Option<&str>) -> Option<String> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    Some(
        value
            .chars()
            .take(VIDEO_TASK_ERROR_CODE_MAX_CHARS)
            .collect(),
    )
}

pub(crate) fn safe_external_http_url(value: &str) -> Option<String> {
    let value = value.trim();
    let parsed = url::Url::parse(value).ok()?;
    // Ark asset links carry credentials in their query string. Never forward
    // those tokens over cleartext HTTP, even when a compatible provider emits
    // such a URL.
    if parsed.scheme() != "https" || !parsed.username().is_empty() || parsed.password().is_some() {
        return None;
    }
    let host = parsed.host()?;
    let blocked = match host {
        url::Host::Domain(host) => {
            let host = host.trim_end_matches('.');
            host.eq_ignore_ascii_case("localhost") || host.ends_with(".local")
        }
        url::Host::Ipv4(ip) => blocked_ipv4(ip),
        url::Host::Ipv6(ip) => blocked_ipv6(ip),
    };
    if blocked {
        return None;
    }
    Some(parsed.to_string())
}

fn blocked_ipv4(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_multicast()
        || ip.octets() == [169, 254, 169, 254]
}

fn blocked_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    let mapped_ipv4 =
        (segments[..5] == [0, 0, 0, 0, 0] && matches!(segments[5], 0 | 0xffff)).then(|| {
            Ipv4Addr::new(
                (segments[6] >> 8) as u8,
                segments[6] as u8,
                (segments[7] >> 8) as u8,
                segments[7] as u8,
            )
        });
    mapped_ipv4.is_some_and(blocked_ipv4)
        || ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_multicast()
        || (segments[0] & 0xfe00) == 0xfc00
        || (segments[0] & 0xffc0) == 0xfe80
}

pub(crate) fn value_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

pub(crate) fn value_i32(value: &Value) -> Option<i32> {
    match value {
        Value::Number(number) => number.as_i64().and_then(|value| i32::try_from(value).ok()),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

pub fn non_empty_owned(value: Option<&String>) -> Option<String> {
    value
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Maximum width of `video_tasks.short_id`.
const SHORT_ID_MAX_LEN: usize = 16;

/// Derives the stored `short_id` for surfaces that have no short-id concept.
///
/// `video_tasks.short_id` is `NOT NULL`, but only the Gemini surface exposes a
/// short operation id to clients. OpenAI and Doubao task ids are longer than the
/// column allows, so a deterministic prefix is derived instead: the same task
/// always yields the same value, which keeps repeated upserts idempotent.
pub fn derive_video_task_short_id(local_task_id: &str) -> String {
    let compact = local_task_id
        .chars()
        .filter(|value| value.is_ascii_alphanumeric())
        .take(SHORT_ID_MAX_LEN)
        .collect::<String>();
    if compact.is_empty() {
        // Never leave the column empty, even for an unexpected id shape.
        return "video".to_string();
    }
    compact
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        derive_video_task_short_id, non_empty_owned, normalize_video_task_error_code,
        safe_external_http_url, value_i32, value_u64, VIDEO_TASK_ERROR_CODE_MAX_CHARS,
    };

    #[test]
    fn reads_unsigned_integer_from_json_number_or_string() {
        assert_eq!(value_u64(&json!(42)), Some(42));
        assert_eq!(value_u64(&json!(" 42 ")), Some(42));
        assert_eq!(value_u64(&json!(-1)), None);
        assert_eq!(value_u64(&json!("not-a-number")), None);
    }

    #[test]
    fn reads_signed_32_bit_integer_from_json_number_or_string() {
        assert_eq!(value_i32(&json!(-1)), Some(-1));
        assert_eq!(value_i32(&json!(24)), Some(24));
        assert_eq!(value_i32(&json!(" 121 ")), Some(121));
        assert_eq!(value_i32(&json!(2_147_483_648_i64)), None);
        assert_eq!(value_i32(&json!("not-a-number")), None);
    }

    #[test]
    fn rejects_non_public_video_asset_urls() {
        for value in [
            "file:///tmp/video.mp4",
            "http://cdn.example.com/video.mp4?X-Sig=secret",
            "http://127.0.0.1/video.mp4",
            "http://[::ffff:127.0.0.1]/video.mp4",
            "http://169.254.169.254/latest/meta-data",
            "http://localhost/video.mp4",
            "http://user:pass@example.com/video.mp4",
        ] {
            assert!(safe_external_http_url(value).is_none(), "{value}");
        }
        assert_eq!(
            safe_external_http_url("https://cdn.example.com/video.mp4").as_deref(),
            Some("https://cdn.example.com/video.mp4")
        );
    }

    #[test]
    fn discards_blank_strings() {
        let empty = String::from("   ");
        let value = String::from(" hello ");

        assert_eq!(non_empty_owned(Some(&empty)), None);
        assert_eq!(non_empty_owned(Some(&value)).as_deref(), Some("hello"));
    }

    #[test]
    fn preserves_provider_error_codes_longer_than_the_legacy_limit() {
        let code = "E".repeat(80);

        assert_eq!(
            normalize_video_task_error_code(Some(&code)).as_deref(),
            Some(code.as_str())
        );
    }

    #[test]
    fn truncates_provider_error_codes_to_the_logical_schema_limit() {
        let code = "E".repeat(VIDEO_TASK_ERROR_CODE_MAX_CHARS + 17);
        let normalized = normalize_video_task_error_code(Some(&code)).unwrap();

        assert_eq!(normalized.chars().count(), VIDEO_TASK_ERROR_CODE_MAX_CHARS);
        assert_eq!(normalized, "E".repeat(VIDEO_TASK_ERROR_CODE_MAX_CHARS));
    }

    #[test]
    fn truncates_provider_error_codes_on_a_unicode_character_boundary() {
        let code = format!(
            "{}\u{754c}\u{9519}",
            "E".repeat(VIDEO_TASK_ERROR_CODE_MAX_CHARS - 1)
        );
        let normalized = normalize_video_task_error_code(Some(&code)).unwrap();

        assert_eq!(normalized.chars().count(), VIDEO_TASK_ERROR_CODE_MAX_CHARS);
        assert!(normalized.ends_with('\u{754c}'));
        assert!(!normalized.ends_with('\u{9519}'));
    }

    #[test]
    fn derives_a_column_sized_short_id() {
        // A Doubao local id: `cgt-` plus a 32-char uuid.
        let short_id = derive_video_task_short_id("cgt-0f9a1b2c3d4e5f60718293a4b5c6d7e8");
        assert_eq!(short_id.len(), 16);
        assert_eq!(short_id, "cgt0f9a1b2c3d4e5");

        // A bare uuid, as used by the OpenAI surface.
        let openai = derive_video_task_short_id("0f9a1b2c-3d4e-5f60-7182-93a4b5c6d7e8");
        assert_eq!(openai.len(), 16);
    }

    #[test]
    fn derivation_is_stable_for_the_same_task() {
        let first = derive_video_task_short_id("cgt-abc123");
        let second = derive_video_task_short_id("cgt-abc123");

        assert_eq!(first, second);
    }

    #[test]
    fn short_ids_stay_non_empty_for_unexpected_shapes() {
        assert_eq!(derive_video_task_short_id("---"), "video");
        assert_eq!(derive_video_task_short_id(""), "video");
    }
}
