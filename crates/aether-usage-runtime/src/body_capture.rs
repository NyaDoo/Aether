use aether_data_contracts::repository::usage::{
    UpsertUsageRecord, UsageBodyCaptureState, UsageBodyField,
};
use serde_json::{json, Map, Value};

use crate::event::UsageEvent;
use crate::runtime::{UsageBodyCapturePolicy, UsageRequestRecordLevel};

const TRUNCATED_BODY_STRING_SUFFIX: &str = "...[truncated]";

#[derive(Debug)]
struct LimitedUsageBodyCapture {
    value: Value,
    source_bytes: Option<u64>,
    stored_bytes: Option<u64>,
    truncated: bool,
    reason: Option<&'static str>,
}

struct UsageBodyCapturePayloadMut<'a> {
    request_body: &'a mut Option<Value>,
    request_body_ref: &'a mut Option<String>,
    request_body_state: &'a mut Option<UsageBodyCaptureState>,
    provider_request_body: &'a mut Option<Value>,
    provider_request_body_ref: &'a mut Option<String>,
    provider_request_body_state: &'a mut Option<UsageBodyCaptureState>,
    response_body: &'a mut Option<Value>,
    response_body_ref: &'a mut Option<String>,
    response_body_state: &'a mut Option<UsageBodyCaptureState>,
    client_response_body: &'a mut Option<Value>,
    client_response_body_ref: &'a mut Option<String>,
    client_response_body_state: &'a mut Option<UsageBodyCaptureState>,
    request_metadata: &'a mut Option<Value>,
}

impl<'a> UsageBodyCapturePayloadMut<'a> {
    fn from_event(event: &'a mut UsageEvent) -> Self {
        Self {
            request_body: &mut event.data.request_body,
            request_body_ref: &mut event.data.request_body_ref,
            request_body_state: &mut event.data.request_body_state,
            provider_request_body: &mut event.data.provider_request_body,
            provider_request_body_ref: &mut event.data.provider_request_body_ref,
            provider_request_body_state: &mut event.data.provider_request_body_state,
            response_body: &mut event.data.response_body,
            response_body_ref: &mut event.data.response_body_ref,
            response_body_state: &mut event.data.response_body_state,
            client_response_body: &mut event.data.client_response_body,
            client_response_body_ref: &mut event.data.client_response_body_ref,
            client_response_body_state: &mut event.data.client_response_body_state,
            request_metadata: &mut event.data.request_metadata,
        }
    }

    fn from_record(record: &'a mut UpsertUsageRecord) -> Self {
        Self {
            request_body: &mut record.request_body,
            request_body_ref: &mut record.request_body_ref,
            request_body_state: &mut record.request_body_state,
            provider_request_body: &mut record.provider_request_body,
            provider_request_body_ref: &mut record.provider_request_body_ref,
            provider_request_body_state: &mut record.provider_request_body_state,
            response_body: &mut record.response_body,
            response_body_ref: &mut record.response_body_ref,
            response_body_state: &mut record.response_body_state,
            client_response_body: &mut record.client_response_body,
            client_response_body_ref: &mut record.client_response_body_ref,
            client_response_body_state: &mut record.client_response_body_state,
            request_metadata: &mut record.request_metadata,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UsageBodyCaptureEngine {
    policy: UsageBodyCapturePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeBodyCaptureStates {
    pub request: UsageBodyCaptureState,
    pub provider_request: UsageBodyCaptureState,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RuntimeBodyCaptureMetadataInput<'a> {
    pub request_has_inline_body: bool,
    pub request_body_ref: Option<&'a str>,
    pub provider_request_has_inline_body: bool,
    pub provider_request_body_ref: Option<&'a str>,
    pub provider_request_source_bytes: Option<u64>,
    pub provider_request_unavailable: bool,
    pub provider_request_unavailable_reason: Option<&'a str>,
}

impl UsageBodyCaptureEngine {
    pub fn new(policy: UsageBodyCapturePolicy) -> Self {
        Self { policy }
    }

    pub fn apply_to_event(self, event: &mut UsageEvent) {
        self.apply_to_payload(UsageBodyCapturePayloadMut::from_event(event));
    }

    pub fn apply_to_record(self, record: &mut UpsertUsageRecord) {
        self.apply_to_payload(UsageBodyCapturePayloadMut::from_record(record));
    }

    fn apply_to_payload(self, payload: UsageBodyCapturePayloadMut<'_>) {
        if matches!(self.policy.record_level, UsageRequestRecordLevel::Basic) {
            disable_usage_body_capture_field(
                UsageBodyField::RequestBody,
                "request",
                payload.request_body,
                payload.request_body_ref,
                payload.request_body_state,
                payload.request_metadata,
            );
            disable_usage_body_capture_field(
                UsageBodyField::ProviderRequestBody,
                "provider_request",
                payload.provider_request_body,
                payload.provider_request_body_ref,
                payload.provider_request_body_state,
                payload.request_metadata,
            );
            disable_usage_body_capture_field(
                UsageBodyField::ResponseBody,
                "response",
                payload.response_body,
                payload.response_body_ref,
                payload.response_body_state,
                payload.request_metadata,
            );
            disable_usage_body_capture_field(
                UsageBodyField::ClientResponseBody,
                "client_response",
                payload.client_response_body,
                payload.client_response_body_ref,
                payload.client_response_body_state,
                payload.request_metadata,
            );
            return;
        }

        apply_usage_body_capture_limit(
            UsageBodyField::RequestBody,
            "request",
            self.policy.max_request_body_bytes,
            payload.request_body,
            payload.request_body_ref,
            payload.request_body_state,
            payload.request_metadata,
        );
        apply_usage_body_capture_limit(
            UsageBodyField::ProviderRequestBody,
            "provider_request",
            self.policy.max_request_body_bytes,
            payload.provider_request_body,
            payload.provider_request_body_ref,
            payload.provider_request_body_state,
            payload.request_metadata,
        );
        apply_usage_body_capture_limit(
            UsageBodyField::ResponseBody,
            "response",
            self.policy.max_response_body_bytes,
            payload.response_body,
            payload.response_body_ref,
            payload.response_body_state,
            payload.request_metadata,
        );
        apply_usage_body_capture_limit(
            UsageBodyField::ClientResponseBody,
            "client_response",
            self.policy.max_response_body_bytes,
            payload.client_response_body,
            payload.client_response_body_ref,
            payload.client_response_body_state,
            payload.request_metadata,
        );
    }
}

pub fn apply_usage_body_capture_policy_to_event(
    policy: UsageBodyCapturePolicy,
    event: &mut UsageEvent,
) {
    UsageBodyCaptureEngine::new(policy).apply_to_event(event);
}

pub fn apply_usage_body_capture_policy_to_record(
    policy: UsageBodyCapturePolicy,
    record: &mut UpsertUsageRecord,
) {
    UsageBodyCaptureEngine::new(policy).apply_to_record(record);
}

fn disable_usage_body_capture_field(
    field: UsageBodyField,
    metadata_key: &str,
    body: &mut Option<Value>,
    body_ref: &mut Option<String>,
    state: &mut Option<UsageBodyCaptureState>,
    request_metadata: &mut Option<Value>,
) {
    *body = None;
    *body_ref = None;
    *state = Some(UsageBodyCaptureState::Disabled);
    sync_usage_body_ref_metadata(request_metadata, field, None);
    upsert_body_capture_metadata_value_entry(
        request_metadata,
        metadata_key,
        Some(UsageBodyCaptureState::Disabled),
        None,
        None,
        Some("request_record_level_basic"),
    );
}

fn apply_usage_body_capture_limit(
    field: UsageBodyField,
    metadata_key: &str,
    max_bytes: Option<usize>,
    body: &mut Option<Value>,
    body_ref: &mut Option<String>,
    state: &mut Option<UsageBodyCaptureState>,
    request_metadata: &mut Option<Value>,
) {
    *body_ref = sanitize_usage_body_ref(body_ref.take());
    if body.is_some() && body_ref.is_some() {
        *body = None;
    }

    if let Some(body_ref_value) = body_ref.as_ref() {
        *state = Some(UsageBodyCaptureState::Reference);
        sync_usage_body_ref_metadata(request_metadata, field, Some(body_ref_value));
        upsert_body_capture_metadata_value_entry(
            request_metadata,
            metadata_key,
            Some(UsageBodyCaptureState::Reference),
            None,
            None,
            None,
        );
        return;
    }

    let Some(value) = body.take() else {
        if matches!(state, Some(UsageBodyCaptureState::Unavailable)) {
            upsert_body_capture_metadata_value_entry(
                request_metadata,
                metadata_key,
                *state,
                None,
                None,
                None,
            );
        } else if state.is_none() {
            *state = Some(UsageBodyCaptureState::None);
        }
        sync_usage_body_ref_metadata(request_metadata, field, None);
        return;
    };

    let limited = limit_usage_body_capture_value(value, max_bytes);
    let next_state = if limited.truncated {
        UsageBodyCaptureState::Truncated
    } else {
        UsageBodyCaptureState::Inline
    };
    *state = Some(next_state);
    *body = Some(limited.value);
    sync_usage_body_ref_metadata(request_metadata, field, None);
    upsert_body_capture_metadata_value_entry(
        request_metadata,
        metadata_key,
        Some(next_state),
        limited.stored_bytes,
        limited.source_bytes,
        limited.reason,
    );
}

fn limit_usage_body_capture_value(
    value: Value,
    max_bytes: Option<usize>,
) -> LimitedUsageBodyCapture {
    let source_bytes = serde_json::to_vec(&value)
        .ok()
        .map(|bytes| bytes.len() as u64);
    let Some(limit) = max_bytes.filter(|value| *value > 0) else {
        return LimitedUsageBodyCapture {
            stored_bytes: source_bytes,
            source_bytes,
            value,
            truncated: false,
            reason: None,
        };
    };
    let Some(source_len) = source_bytes else {
        return LimitedUsageBodyCapture {
            stored_bytes: None,
            source_bytes: None,
            value,
            truncated: false,
            reason: None,
        };
    };
    if source_len <= limit as u64 {
        return LimitedUsageBodyCapture {
            stored_bytes: Some(source_len),
            source_bytes: Some(source_len),
            value,
            truncated: false,
            reason: None,
        };
    }

    let truncated_value = match value {
        Value::String(text) => Value::String(truncate_usage_body_string(&text, limit)),
        other => json!({
            "truncated": true,
            "reason": "body_capture_limit_exceeded",
            "max_bytes": limit,
            "source_bytes": source_len,
            "value_kind": usage_value_kind(&other),
        }),
    };
    let stored_bytes = serde_json::to_vec(&truncated_value)
        .ok()
        .map(|bytes| bytes.len() as u64);
    LimitedUsageBodyCapture {
        value: truncated_value,
        source_bytes: Some(source_len),
        stored_bytes,
        truncated: true,
        reason: Some("body_capture_limit_exceeded"),
    }
}

fn truncate_usage_body_string(value: &str, max_bytes: usize) -> String {
    if serde_json::to_vec(&Value::String(value.to_string()))
        .ok()
        .is_some_and(|bytes| bytes.len() <= max_bytes)
    {
        return value.to_string();
    }

    let mut end = value.len();
    while end > 0 {
        while end > 0 && !value.is_char_boundary(end) {
            end -= 1;
        }
        let mut candidate = value[..end].to_string();
        candidate.push_str(TRUNCATED_BODY_STRING_SUFFIX);
        if serde_json::to_vec(&Value::String(candidate.clone()))
            .ok()
            .is_some_and(|bytes| bytes.len() <= max_bytes)
        {
            return candidate;
        }
        end = value[..end]
            .char_indices()
            .last()
            .map(|(index, _)| index)
            .unwrap_or(0);
        if end == 0 {
            break;
        }
    }

    json!({
        "truncated": true,
        "reason": "body_capture_limit_exceeded",
        "max_bytes": max_bytes,
        "value_kind": "string",
    })
    .to_string()
}

pub(crate) fn sync_usage_body_ref_metadata(
    metadata: &mut Option<Value>,
    field: UsageBodyField,
    body_ref: Option<&str>,
) {
    let Some(body_ref) = body_ref.map(str::trim).filter(|value| !value.is_empty()) else {
        if let Some(object) = metadata.as_mut().and_then(Value::as_object_mut) {
            object.remove(field.as_ref_key());
        }
        return;
    };
    let object = metadata
        .get_or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut();
    let Some(object) = object else {
        return;
    };
    object.insert(
        field.as_ref_key().to_string(),
        Value::String(body_ref.to_string()),
    );
}

pub(crate) fn build_payload_body_capture_metadata(
    provider_body_base64: Option<&str>,
    client_body_base64: Option<&str>,
    provider_body_state: Option<UsageBodyCaptureState>,
    client_body_state: Option<UsageBodyCaptureState>,
) -> Option<Value> {
    let mut metadata = Map::new();
    if let Some(decoded_len) = provider_body_base64.and_then(decoded_base64_len_hint) {
        metadata.insert(
            "provider_response_body_base64_bytes".to_string(),
            Value::Number(decoded_len.into()),
        );
    }
    if let Some(decoded_len) = client_body_base64.and_then(decoded_base64_len_hint) {
        metadata.insert(
            "client_response_body_base64_bytes".to_string(),
            Value::Number(decoded_len.into()),
        );
    }

    let mut body_capture = Map::new();
    append_body_capture_metadata_entry(
        &mut body_capture,
        "response",
        provider_body_state,
        provider_body_base64.and_then(decoded_base64_len_hint),
        provider_body_base64.and_then(decoded_base64_len_hint),
    );
    append_body_capture_metadata_entry(
        &mut body_capture,
        "client_response",
        client_body_state,
        client_body_base64.and_then(decoded_base64_len_hint),
        client_body_base64.and_then(decoded_base64_len_hint),
    );
    if !body_capture.is_empty() {
        metadata.insert("body_capture".to_string(), Value::Object(body_capture));
    }

    (!metadata.is_empty()).then_some(Value::Object(metadata))
}

pub(crate) fn build_runtime_body_capture_states(
    request_has_inline_body: bool,
    request_body_ref: Option<&str>,
    provider_request_has_inline_body: bool,
    provider_request_body_ref: Option<&str>,
    provider_request_unavailable: bool,
) -> RuntimeBodyCaptureStates {
    RuntimeBodyCaptureStates {
        request: UsageBodyCaptureState::from_capture_parts(
            request_has_inline_body,
            request_body_ref.is_some(),
            false,
        ),
        provider_request: UsageBodyCaptureState::from_capture_parts(
            provider_request_has_inline_body,
            provider_request_body_ref.is_some(),
            provider_request_unavailable,
        ),
    }
}

pub(crate) fn append_runtime_body_capture_metadata(
    metadata: &mut Map<String, Value>,
    input: RuntimeBodyCaptureMetadataInput<'_>,
) {
    let states = build_runtime_body_capture_states(
        input.request_has_inline_body,
        input.request_body_ref,
        input.provider_request_has_inline_body,
        input.provider_request_body_ref,
        input.provider_request_unavailable,
    );
    upsert_body_capture_metadata_entry(metadata, "request", Some(states.request), None, None, None);
    upsert_body_capture_metadata_entry(
        metadata,
        "provider_request",
        Some(states.provider_request),
        input.provider_request_source_bytes,
        input.provider_request_source_bytes,
        input.provider_request_unavailable_reason,
    );
}

pub(crate) fn build_plan_body_capture_metadata(
    provider_request_body_base64: Option<&str>,
) -> Option<Value> {
    let mut metadata = Map::new();
    append_plan_body_capture_metadata(&mut metadata, provider_request_body_base64);
    (!metadata.is_empty()).then_some(Value::Object(metadata))
}

pub(crate) fn append_plan_body_capture_metadata(
    metadata: &mut Map<String, Value>,
    provider_request_body_base64: Option<&str>,
) {
    if let Some(body_bytes_b64) = provider_request_body_base64 {
        let decoded_len = decoded_base64_len_hint(body_bytes_b64);
        if let Some(decoded_len) = decoded_len {
            metadata.insert(
                "provider_request_body_base64_bytes".to_string(),
                Value::Number(decoded_len.into()),
            );
        }
        upsert_body_capture_metadata_entry(
            metadata,
            "provider_request",
            Some(UsageBodyCaptureState::Unavailable),
            decoded_len,
            decoded_len,
            Some("body_bytes_base64_only"),
        );
    }
}

fn append_body_capture_metadata_entry(
    target: &mut Map<String, Value>,
    key: &str,
    state: Option<UsageBodyCaptureState>,
    stored_bytes: Option<u64>,
    source_bytes: Option<u64>,
) {
    let Some(state) = state else {
        return;
    };
    let mut entry = Map::new();
    entry.insert(
        "state".to_string(),
        Value::String(state.as_str().to_string()),
    );
    if let Some(stored_bytes) = stored_bytes {
        entry.insert("stored_bytes".to_string(), json!(stored_bytes));
    }
    if let Some(source_bytes) = source_bytes {
        entry.insert("source_bytes".to_string(), json!(source_bytes));
    }
    if matches!(state, UsageBodyCaptureState::Truncated) {
        entry.insert(
            "reason".to_string(),
            Value::String("body_capture_limit_exceeded".to_string()),
        );
    }
    target.insert(key.to_string(), Value::Object(entry));
}

pub(crate) fn upsert_body_capture_metadata_entry(
    metadata: &mut Map<String, Value>,
    key: &str,
    state: Option<UsageBodyCaptureState>,
    stored_bytes: Option<u64>,
    source_bytes: Option<u64>,
    reason: Option<&str>,
) {
    let body_capture = metadata
        .entry("body_capture".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(body_capture_object) = body_capture.as_object_mut() else {
        return;
    };
    let Some(state) = state else {
        return;
    };
    let mut entry = Map::new();
    entry.insert(
        "state".to_string(),
        Value::String(state.as_str().to_string()),
    );
    if let Some(bytes) = stored_bytes {
        entry.insert("stored_bytes".to_string(), json!(bytes));
    }
    if let Some(bytes) = source_bytes {
        entry.insert("source_bytes".to_string(), json!(bytes));
    }
    if let Some(reason) = reason {
        entry.insert("reason".to_string(), Value::String(reason.to_string()));
    }
    body_capture_object.insert(key.to_string(), Value::Object(entry));
}

fn upsert_body_capture_metadata_value_entry(
    metadata: &mut Option<Value>,
    key: &str,
    state: Option<UsageBodyCaptureState>,
    stored_bytes: Option<u64>,
    source_bytes: Option<u64>,
    reason: Option<&str>,
) {
    let Some(state) = state else {
        return;
    };
    let metadata_object = metadata
        .get_or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut();
    let Some(metadata_object) = metadata_object else {
        return;
    };
    upsert_body_capture_metadata_entry(
        metadata_object,
        key,
        Some(state),
        stored_bytes,
        source_bytes,
        reason,
    );
}

pub(crate) fn decoded_base64_len_hint(body_base64: &str) -> Option<u64> {
    let body_base64 = body_base64.trim();
    if body_base64.is_empty() {
        return None;
    }

    let usable_len = body_base64.len();
    if usable_len % 4 == 1 {
        return None;
    }

    let padding = body_base64
        .chars()
        .rev()
        .take_while(|char| *char == '=')
        .count();
    let full_quads = usable_len / 4;
    let remainder = usable_len % 4;
    let base_len = full_quads.saturating_mul(3);
    let remainder_len = match remainder {
        0 => 0,
        2 => 1,
        3 => 2,
        _ => return None,
    };
    let decoded_len = base_len
        .saturating_add(remainder_len)
        .saturating_sub(padding.min(2));

    Some(decoded_len as u64)
}

fn sanitize_usage_body_ref(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn usage_value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
