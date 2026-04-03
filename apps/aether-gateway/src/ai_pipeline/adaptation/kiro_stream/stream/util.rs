use serde_json::Value;

use crate::gateway::GatewayError;

use super::QUOTE_CHARS;

pub(super) fn encode_events(events: Vec<Value>) -> Result<Vec<u8>, GatewayError> {
    let mut output = Vec::new();
    for event in events {
        output.extend(encode_sse_event(&event)?);
    }
    Ok(output)
}

pub(super) fn encode_sse_event(event: &Value) -> Result<Vec<u8>, GatewayError> {
    let encoded =
        serde_json::to_string(event).map_err(|err| GatewayError::Internal(err.to_string()))?;
    if let Some(event_type) = event.get("type").and_then(Value::as_str) {
        Ok(format!("event: {event_type}\ndata: {encoded}\n\n").into_bytes())
    } else {
        Ok(format!("data: {encoded}\n\n").into_bytes())
    }
}

pub(super) fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    let mut chinese = 0usize;
    let mut other = 0usize;
    for ch in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            chinese += 1;
        } else {
            other += 1;
        }
    }
    let chinese_tokens = (chinese * 2).div_ceil(3);
    let other_tokens = other.div_ceil(4);
    (chinese_tokens + other_tokens).max(1)
}

pub(super) fn is_quote_char(buffer: &str, pos: usize) -> bool {
    buffer
        .as_bytes()
        .get(pos)
        .map(|byte| QUOTE_CHARS.as_bytes().contains(byte))
        .unwrap_or(false)
}

pub(super) fn find_real_thinking_start_tag(buffer: &str) -> Option<usize> {
    let tag = "<thinking>";
    let mut search = 0usize;
    loop {
        let pos = buffer[search..].find(tag).map(|value| value + search)?;
        let has_before = pos > 0 && is_quote_char(buffer, pos - 1);
        let after_pos = pos + tag.len();
        let has_after = is_quote_char(buffer, after_pos);
        if !has_before && !has_after {
            return Some(pos);
        }
        search = pos + 1;
    }
}

pub(super) fn find_real_thinking_end_tag(buffer: &str) -> Option<usize> {
    let tag = "</thinking>";
    let mut search = 0usize;
    loop {
        let pos = buffer[search..].find(tag).map(|value| value + search)?;
        let has_before = pos > 0 && is_quote_char(buffer, pos - 1);
        let after_pos = pos + tag.len();
        let has_after = is_quote_char(buffer, after_pos);
        if has_before || has_after {
            search = pos + 1;
            continue;
        }
        let after = &buffer[after_pos..];
        if after.len() < 2 {
            return None;
        }
        if after.starts_with("\n\n") {
            return Some(pos);
        }
        search = pos + 1;
    }
}

pub(super) fn find_real_thinking_end_tag_at_buffer_end(buffer: &str) -> Option<usize> {
    let tag = "</thinking>";
    let mut search = 0usize;
    loop {
        let pos = buffer[search..].find(tag).map(|value| value + search)?;
        let has_before = pos > 0 && is_quote_char(buffer, pos - 1);
        let after_pos = pos + tag.len();
        let has_after = is_quote_char(buffer, after_pos);
        if has_before || has_after {
            search = pos + 1;
            continue;
        }
        if buffer[after_pos..].trim().is_empty() {
            return Some(pos);
        }
        search = pos + 1;
    }
}

pub(super) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = if crc & 1 == 1 { 0xedb8_8320 } else { 0 };
            crc = (crc >> 1) ^ mask;
        }
    }
    !crc
}
