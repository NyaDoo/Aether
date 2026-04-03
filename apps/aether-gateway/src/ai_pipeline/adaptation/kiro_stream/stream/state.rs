use serde_json::{json, Value};
use uuid::Uuid;

use crate::gateway::GatewayError;

use super::util::{
    encode_events, estimate_tokens, find_real_thinking_end_tag,
    find_real_thinking_end_tag_at_buffer_end, find_real_thinking_start_tag,
};
use super::{
    AwsEventFrame, EventStreamDecoder, KiroClaudeStreamState, KiroToClaudeCliStreamState,
    CONTEXT_WINDOW_TOKENS, MAX_THINKING_BUFFER,
};

impl KiroToClaudeCliStreamState {
    pub(crate) fn new(report_context: &Value) -> Self {
        Self {
            decoder: EventStreamDecoder::default(),
            state: KiroClaudeStreamState::new(report_context),
            started: false,
        }
    }

    pub(crate) fn push_chunk(
        &mut self,
        _report_context: &Value,
        chunk: &[u8],
    ) -> Result<Vec<u8>, GatewayError> {
        let mut output = Vec::new();
        if !self.started {
            self.started = true;
            output.extend(self.state.generate_initial_bytes()?);
        }

        if let Err(err) = self.decoder.feed(chunk) {
            output.extend(
                self.state
                    .emit_stream_error("upstream_stream_error", &err)?,
            );
            return Ok(output);
        }

        match self.decoder.decode_available() {
            Ok(frames) => {
                for frame in frames {
                    output.extend(self.state.process_frame(frame)?);
                }
            }
            Err(err) => {
                output.extend(
                    self.state
                        .emit_stream_error("upstream_stream_error", &err)?,
                );
            }
        }

        Ok(output)
    }

    pub(crate) fn finish(&mut self, _report_context: &Value) -> Result<Vec<u8>, GatewayError> {
        if !self.started || self.state.had_error {
            return Ok(Vec::new());
        }
        self.state.finalize()
    }
}

impl KiroClaudeStreamState {
    fn new(report_context: &Value) -> Self {
        let model = report_context
            .get("mapped_model")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                report_context
                    .get("model")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
            })
            .unwrap_or("unknown")
            .to_string();
        let thinking_enabled = report_context
            .get("original_request_body")
            .and_then(Value::as_object)
            .and_then(|body| body.get("thinking"))
            .and_then(Value::as_object)
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str)
            .map(|value| {
                value.trim().eq_ignore_ascii_case("enabled")
                    || value.trim().eq_ignore_ascii_case("adaptive")
            })
            .unwrap_or(false);
        let estimated_input_tokens = report_context
            .get("input_tokens")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(0);
        Self {
            model,
            thinking_enabled,
            estimated_input_tokens,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            ..Self::default()
        }
    }

    fn generate_initial_bytes(&mut self) -> Result<Vec<u8>, GatewayError> {
        let mut events = vec![json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": self.estimated_input_tokens as u64,
                    "output_tokens": 1,
                },
            }
        })];
        if !self.thinking_enabled {
            events.extend(self.ensure_text_block_open());
        }
        encode_events(events)
    }

    fn emit_stream_error(
        &mut self,
        error_type: &str,
        message: &str,
    ) -> Result<Vec<u8>, GatewayError> {
        if self.had_error {
            return Ok(Vec::new());
        }
        self.had_error = true;
        encode_events(vec![json!({
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            }
        })])
    }

    fn process_frame(&mut self, frame: AwsEventFrame) -> Result<Vec<u8>, GatewayError> {
        let message_type = frame.headers.message_type().unwrap_or("event");
        match message_type {
            "event" => self.process_event_frame(frame),
            "exception" => self.process_exception_frame(frame),
            "error" => self.process_error_frame(frame),
            _ => Ok(Vec::new()),
        }
    }

    fn process_event_frame(&mut self, frame: AwsEventFrame) -> Result<Vec<u8>, GatewayError> {
        let event_type = frame.headers.event_type().unwrap_or_default();
        let payload: Value = if frame.payload.is_empty() {
            json!({})
        } else {
            serde_json::from_slice(&frame.payload).unwrap_or_else(|_| json!({}))
        };
        let payload_object = payload.as_object();
        let mut events = Vec::new();
        match event_type {
            "assistantResponseEvent" => {
                if let Some(content) = payload_object
                    .and_then(|value| value.get("content"))
                    .and_then(Value::as_str)
                {
                    events.extend(self.process_assistant_response(content));
                }
            }
            "toolUseEvent" => {
                if let Some(payload_object) = payload_object {
                    let name = payload_object
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let tool_use_id = payload_object
                        .get("toolUseId")
                        .or_else(|| payload_object.get("tool_use_id"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let input_json = match payload_object.get("input") {
                        None | Some(Value::Null) => String::new(),
                        Some(Value::String(text)) => text.clone(),
                        Some(other) => serde_json::to_string(other)
                            .map_err(|err| GatewayError::Internal(err.to_string()))?,
                    };
                    let stop = payload_object
                        .get("stop")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    events.extend(self.process_tool_use(name, tool_use_id, &input_json, stop));
                }
            }
            "contextUsageEvent" => {
                if let Some(percentage) = payload_object
                    .and_then(|value| value.get("contextUsagePercentage"))
                    .and_then(Value::as_f64)
                {
                    self.context_input_tokens =
                        Some(((percentage * CONTEXT_WINDOW_TOKENS) / 100.0) as usize);
                }
            }
            _ => {}
        }
        encode_events(events)
    }

    fn process_exception_frame(&mut self, frame: AwsEventFrame) -> Result<Vec<u8>, GatewayError> {
        let exception_type = frame
            .headers
            .exception_type()
            .unwrap_or("UnknownException")
            .to_string();
        if exception_type == "ContentLengthExceededException" {
            self.stop_reason_override = Some("max_tokens".to_string());
            return Ok(Vec::new());
        }
        self.emit_stream_error("upstream_exception", &exception_type)
    }

    fn process_error_frame(&mut self, frame: AwsEventFrame) -> Result<Vec<u8>, GatewayError> {
        let error_code = frame
            .headers
            .error_code()
            .unwrap_or("UnknownError")
            .to_string();
        self.emit_stream_error("upstream_error", &error_code)
    }

    fn process_assistant_response(&mut self, content: &str) -> Vec<Value> {
        if content.is_empty() || content == self.last_content {
            return Vec::new();
        }
        self.last_content = content.to_string();
        self.output_tokens += estimate_tokens(content);

        if !self.thinking_enabled {
            return self.emit_text_delta(content);
        }

        self.thinking_buffer.push_str(content);
        if self.thinking_buffer.len() > MAX_THINKING_BUFFER {
            let overflow = std::mem::take(&mut self.thinking_buffer);
            if self.in_thinking_block {
                let mut events = self.emit_thinking_delta(&overflow);
                events.extend(self.close_thinking_block());
                self.in_thinking_block = false;
                self.thinking_extracted = true;
                return events;
            }
            return self.emit_text_delta(&overflow);
        }

        let mut events = Vec::new();
        loop {
            if !self.in_thinking_block && !self.thinking_extracted {
                if let Some(start_pos) = find_real_thinking_start_tag(&self.thinking_buffer) {
                    let before = self.thinking_buffer[..start_pos].to_string();
                    if !before.trim().is_empty() {
                        events.extend(self.emit_text_delta(&before));
                    }
                    self.in_thinking_block = true;
                    self.strip_thinking_leading_newline = true;
                    self.thinking_buffer =
                        self.thinking_buffer[start_pos + "<thinking>".len()..].to_string();
                    events.extend(self.ensure_thinking_block_open());
                    continue;
                }

                let keep = "<thinking>".len();
                if self.thinking_buffer.len() > keep {
                    let split = self.thinking_buffer.len() - keep;
                    let safe = self.thinking_buffer[..split].to_string();
                    if !safe.trim().is_empty() {
                        events.extend(self.emit_text_delta(&safe));
                        self.thinking_buffer = self.thinking_buffer[split..].to_string();
                    }
                }
                break;
            }

            if self.in_thinking_block {
                if self.strip_thinking_leading_newline {
                    if self.thinking_buffer.starts_with('\n') {
                        self.thinking_buffer.remove(0);
                        self.strip_thinking_leading_newline = false;
                    } else if !self.thinking_buffer.is_empty() {
                        self.strip_thinking_leading_newline = false;
                    }
                }

                if let Some(end_pos) = find_real_thinking_end_tag(&self.thinking_buffer) {
                    let thinking_text = self.thinking_buffer[..end_pos].to_string();
                    if !thinking_text.is_empty() {
                        events.extend(self.emit_thinking_delta(&thinking_text));
                    }
                    events.extend(self.close_thinking_block());
                    self.in_thinking_block = false;
                    self.thinking_extracted = true;
                    self.thinking_buffer =
                        self.thinking_buffer[end_pos + "</thinking>".len()..].to_string();
                    continue;
                }

                let keep = "</thinking>".len();
                if self.thinking_buffer.len() > keep {
                    let split = self.thinking_buffer.len() - keep;
                    let safe = self.thinking_buffer[..split].to_string();
                    if !safe.is_empty() {
                        events.extend(self.emit_thinking_delta(&safe));
                        self.thinking_buffer = self.thinking_buffer[split..].to_string();
                    }
                }
                break;
            }

            if !self.thinking_buffer.is_empty() {
                let remaining = std::mem::take(&mut self.thinking_buffer);
                events.extend(self.emit_text_delta(&remaining));
            }
            break;
        }

        events
    }

    fn process_tool_use(
        &mut self,
        name: &str,
        tool_use_id: &str,
        input_json: &str,
        stop: bool,
    ) -> Vec<Value> {
        if tool_use_id.is_empty() {
            return Vec::new();
        }

        self.has_tool_use = true;
        let mut events = Vec::new();

        if self.thinking_enabled && self.in_thinking_block && !self.thinking_buffer.is_empty() {
            if let Some(end_pos) = find_real_thinking_end_tag_at_buffer_end(&self.thinking_buffer) {
                let thinking_text = self.thinking_buffer[..end_pos].to_string();
                if !thinking_text.is_empty() {
                    events.extend(self.emit_thinking_delta(&thinking_text));
                }
                events.extend(self.close_thinking_block());
                let remaining = self.thinking_buffer[end_pos + "</thinking>".len()..].to_string();
                self.thinking_buffer.clear();
                self.in_thinking_block = false;
                self.thinking_extracted = true;
                if !remaining.is_empty() {
                    events.extend(self.emit_text_delta(&remaining));
                }
            } else {
                let thinking = std::mem::take(&mut self.thinking_buffer);
                events.extend(self.emit_thinking_delta(&thinking));
                events.extend(self.close_thinking_block());
                self.in_thinking_block = false;
                self.thinking_extracted = true;
            }
        }

        if self.thinking_enabled
            && !self.in_thinking_block
            && !self.thinking_extracted
            && !self.thinking_buffer.is_empty()
        {
            let buffered = std::mem::take(&mut self.thinking_buffer);
            events.extend(self.emit_text_delta(&buffered));
        }

        if let Some(idx) = self.text_block_index.take() {
            events.extend(self.close_block(idx));
        }

        let block_index = if let Some(block_index) = self.tool_block_indices.get(tool_use_id) {
            *block_index
        } else {
            let block_index = self.next_block_index;
            self.next_block_index += 1;
            self.tool_block_indices
                .insert(tool_use_id.to_string(), block_index);
            block_index
        };

        if let std::collections::btree_map::Entry::Vacant(e) = self.open_blocks.entry(block_index) {
            e.insert("tool_use".to_string());
            events.push(json!({
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": name,
                    "input": {},
                }
            }));
        }

        if !input_json.is_empty() {
            self.output_tokens += estimate_tokens(input_json);
            events.push(json!({
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": input_json,
                }
            }));
        }

        if stop {
            events.extend(self.close_block(block_index));
        }

        events
    }

    fn finalize(&mut self) -> Result<Vec<u8>, GatewayError> {
        if self.thinking_enabled && !self.thinking_buffer.is_empty() {
            let flush_events = if self.in_thinking_block {
                if let Some(end_pos) =
                    find_real_thinking_end_tag_at_buffer_end(&self.thinking_buffer)
                {
                    let thinking_text = self.thinking_buffer[..end_pos].to_string();
                    let mut events = Vec::new();
                    if !thinking_text.is_empty() {
                        events.extend(self.emit_thinking_delta(&thinking_text));
                    }
                    events.extend(self.close_thinking_block());
                    let remaining =
                        self.thinking_buffer[end_pos + "</thinking>".len()..].to_string();
                    if !remaining.is_empty() {
                        events.extend(self.emit_text_delta(&remaining));
                    }
                    events
                } else {
                    let mut events = self.emit_thinking_delta(&self.thinking_buffer.clone());
                    events.extend(self.close_thinking_block());
                    events
                }
            } else {
                self.emit_text_delta(&self.thinking_buffer.clone())
            };
            self.thinking_buffer.clear();
            self.in_thinking_block = false;
            self.thinking_extracted = true;
            let mut output = encode_events(flush_events)?;
            for idx in self
                .open_blocks
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
            {
                output.extend(encode_events(self.close_block(idx))?);
            }
            output.extend(self.final_message_bytes()?);
            return Ok(output);
        }

        let mut output = Vec::new();
        for idx in self
            .open_blocks
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
        {
            output.extend(encode_events(self.close_block(idx))?);
        }
        output.extend(self.final_message_bytes()?);
        Ok(output)
    }

    fn final_message_bytes(&self) -> Result<Vec<u8>, GatewayError> {
        let stop_reason = self.stop_reason_override.clone().unwrap_or_else(|| {
            if self.has_tool_use {
                "tool_use"
            } else {
                "end_turn"
            }
            .to_string()
        });
        let input_tokens = self
            .context_input_tokens
            .unwrap_or(self.estimated_input_tokens) as u64;
        encode_events(vec![
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": Value::Null,
                },
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": self.output_tokens as u64,
                }
            }),
            json!({"type": "message_stop"}),
        ])
    }

    fn ensure_text_block_open(&mut self) -> Vec<Value> {
        if let Some(idx) = self.text_block_index {
            if self
                .open_blocks
                .get(&idx)
                .map(|value| value == "text")
                .unwrap_or(false)
            {
                return Vec::new();
            }
        }
        let idx = self.next_block_index;
        self.next_block_index += 1;
        self.text_block_index = Some(idx);
        self.open_blocks.insert(idx, "text".to_string());
        vec![json!({
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""}
        })]
    }

    fn ensure_thinking_block_open(&mut self) -> Vec<Value> {
        if let Some(idx) = self.thinking_block_index {
            if self
                .open_blocks
                .get(&idx)
                .map(|value| value == "thinking")
                .unwrap_or(false)
            {
                return Vec::new();
            }
        }
        let idx = self.next_block_index;
        self.next_block_index += 1;
        self.thinking_block_index = Some(idx);
        self.open_blocks.insert(idx, "thinking".to_string());
        vec![json!({
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "thinking", "thinking": ""}
        })]
    }

    fn close_block(&mut self, idx: usize) -> Vec<Value> {
        if self.open_blocks.remove(&idx).is_none() {
            return Vec::new();
        }
        vec![json!({"type": "content_block_stop", "index": idx})]
    }

    fn emit_text_delta(&mut self, text: &str) -> Vec<Value> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut events = self.ensure_text_block_open();
        let idx = self.text_block_index.unwrap_or_default();
        events.push(json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": text}
        }));
        events
    }

    fn emit_thinking_delta(&mut self, thinking: &str) -> Vec<Value> {
        if thinking.is_empty() {
            return Vec::new();
        }
        let mut events = self.ensure_thinking_block_open();
        let idx = self.thinking_block_index.unwrap_or_default();
        events.push(json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": thinking}
        }));
        events
    }

    fn close_thinking_block(&mut self) -> Vec<Value> {
        let Some(idx) = self.thinking_block_index else {
            return Vec::new();
        };
        let mut events = vec![json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "thinking_delta", "thinking": ""}
        })];
        events.extend(self.close_block(idx));
        events
    }
}
