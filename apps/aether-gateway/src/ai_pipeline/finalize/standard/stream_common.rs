use serde_json::{json, Value};

use crate::gateway::ai_pipeline::adaptation::private_envelope::transform_provider_private_stream_line as transform_envelope_line;
use crate::gateway::ai_pipeline::adaptation::surfaces::provider_adaptation_should_unwrap_stream_envelope;
use crate::gateway::ai_pipeline::finalize::sse::{encode_done_sse, encode_json_sse};
use crate::gateway::ai_pipeline::finalize::standard::{
    aggregate_standard_chat_stream_sync_response, aggregate_standard_cli_stream_sync_response,
    convert_standard_chat_response, convert_standard_cli_response,
};
use crate::gateway::GatewayError;

#[derive(Default)]
pub(crate) struct BufferedStandardConversionStreamState {
    raw: Vec<u8>,
}

impl BufferedStandardConversionStreamState {
    pub(crate) fn transform_line(
        &mut self,
        report_context: &Value,
        line: Vec<u8>,
    ) -> Result<Vec<u8>, GatewayError> {
        if should_unwrap_envelope(report_context) {
            self.raw
                .extend(transform_envelope_line(report_context, line)?);
        } else {
            self.raw.extend_from_slice(&line);
        }
        Ok(Vec::new())
    }

    pub(crate) fn finish_as_chat(
        &mut self,
        report_context: &Value,
    ) -> Result<Vec<u8>, GatewayError> {
        let provider_api_format = report_context
            .get("provider_api_format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let client_api_format = report_context
            .get("client_api_format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if self.raw.is_empty() {
            return Ok(Vec::new());
        }
        let aggregated =
            aggregate_standard_chat_stream_sync_response(&self.raw, provider_api_format.as_str());
        self.raw.clear();
        let Some(aggregated) = aggregated else {
            return Ok(Vec::new());
        };
        let Some(converted) = convert_standard_chat_response(
            &aggregated,
            provider_api_format.as_str(),
            client_api_format.as_str(),
            report_context,
        ) else {
            return Ok(Vec::new());
        };
        emit_chat_stream_for_client_format(&converted, client_api_format.as_str())
    }

    pub(crate) fn finish_as_cli(
        &mut self,
        report_context: &Value,
    ) -> Result<Vec<u8>, GatewayError> {
        let provider_api_format = report_context
            .get("provider_api_format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        let client_api_format = report_context
            .get("client_api_format")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if self.raw.is_empty() {
            return Ok(Vec::new());
        }
        let aggregated =
            aggregate_standard_cli_stream_sync_response(&self.raw, provider_api_format.as_str());
        self.raw.clear();
        let Some(aggregated) = aggregated else {
            return Ok(Vec::new());
        };
        let Some(converted) = convert_standard_cli_response(
            &aggregated,
            provider_api_format.as_str(),
            client_api_format.as_str(),
            report_context,
        ) else {
            return Ok(Vec::new());
        };
        emit_cli_stream_for_client_format(&converted, client_api_format.as_str())
    }
}

fn should_unwrap_envelope(report_context: &Value) -> bool {
    let envelope_name = report_context
        .get("envelope_name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let provider_api_format = report_context
        .get("provider_api_format")
        .and_then(Value::as_str)
        .unwrap_or_default();
    provider_adaptation_should_unwrap_stream_envelope(envelope_name, provider_api_format)
}

fn emit_chat_stream_for_client_format(
    response_body: &Value,
    client_api_format: &str,
) -> Result<Vec<u8>, GatewayError> {
    match client_api_format {
        "openai:chat" => emit_openai_chat_stream(response_body),
        "claude:chat" | "claude:cli" => emit_claude_message_stream(response_body),
        "gemini:chat" | "gemini:cli" => encode_json_sse(None, response_body),
        _ => Ok(Vec::new()),
    }
}

fn emit_cli_stream_for_client_format(
    response_body: &Value,
    client_api_format: &str,
) -> Result<Vec<u8>, GatewayError> {
    match client_api_format {
        "openai:cli" | "openai:compact" => encode_json_sse(
            Some("response.completed"),
            &json!({
                "type": "response.completed",
                "response": response_body,
            }),
        ),
        "claude:cli" => emit_claude_message_stream(response_body),
        "gemini:cli" => encode_json_sse(None, response_body),
        _ => Ok(Vec::new()),
    }
}

fn emit_openai_chat_stream(response_body: &Value) -> Result<Vec<u8>, GatewayError> {
    let body = match response_body.as_object() {
        Some(body) => body,
        None => return Ok(Vec::new()),
    };
    let choice = match body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
    {
        Some(choice) => choice,
        None => return Ok(Vec::new()),
    };
    let message = match choice.get("message").and_then(Value::as_object) {
        Some(message) => message,
        None => return Ok(Vec::new()),
    };
    let content = match extract_openai_chat_content_text(message.get("content")) {
        Some(content) => content,
        None => return Ok(Vec::new()),
    };
    let mut delta = serde_json::Map::new();
    delta.insert("role".to_string(), Value::String("assistant".to_string()));
    if !content.is_empty() {
        delta.insert("content".to_string(), Value::String(content));
    } else if message.get("tool_calls").is_none() {
        delta.insert("content".to_string(), Value::String(String::new()));
    }
    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        delta.insert("tool_calls".to_string(), Value::Array(tool_calls.clone()));
    }
    let chunk = json!({
        "id": body.get("id").cloned().unwrap_or_else(|| Value::String("chatcmpl-local-stream".to_string())),
        "object": "chat.completion.chunk",
        "model": body.get("model").cloned().unwrap_or_else(|| Value::String("unknown".to_string())),
        "choices": [{
            "index": choice.get("index").cloned().unwrap_or_else(|| Value::from(0_u64)),
            "delta": Value::Object(delta),
            "finish_reason": choice.get("finish_reason").cloned().unwrap_or(Value::Null),
        }]
    });
    let mut out = encode_json_sse(None, &chunk)?;
    out.extend(encode_done_sse());
    Ok(out)
}

fn emit_claude_message_stream(response_body: &Value) -> Result<Vec<u8>, GatewayError> {
    let body = match response_body.as_object() {
        Some(body) => body,
        None => return Ok(Vec::new()),
    };
    let message_id = body
        .get("id")
        .cloned()
        .unwrap_or_else(|| Value::String("msg-local-stream".to_string()));
    let model = body
        .get("model")
        .cloned()
        .unwrap_or_else(|| Value::String("unknown".to_string()));
    let content_blocks = match body.get("content").and_then(Value::as_array) {
        Some(content) => content,
        None => return Ok(Vec::new()),
    };

    let mut out = encode_json_sse(
        Some("message_start"),
        &json!({
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
            }
        }),
    )?;

    for (index, block) in content_blocks.iter().enumerate() {
        let Some(block_object) = block.as_object() else {
            continue;
        };
        match block_object
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("text")
        {
            "text" => {
                out.extend(encode_json_sse(
                    Some("content_block_start"),
                    &json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "text",
                            "text": "",
                        }
                    }),
                )?);
                if let Some(text) = block_object.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        out.extend(encode_json_sse(
                            Some("content_block_delta"),
                            &json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "text_delta",
                                    "text": text,
                                }
                            }),
                        )?);
                    }
                }
                out.extend(encode_json_sse(
                    Some("content_block_stop"),
                    &json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                )?);
            }
            "tool_use" => {
                out.extend(encode_json_sse(
                    Some("content_block_start"),
                    &json!({
                        "type": "content_block_start",
                        "index": index,
                        "content_block": block_object,
                    }),
                )?);
                out.extend(encode_json_sse(
                    Some("content_block_stop"),
                    &json!({
                        "type": "content_block_stop",
                        "index": index,
                    }),
                )?);
            }
            _ => {}
        }
    }

    let mut delta = serde_json::Map::new();
    delta.insert(
        "stop_reason".to_string(),
        body.get("stop_reason").cloned().unwrap_or(Value::Null),
    );
    if let Some(stop_sequence) = body.get("stop_sequence").cloned() {
        delta.insert("stop_sequence".to_string(), stop_sequence);
    }
    let mut message_delta = serde_json::Map::new();
    message_delta.insert(
        "type".to_string(),
        Value::String("message_delta".to_string()),
    );
    message_delta.insert("delta".to_string(), Value::Object(delta));
    if let Some(usage) = body.get("usage").cloned() {
        message_delta.insert("usage".to_string(), usage);
    }
    out.extend(encode_json_sse(
        Some("message_delta"),
        &Value::Object(message_delta),
    )?);
    out.extend(encode_json_sse(
        Some("message_stop"),
        &json!({
            "type": "message_stop",
        }),
    )?);
    Ok(out)
}

fn extract_openai_chat_content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::Null => Some(String::new()),
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let part = part.as_object()?;
                let part_type = part
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .trim()
                    .to_ascii_lowercase();
                if matches!(part_type.as_str(), "text" | "output_text") {
                    if let Some(piece) = part.get("text").and_then(Value::as_str) {
                        text.push_str(piece);
                    }
                }
            }
            Some(text)
        }
        _ => None,
    }
}
