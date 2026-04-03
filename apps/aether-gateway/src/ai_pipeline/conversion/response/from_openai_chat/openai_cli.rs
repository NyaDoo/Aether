use serde_json::{json, Value};

use super::shared::{build_openai_cli_response, canonicalize_tool_arguments};

pub(crate) fn convert_openai_chat_response_to_openai_cli(
    body_json: &Value,
    report_context: &Value,
    compact: bool,
) -> Option<Value> {
    let body = body_json.as_object()?;
    let choices = body.get("choices")?.as_array()?;
    let first_choice = choices.first()?.as_object()?;
    let message = first_choice.get("message")?.as_object()?;
    let mut text = String::new();
    match message.get("content") {
        Some(Value::String(value)) => text.push_str(value),
        Some(Value::Array(parts)) => {
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
        }
        Some(Value::Null) | None => {}
        _ => return None,
    }

    let mut function_calls = Vec::new();
    if let Some(tool_call_values) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_call_values {
            let tool_call = tool_call.as_object()?;
            let function = tool_call.get("function")?.as_object()?;
            let tool_name = function
                .get("name")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            function_calls.push(json!({
                "type": "function_call",
                "id": tool_call.get("id").cloned().unwrap_or(Value::Null),
                "call_id": tool_call.get("id").cloned().unwrap_or(Value::Null),
                "name": tool_name,
                "arguments": canonicalize_tool_arguments(function.get("arguments").cloned()),
            }));
        }
    }

    let usage = body.get("usage").and_then(Value::as_object);
    let prompt_tokens = usage
        .and_then(|value| value.get("prompt_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|value| value.get("completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_tokens = usage
        .and_then(|value| value.get("total_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(prompt_tokens + output_tokens);
    let response_id = if compact {
        body.get("id")
            .and_then(Value::as_str)
            .map(|value| value.replace("chatcmpl", "resp"))
            .unwrap_or_else(|| "resp-local-finalize".to_string())
    } else {
        body.get("id")
            .and_then(Value::as_str)
            .map(|value| value.replace("chatcmpl", "resp"))
            .unwrap_or_else(|| "resp-local-finalize".to_string())
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .or_else(|| report_context.get("mapped_model").and_then(Value::as_str))
        .or_else(|| report_context.get("model").and_then(Value::as_str))
        .unwrap_or("unknown");

    Some(build_openai_cli_response(
        &response_id,
        model,
        &text,
        function_calls,
        prompt_tokens,
        output_tokens,
        total_tokens,
    ))
}
