use serde_json::{json, Map, Value};
use uuid::Uuid;

use super::super::to_openai_chat::{extract_openai_text_content, parse_openai_tool_result_content};
use super::shared::parse_openai_tool_arguments;
use crate::gateway::ai_pipeline::planner::standard::{
    copy_request_number_field, map_openai_reasoning_effort_to_claude_output,
    parse_openai_stop_sequences, resolve_openai_chat_max_tokens,
};

pub(crate) fn convert_openai_chat_request_to_claude_request(
    body_json: &Value,
    mapped_model: &str,
    upstream_is_stream: bool,
) -> Option<Value> {
    let request = body_json.as_object()?;
    let mut system_segments = Vec::new();
    let mut messages = Vec::new();

    if let Some(message_values) = request.get("messages").and_then(Value::as_array) {
        for message in message_values {
            let message_object = message.as_object()?;
            let role = message_object
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase();
            match role.as_str() {
                "system" | "developer" => {
                    let text = extract_openai_text_content(message_object.get("content"))?;
                    if !text.trim().is_empty() {
                        system_segments.push(text);
                    }
                }
                "user" => {
                    let blocks = convert_openai_content_to_claude_blocks(
                        message_object.get("content"),
                        true,
                    )?;
                    if !blocks.is_empty() {
                        messages.push(build_claude_message("user", blocks));
                    }
                }
                "assistant" => {
                    let mut blocks = convert_openai_content_to_claude_blocks(
                        message_object.get("content"),
                        false,
                    )?;
                    if let Some(tool_calls) =
                        message_object.get("tool_calls").and_then(Value::as_array)
                    {
                        for tool_call in tool_calls {
                            let tool_call_object = tool_call.as_object()?;
                            let function = tool_call_object.get("function")?.as_object()?;
                            let tool_name = function
                                .get("name")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|value| !value.is_empty())?
                                .to_string();
                            let tool_call_id = tool_call_object
                                .get("id")
                                .and_then(Value::as_str)
                                .map(str::trim)
                                .filter(|value| !value.is_empty())
                                .map(ToOwned::to_owned)
                                .unwrap_or_else(|| format!("toolu_{}", Uuid::new_v4().simple()));
                            let tool_input =
                                parse_openai_tool_arguments(function.get("arguments"))?;
                            blocks.push(json!({
                                "type": "tool_use",
                                "id": tool_call_id,
                                "name": tool_name,
                                "input": tool_input,
                            }));
                        }
                    }
                    if !blocks.is_empty() {
                        messages.push(build_claude_message("assistant", blocks));
                    }
                }
                "tool" => {
                    let tool_use_id = message_object
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())?
                        .to_string();
                    let tool_result =
                        parse_openai_tool_result_content(message_object.get("content"));
                    messages.push(json!({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_result,
                            "is_error": false,
                        }],
                    }));
                }
                _ => {}
            }
        }
    }

    let mut output = Map::new();
    output.insert("model".to_string(), Value::String(mapped_model.to_string()));
    output.insert(
        "messages".to_string(),
        Value::Array(compact_claude_messages(messages)),
    );
    output.insert(
        "max_tokens".to_string(),
        Value::from(resolve_openai_chat_max_tokens(request)),
    );

    let system_text = system_segments
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system_text.is_empty() {
        output.insert("system".to_string(), Value::String(system_text));
    }
    if upstream_is_stream {
        output.insert("stream".to_string(), Value::Bool(true));
    }
    copy_request_number_field(request, &mut output, "temperature");
    copy_request_number_field(request, &mut output, "top_p");
    copy_request_number_field(request, &mut output, "top_k");
    if let Some(stop_sequences) = parse_openai_stop_sequences(request.get("stop")) {
        output.insert("stop_sequences".to_string(), Value::Array(stop_sequences));
    }
    if let Some(tools) = convert_openai_tools_to_claude(request.get("tools")) {
        output.insert("tools".to_string(), Value::Array(tools));
    }
    if let Some(tool_choice) = convert_openai_tool_choice_to_claude(request.get("tool_choice")) {
        output.insert("tool_choice".to_string(), tool_choice);
    }
    if let Some(metadata) = request.get("metadata").cloned() {
        output.insert("metadata".to_string(), metadata);
    }
    if let Some(reasoning_effort) = request.get("reasoning_effort").and_then(Value::as_str) {
        if let Some(output_effort) = map_openai_reasoning_effort_to_claude_output(reasoning_effort)
        {
            output.insert(
                "output_config".to_string(),
                json!({ "effort": output_effort }),
            );
        }
    }

    Some(Value::Object(output))
}

fn convert_openai_content_to_claude_blocks(
    content: Option<&Value>,
    allow_images: bool,
) -> Option<Vec<Value>> {
    match content {
        None | Some(Value::Null) => Some(Vec::new()),
        Some(Value::String(text)) => {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                Some(Vec::new())
            } else {
                Some(vec![json!({ "type": "text", "text": text })])
            }
        }
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::new();
            for part in parts {
                let part_object = part.as_object()?;
                let part_type = part_object
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match part_type {
                    "text" | "input_text" => {
                        if let Some(text) = part_object.get("text").and_then(Value::as_str) {
                            if !text.trim().is_empty() {
                                blocks.push(json!({ "type": "text", "text": text }));
                            }
                        }
                    }
                    "image_url" | "input_image" if allow_images => {
                        let url = part_object
                            .get("image_url")
                            .and_then(|value| {
                                value.as_str().map(ToOwned::to_owned).or_else(|| {
                                    value
                                        .as_object()
                                        .and_then(|object| object.get("url"))
                                        .and_then(Value::as_str)
                                        .map(ToOwned::to_owned)
                                })
                            })
                            .filter(|value| !value.trim().is_empty())?;
                        blocks.push(json!({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            }
                        }));
                    }
                    _ => {}
                }
            }
            Some(blocks)
        }
        _ => None,
    }
}

fn convert_openai_tools_to_claude(tools: Option<&Value>) -> Option<Vec<Value>> {
    let tool_values = tools?.as_array()?;
    let mut converted = Vec::new();
    for tool in tool_values {
        let tool_object = tool.as_object()?;
        if tool_object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|value| value != "function")
        {
            continue;
        }
        let function = tool_object.get("function")?.as_object()?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())?;
        let mut converted_tool = Map::new();
        converted_tool.insert("name".to_string(), Value::String(name.to_string()));
        if let Some(description) = function.get("description").cloned() {
            converted_tool.insert("description".to_string(), description);
        }
        converted_tool.insert(
            "input_schema".to_string(),
            function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({})),
        );
        converted.push(Value::Object(converted_tool));
    }
    (!converted.is_empty()).then_some(converted)
}

fn convert_openai_tool_choice_to_claude(tool_choice: Option<&Value>) -> Option<Value> {
    let tool_choice = tool_choice?;
    match tool_choice {
        Value::String(value) => match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(json!({ "type": "none" })),
            "required" => Some(json!({ "type": "any" })),
            "auto" => Some(json!({ "type": "auto" })),
            _ => None,
        },
        Value::Object(object) => {
            let function_name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            Some(json!({
                "type": "tool",
                "name": function_name,
            }))
        }
        _ => None,
    }
}

fn compact_claude_messages(messages: Vec<Value>) -> Vec<Value> {
    let mut compact: Vec<Value> = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if let Some(last) = compact.last_mut() {
            let last_role = last
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if last_role == role {
                merge_claude_message_content(last, message);
                continue;
            }
        }
        compact.push(message);
    }
    if compact
        .first()
        .and_then(|value| value.get("role"))
        .and_then(Value::as_str)
        .is_some_and(|value| value == "assistant")
    {
        compact.insert(0, json!({ "role": "user", "content": "" }));
    }
    compact
}

fn merge_claude_message_content(target: &mut Value, message: Value) {
    let Some(target_object) = target.as_object_mut() else {
        return;
    };
    let incoming_content = message.get("content").cloned().unwrap_or(Value::Null);
    let merged_blocks = extract_claude_content_blocks(target_object.get("content"))
        .into_iter()
        .chain(extract_claude_content_blocks(Some(&incoming_content)))
        .collect::<Vec<_>>();
    target_object.insert(
        "content".to_string(),
        simplify_claude_content(merged_blocks),
    );
}

fn build_claude_message(role: &str, blocks: Vec<Value>) -> Value {
    json!({
        "role": role,
        "content": simplify_claude_content(blocks),
    })
}

fn simplify_claude_content(blocks: Vec<Value>) -> Value {
    if blocks.is_empty() {
        return Value::String(String::new());
    }
    let mut text_values = Vec::new();
    for block in &blocks {
        let Some(block_object) = block.as_object() else {
            return Value::Array(blocks);
        };
        if block_object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|value| value == "text")
        {
            if let Some(text) = block_object.get("text").and_then(Value::as_str) {
                text_values.push(text.to_string());
                continue;
            }
        }
        return Value::Array(blocks);
    }
    Value::String(text_values.join("\n"))
}

fn extract_claude_content_blocks(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) if !text.is_empty() => vec![json!({
            "type": "text",
            "text": text,
        })],
        Some(Value::Array(blocks)) => blocks.clone(),
        _ => Vec::new(),
    }
}
