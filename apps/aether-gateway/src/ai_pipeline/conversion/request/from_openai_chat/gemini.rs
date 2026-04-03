use std::collections::BTreeMap;

use serde_json::{json, Map, Value};
use uuid::Uuid;

use super::super::to_openai_chat::{extract_openai_text_content, parse_openai_tool_result_content};
use super::shared::parse_openai_tool_arguments;
use crate::gateway::ai_pipeline::planner::standard::{
    copy_request_number_field_as, map_openai_reasoning_effort_to_gemini_budget,
    parse_openai_stop_sequences, value_as_u64,
};

pub(crate) fn convert_openai_chat_request_to_gemini_request(
    body_json: &Value,
    mapped_model: &str,
    upstream_is_stream: bool,
) -> Option<Value> {
    let request = body_json.as_object()?;
    let mut system_segments = Vec::new();
    let mut tool_name_by_id = BTreeMap::new();
    let mut contents = Vec::new();

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
                    let parts = convert_openai_content_to_gemini_parts(
                        message_object.get("content"),
                        true,
                    )?;
                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "user",
                            "parts": parts,
                        }));
                    }
                }
                "assistant" => {
                    let mut parts = convert_openai_content_to_gemini_parts(
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
                            tool_name_by_id.insert(tool_call_id.clone(), tool_name.clone());
                            parts.push(json!({
                                "functionCall": {
                                    "name": tool_name,
                                    "args": tool_input,
                                    "id": tool_call_id,
                                }
                            }));
                        }
                    }
                    if !parts.is_empty() {
                        contents.push(json!({
                            "role": "model",
                            "parts": parts,
                        }));
                    }
                }
                "tool" => {
                    let tool_use_id = message_object
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|value| !value.is_empty())?
                        .to_string();
                    let tool_name = tool_name_by_id
                        .get(&tool_use_id)
                        .cloned()
                        .unwrap_or_else(|| tool_use_id.clone());
                    let tool_result =
                        parse_openai_tool_result_content(message_object.get("content"));
                    contents.push(json!({
                        "role": "user",
                        "parts": [{
                            "functionResponse": {
                                "name": tool_name,
                                "id": tool_use_id,
                                "response": {
                                    "result": tool_result,
                                },
                            }
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
        "contents".to_string(),
        Value::Array(compact_gemini_contents(contents)),
    );
    if upstream_is_stream {
        output.insert("stream".to_string(), Value::Bool(true));
    }
    let system_text = system_segments
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !system_text.is_empty() {
        output.insert(
            "systemInstruction".to_string(),
            json!({ "parts": [{ "text": system_text }] }),
        );
    }

    let mut generation_config = Map::new();
    if let Some(max_tokens) = request
        .get("max_completion_tokens")
        .and_then(value_as_u64)
        .or_else(|| request.get("max_tokens").and_then(value_as_u64))
    {
        generation_config.insert("maxOutputTokens".to_string(), Value::from(max_tokens));
    }
    copy_request_number_field_as(
        request,
        &mut generation_config,
        "temperature",
        "temperature",
    );
    copy_request_number_field_as(request, &mut generation_config, "top_p", "topP");
    copy_request_number_field_as(request, &mut generation_config, "top_k", "topK");
    if let Some(stop_sequences) = parse_openai_stop_sequences(request.get("stop")) {
        generation_config.insert("stopSequences".to_string(), Value::Array(stop_sequences));
    }
    if let Some(reasoning_effort) = request.get("reasoning_effort").and_then(Value::as_str) {
        if let Some(thinking_budget) =
            map_openai_reasoning_effort_to_gemini_budget(reasoning_effort)
        {
            generation_config.insert(
                "thinkingConfig".to_string(),
                json!({
                    "includeThoughts": true,
                    "thinkingBudget": thinking_budget,
                }),
            );
        }
    }
    if !generation_config.is_empty() {
        output.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }
    if let Some(tools) = convert_openai_tools_to_gemini(request.get("tools")) {
        output.insert("tools".to_string(), tools);
    }
    if let Some(tool_config) = convert_openai_tool_choice_to_gemini(request.get("tool_choice")) {
        output.insert("toolConfig".to_string(), tool_config);
    }
    if let Some(extra_body) = request.get("extra_body").and_then(Value::as_object) {
        if let Some(google) = extra_body.get("google").and_then(Value::as_object) {
            if let Some(existing) = output
                .get_mut("generationConfig")
                .and_then(Value::as_object_mut)
            {
                if let Some(response_modalities) = google.get("response_modalities").cloned() {
                    existing.insert("responseModalities".to_string(), response_modalities);
                }
                if let Some(thinking_config) = google.get("thinking_config").cloned() {
                    existing
                        .entry("thinkingConfig".to_string())
                        .or_insert(thinking_config);
                }
            }
        }
    }

    Some(Value::Object(output))
}

fn convert_openai_content_to_gemini_parts(
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
                Some(vec![json!({ "text": text })])
            }
        }
        Some(Value::Array(parts)) => {
            let mut converted = Vec::new();
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
                                converted.push(json!({ "text": text }));
                            }
                        }
                    }
                    "image_url" | "input_image" if allow_images => return None,
                    _ => {}
                }
            }
            Some(converted)
        }
        _ => None,
    }
}

fn convert_openai_tools_to_gemini(tools: Option<&Value>) -> Option<Value> {
    let tool_values = tools?.as_array()?;
    let mut declarations = Vec::new();
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
        let mut declaration = Map::new();
        declaration.insert("name".to_string(), Value::String(name.to_string()));
        if let Some(description) = function.get("description").cloned() {
            declaration.insert("description".to_string(), description);
        }
        declaration.insert(
            "parameters".to_string(),
            function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({})),
        );
        declarations.push(Value::Object(declaration));
    }
    (!declarations.is_empty()).then(|| json!([{ "functionDeclarations": declarations }]))
}

fn convert_openai_tool_choice_to_gemini(tool_choice: Option<&Value>) -> Option<Value> {
    let tool_choice = tool_choice?;
    match tool_choice {
        Value::String(value) => {
            let mode = match value.trim().to_ascii_lowercase().as_str() {
                "none" => "NONE",
                "required" => "ANY",
                "auto" => "AUTO",
                _ => return None,
            };
            Some(json!({
                "functionCallingConfig": {
                    "mode": mode,
                }
            }))
        }
        Value::Object(object) => {
            let function_name = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())?;
            Some(json!({
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [function_name],
                }
            }))
        }
        _ => None,
    }
}

fn compact_gemini_contents(contents: Vec<Value>) -> Vec<Value> {
    let mut compact: Vec<Value> = Vec::new();
    for content in contents {
        let role = content
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let parts = content
            .get("parts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if parts.is_empty() {
            continue;
        }
        if let Some(last) = compact.last_mut() {
            let last_role = last
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if last_role == role {
                if let Some(last_parts) = last.get_mut("parts").and_then(Value::as_array_mut) {
                    last_parts.extend(parts);
                }
                continue;
            }
        }
        compact.push(content);
    }
    compact
}
