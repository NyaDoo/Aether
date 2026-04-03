//! Standard finalize surface for standard contract sync/stream compilation.

use std::collections::BTreeMap;

use serde_json::Value;
use serde_json::{json, Map};

use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

#[path = "claude/chat.rs"]
mod claude_chat;
#[path = "claude/chat_stream.rs"]
mod claude_chat_stream;
#[path = "claude/cli.rs"]
mod claude_cli;
#[path = "claude/cli_stream.rs"]
mod claude_cli_stream;
#[path = "gemini/chat.rs"]
mod gemini_chat;
#[path = "gemini/chat_stream.rs"]
mod gemini_chat_stream;
#[path = "gemini/cli.rs"]
mod gemini_cli;
#[path = "gemini/cli_stream.rs"]
mod gemini_cli_stream;
#[path = "openai/chat.rs"]
mod openai_chat;
#[path = "openai/chat_stream.rs"]
mod openai_chat_stream;
#[path = "openai/cli.rs"]
mod openai_cli;
#[path = "openai/cli_stream.rs"]
mod openai_cli_stream;

#[path = "stream_common.rs"]
mod stream_common;

pub(crate) use crate::gateway::ai_pipeline::conversion::response::{
    build_openai_cli_response, convert_claude_chat_response_to_openai_chat,
    convert_claude_cli_response_to_openai_cli, convert_gemini_chat_response_to_openai_chat,
    convert_gemini_cli_response_to_openai_cli, convert_openai_chat_response_to_claude_chat,
    convert_openai_chat_response_to_gemini_chat, convert_openai_chat_response_to_openai_cli,
    convert_openai_cli_response_to_openai_chat,
};
pub(crate) use claude_chat::maybe_build_local_claude_sync_response;
pub(crate) use claude_chat_stream::{
    aggregate_claude_stream_sync_response, maybe_build_local_claude_stream_sync_response,
};
pub(crate) use claude_cli_stream::maybe_build_local_claude_cli_stream_sync_response;
pub(crate) use gemini_chat::maybe_build_local_gemini_sync_response;
pub(crate) use gemini_chat_stream::{
    aggregate_gemini_stream_sync_response, maybe_build_local_gemini_stream_sync_response,
};
pub(crate) use gemini_cli_stream::maybe_build_local_gemini_cli_stream_sync_response;
pub(crate) use openai_chat::{
    aggregate_openai_chat_stream_sync_response,
    maybe_build_local_openai_chat_cross_format_stream_sync_response,
    maybe_build_local_openai_chat_cross_format_sync_response,
    maybe_build_local_openai_chat_stream_sync_response,
    maybe_build_local_openai_chat_sync_response,
};
pub(crate) use openai_chat_stream::{
    ClaudeToOpenAIChatStreamState, GeminiToOpenAIChatStreamState, OpenAICliToOpenAIChatStreamState,
};
pub(crate) use openai_cli::{
    aggregate_openai_cli_stream_sync_response,
    maybe_build_local_openai_cli_cross_format_stream_sync_response,
    maybe_build_local_openai_cli_cross_format_sync_response,
    maybe_build_local_openai_cli_stream_sync_response,
};
pub(crate) use openai_cli_stream::BufferedCliConversionStreamState;
pub(crate) use stream_common::BufferedStandardConversionStreamState;

pub(crate) fn aggregate_standard_chat_stream_sync_response(
    body: &[u8],
    provider_api_format: &str,
) -> Option<Value> {
    match provider_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:chat" => aggregate_openai_chat_stream_sync_response(body),
        "openai:cli" | "openai:compact" => aggregate_openai_cli_stream_sync_response(body),
        "claude:chat" | "claude:cli" => aggregate_claude_stream_sync_response(body),
        "gemini:chat" | "gemini:cli" => aggregate_gemini_stream_sync_response(body),
        _ => None,
    }
}

pub(crate) fn convert_standard_chat_response(
    body_json: &Value,
    provider_api_format: &str,
    client_api_format: &str,
    report_context: &Value,
) -> Option<Value> {
    let canonical = match provider_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:chat" => body_json.clone(),
        "openai:cli" | "openai:compact" => {
            convert_openai_cli_response_to_openai_chat(body_json, report_context)?
        }
        "claude:chat" | "claude:cli" => {
            convert_claude_chat_response_to_openai_chat(body_json, report_context)?
        }
        "gemini:chat" | "gemini:cli" => {
            convert_gemini_chat_response_to_openai_chat(body_json, report_context)?
        }
        _ => return None,
    };

    match client_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:chat" => Some(canonical),
        "claude:chat" => convert_openai_chat_response_to_claude_chat(&canonical, report_context),
        "gemini:chat" => convert_openai_chat_response_to_gemini_chat(&canonical, report_context),
        _ => None,
    }
}

pub(crate) fn aggregate_standard_cli_stream_sync_response(
    body: &[u8],
    provider_api_format: &str,
) -> Option<Value> {
    aggregate_standard_chat_stream_sync_response(body, provider_api_format)
}

pub(crate) fn convert_standard_cli_response(
    body_json: &Value,
    provider_api_format: &str,
    client_api_format: &str,
    report_context: &Value,
) -> Option<Value> {
    let canonical = match provider_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:cli" | "openai:compact" => {
            convert_openai_cli_response_to_openai_chat(body_json, report_context)?
        }
        _ => convert_standard_chat_response(
            body_json,
            provider_api_format,
            "openai:chat",
            report_context,
        )?,
    };

    match client_api_format.trim().to_ascii_lowercase().as_str() {
        "openai:cli" => {
            convert_openai_chat_response_to_openai_cli(&canonical, report_context, false)
        }
        "openai:compact" => {
            convert_openai_chat_response_to_openai_cli(&canonical, report_context, true)
        }
        "claude:cli" => convert_openai_chat_response_to_claude_chat(&canonical, report_context),
        "gemini:cli" => convert_openai_chat_response_to_gemini_chat(&canonical, report_context),
        _ => None,
    }
}
