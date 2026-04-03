use serde_json::Value;

use crate::gateway::ai_pipeline::adaptation::private_envelope::transform_provider_private_stream_line as transform_envelope_line;
use crate::gateway::ai_pipeline::finalize::sse::{encode_done_sse, encode_json_sse};
use crate::gateway::ai_pipeline::finalize::standard::{
    BufferedCliConversionStreamState, BufferedStandardConversionStreamState,
    ClaudeToOpenAIChatStreamState, GeminiToOpenAIChatStreamState, OpenAICliToOpenAIChatStreamState,
};
use crate::gateway::ai_pipeline::runtime::KiroToClaudeCliStreamState;
use crate::gateway::GatewayError;

use super::sync::{
    aggregate_claude_stream_sync_response, aggregate_gemini_stream_sync_response,
    convert_claude_cli_response_to_openai_cli, convert_gemini_cli_response_to_openai_cli,
};
enum RewriteMode {
    EnvelopeUnwrap,
    ClaudeToOpenAIChat(ClaudeToOpenAIChatStreamState),
    GeminiToOpenAIChat(GeminiToOpenAIChatStreamState),
    OpenAICliToOpenAIChat(OpenAICliToOpenAIChatStreamState),
    ClaudeToOpenAICli(BufferedCliConversionStreamState),
    GeminiToOpenAICli(BufferedCliConversionStreamState),
    AntigravityGeminiToOpenAIChat(GeminiToOpenAIChatStreamState),
    AntigravityGeminiToOpenAICli(BufferedCliConversionStreamState),
    KiroToClaudeCli(KiroToClaudeCliStreamState),
    StandardChat(BufferedStandardConversionStreamState),
    StandardCli(BufferedStandardConversionStreamState),
}

pub(crate) struct LocalStreamRewriter {
    report_context: Value,
    buffered: Vec<u8>,
    mode: RewriteMode,
}

pub(crate) fn maybe_build_local_stream_rewriter(
    report_context: Option<&Value>,
) -> Option<LocalStreamRewriter> {
    let report_context = report_context?;
    let needs_conversion = report_context
        .get("needs_conversion")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let envelope_name = report_context
        .get("envelope_name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
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

    let mode = if needs_conversion {
        match (
            envelope_name.as_str(),
            provider_api_format.as_str(),
            client_api_format.as_str(),
        ) {
            ("", "claude:chat", "openai:chat") => {
                RewriteMode::ClaudeToOpenAIChat(ClaudeToOpenAIChatStreamState::default())
            }
            ("", "gemini:chat", "openai:chat") => {
                RewriteMode::GeminiToOpenAIChat(GeminiToOpenAIChatStreamState::default())
            }
            ("", "openai:cli", "openai:chat") | ("", "openai:compact", "openai:chat") => {
                RewriteMode::OpenAICliToOpenAIChat(OpenAICliToOpenAIChatStreamState::default())
            }
            ("", "claude:cli", "openai:cli") => {
                RewriteMode::ClaudeToOpenAICli(BufferedCliConversionStreamState::default())
            }
            ("", "claude:cli", "openai:compact") => {
                RewriteMode::ClaudeToOpenAICli(BufferedCliConversionStreamState::default())
            }
            ("", "gemini:cli", "openai:cli") => {
                RewriteMode::GeminiToOpenAICli(BufferedCliConversionStreamState::default())
            }
            ("", "gemini:cli", "openai:compact") => {
                RewriteMode::GeminiToOpenAICli(BufferedCliConversionStreamState::default())
            }
            ("antigravity:v1internal", "gemini:chat", "openai:chat") => {
                RewriteMode::AntigravityGeminiToOpenAIChat(GeminiToOpenAIChatStreamState::default())
            }
            ("antigravity:v1internal", "gemini:cli", "openai:cli") => {
                RewriteMode::AntigravityGeminiToOpenAICli(
                    BufferedCliConversionStreamState::default(),
                )
            }
            ("antigravity:v1internal", "gemini:cli", "openai:compact") => {
                RewriteMode::AntigravityGeminiToOpenAICli(
                    BufferedCliConversionStreamState::default(),
                )
            }
            _ if is_standard_chat_client_api_format(client_api_format.as_str())
                && is_standard_provider_api_format(provider_api_format.as_str()) =>
            {
                RewriteMode::StandardChat(BufferedStandardConversionStreamState::default())
            }
            _ if is_standard_cli_client_api_format(client_api_format.as_str())
                && is_standard_provider_api_format(provider_api_format.as_str()) =>
            {
                RewriteMode::StandardCli(BufferedStandardConversionStreamState::default())
            }
            _ => return None,
        }
    } else {
        match envelope_name.as_str() {
            "antigravity:v1internal" => {
                if provider_api_format == client_api_format
                    && matches!(provider_api_format.as_str(), "gemini:chat" | "gemini:cli")
                {
                    RewriteMode::EnvelopeUnwrap
                } else {
                    return None;
                }
            }
            "gemini_cli:v1internal" => {
                if provider_api_format == "gemini:cli" && client_api_format == "gemini:cli" {
                    RewriteMode::EnvelopeUnwrap
                } else {
                    return None;
                }
            }
            "kiro:generateassistantresponse" => {
                if provider_api_format == "claude:cli" && client_api_format == "claude:cli" {
                    RewriteMode::KiroToClaudeCli(KiroToClaudeCliStreamState::new(report_context))
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    };

    Some(LocalStreamRewriter {
        report_context: report_context.clone(),
        buffered: Vec::new(),
        mode,
    })
}

impl LocalStreamRewriter {
    pub(crate) fn push_chunk(&mut self, chunk: &[u8]) -> Result<Vec<u8>, GatewayError> {
        if let RewriteMode::KiroToClaudeCli(state) = &mut self.mode {
            return state.push_chunk(&self.report_context, chunk);
        }
        self.buffered.extend_from_slice(chunk);
        let mut output = Vec::new();
        while let Some(line_end) = self.buffered.iter().position(|byte| *byte == b'\n') {
            let line = self.buffered.drain(..=line_end).collect::<Vec<_>>();
            output.extend(self.transform_line(line)?);
        }
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u8>, GatewayError> {
        if let RewriteMode::KiroToClaudeCli(state) = &mut self.mode {
            return state.finish(&self.report_context);
        }
        if self.buffered.is_empty() {
            match &mut self.mode {
                RewriteMode::ClaudeToOpenAIChat(state) => return Ok(state.finish()),
                RewriteMode::GeminiToOpenAIChat(state) => {
                    return state.finish(&self.report_context);
                }
                RewriteMode::OpenAICliToOpenAIChat(state) => {
                    return state.finish(&self.report_context);
                }
                RewriteMode::ClaudeToOpenAICli(state) => {
                    return state.finish(
                        &self.report_context,
                        aggregate_claude_stream_sync_response,
                        convert_claude_cli_response_to_openai_cli,
                    );
                }
                RewriteMode::GeminiToOpenAICli(state) => {
                    return state.finish(
                        &self.report_context,
                        aggregate_gemini_stream_sync_response,
                        convert_gemini_cli_response_to_openai_cli,
                    );
                }
                RewriteMode::AntigravityGeminiToOpenAIChat(state) => {
                    return state.finish(&self.report_context);
                }
                RewriteMode::AntigravityGeminiToOpenAICli(state) => {
                    return state.finish(
                        &self.report_context,
                        aggregate_gemini_stream_sync_response,
                        convert_gemini_cli_response_to_openai_cli,
                    );
                }
                RewriteMode::KiroToClaudeCli(_) => {}
                RewriteMode::StandardChat(state) => {
                    return state.finish_as_chat(&self.report_context)
                }
                RewriteMode::StandardCli(state) => {
                    return state.finish_as_cli(&self.report_context)
                }
                RewriteMode::EnvelopeUnwrap => {}
            }
            return Ok(Vec::new());
        }
        let line = std::mem::take(&mut self.buffered);
        let mut output = self.transform_line(line)?;
        match &mut self.mode {
            RewriteMode::ClaudeToOpenAIChat(state) => {
                output.extend(state.finish());
            }
            RewriteMode::GeminiToOpenAIChat(state) => {
                output.extend(state.finish(&self.report_context)?);
            }
            RewriteMode::OpenAICliToOpenAIChat(state) => {
                output.extend(state.finish(&self.report_context)?);
            }
            RewriteMode::ClaudeToOpenAICli(state) => {
                output.extend(state.finish(
                    &self.report_context,
                    aggregate_claude_stream_sync_response,
                    convert_claude_cli_response_to_openai_cli,
                )?);
            }
            RewriteMode::GeminiToOpenAICli(state) => {
                output.extend(state.finish(
                    &self.report_context,
                    aggregate_gemini_stream_sync_response,
                    convert_gemini_cli_response_to_openai_cli,
                )?);
            }
            RewriteMode::AntigravityGeminiToOpenAIChat(state) => {
                output.extend(state.finish(&self.report_context)?);
            }
            RewriteMode::AntigravityGeminiToOpenAICli(state) => {
                output.extend(state.finish(
                    &self.report_context,
                    aggregate_gemini_stream_sync_response,
                    convert_gemini_cli_response_to_openai_cli,
                )?);
            }
            RewriteMode::KiroToClaudeCli(_) => {}
            RewriteMode::StandardChat(state) => {
                output.extend(state.finish_as_chat(&self.report_context)?);
            }
            RewriteMode::StandardCli(state) => {
                output.extend(state.finish_as_cli(&self.report_context)?);
            }
            RewriteMode::EnvelopeUnwrap => {}
        }
        Ok(output)
    }

    fn transform_line(&mut self, line: Vec<u8>) -> Result<Vec<u8>, GatewayError> {
        match &mut self.mode {
            RewriteMode::EnvelopeUnwrap => transform_envelope_line(&self.report_context, line),
            RewriteMode::ClaudeToOpenAIChat(state) => {
                state.transform_line(&self.report_context, line)
            }
            RewriteMode::GeminiToOpenAIChat(state) => {
                state.transform_line(&self.report_context, line)
            }
            RewriteMode::OpenAICliToOpenAIChat(state) => {
                state.transform_line(&self.report_context, line)
            }
            RewriteMode::ClaudeToOpenAICli(state) | RewriteMode::GeminiToOpenAICli(state) => {
                state.transform_line(line)
            }
            RewriteMode::AntigravityGeminiToOpenAIChat(state) => {
                let unwrapped = transform_envelope_line(&self.report_context, line)?;
                if unwrapped.is_empty() {
                    Ok(Vec::new())
                } else {
                    state.transform_line(&self.report_context, unwrapped)
                }
            }
            RewriteMode::AntigravityGeminiToOpenAICli(state) => {
                let unwrapped = transform_envelope_line(&self.report_context, line)?;
                if unwrapped.is_empty() {
                    Ok(Vec::new())
                } else {
                    state.transform_line(unwrapped)
                }
            }
            RewriteMode::StandardChat(state) | RewriteMode::StandardCli(state) => {
                state.transform_line(&self.report_context, line)
            }
            RewriteMode::KiroToClaudeCli(_) => Ok(Vec::new()),
        }
    }
}

fn is_standard_provider_api_format(api_format: &str) -> bool {
    matches!(
        api_format,
        "openai:chat"
            | "openai:cli"
            | "openai:compact"
            | "claude:chat"
            | "claude:cli"
            | "gemini:chat"
            | "gemini:cli"
    )
}

fn is_standard_chat_client_api_format(api_format: &str) -> bool {
    matches!(api_format, "openai:chat" | "claude:chat" | "gemini:chat")
}

fn is_standard_cli_client_api_format(api_format: &str) -> bool {
    matches!(
        api_format,
        "openai:cli" | "openai:compact" | "claude:cli" | "gemini:cli"
    )
}

#[cfg(test)]
#[path = "../tests_stream.rs"]
mod tests;
