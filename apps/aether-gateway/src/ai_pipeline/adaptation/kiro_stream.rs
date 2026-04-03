use std::collections::BTreeMap;

const CONTEXT_WINDOW_TOKENS: f64 = 200_000.0;
const MAX_THINKING_BUFFER: usize = 1024 * 1024;
const MAX_MESSAGE_SIZE: usize = 16 * 1024 * 1024;
const MAX_BUFFER_SIZE: usize = MAX_MESSAGE_SIZE;
const MAX_ERRORS: usize = 5;
const QUOTE_CHARS: &str = "`\"'\\#!@$%^&*()-_=+[]{};:<>,.?/";

#[derive(Default)]
pub(crate) struct KiroToClaudeCliStreamState {
    decoder: EventStreamDecoder,
    state: KiroClaudeStreamState,
    started: bool,
}

#[derive(Default)]
struct KiroClaudeStreamState {
    model: String,
    thinking_enabled: bool,
    estimated_input_tokens: usize,
    message_id: String,
    output_tokens: usize,
    context_input_tokens: Option<usize>,
    next_block_index: usize,
    open_blocks: BTreeMap<usize, String>,
    text_block_index: Option<usize>,
    thinking_block_index: Option<usize>,
    tool_block_indices: BTreeMap<String, usize>,
    thinking_buffer: String,
    in_thinking_block: bool,
    thinking_extracted: bool,
    strip_thinking_leading_newline: bool,
    has_tool_use: bool,
    stop_reason_override: Option<String>,
    had_error: bool,
    last_content: String,
}

#[derive(Default)]
struct EventStreamDecoder {
    buffer: Vec<u8>,
    error_count: usize,
    stopped: bool,
}

#[derive(Default)]
struct AwsHeaders {
    values: BTreeMap<String, AwsHeaderValue>,
}

enum AwsHeaderValue {
    Ignored,
    String(String),
}

struct AwsEventFrame {
    headers: AwsHeaders,
    payload: Vec<u8>,
}

enum FrameParseError {
    Incomplete,
    Invalid(String),
}

#[path = "kiro_stream/stream/decoder.rs"]
mod decoder;
#[path = "kiro_stream/stream/state.rs"]
mod state;
#[path = "kiro_stream/stream/util.rs"]
mod util;

#[cfg(test)]
#[path = "kiro_stream/stream/tests.rs"]
mod tests;
