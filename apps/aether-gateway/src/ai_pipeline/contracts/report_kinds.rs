use super::plan_kinds::{
    CLAUDE_CHAT_SYNC_PLAN_KIND, CLAUDE_CLI_SYNC_PLAN_KIND, GEMINI_CHAT_SYNC_PLAN_KIND,
    GEMINI_CLI_SYNC_PLAN_KIND, OPENAI_CHAT_SYNC_PLAN_KIND, OPENAI_CLI_SYNC_PLAN_KIND,
    OPENAI_COMPACT_SYNC_PLAN_KIND,
};

pub(crate) const OPENAI_CHAT_SYNC_FINALIZE_REPORT_KIND: &str = "openai_chat_sync_finalize";
pub(crate) const CLAUDE_CHAT_SYNC_FINALIZE_REPORT_KIND: &str = "claude_chat_sync_finalize";
pub(crate) const GEMINI_CHAT_SYNC_FINALIZE_REPORT_KIND: &str = "gemini_chat_sync_finalize";
pub(crate) const OPENAI_CLI_SYNC_FINALIZE_REPORT_KIND: &str = "openai_cli_sync_finalize";
pub(crate) const OPENAI_COMPACT_SYNC_FINALIZE_REPORT_KIND: &str = "openai_compact_sync_finalize";
pub(crate) const CLAUDE_CLI_SYNC_FINALIZE_REPORT_KIND: &str = "claude_cli_sync_finalize";
pub(crate) const GEMINI_CLI_SYNC_FINALIZE_REPORT_KIND: &str = "gemini_cli_sync_finalize";
pub(crate) const OPENAI_VIDEO_CREATE_SYNC_FINALIZE_REPORT_KIND: &str =
    "openai_video_create_sync_finalize";
pub(crate) const GEMINI_VIDEO_CREATE_SYNC_FINALIZE_REPORT_KIND: &str =
    "gemini_video_create_sync_finalize";

pub(crate) const OPENAI_CHAT_SYNC_SUCCESS_REPORT_KIND: &str = "openai_chat_sync_success";
pub(crate) const CLAUDE_CHAT_SYNC_SUCCESS_REPORT_KIND: &str = "claude_chat_sync_success";
pub(crate) const GEMINI_CHAT_SYNC_SUCCESS_REPORT_KIND: &str = "gemini_chat_sync_success";
pub(crate) const OPENAI_CLI_SYNC_SUCCESS_REPORT_KIND: &str = "openai_cli_sync_success";
pub(crate) const CLAUDE_CLI_SYNC_SUCCESS_REPORT_KIND: &str = "claude_cli_sync_success";
pub(crate) const GEMINI_CLI_SYNC_SUCCESS_REPORT_KIND: &str = "gemini_cli_sync_success";

pub(crate) const OPENAI_CHAT_STREAM_SUCCESS_REPORT_KIND: &str = "openai_chat_stream_success";
pub(crate) const CLAUDE_CHAT_STREAM_SUCCESS_REPORT_KIND: &str = "claude_chat_stream_success";
pub(crate) const GEMINI_CHAT_STREAM_SUCCESS_REPORT_KIND: &str = "gemini_chat_stream_success";
pub(crate) const OPENAI_CLI_STREAM_SUCCESS_REPORT_KIND: &str = "openai_cli_stream_success";
pub(crate) const CLAUDE_CLI_STREAM_SUCCESS_REPORT_KIND: &str = "claude_cli_stream_success";
pub(crate) const GEMINI_CLI_STREAM_SUCCESS_REPORT_KIND: &str = "gemini_cli_stream_success";

pub(crate) const OPENAI_CHAT_SYNC_ERROR_REPORT_KIND: &str = "openai_chat_sync_error";
pub(crate) const CLAUDE_CHAT_SYNC_ERROR_REPORT_KIND: &str = "claude_chat_sync_error";
pub(crate) const GEMINI_CHAT_SYNC_ERROR_REPORT_KIND: &str = "gemini_chat_sync_error";
pub(crate) const OPENAI_CLI_SYNC_ERROR_REPORT_KIND: &str = "openai_cli_sync_error";
pub(crate) const OPENAI_COMPACT_SYNC_ERROR_REPORT_KIND: &str = "openai_compact_sync_error";
pub(crate) const CLAUDE_CLI_SYNC_ERROR_REPORT_KIND: &str = "claude_cli_sync_error";
pub(crate) const GEMINI_CLI_SYNC_ERROR_REPORT_KIND: &str = "gemini_cli_sync_error";

pub(crate) fn implicit_sync_finalize_report_kind(plan_kind: &str) -> Option<&'static str> {
    match plan_kind {
        OPENAI_CHAT_SYNC_PLAN_KIND => Some(OPENAI_CHAT_SYNC_FINALIZE_REPORT_KIND),
        CLAUDE_CHAT_SYNC_PLAN_KIND => Some(CLAUDE_CHAT_SYNC_FINALIZE_REPORT_KIND),
        GEMINI_CHAT_SYNC_PLAN_KIND => Some(GEMINI_CHAT_SYNC_FINALIZE_REPORT_KIND),
        OPENAI_CLI_SYNC_PLAN_KIND => Some(OPENAI_CLI_SYNC_FINALIZE_REPORT_KIND),
        OPENAI_COMPACT_SYNC_PLAN_KIND => Some(OPENAI_COMPACT_SYNC_FINALIZE_REPORT_KIND),
        CLAUDE_CLI_SYNC_PLAN_KIND => Some(CLAUDE_CLI_SYNC_FINALIZE_REPORT_KIND),
        GEMINI_CLI_SYNC_PLAN_KIND => Some(GEMINI_CLI_SYNC_FINALIZE_REPORT_KIND),
        _ => None,
    }
}

pub(crate) fn core_error_default_client_api_format(report_kind: &str) -> Option<&'static str> {
    match report_kind {
        OPENAI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some("openai:chat"),
        CLAUDE_CHAT_SYNC_FINALIZE_REPORT_KIND => Some("claude:chat"),
        GEMINI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some("gemini:chat"),
        OPENAI_CLI_SYNC_FINALIZE_REPORT_KIND => Some("openai:cli"),
        OPENAI_COMPACT_SYNC_FINALIZE_REPORT_KIND => Some("openai:compact"),
        CLAUDE_CLI_SYNC_FINALIZE_REPORT_KIND => Some("claude:cli"),
        GEMINI_CLI_SYNC_FINALIZE_REPORT_KIND => Some("gemini:cli"),
        _ => None,
    }
}

pub(crate) fn core_error_background_report_kind(report_kind: &str) -> Option<&'static str> {
    match report_kind {
        OPENAI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(OPENAI_CHAT_SYNC_ERROR_REPORT_KIND),
        CLAUDE_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(CLAUDE_CHAT_SYNC_ERROR_REPORT_KIND),
        GEMINI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(GEMINI_CHAT_SYNC_ERROR_REPORT_KIND),
        OPENAI_CLI_SYNC_FINALIZE_REPORT_KIND => Some(OPENAI_CLI_SYNC_ERROR_REPORT_KIND),
        OPENAI_COMPACT_SYNC_FINALIZE_REPORT_KIND => Some(OPENAI_COMPACT_SYNC_ERROR_REPORT_KIND),
        CLAUDE_CLI_SYNC_FINALIZE_REPORT_KIND => Some(CLAUDE_CLI_SYNC_ERROR_REPORT_KIND),
        GEMINI_CLI_SYNC_FINALIZE_REPORT_KIND => Some(GEMINI_CLI_SYNC_ERROR_REPORT_KIND),
        _ => None,
    }
}

pub(crate) fn core_success_background_report_kind(report_kind: &str) -> Option<&'static str> {
    match report_kind {
        OPENAI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(OPENAI_CHAT_SYNC_SUCCESS_REPORT_KIND),
        CLAUDE_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(CLAUDE_CHAT_SYNC_SUCCESS_REPORT_KIND),
        GEMINI_CHAT_SYNC_FINALIZE_REPORT_KIND => Some(GEMINI_CHAT_SYNC_SUCCESS_REPORT_KIND),
        OPENAI_CLI_SYNC_FINALIZE_REPORT_KIND | OPENAI_COMPACT_SYNC_FINALIZE_REPORT_KIND => {
            Some(OPENAI_CLI_SYNC_SUCCESS_REPORT_KIND)
        }
        CLAUDE_CLI_SYNC_FINALIZE_REPORT_KIND => Some(CLAUDE_CLI_SYNC_SUCCESS_REPORT_KIND),
        GEMINI_CLI_SYNC_FINALIZE_REPORT_KIND => Some(GEMINI_CLI_SYNC_SUCCESS_REPORT_KIND),
        _ => None,
    }
}
