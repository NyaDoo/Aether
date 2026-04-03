use crate::gateway::ai_pipeline::planner::{
    CLAUDE_CHAT_STREAM_PLAN_KIND, CLAUDE_CHAT_SYNC_PLAN_KIND,
};

use super::super::family::{LocalStandardSourceFamily, LocalStandardSourceMode, LocalStandardSpec};

pub(crate) fn resolve_sync_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        CLAUDE_CHAT_SYNC_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "claude:chat",
            decision_kind: CLAUDE_CHAT_SYNC_PLAN_KIND,
            report_kind: "claude_chat_sync_finalize",
            family: LocalStandardSourceFamily::Standard,
            mode: LocalStandardSourceMode::Chat,
            require_streaming: false,
        }),
        _ => None,
    }
}

pub(crate) fn resolve_stream_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        CLAUDE_CHAT_STREAM_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "claude:chat",
            decision_kind: CLAUDE_CHAT_STREAM_PLAN_KIND,
            report_kind: "claude_chat_stream_success",
            family: LocalStandardSourceFamily::Standard,
            mode: LocalStandardSourceMode::Chat,
            require_streaming: true,
        }),
        _ => None,
    }
}
