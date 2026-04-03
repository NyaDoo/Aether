use crate::gateway::ai_pipeline::planner::{
    CLAUDE_CLI_STREAM_PLAN_KIND, CLAUDE_CLI_SYNC_PLAN_KIND,
};

use super::super::family::{LocalStandardSourceFamily, LocalStandardSourceMode, LocalStandardSpec};

pub(crate) fn resolve_sync_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        CLAUDE_CLI_SYNC_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "claude:cli",
            decision_kind: CLAUDE_CLI_SYNC_PLAN_KIND,
            report_kind: "claude_cli_sync_finalize",
            family: LocalStandardSourceFamily::Standard,
            mode: LocalStandardSourceMode::Cli,
            require_streaming: false,
        }),
        _ => None,
    }
}

pub(crate) fn resolve_stream_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        CLAUDE_CLI_STREAM_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "claude:cli",
            decision_kind: CLAUDE_CLI_STREAM_PLAN_KIND,
            report_kind: "claude_cli_stream_success",
            family: LocalStandardSourceFamily::Standard,
            mode: LocalStandardSourceMode::Cli,
            require_streaming: true,
        }),
        _ => None,
    }
}
