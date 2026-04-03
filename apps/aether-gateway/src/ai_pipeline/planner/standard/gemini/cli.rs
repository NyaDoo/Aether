use crate::gateway::ai_pipeline::planner::{
    GEMINI_CLI_STREAM_PLAN_KIND, GEMINI_CLI_SYNC_PLAN_KIND,
};

use super::super::family::{LocalStandardSourceFamily, LocalStandardSourceMode, LocalStandardSpec};

pub(crate) fn resolve_sync_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        GEMINI_CLI_SYNC_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "gemini:cli",
            decision_kind: GEMINI_CLI_SYNC_PLAN_KIND,
            report_kind: "gemini_cli_sync_finalize",
            family: LocalStandardSourceFamily::Gemini,
            mode: LocalStandardSourceMode::Cli,
            require_streaming: false,
        }),
        _ => None,
    }
}

pub(crate) fn resolve_stream_spec(plan_kind: &str) -> Option<LocalStandardSpec> {
    match plan_kind {
        GEMINI_CLI_STREAM_PLAN_KIND => Some(LocalStandardSpec {
            api_format: "gemini:cli",
            decision_kind: GEMINI_CLI_STREAM_PLAN_KIND,
            report_kind: "gemini_cli_stream_success",
            family: LocalStandardSourceFamily::Gemini,
            mode: LocalStandardSourceMode::Cli,
            require_streaming: true,
        }),
        _ => None,
    }
}
