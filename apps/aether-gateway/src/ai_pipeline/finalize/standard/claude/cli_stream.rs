use crate::gateway::ai_pipeline::finalize::common::LocalCoreSyncFinalizeOutcome;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) fn maybe_build_local_claude_cli_stream_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    super::claude_cli::maybe_build_local_claude_cli_stream_sync_response(
        trace_id, decision, payload,
    )
}
