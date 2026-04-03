use crate::gateway::ai_pipeline::finalize::common::LocalCoreSyncFinalizeOutcome;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) fn maybe_build_local_gemini_cli_stream_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    super::gemini_cli::maybe_build_local_gemini_cli_stream_sync_response(
        trace_id, decision, payload,
    )
}
