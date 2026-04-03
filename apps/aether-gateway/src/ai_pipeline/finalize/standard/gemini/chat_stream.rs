use serde_json::Value;

use crate::gateway::ai_pipeline::finalize::common::LocalCoreSyncFinalizeOutcome;
use crate::gateway::{GatewayControlDecision, GatewayError, GatewaySyncReportRequest};

pub(crate) fn aggregate_gemini_stream_sync_response(body: &[u8]) -> Option<Value> {
    super::gemini_chat::aggregate_gemini_stream_sync_response(body)
}

pub(crate) fn maybe_build_local_gemini_stream_sync_response(
    trace_id: &str,
    decision: &GatewayControlDecision,
    payload: &GatewaySyncReportRequest,
) -> Result<Option<LocalCoreSyncFinalizeOutcome>, GatewayError> {
    super::gemini_chat::maybe_build_local_gemini_stream_sync_response(trace_id, decision, payload)
}
