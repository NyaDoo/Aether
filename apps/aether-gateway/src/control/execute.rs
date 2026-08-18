use axum::body::{Body, Bytes};
use axum::http::{HeaderName, HeaderValue, Response};

use crate::constants::CONTROL_EXECUTED_HEADER;
use crate::control::GatewayControlDecision;
use crate::executor::{
    maybe_execute_stream_request, maybe_execute_sync_request, LocalExecutionRequestOutcome,
};
use crate::{AppState, GatewayError};

use super::resolve_execution_runtime_auth_context;

pub(crate) fn allows_control_execute_emergency(decision: &GatewayControlDecision) -> bool {
    decision.is_execution_runtime_candidate()
}

pub(crate) async fn maybe_execute_via_control(
    state: &AppState,
    parts: &http::request::Parts,
    body_bytes: Bytes,
    trace_id: &str,
    decision: Option<&GatewayControlDecision>,
    require_stream: bool,
) -> Result<LocalExecutionRequestOutcome, GatewayError> {
    let Some(decision) = decision else {
        return Ok(LocalExecutionRequestOutcome::NoPath);
    };

    let mut local_decision = decision.clone();
    if let Some(auth_context) = resolve_execution_runtime_auth_context(
        state,
        &local_decision,
        &parts.method,
        &parts.headers,
        &parts.uri,
        trace_id,
    )
    .await?
    {
        local_decision.auth_context = Some(auth_context);
        local_decision.local_auth_rejection = None;
    }

    let response = if require_stream {
        maybe_execute_stream_request(state, parts, &body_bytes, trace_id, Some(&local_decision))
            .await?
    } else {
        maybe_execute_sync_request(state, parts, &body_bytes, trace_id, Some(&local_decision))
            .await?
    };

    Ok(match response {
        LocalExecutionRequestOutcome::Responded(response) => {
            LocalExecutionRequestOutcome::Responded(mark_control_executed(response))
        }
        LocalExecutionRequestOutcome::Exhausted(outcome) => {
            LocalExecutionRequestOutcome::Exhausted(outcome)
        }
        LocalExecutionRequestOutcome::NoPath => LocalExecutionRequestOutcome::NoPath,
    })
}

fn mark_control_executed(mut response: Response<Body>) -> Response<Body> {
    response.headers_mut().insert(
        HeaderName::from_static(CONTROL_EXECUTED_HEADER),
        HeaderValue::from_static("true"),
    );
    response
}
