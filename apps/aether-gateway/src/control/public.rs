use axum::http::Uri;

use crate::constants::REALTIME_ADMISSION_ID_HEADER;
use crate::{AppState, GatewayError};

use super::{resolve_control_route, GatewayControlDecision};

pub(crate) type GatewayPublicRequestContext =
    aether_gateway_control::PublicRequestContext<GatewayControlDecision>;

pub(crate) async fn resolve_public_request_context(
    state: &AppState,
    method: &http::Method,
    uri: &Uri,
    headers: &http::HeaderMap,
    trace_id: &str,
) -> Result<GatewayPublicRequestContext, GatewayError> {
    let control_decision = resolve_control_route(state, method, uri, headers, trace_id).await?;
    let mut context = GatewayPublicRequestContext::from_request_parts(
        trace_id,
        method,
        uri,
        headers,
        control_decision,
    );
    // The outer frontdoor creates this value afresh for every HTTP request.
    // Carry it through the request context so every response/error path can
    // settle the same marker without relying on a caller-controlled trace ID.
    context.realtime_admission_id = headers
        .get(REALTIME_ADMISSION_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Ok(context)
}
