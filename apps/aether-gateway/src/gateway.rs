#[path = "ai_pipeline/mod.rs"]
pub(crate) mod ai_pipeline;
#[path = "api/mod.rs"]
mod api;
#[path = "async_task/mod.rs"]
mod async_task;
#[path = "audit/mod.rs"]
mod audit;
#[path = "auth/mod.rs"]
mod auth;
#[path = "billing_runtime/mod.rs"]
mod billing_runtime;
#[path = "constants.rs"]
mod constants;
#[path = "control/mod.rs"]
mod control;
#[path = "error.rs"]
mod error;
#[path = "execution_runtime/mod.rs"]
mod execution_runtime;
#[path = "executor/mod.rs"]
mod executor;
#[path = "fallback_metrics.rs"]
mod fallback_metrics;
#[path = "gateway_cache/mod.rs"]
mod gateway_cache;
#[path = "gateway_data/mod.rs"]
mod gateway_data;
#[path = "handlers/mod.rs"]
mod handlers;
#[path = "headers.rs"]
mod headers;
#[path = "hooks/mod.rs"]
mod hooks;
#[path = "intent/mod.rs"]
mod intent;
#[path = "maintenance/mod.rs"]
mod maintenance;
#[path = "middleware/mod.rs"]
pub(crate) mod middleware;
#[path = "model_fetch/mod.rs"]
mod model_fetch;
#[path = "provider_transport/mod.rs"]
mod provider_transport;
#[path = "rate_limit.rs"]
mod rate_limit;
#[path = "gateway/router.rs"]
mod router;
#[path = "scheduler/mod.rs"]
mod scheduler;
#[path = "gateway/state.rs"]
mod state;
#[path = "tunnel/mod.rs"]
mod tunnel;
#[path = "usage/mod.rs"]
mod usage;
#[path = "wallet_runtime/mod.rs"]
mod wallet_runtime;

#[path = "gateway/exports.rs"]
pub(crate) mod exports;
pub(crate) use self::exports::*;

use axum::http::header::{HeaderName, HeaderValue};

fn insert_header_if_missing(
    headers: &mut http::HeaderMap,
    key: &'static str,
    value: &str,
) -> Result<(), GatewayError> {
    if headers.contains_key(key) {
        return Ok(());
    }
    let name = HeaderName::from_static(key);
    let value =
        HeaderValue::from_str(value).map_err(|err| GatewayError::Internal(err.to_string()))?;
    headers.insert(name, value);
    Ok(())
}

#[cfg(test)]
#[path = "execution_runtime/tests.rs"]
mod execution_runtime_contract_tests;

#[cfg(test)]
mod tests;
