mod mutations;
mod quota;
mod reads;

use std::{future::Future, pin::Pin};

use crate::handlers::admin::request::{AdminAppState, AdminRequestContext};
use crate::GatewayError;
use axum::{
    body::{Body, Bytes},
    response::Response,
};

pub(crate) async fn maybe_build_local_admin_endpoints_keys_response(
    state: &AdminAppState<'_>,
    request_context: &AdminRequestContext<'_>,
    request_body: Option<&Bytes>,
) -> Result<Option<Response<Body>>, GatewayError> {
    if let Some(response) = reads::maybe_handle(state, request_context, request_body).await? {
        return Ok(Some(response));
    }

    if let Some(response) = boxed_mutation_response(state, request_context, request_body).await? {
        return Ok(Some(response));
    }

    if let Some(response) = quota::maybe_handle(state, request_context, request_body).await? {
        return Ok(Some(response));
    }

    Ok(None)
}

type EndpointKeyMutationFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<Response<Body>>, GatewayError>> + Send + 'a>>;

fn boxed_mutation_response<'a>(
    state: &'a AdminAppState<'_>,
    request_context: &'a AdminRequestContext<'_>,
    request_body: Option<&'a Bytes>,
) -> EndpointKeyMutationFuture<'a> {
    Box::pin(mutations::maybe_handle(
        state,
        request_context,
        request_body,
    ))
}
