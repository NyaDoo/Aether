use crate::handlers::admin::request::{AdminRouteRequest, AdminRouteResult};

pub(crate) async fn maybe_build_local_admin_material_assets_response(
    request: AdminRouteRequest<'_>,
) -> AdminRouteResult {
    let state = request.state();
    let request_context = request.request_context();
    Ok(crate::material_assets::maybe_handle_admin_asset_request(
        state.app(),
        request_context.public(),
        request.request_headers(),
        request.request_body(),
    )
    .await)
}
