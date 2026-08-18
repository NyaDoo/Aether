//! Volcengine Ark private asset-library boundary.
//!
//! The provider API is an Action-style POST endpoint.  Keeping the protocol
//! parsing here makes it possible to expose both the native Ark surface and
//! the dashboard's resource API without making either surface responsible for
//! provider credentials or ownership checks.

mod protocol;
mod service;

pub(crate) use protocol::{
    action_from_request, is_asset_library_path, ArkAssetAction, ArkAssetProtocolError,
    ARK_ASSET_API_FORMAT, ARK_ASSET_REQUIRED_CAPABILITY, ARK_ASSET_VERSION,
};

pub(crate) mod protocol_api {
    pub(crate) use super::protocol::{
        build_error_envelope, extract_result, response_status_from_body, sanitize_action_body,
    };
}

pub(crate) use service::{
    maybe_handle_admin_asset_request, maybe_handle_native_asset_request,
    maybe_handle_user_asset_request, project_video_asset_references,
};
