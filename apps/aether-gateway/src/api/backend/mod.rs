mod admin;
mod internal;
mod oauth;
mod public;

pub(crate) use admin::mount_admin_routes;
pub(crate) use internal::mount_internal_routes;
pub(crate) use oauth::mount_oauth_routes;
pub(crate) use public::{mount_public_support_routes, root_query_targets_action};
