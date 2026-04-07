mod api_keys;
mod ldap;
mod oauth_config;
mod oauth_routes;
mod security;

pub(crate) use self::api_keys::maybe_build_local_admin_api_keys_response;
pub(crate) use self::ldap::maybe_build_local_admin_ldap_response;
pub(crate) use self::oauth_config::{
    build_admin_oauth_provider_payload, build_admin_oauth_supported_types_payload,
    build_admin_oauth_upsert_record,
};
pub(crate) use self::oauth_routes::maybe_build_local_admin_oauth_response;
pub(crate) use self::security::maybe_build_local_admin_security_response;
