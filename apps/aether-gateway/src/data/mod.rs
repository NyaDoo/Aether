pub(crate) mod auth;
pub(crate) mod candidate_selection;
pub(crate) mod candidates;
mod config;
pub(crate) mod decision_trace;
pub(crate) mod state;

#[cfg(test)]
mod tests;

pub(crate) use aether_data::repository::asset_library::{
    AssetProviderReference, AssetProviderReferenceCounts,
};
pub use config::GatewayDataConfig;
pub(crate) use state::GatewayDataState;
