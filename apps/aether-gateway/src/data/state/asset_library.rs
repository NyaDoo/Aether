use std::sync::Arc;

use aether_data_contracts::repository::asset_library::{
    AssetLibraryReadRepository, AssetLibraryWriteRepository, AssetProviderReference,
    AssetProviderReferenceCounts,
};
use aether_data_contracts::DataLayerError;

use super::GatewayDataState;

impl GatewayDataState {
    pub(crate) async fn count_asset_provider_references(
        &self,
        reference: AssetProviderReference<'_>,
    ) -> Result<AssetProviderReferenceCounts, DataLayerError> {
        let repository = self.asset_library_read_repository().ok_or_else(|| {
            DataLayerError::InvalidConfiguration(
                "asset library read repository is unavailable".to_string(),
            )
        })?;
        repository.count_provider_references(reference).await
    }

    pub(crate) fn has_asset_library_reader(&self) -> bool {
        self.asset_library_read_repository().is_some()
    }

    pub(crate) fn has_asset_library_writer(&self) -> bool {
        self.asset_library_write_repository().is_some()
    }

    pub(crate) fn asset_library_read_repository(
        &self,
    ) -> Option<Arc<dyn AssetLibraryReadRepository>> {
        self.backends
            .as_ref()
            .and_then(|backends| backends.read().asset_library())
    }

    pub(crate) fn asset_library_write_repository(
        &self,
    ) -> Option<Arc<dyn AssetLibraryWriteRepository>> {
        self.backends
            .as_ref()
            .and_then(|backends| backends.write().asset_library())
    }

    #[cfg(test)]
    pub(crate) fn with_asset_library_repository_for_tests<T>(repository: Arc<T>) -> Self
    where
        T: AssetLibraryReadRepository + AssetLibraryWriteRepository + 'static,
    {
        Self {
            backends: Some(
                aether_data::DataBackends::with_asset_library_repository_for_tests(repository),
            ),
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aether_data::repository::asset_library::{
        AssetLibraryWriteRepository, InMemoryAssetLibraryRepository, UpsertAssetGroupRecord,
    };

    use super::*;

    fn group_record() -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: "group-1".to_string(),
            upstream_group_id: Some("upstream-group-1".to_string()),
            user_id: "user-1".to_string(),
            api_key_id: None,
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            group_type: "face".to_string(),
            name: "Faces".to_string(),
            description: None,
            status: "active".to_string(),
            created_at_unix_secs: 1,
            updated_at_unix_secs: 1,
            deleted_at_unix_secs: None,
        }
    }

    #[tokio::test]
    async fn provider_reference_preflight_uses_asset_repository() {
        let repository = Arc::new(InMemoryAssetLibraryRepository::default());
        repository
            .upsert_group(group_record())
            .await
            .expect("asset group");
        let data = GatewayDataState::with_asset_library_repository_for_tests(repository);

        assert_eq!(
            data.count_asset_provider_references(AssetProviderReference::ProviderId("provider-1"))
                .await
                .expect("provider references"),
            AssetProviderReferenceCounts {
                asset_groups: 1,
                visual_validation_sessions: 0,
            }
        );
    }

    #[tokio::test]
    async fn provider_reference_preflight_fails_closed_without_repository() {
        let error = GatewayDataState::disabled()
            .count_asset_provider_references(AssetProviderReference::ProviderId("provider-1"))
            .await
            .expect_err("missing repository must fail closed");
        assert!(matches!(error, DataLayerError::InvalidConfiguration(_)));
    }
}
