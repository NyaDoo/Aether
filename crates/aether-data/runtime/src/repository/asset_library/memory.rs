use std::collections::BTreeMap;
use std::sync::RwLock;

use async_trait::async_trait;

use super::{
    AssetGroupListQuery, AssetLibraryReadRepository, AssetLibraryWriteRepository, AssetListQuery,
    AssetProviderReference, AssetProviderReferenceCounts, ConsumeArkVisualValidationSessionRecord,
    StoredArkVisualValidationSession, StoredAsset, StoredAssetGroup, StoredAssetGroupListPage,
    StoredAssetListPage, UpsertArkVisualValidationSessionRecord, UpsertAssetGroupRecord,
    UpsertAssetRecord,
};
use crate::DataLayerError;

type GroupUpstreamKey = (String, String);
type AssetUpstreamKey = (String, String);
type SessionUpstreamKey = (String, String);

#[derive(Debug, Default)]
struct MemoryAssetLibraryIndex {
    groups_by_id: BTreeMap<String, StoredAssetGroup>,
    group_upstream_to_id: BTreeMap<GroupUpstreamKey, String>,
    assets_by_id: BTreeMap<String, StoredAsset>,
    asset_upstream_to_id: BTreeMap<AssetUpstreamKey, String>,
    sessions_by_id: BTreeMap<String, StoredArkVisualValidationSession>,
    session_upstream_to_id: BTreeMap<SessionUpstreamKey, String>,
    session_byted_token_to_id: BTreeMap<String, String>,
    session_callback_to_id: BTreeMap<String, String>,
}

#[derive(Debug, Default)]
pub struct InMemoryAssetLibraryRepository {
    index: RwLock<MemoryAssetLibraryIndex>,
}

impl InMemoryAssetLibraryRepository {
    fn group_upstream_key(group: &StoredAssetGroup) -> Option<GroupUpstreamKey> {
        group
            .upstream_group_id
            .as_ref()
            .map(|upstream_group_id| (group.provider_id.clone(), upstream_group_id.clone()))
    }

    fn asset_upstream_key(asset: &StoredAsset) -> Option<AssetUpstreamKey> {
        asset
            .upstream_asset_id
            .as_ref()
            .map(|upstream_asset_id| (asset.group_id.clone(), upstream_asset_id.clone()))
    }

    fn session_upstream_key(session: &StoredArkVisualValidationSession) -> SessionUpstreamKey {
        (session.provider_id.clone(), session.session_id.clone())
    }

    fn ensure_unclaimed(
        index_name: &str,
        existing_id: Option<&String>,
        candidate_id: &str,
    ) -> Result<(), DataLayerError> {
        if existing_id.is_some_and(|existing_id| existing_id != candidate_id) {
            return Err(DataLayerError::InvalidInput(format!(
                "{index_name} is already bound to another record"
            )));
        }
        Ok(())
    }

    fn matches_group(group: &StoredAssetGroup, query: &AssetGroupListQuery) -> bool {
        query
            .user_id
            .as_deref()
            .is_none_or(|value| group.user_id == value)
            && query
                .api_key_id
                .as_deref()
                .is_none_or(|value| group.api_key_id.as_deref() == Some(value))
            && query
                .provider_id
                .as_deref()
                .is_none_or(|value| group.provider_id == value)
            && query
                .group_type
                .as_deref()
                .is_none_or(|value| group.group_type == value)
            && query
                .status
                .as_deref()
                .is_none_or(|value| group.status == value)
            && (query.include_deleted || group.deleted_at_unix_secs.is_none())
            && query.search.as_deref().is_none_or(|search| {
                let search = search.to_ascii_lowercase();
                group.id.to_ascii_lowercase().contains(&search)
                    || group.name.to_ascii_lowercase().contains(&search)
                    || group
                        .upstream_group_id
                        .as_deref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&search))
                    || group
                        .description
                        .as_deref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&search))
            })
    }

    fn matches_asset(asset: &StoredAsset, query: &AssetListQuery) -> bool {
        query
            .group_id
            .as_deref()
            .is_none_or(|value| asset.group_id == value)
            && query
                .user_id
                .as_deref()
                .is_none_or(|value| asset.user_id == value)
            && query
                .api_key_id
                .as_deref()
                .is_none_or(|value| asset.api_key_id.as_deref() == Some(value))
            && query
                .asset_type
                .as_deref()
                .is_none_or(|value| asset.asset_type == value)
            && query
                .status
                .as_deref()
                .is_none_or(|value| asset.status == value)
            && (query.include_deleted || !asset.is_deleted)
            && query.search.as_deref().is_none_or(|search| {
                let search = search.to_ascii_lowercase();
                asset.id.to_ascii_lowercase().contains(&search)
                    || asset.name.to_ascii_lowercase().contains(&search)
                    || asset
                        .upstream_asset_id
                        .as_deref()
                        .is_some_and(|value| value.to_ascii_lowercase().contains(&search))
            })
    }
}

#[async_trait]
impl AssetLibraryReadRepository for InMemoryAssetLibraryRepository {
    async fn count_provider_references(
        &self,
        reference: AssetProviderReference<'_>,
    ) -> Result<AssetProviderReferenceCounts, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        let matches_group = |group: &&StoredAssetGroup| match reference {
            AssetProviderReference::ProviderId(id) => group.provider_id == id,
            AssetProviderReference::EndpointId(id) => group.endpoint_id == id,
            AssetProviderReference::KeyId(id) => group.key_id == id,
        };
        let matches_session = |session: &&StoredArkVisualValidationSession| match reference {
            AssetProviderReference::ProviderId(id) => session.provider_id == id,
            AssetProviderReference::EndpointId(id) => session.endpoint_id == id,
            AssetProviderReference::KeyId(id) => session.key_id == id,
        };
        Ok(AssetProviderReferenceCounts {
            asset_groups: index.groups_by_id.values().filter(matches_group).count() as u64,
            visual_validation_sessions: index
                .sessions_by_id
                .values()
                .filter(matches_session)
                .count() as u64,
        })
    }

    async fn find_group_by_id(
        &self,
        group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .groups_by_id
            .get(group_id)
            .cloned())
    }

    async fn find_group_for_user(
        &self,
        group_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .groups_by_id
            .get(group_id)
            .filter(|group| group.user_id == user_id)
            .cloned())
    }

    async fn find_group_by_upstream(
        &self,
        provider_id: &str,
        endpoint_id: &str,
        key_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .groups_by_id
            .values()
            .find(|group| {
                group.provider_id == provider_id
                    && group.endpoint_id == endpoint_id
                    && group.key_id == key_id
                    && group.upstream_group_id.as_deref() == Some(upstream_group_id)
            })
            .cloned())
    }

    async fn find_group_by_canonical_upstream(
        &self,
        provider_id: &str,
        upstream_group_id: &str,
    ) -> Result<Option<StoredAssetGroup>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .group_upstream_to_id
            .get(&(provider_id.to_string(), upstream_group_id.to_string()))
            .and_then(|id| index.groups_by_id.get(id))
            .cloned())
    }

    async fn list_groups(
        &self,
        query: &AssetGroupListQuery,
    ) -> Result<StoredAssetGroupListPage, DataLayerError> {
        let mut items = self
            .index
            .read()
            .expect("asset library repository lock")
            .groups_by_id
            .values()
            .filter(|group| Self::matches_group(group, query))
            .cloned()
            .collect::<Vec<_>>();
        items.sort_by(|left, right| {
            right
                .created_at_unix_secs
                .cmp(&left.created_at_unix_secs)
                .then_with(|| right.updated_at_unix_secs.cmp(&left.updated_at_unix_secs))
                .then_with(|| left.id.cmp(&right.id))
        });
        let total = items.len();
        let items = items
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();
        Ok(StoredAssetGroupListPage { items, total })
    }

    async fn find_asset_by_id(
        &self,
        asset_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .assets_by_id
            .get(asset_id)
            .cloned())
    }

    async fn find_asset_for_user(
        &self,
        asset_id: &str,
        user_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .assets_by_id
            .get(asset_id)
            .filter(|asset| asset.user_id == user_id)
            .cloned())
    }

    async fn find_asset_by_upstream(
        &self,
        group_id: &str,
        upstream_asset_id: &str,
    ) -> Result<Option<StoredAsset>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .asset_upstream_to_id
            .get(&(group_id.to_string(), upstream_asset_id.to_string()))
            .and_then(|id| index.assets_by_id.get(id))
            .cloned())
    }

    async fn list_assets(
        &self,
        query: &AssetListQuery,
    ) -> Result<StoredAssetListPage, DataLayerError> {
        let mut items = self
            .index
            .read()
            .expect("asset library repository lock")
            .assets_by_id
            .values()
            .filter(|asset| Self::matches_asset(asset, query))
            .cloned()
            .collect::<Vec<_>>();
        items.sort_by(|left, right| {
            right
                .created_at_unix_secs
                .cmp(&left.created_at_unix_secs)
                .then_with(|| right.updated_at_unix_secs.cmp(&left.updated_at_unix_secs))
                .then_with(|| left.id.cmp(&right.id))
        });
        let total = items.len();
        let items = items
            .into_iter()
            .skip(query.offset)
            .take(query.limit)
            .collect();
        Ok(StoredAssetListPage { items, total })
    }

    async fn find_visual_validation_session_by_id(
        &self,
        id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .sessions_by_id
            .get(id)
            .cloned())
    }

    async fn find_visual_validation_session_for_user(
        &self,
        id: &str,
        user_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        Ok(self
            .index
            .read()
            .expect("asset library repository lock")
            .sessions_by_id
            .get(id)
            .filter(|session| session.user_id == user_id)
            .cloned())
    }

    async fn find_visual_validation_session_by_upstream(
        &self,
        provider_id: &str,
        key_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .sessions_by_id
            .values()
            .find(|session| {
                session.provider_id == provider_id
                    && session.key_id == key_id
                    && session.session_id == session_id
            })
            .cloned())
    }

    async fn find_visual_validation_session_by_canonical_upstream(
        &self,
        provider_id: &str,
        session_id: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .session_upstream_to_id
            .get(&(provider_id.to_string(), session_id.to_string()))
            .and_then(|id| index.sessions_by_id.get(id))
            .cloned())
    }

    async fn find_visual_validation_session_by_callback_state_hash(
        &self,
        callback_state_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .session_callback_to_id
            .get(callback_state_hash)
            .and_then(|id| index.sessions_by_id.get(id))
            .cloned())
    }

    async fn find_visual_validation_session_by_byted_token_hash(
        &self,
        byted_token_hash: &str,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        let index = self.index.read().expect("asset library repository lock");
        Ok(index
            .session_byted_token_to_id
            .get(byted_token_hash)
            .and_then(|id| index.sessions_by_id.get(id))
            .cloned())
    }
}

#[async_trait]
impl AssetLibraryWriteRepository for InMemoryAssetLibraryRepository {
    async fn upsert_group(
        &self,
        mut record: UpsertAssetGroupRecord,
    ) -> Result<StoredAssetGroup, DataLayerError> {
        record.validate()?;
        let mut index = self.index.write().expect("asset library repository lock");
        if let Some(existing) = index.groups_by_id.get(&record.id) {
            if !record.has_same_immutable_identity(existing) {
                return Err(DataLayerError::InvalidInput(
                    "asset group immutable identity cannot be changed".to_string(),
                ));
            }
            record.created_at_unix_secs = existing.created_at_unix_secs;
        }
        let group = record.into_stored();
        if let Some(key) = Self::group_upstream_key(&group) {
            Self::ensure_unclaimed(
                "asset group upstream id",
                index.group_upstream_to_id.get(&key),
                &group.id,
            )?;
        }
        if let Some(previous) = index.groups_by_id.insert(group.id.clone(), group.clone()) {
            if let Some(key) = Self::group_upstream_key(&previous) {
                index.group_upstream_to_id.remove(&key);
            }
        }
        if let Some(key) = Self::group_upstream_key(&group) {
            index.group_upstream_to_id.insert(key, group.id.clone());
        }
        Ok(group)
    }

    async fn upsert_asset(
        &self,
        mut record: UpsertAssetRecord,
    ) -> Result<StoredAsset, DataLayerError> {
        record.validate()?;
        let mut index = self.index.write().expect("asset library repository lock");
        let Some(group) = index.groups_by_id.get(&record.group_id) else {
            return Err(DataLayerError::InvalidInput(
                "assets.group_id does not reference an asset group".to_string(),
            ));
        };
        if group.deleted_at_unix_secs.is_some() {
            return Err(DataLayerError::InvalidInput(
                "assets.group_id references a deleted asset group".to_string(),
            ));
        }
        if group.user_id != record.user_id {
            return Err(DataLayerError::InvalidInput(
                "assets.user_id must match the asset group owner".to_string(),
            ));
        }
        if let Some(existing) = index.assets_by_id.get(&record.id) {
            if !record.has_same_immutable_identity(existing) {
                return Err(DataLayerError::InvalidInput(
                    "asset immutable identity cannot be changed".to_string(),
                ));
            }
            record.created_at_unix_secs = existing.created_at_unix_secs;
        }
        let asset = record.into_stored();
        if let Some(key) = Self::asset_upstream_key(&asset) {
            Self::ensure_unclaimed(
                "asset upstream id",
                index.asset_upstream_to_id.get(&key),
                &asset.id,
            )?;
        }
        if let Some(previous) = index.assets_by_id.insert(asset.id.clone(), asset.clone()) {
            if let Some(key) = Self::asset_upstream_key(&previous) {
                index.asset_upstream_to_id.remove(&key);
            }
        }
        if let Some(key) = Self::asset_upstream_key(&asset) {
            index.asset_upstream_to_id.insert(key, asset.id.clone());
        }
        Ok(asset)
    }

    async fn soft_delete_group(
        &self,
        group_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        if deleted_at_unix_secs == 0 {
            return Err(DataLayerError::InvalidInput(
                "asset group deletion timestamp is empty".to_string(),
            ));
        }
        let mut index = self.index.write().expect("asset library repository lock");
        let Some(group) = index.groups_by_id.get_mut(group_id) else {
            return Ok(false);
        };
        if group.deleted_at_unix_secs.is_some() {
            return Ok(false);
        }
        group.status = "deleted".to_string();
        group.deleted_at_unix_secs = Some(deleted_at_unix_secs);
        group.updated_at_unix_secs = deleted_at_unix_secs;
        for asset in index
            .assets_by_id
            .values_mut()
            .filter(|asset| asset.group_id == group_id && !asset.is_deleted)
        {
            asset.status = "deleted".to_string();
            asset.is_deleted = true;
            asset.deleted_at_unix_secs = Some(deleted_at_unix_secs);
            asset.updated_at_unix_secs = deleted_at_unix_secs;
        }
        Ok(true)
    }

    async fn soft_delete_asset(
        &self,
        asset_id: &str,
        deleted_at_unix_secs: u64,
    ) -> Result<bool, DataLayerError> {
        if deleted_at_unix_secs == 0 {
            return Err(DataLayerError::InvalidInput(
                "asset deletion timestamp is empty".to_string(),
            ));
        }
        let mut index = self.index.write().expect("asset library repository lock");
        let Some(asset) = index.assets_by_id.get_mut(asset_id) else {
            return Ok(false);
        };
        if asset.is_deleted {
            return Ok(false);
        }
        asset.status = "deleted".to_string();
        asset.is_deleted = true;
        asset.deleted_at_unix_secs = Some(deleted_at_unix_secs);
        asset.updated_at_unix_secs = deleted_at_unix_secs;
        Ok(true)
    }

    async fn upsert_visual_validation_session(
        &self,
        mut record: UpsertArkVisualValidationSessionRecord,
    ) -> Result<StoredArkVisualValidationSession, DataLayerError> {
        record.validate()?;
        let mut index = self.index.write().expect("asset library repository lock");
        if let Some(group_id) = record.group_id.as_deref() {
            let Some(group) = index.groups_by_id.get(group_id) else {
                return Err(DataLayerError::InvalidInput(
                    "validation session group does not exist".to_string(),
                ));
            };
            if group.user_id != record.user_id
                || group.provider_id != record.provider_id
                || group.endpoint_id != record.endpoint_id
                || group.key_id != record.key_id
                || group.project_name != record.project_name
                || group.deleted_at_unix_secs.is_some()
            {
                return Err(DataLayerError::InvalidInput(
                    "validation session group owner or provider binding is invalid".to_string(),
                ));
            }
        }
        if let Some(existing) = index.sessions_by_id.get(&record.id) {
            let immutable_matches = record.has_same_immutable_identity(existing);
            if !immutable_matches {
                return Err(DataLayerError::InvalidInput(
                    "validation session immutable identity cannot be changed".to_string(),
                ));
            }
            if existing.consumed_at_unix_secs.is_some() {
                return Ok(existing.clone());
            }
            record.created_at_unix_secs = existing.created_at_unix_secs;
        }
        let session = record.into_stored();
        let upstream_key = Self::session_upstream_key(&session);
        Self::ensure_unclaimed(
            "visual validation upstream session",
            index.session_upstream_to_id.get(&upstream_key),
            &session.id,
        )?;
        Self::ensure_unclaimed(
            "visual validation callback state",
            index
                .session_callback_to_id
                .get(&session.callback_state_hash),
            &session.id,
        )?;
        Self::ensure_unclaimed(
            "visual validation BytedToken hash",
            index
                .session_byted_token_to_id
                .get(&session.byted_token_hash),
            &session.id,
        )?;
        if let Some(previous) = index
            .sessions_by_id
            .insert(session.id.clone(), session.clone())
        {
            index
                .session_upstream_to_id
                .remove(&Self::session_upstream_key(&previous));
            index
                .session_callback_to_id
                .remove(&previous.callback_state_hash);
            index
                .session_byted_token_to_id
                .remove(&previous.byted_token_hash);
        }
        index
            .session_upstream_to_id
            .insert(upstream_key, session.id.clone());
        index
            .session_callback_to_id
            .insert(session.callback_state_hash.clone(), session.id.clone());
        index
            .session_byted_token_to_id
            .insert(session.byted_token_hash.clone(), session.id.clone());
        Ok(session)
    }

    async fn consume_visual_validation_session(
        &self,
        record: ConsumeArkVisualValidationSessionRecord,
    ) -> Result<Option<StoredArkVisualValidationSession>, DataLayerError> {
        record.validate()?;
        let mut index = self.index.write().expect("asset library repository lock");
        let Some(id) = index
            .session_callback_to_id
            .get(&record.callback_state_hash)
            .cloned()
        else {
            return Ok(None);
        };
        let Some(session) = index.sessions_by_id.get_mut(&id) else {
            return Ok(None);
        };
        if session.consumed_at_unix_secs.is_some()
            || session.expires_at_unix_secs <= record.consumed_at_unix_secs
        {
            return Ok(None);
        }
        session.status = record.status;
        session.consumed_at_unix_secs = Some(record.consumed_at_unix_secs);
        session.sanitized_result = record.sanitized_result;
        session.updated_at_unix_secs = record.updated_at_unix_secs;
        Ok(Some(session.clone()))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn group_record(id: &str, user_id: &str) -> UpsertAssetGroupRecord {
        UpsertAssetGroupRecord {
            id: id.to_string(),
            upstream_group_id: Some(format!("upstream-{id}")),
            user_id: user_id.to_string(),
            api_key_id: Some("api-key-1".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            project_name: "default".to_string(),
            group_type: "face".to_string(),
            name: format!("Group {id}"),
            description: None,
            status: "active".to_string(),
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
            deleted_at_unix_secs: None,
        }
    }

    fn asset_record(id: &str, group_id: &str, user_id: &str) -> UpsertAssetRecord {
        UpsertAssetRecord {
            id: id.to_string(),
            upstream_asset_id: Some(format!("upstream-{id}")),
            group_id: group_id.to_string(),
            user_id: user_id.to_string(),
            api_key_id: Some("api-key-1".to_string()),
            asset_type: "image".to_string(),
            name: format!("Asset {id}"),
            status: "active".to_string(),
            error_code: None,
            error_message: None,
            moderation: Some(json!({"result": "pass"})),
            last_inference_at_unix_secs: None,
            source_url_fingerprint: Some("fingerprint".to_string()),
            provider_url: None,
            provider_url_expires_at_unix_secs: None,
            sanitized_metadata: Some(json!({"width": 100})),
            is_deleted: false,
            deleted_at_unix_secs: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
        }
    }

    fn validation_record() -> UpsertArkVisualValidationSessionRecord {
        UpsertArkVisualValidationSessionRecord {
            id: "validation-1".to_string(),
            session_id: "session-1".to_string(),
            user_id: "user-1".to_string(),
            api_key_id: Some("api-key-1".to_string()),
            provider_id: "provider-1".to_string(),
            endpoint_id: "endpoint-1".to_string(),
            key_id: "key-1".to_string(),
            project_name: "default".to_string(),
            byted_token_hash: "token-hash".to_string(),
            encrypted_byted_token: "encrypted-token".to_string(),
            callback_state_hash: "state-hash".to_string(),
            status: "pending".to_string(),
            expires_at_unix_secs: 100,
            consumed_at_unix_secs: None,
            group_id: Some("group-1".to_string()),
            sanitized_result: None,
            created_at_unix_secs: 10,
            updated_at_unix_secs: 10,
        }
    }

    #[tokio::test]
    async fn stores_and_scopes_groups_and_assets() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_asset(asset_record("asset-1", "group-1", "user-1"))
            .await?;

        assert!(repository
            .find_group_for_user("group-1", "user-1")
            .await?
            .is_some());
        assert!(repository
            .find_group_for_user("group-1", "user-2")
            .await?
            .is_none());
        assert_eq!(
            repository
                .list_assets(&AssetListQuery {
                    group_id: Some("group-1".to_string()),
                    limit: 20,
                    ..AssetListQuery::default()
                })
                .await?
                .total,
            1
        );
        assert!(repository.soft_delete_asset("asset-1", 20).await?);
        assert_eq!(
            repository
                .list_assets(&AssetListQuery {
                    group_id: Some("group-1".to_string()),
                    limit: 20,
                    ..AssetListQuery::default()
                })
                .await?
                .total,
            0
        );
        Ok(())
    }

    #[tokio::test]
    async fn consumes_validation_session_once_before_expiry() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_visual_validation_session(validation_record())
            .await?;
        let consume = ConsumeArkVisualValidationSessionRecord {
            callback_state_hash: "state-hash".to_string(),
            status: "succeeded".to_string(),
            consumed_at_unix_secs: 50,
            sanitized_result: Some(json!({"verified": true})),
            updated_at_unix_secs: 50,
        };

        let consumed = repository
            .consume_visual_validation_session(consume.clone())
            .await?
            .expect("session should be consumed");
        assert_eq!(consumed.status, "succeeded");
        assert_eq!(
            repository
                .find_visual_validation_session_by_byted_token_hash("token-hash")
                .await?
                .expect("BytedToken hash lookup")
                .id,
            "validation-1"
        );
        assert!(repository
            .consume_visual_validation_session(consume)
            .await?
            .is_none());

        let mut retry = validation_record();
        retry.status = "pending".to_string();
        retry.updated_at_unix_secs = 60;
        let stored = repository.upsert_visual_validation_session(retry).await?;
        assert_eq!(stored.status, "succeeded");
        assert_eq!(stored.consumed_at_unix_secs, Some(50));
        Ok(())
    }

    #[tokio::test]
    async fn counts_provider_references_including_soft_deleted_groups() -> Result<(), DataLayerError>
    {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_visual_validation_session(validation_record())
            .await?;
        assert!(repository.soft_delete_group("group-1", 20).await?);

        for reference in [
            AssetProviderReference::ProviderId("provider-1"),
            AssetProviderReference::EndpointId("endpoint-1"),
            AssetProviderReference::KeyId("key-1"),
        ] {
            assert_eq!(
                repository.count_provider_references(reference).await?,
                AssetProviderReferenceCounts {
                    asset_groups: 1,
                    visual_validation_sessions: 1,
                }
            );
        }
        assert_eq!(
            repository
                .count_provider_references(AssetProviderReference::ProviderId("missing"))
                .await?,
            AssetProviderReferenceCounts::default()
        );
        Ok(())
    }

    #[tokio::test]
    async fn rejects_expired_validation_session_consumption() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_visual_validation_session(validation_record())
            .await?;

        assert!(repository
            .consume_visual_validation_session(ConsumeArkVisualValidationSessionRecord {
                callback_state_hash: "state-hash".to_string(),
                status: "succeeded".to_string(),
                consumed_at_unix_secs: 100,
                sanitized_result: Some(json!({"verified": true})),
                updated_at_unix_secs: 100,
            })
            .await?
            .is_none());
        Ok(())
    }

    #[tokio::test]
    async fn rejects_asset_with_different_group_owner() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;

        let error = repository
            .upsert_asset(asset_record("asset-1", "group-1", "user-2"))
            .await
            .expect_err("cross-owner asset must be rejected");
        assert!(error
            .to_string()
            .contains("must match the asset group owner"));
        Ok(())
    }

    #[tokio::test]
    async fn canonical_group_identity_survives_endpoint_and_key_rotation(
    ) -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        let original = group_record("group-1", "user-1");
        repository.upsert_group(original.clone()).await?;

        let mut duplicate = original;
        duplicate.id = "group-2".to_string();
        duplicate.endpoint_id = "endpoint-2".to_string();
        duplicate.key_id = "key-2".to_string();
        assert!(repository.upsert_group(duplicate).await.is_err());
        assert_eq!(
            repository
                .find_group_by_canonical_upstream("provider-1", "upstream-group-1",)
                .await?
                .expect("canonical group")
                .id,
            "group-1"
        );
        Ok(())
    }

    #[tokio::test]
    async fn rejects_local_id_identity_and_tombstone_rewrites() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_group(group_record("group-2", "user-1"))
            .await?;
        repository
            .upsert_asset(asset_record("asset-1", "group-1", "user-1"))
            .await?;

        let mut stolen_group = group_record("group-1", "user-2");
        stolen_group.updated_at_unix_secs = 11;
        assert!(repository.upsert_group(stolen_group).await.is_err());

        let mut moved = asset_record("asset-1", "group-2", "user-1");
        moved.updated_at_unix_secs = 11;
        assert!(repository.upsert_asset(moved).await.is_err());

        let mut mutable_asset = asset_record("asset-1", "group-1", "user-1");
        mutable_asset.name = "Updated name".to_string();
        mutable_asset.status = "processing".to_string();
        mutable_asset.sanitized_metadata = Some(json!({"width": 200}));
        mutable_asset.updated_at_unix_secs = 12;
        let updated = repository.upsert_asset(mutable_asset).await?;
        assert_eq!(updated.name, "Updated name");
        assert_eq!(updated.sanitized_metadata, Some(json!({"width": 200})));

        repository
            .upsert_visual_validation_session(validation_record())
            .await?;
        let mut replaced_session = validation_record();
        replaced_session.callback_state_hash = "other-state".to_string();
        replaced_session.updated_at_unix_secs = 13;
        assert!(repository
            .upsert_visual_validation_session(replaced_session)
            .await
            .is_err());

        repository.soft_delete_asset("asset-1", 20).await?;
        let mut resurrected = asset_record("asset-1", "group-1", "user-1");
        resurrected.updated_at_unix_secs = 21;
        assert!(repository.upsert_asset(resurrected).await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn validation_group_must_match_complete_transport_and_project(
    ) -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;

        let mut wrong_provider = validation_record();
        wrong_provider.provider_id = "provider-2".to_string();
        assert!(repository
            .upsert_visual_validation_session(wrong_provider)
            .await
            .is_err());

        let mut wrong_endpoint = validation_record();
        wrong_endpoint.endpoint_id = "endpoint-2".to_string();
        assert!(repository
            .upsert_visual_validation_session(wrong_endpoint)
            .await
            .is_err());

        let mut wrong_key = validation_record();
        wrong_key.key_id = "key-2".to_string();
        assert!(repository
            .upsert_visual_validation_session(wrong_key)
            .await
            .is_err());

        let mut wrong_project = validation_record();
        wrong_project.project_name = "another-project".to_string();
        assert!(repository
            .upsert_visual_validation_session(wrong_project)
            .await
            .is_err());
        Ok(())
    }

    #[tokio::test]
    async fn soft_deleting_group_hides_its_assets() -> Result<(), DataLayerError> {
        let repository = InMemoryAssetLibraryRepository::default();
        repository
            .upsert_group(group_record("group-1", "user-1"))
            .await?;
        repository
            .upsert_asset(asset_record("asset-1", "group-1", "user-1"))
            .await?;

        assert!(repository.soft_delete_group("group-1", 20).await?);
        assert!(!repository.soft_delete_group("group-1", 21).await?);
        assert_eq!(
            repository
                .list_assets(&AssetListQuery {
                    group_id: Some("group-1".to_string()),
                    limit: 20,
                    ..AssetListQuery::default()
                })
                .await?
                .total,
            0
        );
        let asset = repository
            .find_asset_by_id("asset-1")
            .await?
            .expect("soft-deleted asset remains available for internal lookup");
        assert!(asset.is_deleted);
        assert_eq!(asset.deleted_at_unix_secs, Some(20));
        Ok(())
    }
}
